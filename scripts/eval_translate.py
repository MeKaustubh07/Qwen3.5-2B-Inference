import argparse
import hashlib
import json
import math
import os
import queue
import random
import re
import struct
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

IN22 = dict(repo="ai4bharat/IN22-Gen", revision="e042ab3d3063110b1a85efa0a59bdbf8553bb928",
            file="data/train-00000-of-00001.parquet",
            sha256="e79c7889abe97c78cf03c09f04e50874b7932b381b49a08a4431e58aefc5fb5c")
DATA, RUNS = ROOT / "models/eval/in22-gen", ROOT / "build/eval_translate"
OUT = ROOT / "docs/bench/raw/sarvam_translate"
ENG = "eng"
NAMES = {"eng": "English", "asm": "Assamese", "ben": "Bengali", "brx": "Bodo", "doi": "Dogri", "gom": "Konkani",
         "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada", "kas": "Kashmiri", "mai": "Maithili", "mal": "Malayalam",
         "mar": "Marathi", "mni": "Manipuri", "npi": "Nepali", "ory": "Odia", "pan": "Punjabi", "san": "Sanskrit",
         "sat": "Santali", "snd": "Sindhi", "tam": "Tamil", "tel": "Telugu", "urd": "Urdu"}
LANGS = [c for c in NAMES if c != ENG]
SCRIPT_PREF = {"mni": ("Mtei", "Meitei Mayek"), "snd": ("Deva", "Devanagari"), "kas": ("Arab", "Perso-Arabic"),
               "sat": ("Olck", "Ol Chiki")}    # written in more than one script: IN22-Gen's column, named in the prompt
SCRIPTS = {"Latn": "LATIN", "Deva": "DEVANAGARI", "Beng": "BENGALI", "Guru": "GURMUKHI", "Gujr": "GUJARATI",
           "Orya": "ORIYA", "Taml": "TAMIL", "Telu": "TELUGU", "Knda": "KANNADA", "Mlym": "MALAYALAM",
           "Arab": "ARABIC", "Mtei": "MEETEI MAYEK", "Olck": "OL CHIKI"}           # Unicode name prefixes
LANG_COL = re.compile(r"(?:sentence_)?([a-z]{3})_([A-Z][a-z]{3})")
META, STRATA = ("id", "domain", "source", "bucket", "num_words"), ("domain", "source")
CONFIGS = {   # name: (registry model, backend, INT4 policy its weights must have: "calibrated" | "flat" | None)
    "sarvam-int8": ("sarvam-translate", "metal-int8", None),
    "sarvam-int4": ("sarvam-translate", "metal-int4", "calibrated"),
    "sarvam-int4-flat": ("sarvam-translate", "metal-int4", "flat"),
    "aya-int8": ("tiny-aya-global", "metal-int8", None),
    "aya-int4": ("tiny-aya-global", "metal-int4", "calibrated"),
    "qwen-int8": ("qwen3.5-2b", "metal-int8", None),
    "qwen-int4": ("qwen3.5-2b", "metal-int4", "calibrated")}
QT_NAME = {None: "model.int8.qt", "calibrated": "model.int4.qt", "flat": "model.int4-flat.qt"}   # by policy
CARD_PROMPT = {"sarvam-translate"}
SUPPORT = {"sarvam-translate": set(NAMES),            # the card's list; Tiny Aya's README; Qwen3.5: 201, no list
           "tiny-aya-global": {"eng", "hin", "mar", "ben", "guj", "pan", "tam", "tel", "npi", "urd"}}
MAX_NEW = [32, 4.0]
PACK_TOKENS = 1024                                     # scored tokens per packed forward


def sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def script_share(text: str, script: str) -> float:
    """Of the letters and marks in text, the share in `script` (1.0 for no letters)."""
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    return sum(unicodedata.name(c, "").startswith(SCRIPTS[script]) for c in letters) / len(letters) if letters else 1.0


def read_jsonl(path: Path) -> list[dict]:
    """A results file's complete records: a last line without its newline (being written, or torn) is skipped."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    return [json.loads(line) for line in raw[:raw.rfind(b"\n") + 1].decode("utf-8").splitlines() if line]


def repair_jsonl(path: Path) -> None:
    """Cut a torn last line (the writer died mid-line) so appends stay clean; only the writer calls this."""
    if (raw := path.read_bytes()) and not raw.endswith(b"\n"):
        os.truncate(path, raw.rfind(b"\n") + 1)


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- data ---------------------------------------------------------------------------------------------------------

def write_dataset(rows: list[dict], out: Path, source: dict) -> dict:
    """IN22-Gen rows (a column per language) -> out/in22_gen.jsonl ({id, domain, ..., text: {639-3 code: text}}, texts
    trimmed) and out/manifest.json (each language's column, its script checked on the text, SHA-256s, strata)."""
    if "id" not in rows[0]:                            # IN22-Gen has none: the row's place in the pinned parquet
        rows = [{"id": i} | r for i, r in enumerate(rows)]
    cols ={m.group(1, 2): c for c in rows[0] if (m := LANG_COL.fullmatch(c))}
    langs = {}
    for code in NAMES:
        if not (scripts := [s for (lang, s) in cols if lang == code]):
            raise ValueError(f"no {code} column among {sorted(rows[0])}")
        pick = SCRIPT_PREF[code][0] if code in SCRIPT_PREF else scripts[0]
        if pick not in scripts or (len(scripts) > 1 and code not in SCRIPT_PREF):
            raise ValueError(f"{code}: columns in {scripts}, the prompts name {pick}: set SCRIPT_PREF")
        texts = [(r[cols[code, pick]] or "").strip() for r in rows]
        langs[code] = dict(column=cols[code, pick], script=pick, other_scripts=sorted(set(scripts) - {pick}),
                           script_share=round(script_share(" ".join(texts), pick), 4), empty=texts.count(""))
    items = [{k: r[k] for k in META if k in r} | {"text": {c: (r[v["column"]] or "").strip()
                                                           for c, v in langs.items()}} for r in rows]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "in22_gen.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    strata = Counter("/".join(str(it.get(s)) for s in STRATA) for it in items)
    manifest = dict(source=source, rows=len(items), columns=list(rows[0]), languages=langs, strata=dict(sorted(
        strata.items())), jsonl="in22_gen.jsonl", jsonl_sha256=sha256(out / "in22_gen.jsonl"))
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    return manifest


def cmd_data(a) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    try:
        path = hf_hub_download(IN22["repo"], IN22["file"], repo_type="dataset", revision=IN22["revision"],
                               local_dir=a.data)
    except GatedRepoError:
        sys.exit(f"{IN22['repo']} is gated: accept its terms at https://huggingface.co/datasets/{IN22['repo']} "
                 "(approved automatically) with the account of your HF token, then rerun")
    if (got := sha256(path)) != IN22["sha256"]:
        sys.exit(f"{path}: SHA-256 {got}, pinned {IN22['sha256']}")
    import pyarrow.parquet as pq
    m = write_dataset(pq.read_table(path).to_pylist(), Path(a.data), IN22)
    print(json.dumps(m["languages"], indent=1) + f"\n{m['rows']} rows, strata {m['strata']}\nwrote {a.data}/"
          f"in22_gen.jsonl (sha256 {m['jsonl_sha256']}) and manifest.json")


def load_data(data: str) -> tuple[dict[int, dict], dict]:
    d = Path(data)
    if not (d / "manifest.json").exists():
        sys.exit(f"no dataset in {d}: run `eval_translate.py data` first")
    manifest = json.load(open(d / "manifest.json"))
    if sha256(d / manifest["jsonl"]) != manifest["jsonl_sha256"]:
        sys.exit(f"{d / manifest['jsonl']} does not match its manifest's SHA-256")
    return {it["id"]: it for it in read_jsonl(d / manifest["jsonl"])}, manifest


def sample_ids(items: dict[int, dict], n: int, seed: int) -> list[int]:
    """The first n of one seeded order of the ids in which every stratum is spread evenly (see the module doc)."""
    rng, cells, keyed = random.Random(seed), defaultdict(list), []
    for i, it in sorted(items.items()):
        cells[tuple(str(it.get(s)) for s in STRATA)].append(i)
    for key in sorted(cells):
        rng.shuffle(ids := cells[key])
        u = rng.random()
        keyed += [((k + u) / len(ids), key, i) for k, i in enumerate(ids)]
    return [i for *_, i in sorted(keyed)][:n]


def pairs(items: dict, ids: list[int], langs: list[str], done_in):
    """-> (src, tgt, path, ids to do) per direction, English -> X first; done_in(src, tgt) -> (path, ids done).
    Items without text on both sides are skipped (the manifest counts empty texts)."""
    for src, tgt in [(ENG, x) for x in langs] + [(x, ENG) for x in langs]:
        path, done = done_in(src, tgt)
        yield src, tgt, path, [i for i in ids if i not in done and items[i]["text"][src] and items[i]["text"][tgt]]


# --- models -------------------------------------------------------------------------------------------------------

def label(code: str) -> str:
    return f"{NAMES[code]} ({SCRIPT_PREF[code][1]} script)" if code in SCRIPT_PREF else NAMES[code]


def messages(style: str, source: str, target: str, text: str) -> list[dict]:
    """The chat for one translation; style "card" (Sarvam-Translate's) or "neutral"; source / target: language names."""
    if style == "card":
        return [{"role": "system", "content": f"Translate the text below to {target}."},
                {"role": "user", "content": text}]
    return [{"role": "user", "content": f"Translate the following text from {source} to {target}. Reply with the "
                                        f"translation only.\n\n{text}"}]


def prompt_style(model: str, override: str | None = None) -> str:
    return override or ("card" if model in CARD_PROMPT else "neutral")


def prompt(eng, src: str, tgt: str, text: str, style: str | None = None) -> str:
    """The rendered prompt (the template writes BOS); thinking off: Qwen3.5 gets an empty think block. The card's
    prompt names the target as the card does; the neutral one adds the script where there is a choice."""
    from chat import format_chat
    style = prompt_style(eng.name, style)
    names = (NAMES[src], NAMES[tgt]) if style == "card" else (label(src), label(tgt))
    return format_chat(messages(style, *names, text), style=eng.chat_style, enable_thinking=False)


def end_of_answer(eng) -> int:
    """The token the chat template writes right after an assistant's answer (one of the model's stops)."""
    from chat import format_chat
    mark = "\ue000"                                  # a private-use character: in no template
    s = format_chat([{"role": "user", "content": "u"}, {"role": "assistant", "content": mark}],
                    add_generation_prompt=False, style=eng.chat_style)
    if not (ids := eng.tokenizer.encode(s[s.index(mark) + 1:])) or ids[0] not in eng.eos_ids:
        sys.exit(f"{eng.name}: the template ends an answer with {ids[:2]}, not one of the stops {sorted(eng.eos_ids)}")
    return ids[0]


def added_tokens(model: str) -> tuple[set[int], set[int]]:
    """A tokenizer's added tokens: those flagged special (skip_special_tokens drops them from an output), and the
    others that are not whitespace (kept, as there: Gemma 3's HTML tags and unused slots, Qwen3.5's <think>)."""
    from engine import MODELS
    spec = json.load(open(ROOT / MODELS[model]["dir"] / "tokenizer.json", encoding="utf-8"))["added_tokens"]
    return ({t["id"] for t in spec if t["special"]},
            {t["id"] for t in spec if not t["special"] and t["content"].replace("\u2581", " ").strip()})


def qt_header(path) -> tuple[str, dict]:
    """A .qt file's header (every tensor's name, scheme, shape and offsets) and its SHA-256."""
    with open(path, "rb") as f:
        raw = f.read(struct.unpack("<Q", f.read(8))[0])
    return hashlib.sha256(raw).hexdigest(), json.loads(raw)


def weights_identity(config: str, weights: str | None) -> dict:
    """What the config's weights are: the .qt by its header, or quantized at load (with the policy file, if any).
    An INT4 file is calibrated if it keeps a tensor at INT8 beyond those every INT4 file keeps (quant.scheme_for: the
    tied embedding). Refuses a file of another scheme, a calibrated config whose weights are flat, and the reverse."""
    from engine import MODELS
    from quant import scheme_for
    model, backend, policy = CONFIGS[config]
    if weights:
        head_sha, head = qt_header(weights)
        if (scheme := head.pop("__metadata__")["scheme"]) != backend.removeprefix("metal-"):
            sys.exit(f"{weights} is {scheme}, {config} runs {backend}")
        q = {k.split("::")[0]: v["scheme"] for k, v in head.items() if "scheme" in v}
        kept = sum(s == "int8" and scheme_for(n, "int4") == "int4" for n, s in q.items())
        w = dict(file=Path(weights).name, bytes=Path(weights).stat().st_size, header_sha256=head_sha,
                 schemes=Counter(q.values()), policy_int8=kept if scheme == "int4" else None)
        mixed, what = kept > 0, Path(weights).name
    else:
        p = MODELS[model].get("policy") if backend == "metal-int4" else None
        p = p if p and (ROOT / p).exists() else None
        w = dict(file=None, policy=p, policy_sha256=p and sha256(ROOT / p))
        mixed, what = p is not None, f"quantizing at load ({p or 'no policy file'})"
    if policy and mixed != (policy == "calibrated"):
        sys.exit(f"{config} needs {policy} INT4 weights, but {what} is {'calibrated' if mixed else 'flat'}: "
                 + (f"write {QT_NAME['flat']} with scripts/quantize.py --scheme int4 and no --policy" if mixed else
                    "write the policy with scripts/calibrate_quant.py and quantize with --policy"))
    return w


def code_version() -> dict:
    """What made a run segment: git describe, and a SHA-256 of the code that makes the outputs (src/, this script)."""
    h = hashlib.sha256()
    for f in sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]:
        h.update(str(f.relative_to(ROOT)).encode() + b"\0" + f.read_bytes())
    git = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True)
    return dict(git=git.stdout.strip(), code_sha256=h.hexdigest()[:16])


def open_run(a, kind: str, manifest: dict):
    """-> (engine, run directory, (src, tgt) -> (results file, ids done)). Results of another identity (weights,
    policy, prompts, stops, limits, seed, dataset) in the directory are refused; torn lines are cut."""
    from engine import MODELS, load_engine
    model, backend, policy = CONFIGS[a.config]
    qt = ROOT / MODELS[model]["dir"] / QT_NAME[policy]
    weights = a.weights or (str(qt) if backend.startswith("metal-int") and qt.exists() else None)
    w = weights_identity(a.config, weights)                       # before loading: a refusal costs nothing
    eng = load_engine(model, backend, weights)
    style = prompt_style(model, a.prompt)
    identity = json.loads(json.dumps(dict(
        config=a.config, model=model, backend=backend, weights=w, eos=sorted(eng.eos_ids),
        end_of_answer=end_of_answer(eng), max_new=MAX_NEW, prompt=style,
        messages=messages(style, "{Source}", "{Target}", "{text}"),
        example_prompt=prompt(eng, ENG, "mni", "{text}", style), seed=a.seed, dataset=manifest["jsonl_sha256"])))
    run = Path(a.runs) / (a.config + ("" if style == prompt_style(model) else f"@{style}"))
    run.mkdir(parents=True, exist_ok=True)
    if (meta := run / "meta.json").exists() and (old := json.load(open(meta))) != identity:
        diff = {k: old.get(k) for k in identity if old.get(k) != identity[k]}
        sys.exit(f"{run} holds results of another build ({diff}); move it away first")
    json.dump(identity, open(meta, "w"), indent=1, ensure_ascii=False)
    for f in run.rglob("*.jsonl"):
        repair_jsonl(f)

    def done_in(src, tgt):
        path = run / kind / f"{src}-{tgt}.jsonl"
        return path, {r["id"] for r in read_jsonl(path)}
    return eng, run, done_in


def run_batched(eng, jobs: list, batch: int, kv_blocks: int, prefix_cache: bool, on_done) -> dict:
    """jobs [(key, prompt ids, max_new)] through the server's scheduler, `batch` in flight; on_done(key, ids, finish,
    ttft) as each finishes. -> the scheduler's facts (preemptions, prefix cache)."""
    from sampler import SamplingParams
    from server.metrics import Metrics
    from server.scheduler import Scheduler
    metrics, events = Metrics(), queue.Queue()
    sched = Scheduler(eng, metrics, max_batch=batch, max_waiting=batch + 1, kv_blocks=kv_blocks,
                      max_model_len=eng.max_model_len or 8192, prefix_cache=prefix_cache)
    greedy, pending, live = SamplingParams(temperature=0.0, top_k=1, top_p=1.0), iter(jobs), {}

    def submit_next() -> None:
        if (job := next(pending, None)) is not None:     # its channel tags events with the job's key
            out = SimpleNamespace(put=lambda ev, key=job[0]: events.put((key, ev)))
            live[job[0]] = sched.submit(job[1], greedy, job[2], out=out)
    try:
        for _ in range(batch):
            submit_next()
        while live:
            key, (kind, payload) = events.get()
            if kind == "token":
                continue
            req = live.pop(key)
            if kind == "error":
                raise RuntimeError(f"{key}: {payload}")
            on_done(key, req.generated, req.finish_reason,
                    req.first_token_at - req.arrived if req.first_token_at is not None else None)
            submit_next()
    finally:
        sched.shutdown(drain_timeout=0)
    return dict(preempted=metrics.counters["requests_preempted_total"], prefix_cache=sched.prefix_note)


def cmd_generate(a) -> None:
    import torch
    from bench import footprint_mib
    items, manifest = load_data(a.data)
    eng, run, done_in = open_run(a, "gen", manifest)
    tok, cap, jobs = eng.tokenizer, eng.max_model_len or 8192, []
    special, added = added_tokens(eng.name)
    for src, tgt, path, todo in pairs(items, sample_ids(items, a.n, a.seed), a.langs, done_in):
        for i in todo:
            text = items[i]["text"][src]
            ids = tok.encode(prompt(eng, src, tgt, text, a.prompt))
            jobs.append(((path, i), ids, min(MAX_NEW[0] + math.ceil(MAX_NEW[1] * len(text)), cap - len(ids))))
    print(f"{a.config}: {len(jobs)} translations to make", flush=True)
    if not jobs:
        return
    totals, t0 = Counter(), time.perf_counter()

    def on_done(key, out, finish, ttft):
        keep = [t for t in out if t not in special]
        hyp = tok.decode(keep).strip()
        append_jsonl(key[0], dict(id=key[1], hyp=hyp, tokens=len(out), stop=finish, ttft=ttft,
                                  special=len(out) - len(keep), added=sum(t in added for t in keep)))
        totals.update(requests=1, tokens=len(out), chars=len(hyp))
        if totals["requests"] % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {totals['requests']}/{len(jobs)}  {totals['tokens'] / el:.1f} tok/s  [{el:.0f} s]", flush=True)
    facts = run_batched(eng, jobs, a.batch, a.kv_blocks, not a.no_prefix_cache, on_done)
    secs, gpu = time.perf_counter() - t0, None
    if eng.model.b.device.type == "mps":                 # the allocator's peak + what Metal holds outside it now
        torch.mps.synchronize()                          # (tests/serving_targets.py gpu_peak)
        A = torch.accelerator
        gpu = A.max_memory_reserved() + max(0, torch.mps.driver_allocated_memory() - A.memory_reserved())
    weights = sum(w.nbytes if hasattr(w, "scheme") else w.numel() * w.element_size()    # as bench.py counts them
                  for w in eng.model._cache.values())
    append_jsonl(run / "speed.jsonl", dict(
        totals, secs=secs, batch=a.batch, kv_blocks=a.kv_blocks, weights_bytes=weights, gpu_peak_bytes=gpu,
        footprint_peak_mib=footprint_mib()[1], date=time.strftime("%Y-%m-%d %H:%M"), **facts))
    append_jsonl(run / "segments.jsonl", dict(kind="gen", items=len(jobs), date=time.strftime("%Y-%m-%d %H:%M"),
                                              **code_version()))
    print(f"{a.config}: {totals['tokens'] / secs:.1f} tok/s, {totals['chars'] / secs:.0f} chars/s over {secs:.0f} s; "
          f"{facts}", flush=True)


def teacher_forced(model, seqs: list, vocab: int):
    """seqs [(key, prompt ids, continuation ids)] -> yields (key, -ln P(token | prompt, the tokens before it) for each
    continuation token, float64). The prompts' common
    prefix (a chat template's preamble) is prefilled once and forked for each sequence; the rest runs packed."""
    import torch
    from decision import FORK_BUDGET, PREFILL_CHUNK, _fork_bytes, _token_logprobs
    first = [p for _, p, _ in seqs]
    common = next((i for i, col in enumerate(zip(*first)) if len(set(col)) > 1), len(first[0]))
    common = min(common, min(map(len, first)) - 1)             # each sequence keeps its last prompt token
    base = model.new_state(common + max(len(p) - common + len(c) for _, p, c in seqs))
    for i in range(0, common, PREFILL_CHUNK):
        model.forward(torch.tensor(first[0][i:min(i + PREFILL_CHUNK, common)]), state=base, last_only=True)
    per_pack = max(1, FORK_BUDGET // _fork_bytes(base))

    def run(pack):
        h = model.packed_hidden([(torch.tensor(p[common:] + c[:-1]), base.fork()) for _, p, c in pack])
        rows, off = [], 0
        for _, p, c in pack:                                    # row j predicts the chunk's token j + 1
            rows += range(off + len(p) - common - 1, off + len(p) - common - 1 + len(c))
            off += len(p) - common + len(c) - 1
        lp = _token_logprobs(model, h[torch.tensor(rows, device=h.device)], [t for *_, c in pack for t in c], vocab)
        ends = np.cumsum([len(c) for *_, c in pack]).tolist()
        yield from ((key, -lp[e - len(c):e].double().numpy()) for (key, _, c), e in zip(pack, ends))
    pack, n = [], 0
    for s in seqs:
        size = len(s[1]) - common + len(s[2])
        if pack and (len(pack) == per_pack or n + size > PACK_TOKENS):
            yield from run(pack)
            pack, n = [], 0
        pack.append(s)
        n += size
    if pack:
        yield from run(pack)


def cmd_score(a) -> None:
    items, manifest = load_data(a.data)
    eng, run, done_in = open_run(a, "score", manifest)
    tok, t0, eoa, made = eng.tokenizer, time.perf_counter(), end_of_answer(eng), 0
    for src, tgt, path, todo in pairs(items, sample_ids(items, a.n, a.seed), a.langs, done_in):
        seqs = [(i, tok.encode(prompt(eng, src, tgt, items[i]["text"][src], a.prompt)),
                 tok.encode(items[i]["text"][tgt]) + [eoa]) for i in todo]
        for (i, p, c), (_, nats) in zip(seqs, teacher_forced(eng.model, seqs, tok.vocab_size()) if seqs else ()):
            ref = items[i]["text"][tgt]
            append_jsonl(path, dict(id=i, prompt_tokens=len(p), ref_tokens=len(c) - 1, nats=float(nats[:-1].sum()),
                                    end_nats=float(nats[-1]), chars=len(ref), roundtrip=tok.decode(c[:-1]) == ref))
        made += len(seqs)
        if seqs:
            rs = read_jsonl(path)
            print(f"{src}-{tgt}: {sum(r['nats'] for r in rs) / math.log(2) / sum(r['chars'] for r in rs):.3f} "
                  f"bits/char ({len(rs)} references) [{time.perf_counter() - t0:.0f} s]", flush=True)
    if made:
        append_jsonl(run / "segments.jsonl", dict(kind="score", items=made, date=time.strftime("%Y-%m-%d %H:%M"),
                                                  **code_version()))


def main(argv=None) -> None:
    split = lambda s: s.split(",")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=str(DATA))
    common.add_argument("--runs", default=str(RUNS))
    common.add_argument("--seed", type=int, default=0)
    model = argparse.ArgumentParser(add_help=False, parents=[common])
    model.add_argument("config", choices=list(CONFIGS))
    model.add_argument("--langs", type=split, default=LANGS)
    model.add_argument("--weights", help="a .qt file in place of the model's own")
    model.add_argument("--prompt", choices=("card", "neutral"), help="a control: another prompt than the model's")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", parents=[common])
    gen = sub.add_parser("generate", parents=[model])
    gen.add_argument("--n", type=int, default=50)
    gen.add_argument("--batch", type=int, default=8)
    gen.add_argument("--kv-blocks", type=int, default=384)
    gen.add_argument("--no-prefix-cache", action="store_true")
    sub.add_parser("score", parents=[model]).add_argument("--n", type=int, default=200)
    rep = sub.add_parser("report", parents=[common])
    rep.add_argument("--configs", type=split)
    rep.add_argument("--baseline", help="the config deltas are taken against (default sarvam-int8)")
    rep.add_argument("--boot", type=int, default=1000)
    rep.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if bad := set(getattr(a, "langs", ())) - set(LANGS):
        ap.error(f"unknown languages {sorted(bad)}; known: {','.join(LANGS)}")
    if a.cmd == "report":
        from eval_translate_report import cmd_report
        return cmd_report(a)
    {"data": cmd_data, "generate": cmd_generate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
