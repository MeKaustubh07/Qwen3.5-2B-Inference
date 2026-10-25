import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
sys.path.insert(0, "scripts")
import decision
import engine
import eval_translate as ev
import eval_translate_report as report
from sacrebleu.metrics import CHRF
from serving_targets import TEMPLATE, train_tokenizer
from tokenizer import Tokenizer

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


# synthetic text: English words, and the same words spelled letter for letter in each language's script
def letters(script: str) -> list[str]:
    import unicodedata
    blocks = {"Latn": (0x61, 0x7B), "Arab": (0x620, 0x64B), "Mtei": (0xABC0, 0xABDB), "Olck": (0x1C5A, 0x1C78)}
    lo, hi = blocks.get(script, (0x900, 0xD80))
    return [chr(c) for c in range(lo, hi) if unicodedata.category(chr(c)) == "Lo"
            and unicodedata.name(chr(c), "").startswith(ev.SCRIPTS[script] + " LETTER")]


WORDS = "the river city school market rain festival court train doctor music field".split()
SCRIPT_OF = {"eng": "Latn", "asm": "Beng", "ben": "Beng", "guj": "Gujr", "kan": "Knda", "kas": "Arab", "mal": "Mlym",
             "mni": "Mtei", "ory": "Orya", "pan": "Guru", "sat": "Olck", "tam": "Taml", "tel": "Telu", "urd": "Arab"}
DOMAINS = ["culture", "economy", "education", "entertainment", "geography", "governments", "health", "industry",
           "legal", "news", "religion", "sports", "tourism"]


def spell(text: str, script: str) -> str:
    al = letters(script)
    return "".join(al[(ord(ch) - 97) % len(al)] if ch.isalpha() else ch for ch in text)


rng = np.random.default_rng(0)
rows = []
for k in range(104):                                   # 13 domains x 2 sources x 4
    en = " ".join(rng.choice(WORDS, size=int(rng.integers(3, 6)))).capitalize() + "."
    row = dict(id=k + 1, context="ctx", source=("web", "wikipedia")[k % 2], url="", domain=DOMAINS[k // 2 % 13],
               num_words=len(en.split()), bucket="1 - 10")
    for code in ev.NAMES:
        script = SCRIPT_OF.get(code, "Deva")
        row[f"sentence_{code}_{script}"] = en if code == "eng" else spell(en.lower(), script)
    row["sentence_mni_Beng"] = spell(en.lower(), "Beng")   # a second script for Manipuri: Meitei is picked
    rows.append(row)

tmp = tempfile.TemporaryDirectory()
T = Path(tmp.name)
D, R, O = T / "data", T / "runs", T / "out"

# 1. data: the parquet through cmd_data, the download stubbed
import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
(T / "dl/data").mkdir(parents=True)
pq.write_table(pa.Table.from_pylist(rows), T / "dl/data/train.parquet")
huggingface_hub.hf_hub_download = lambda *a, **k: str(T / "dl/data/train.parquet")
ev.IN22 = dict(ev.IN22, sha256=ev.sha256(T / "dl/data/train.parquet"))
ev.main(["data", "--data", str(D)])
items, man = ev.load_data(str(D))
langs = man["languages"]
check("data: 104 rows, all 23 languages found, Manipuri in Meitei (Bengali-script column kept as the alternative)",
      len(items) == 104 and set(langs) == set(ev.NAMES) and langs["mni"]["script"] == "Mtei"
      and langs["mni"]["other_scripts"] == ["Beng"] and langs["snd"]["column"] == "sentence_snd_Deva")
check("data: every language's text is in its column's script (letters), none empty",
      all(v["script_share"] == 1.0 and v["empty"] == 0 for v in langs.values()))
check("data: 26 strata of 4 ids; metadata kept, the context dropped",
      len(man["strata"]) == 26 and set(man["strata"].values()) == {4} and "context" not in items[1]
      and items[1]["domain"] == "culture")
bad = []
for drop, add in (("sentence_mni_Mtei", {}), (None, {"sentence_hin_Latn": "x"})):
    try:
        ev.write_dataset([{k: v for k, v in r.items() if k != drop} | add for r in rows[:2]], T / "bad", ev.IN22)
        bad.append(False)
    except ValueError:
        bad.append(True)
check("data: a multi-script language without its prompt's column, and a second script for another, are refused",
      all(bad))
(D / "in22_gen.jsonl").write_text((D / "in22_gen.jsonl").read_text().replace("River", "Rivers", 1))
try:
    ev.load_data(str(D)); refused = False
except SystemExit:
    refused = True
check("data: a changed in22_gen.jsonl is refused (SHA-256 in the manifest)", refused)
ev.main(["data", "--data", str(D)])                     # rewritten from the parquet
ev.IN22["sha256"] = "0" * 64
try:
    ev.main(["data", "--data", str(D)]); refused = False
except SystemExit:
    refused = True
check("data: a parquet that is not the pinned one is refused", refused)
items, man = ev.load_data(str(D))

# 2. the sample
s26, s200, s50 = ev.sample_ids(items, 26, 0), ev.sample_ids(items, 200, 0), ev.sample_ids(items, 50, 0)
cells = {(items[i]["domain"], items[i]["source"]) for i in s26}
check("sample: seeded (same order twice), nested (n=50 is the first 50 of n=200), another seed differs",
      s50 == ev.sample_ids(items, 50, 0) == s200[:50] and len(s200) == 104 and ev.sample_ids(items, 50, 1) != s50)
check("sample: stratified (26 ids hit all 26 domain x source cells once)", len(cells) == 26 and len(set(s26)) == 26)

# 3. prompts through the real chat templates == transformers'
from transformers import AutoTokenizer
TEXT = {"eng": "The court opened the river festival in 2026.", "mni": "ꯃꯤꯇꯩꯂꯣꯟ ꯑꯁꯤ ꯌꯥꯝꯅ ꯐꯖꯩ꯫", "hin": "नदी उत्सव।"}
for model in ("sarvam-translate", "tiny-aya-global", "qwen3.5-2b"):
    d = engine.ROOT / engine.MODELS[model]["dir"]
    if not (d / "tokenizer.json").exists():
        print(f"SKIP  prompts of {model}: no tokenizer files"); continue
    tok, hf = Tokenizer(str(d / "tokenizer.json")), AutoTokenizer.from_pretrained(str(d))
    eng = SimpleNamespace(name=model, chat_style=engine.chat_style(engine.MODELS[model], d), tokenizer=tok,
                          eos_ids=set(engine.MODELS[model]["eos"]))
    special, added = ev.added_tokens(model)
    eoa = ev.end_of_answer(eng)
    ok = [eoa == {"sarvam-translate": 106, "tiny-aya-global": 261001, "qwen3.5-2b": 248046}[model], eoa in special]
    if model == "qwen3.5-2b":
        ok.append(tok.special["<think>"] in added and tok.special["<|im_end|>"] in special)
    for src, tgt in (("eng", "hin"), ("eng", "mni"), ("mni", "eng")):
        p = ev.prompt(eng, src, tgt, TEXT[src])
        style = ev.prompt_style(model)
        names = (ev.NAMES[src], ev.NAMES[tgt]) if style == "card" else (ev.label(src), ev.label(tgt))
        msgs = ev.messages(style, *names, TEXT[src])
        ref = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ref_ids = hf(ref, add_special_tokens=False)["input_ids"]
        ids = tok.encode(p)
        bos = [i for i in ids if i == tok.bos_id] if tok.bos_id is not None else []
        ok += [p == ref, ids == ref_ids, TEXT[src] in p, len(bos) == (tok.bos_id is not None)]
        if model == "sarvam-translate":
            ok.append(p.startswith(f"<bos><start_of_turn>user\nTranslate the text below to {ev.NAMES[tgt]}.\n\n"))
        else:
            ok.append(f"Translate the following text from {names[0]} to {names[1]}. Reply with the "
                      f"translation only.\n\n{TEXT[src]}" in p and ("Manipuri (Meitei Mayek script)" in p) == (
                          "mni" in (src, tgt)))
        if model == "qwen3.5-2b":
            ok.append(p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    kind = "card" if model in ev.CARD_PROMPT else "neutral"
    check(f"prompts: {model} == apply_chat_template (text and ids, BOS once), the {kind} instruction, both directions; "
          f"end of answer {eoa}, a special token", all(ok))

# 3b. the weights' INT4 policy, from .qt headers or the registry's policy file (no model is loaded)
def fake_qt(path, schemes, scheme="int4"):
    """A .qt header without data: the tied embedding first (INT8 in every INT4 file), then layer tensors."""
    names = ["model.embed_tokens.weight"] + [f"model.layers.{k}.mlp.down_proj.weight" for k in range(len(schemes) - 1)]
    head = json.dumps({"__metadata__": {"scheme": scheme, "block": "32"}} | {
        f"{n}::q": {"dtype": "U8", "shape": [1, 1], "data_offsets": [0, 0], "scheme": s} for n, s in zip(names, schemes)})
    path.write_bytes(len(head).to_bytes(8, "little") + head.encode())
    return str(path)


def refused(fn):
    try:
        fn(); return False
    except SystemExit:
        return True


mixed, flat = fake_qt(T / "mixed.qt", ["int8", "int4", "int8", "int4"]), fake_qt(T / "flat.qt", ["int8", "int4", "int4"])
int8 = fake_qt(T / "int8.qt", ["int8", "int8"], "int8")
spec, pol = engine.MODELS["sarvam-translate"], T / "policy.json"
pol.write_text("{}")
saved = spec.get("policy")
w_mixed, w_flat = ev.weights_identity("sarvam-int4", mixed), ev.weights_identity("sarvam-int4-flat", flat)
spec["policy"] = str(T / "absent.json")
no_policy = ev.weights_identity("sarvam-int4-flat", None)
ok = [w_mixed["schemes"] == {"int4": 2, "int8": 2} and w_mixed["policy_int8"] == 1 and len(w_mixed["header_sha256"]) == 64,
      w_flat["schemes"] == {"int8": 1, "int4": 2} and w_flat["policy_int8"] == 0,
      refused(lambda: ev.weights_identity("sarvam-int4", flat)),
      refused(lambda: ev.weights_identity("sarvam-int4-flat", mixed)),
      refused(lambda: ev.weights_identity("sarvam-int4", int8)), ev.weights_identity("sarvam-int8", int8)["file"],
      refused(lambda: ev.weights_identity("sarvam-int4", None)), no_policy["policy"] is None]
spec["policy"] = str(pol)
ok += [ev.weights_identity("sarvam-int4", None)["policy_sha256"] == ev.sha256(pol),
       refused(lambda: ev.weights_identity("sarvam-int4-flat", None)),
       ev.weights_identity("sarvam-int8", None)["policy"] is None]
spec["policy"] = saved
check("weights: calibrated INT4 refuses flat weights and flat refuses calibrated (INT8 tensors beyond the tied "
      "embedding in the .qt, or a policy file at load); a file of another scheme refused; identity holds the header "
      "SHA-256 and scheme counts, or the policy's SHA-256", all(ok))

# 4. tiny models in a fake registry entry: generate, resume, score
import tiny_cohere2
import tiny_gemma3


def build_qwen35(d: str) -> None:
    """A random 4-layer Qwen3.5 (3 DeltaNet layers, 1 full attention) saved by HF, rewritten to the checkpoint's layout
    (config under text_config, tensors under model.language_model.)."""
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    Qwen3_5ForCausalLM._from_config(Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=2, layer_types=["linear_attention"] * 3 + ["full_attention"], initializer_range=0.1,
        tie_word_embeddings=True)).save_pretrained(d)
    json.dump({"text_config": json.load(open(f"{d}/config.json")), "tie_word_embeddings": True},
              open(f"{d}/config.json", "w"))
    tensors = load_file(f"{d}/model.safetensors")
    save_file({k.replace("model.", "model.language_model.", 1): v for k, v in tensors.items()},
              f"{d}/model.safetensors")


# every tiny model speaks the test template (its tokenizer has <|end|>, not Qwen's <|im_end|>; the qwen3.5 style is
# checked on the real tokenizer above)
for fam, build in (("gemma3", lambda d: tiny_gemma3.build(d, window=16)),
                   ("cohere2", lambda d: tiny_cohere2.build(d, window=8, vocab_size=512)), ("qwen3_5", build_qwen35)):
    d = T / f"tiny-{fam}"
    d.mkdir()
    train_tokenizer(str(d / "tokenizer.json"))
    json.dump({"chat_template": TEMPLATE, "bos_token": "<bos>", "eos_token": "<eos>"},
              open(d / "tokenizer_config.json", "w"))
    build(str(d))
    t = Tokenizer(str(d / "tokenizer.json"))
    engine.MODELS[f"tiny-{fam}-eval"] = dict(dir=str(d), family=fam, weights="model.safetensors", chat="template",
                                             eos=[t.special["<eos>"], t.special["<|end|>"]], sampling={},
                                             max_model_len=2048)
    ev.CONFIGS[f"tiny-{fam}"] = (f"tiny-{fam}-eval", "cpu", None)
ev.CARD_PROMPT.add("tiny-gemma3-eval")                  # the card's shape: a system turn
COMMON = ["--data", str(D), "--runs", str(R), "--langs", "hin,mni"]


def gen_lines(cfg):
    return {f.stem: ev.read_jsonl(f) for f in sorted((R / cfg / "gen").glob("*.jsonl"))}


def outputs(cfg):
    """Each direction's records without their timing, in file order."""
    return {k: [json.dumps({f: x for f, x in r.items() if f != "ttft"}) for r in v] for k, v in gen_lines(cfg).items()}


def greedy(eng, ids, max_new):
    st, out = eng.model.new_state(len(ids) + max_new), []
    lg = eng.model.forward(torch.tensor(ids), state=st, last_only=True)[0]
    while len(out) < max_new and (nxt := int(lg[:eng.tokenizer.vocab_size()].argmax())) not in eng.eos_ids:
        out.append(nxt)
        lg = eng.model.forward(torch.tensor([nxt]), state=st, last_only=True)[0]
    return out


for cfg in ("tiny-gemma3", "tiny-cohere2", "tiny-qwen3_5"):
    ev.main(["generate", cfg, *COMMON, "--n", "6", "--batch", "4", "--kv-blocks", "128"])
    g = gen_lines(cfg)
    speed, seg = ev.read_jsonl(R / cfg / "speed.jsonl"), ev.read_jsonl(R / cfg / "segments.jsonl")
    check(f"{cfg} generate: 4 directions x 6 ids, every record complete, one speed record of 24 requests, one segment "
          "with its code version",
          sorted(g) == ["eng-hin", "eng-mni", "hin-eng", "mni-eng"] and all(len(v) == 6 for v in g.values())
          and all(set(r) == {"id", "hyp", "tokens", "stop", "ttft", "special", "added"} for v in g.values() for r in v)
          and len(speed) == 1 and speed[0]["requests"] == 24 and speed[0]["tokens"] > 0
          and len(seg) == 1 and seg[0]["kind"] == "gen" and seg[0]["items"] == 24 and len(seg[0]["code_sha256"]) == 16)
    args = SimpleNamespace(config=cfg, weights=None, runs=str(R), prompt=None, seed=0)
    eng, run, _ = ev.open_run(args, "gen", man)
    meta = json.load(open(run / "meta.json"))
    check(f"{cfg} generate: meta records the prompt messages ({'card' if 'gemma' in cfg else 'neutral'}), the rendered "
          "template, the seed, the dataset and the end-of-answer token", meta["config"] == cfg
          and meta["messages"][0]["role"] == ("system" if "gemma" in cfg else "user") and "{text}" in meta["example_prompt"]
          and meta["seed"] == 0 and meta["dataset"] == man["jsonl_sha256"]
          and meta["end_of_answer"] == eng.tokenizer.special["<|end|>"] and meta["weights"]["file"] is None)
    special, added = ev.added_tokens(eng.name)
    same, stripped = True, 0
    for d_, recs in g.items():
        src, tgt = d_.split("-")
        for r in recs[:3]:
            text = items[r["id"]]["text"][src]
            ids = eng.tokenizer.encode(ev.prompt(eng, src, tgt, text))
            max_new = min(ev.MAX_NEW[0] + int(np.ceil(ev.MAX_NEW[1] * len(text))), 2048 - len(ids))
            out = greedy(eng, ids, max_new)
            keep = [t for t in out if t not in special]
            stripped += len(out) - len(keep)
            same &= eng.tokenizer.decode(keep).strip() == r["hyp"] and len(out) == r["tokens"] \
                and r["stop"] == ("length" if len(out) == max_new else "stop") \
                and r["special"] == len(out) - len(keep) and r["added"] == sum(t in added for t in keep)
    check(f"{cfg} generate: batched through the scheduler == a sequential greedy loop (12 items, tokens, stop), special "
          f"tokens removed and counted ({stripped})", same)

    before = outputs(cfg)
    ev.main(["generate", cfg, *COMMON, "--n", "6", "--batch", "4", "--kv-blocks", "128"])
    check(f"{cfg} resume: a rerun makes nothing (files, speed records and segments unchanged)",
          outputs(cfg) == before and len(ev.read_jsonl(R / cfg / "speed.jsonl")) == 1
          and len(ev.read_jsonl(R / cfg / "segments.jsonl")) == 1)
    f = R / cfg / "gen" / "hin-eng.jsonl"
    f.write_bytes(torn := f.read_bytes()[:-7])          # the last record torn mid-line
    check(f"{cfg} read_jsonl: a torn last line is skipped and the file left as it is",
          len(ev.read_jsonl(f)) == 5 and f.read_bytes() == torn)
    ev.main(["generate", cfg, *COMMON, "--n", "6", "--batch", "4", "--kv-blocks", "128"])
    speed = ev.read_jsonl(R / cfg / "speed.jsonl")
    after = outputs(cfg)
    check(f"{cfg} resume: the writer cuts a torn last line and makes only that item again, identical",
          len(speed) == 2 and speed[1]["requests"] == 1 and sorted(after["hin-eng"]) == sorted(before["hin-eng"])
          and all(after[k] == before[k] for k in before if k != "hin-eng"))

    ev.main(["score", cfg, *COMMON, "--n", "6"])
    sc = {f.stem: ev.read_jsonl(f) for f in sorted((R / cfg / "score").glob("*.jsonl"))}
    worst, eoa = 0.0, meta["end_of_answer"]
    for d_, recs in sc.items():
        src, tgt = d_.split("-")
        for r in recs:
            p = eng.tokenizer.encode(ev.prompt(eng, src, tgt, items[r["id"]]["text"][src]))
            c = eng.tokenizer.encode(items[r["id"]]["text"][tgt])
            lp = torch.log_softmax(eng.model.forward(torch.tensor(p + c + [eoa])).double(), -1)
            hand = -sum(lp[len(p) - 1 + k, t].item() for k, t in enumerate(c))
            end = -lp[len(p) - 1 + len(c), eoa].item()
            worst = max(worst, abs(r["nats"] - hand) / hand, abs(r["end_nats"] - end) / end)
            worst = worst if r["ref_tokens"] == len(c) else float("inf")
    check(f"{cfg} score: nats and end_nats == -log p from full-sequence logits (24 references and their "
          f"end-of-answer token, worst rel {worst:.1e} < 1e-5); one score segment",
          len(sc) == 4 and all(len(v) == 6 for v in sc.values()) and worst < 1e-5
          and all(r["roundtrip"] and r["chars"] == len(items[r["id"]]["text"][k.split("-")[1]])
                  for k, v in sc.items() for r in v)
          and [x["kind"] for x in ev.read_jsonl(R / cfg / "segments.jsonl")] == ["gen", "gen", "score"])
    seqs = [(r["id"], eng.tokenizer.encode(ev.prompt(eng, "eng", "mni", items[r["id"]]["text"]["eng"])),
             eng.tokenizer.encode(items[r["id"]]["text"]["mni"]) + [eoa]) for r in sc["eng-mni"]]
    packed = dict(ev.teacher_forced(eng.model, seqs, eng.tokenizer.vocab_size()))
    budget, decision.FORK_BUDGET = decision.FORK_BUDGET, 1
    alone = dict(ev.teacher_forced(eng.model, seqs, eng.tokenizer.vocab_size()))
    decision.FORK_BUDGET = budget
    common = len(os.path.commonprefix([p for _, p, _ in seqs]))
    check(f"{cfg} score: packed forks == one at a time, token by token (common prefix {common} tokens)",
          all(len(packed[k]) == len(c) and np.allclose(packed[k], alone[k], rtol=1e-5, atol=1e-6)
              for (k, _, c) in seqs))
    json.dump(meta | {"weights": "another.qt"}, open(run / "meta.json", "w"))
    other = refused(lambda: ev.open_run(args, "gen", man))
    json.dump(meta, open(run / "meta.json", "w"))
    check(f"{cfg}: results of another build, or of another seed, in the same run directory are refused",
          other and refused(lambda: ev.open_run(SimpleNamespace(**vars(args) | {"seed": 1}), "gen", man)))

# 5. report: chrF++ on IN22's protocol vs sacrebleu, macro rows, diagnostics; files read, never written
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.indic_tokenize import trivial_tokenize


def in22(text, tgt):
    """IndicTrans2's preprocess_translate.py for hin_Deva and mni_Mtei (both "hi" in its code map); English raw."""
    if tgt == "eng":
        return text
    return " ".join(trivial_tokenize(IndicNormalizerFactory().get_normalizer("hi").normalize(text.strip()), "hi"))


def snapshot():
    return {f: f.read_bytes() for f in sorted(R.rglob("*")) if f.is_file()}


f = R / "tiny-cohere2" / "gen" / "eng-hin.jsonl"
f.write_bytes(f.read_bytes() + b'{"id": 99, "hy')     # a record being written while the report reads
files = snapshot()
ev.main(["report", "--data", str(D), "--runs", str(R), "--out", str(O), "--boot", "200", "--baseline", "tiny-gemma3"])
check("report: reads only (every run file byte-identical after it, a line being written skipped)", snapshot() == files)
f.write_bytes(f.read_bytes()[:f.read_bytes().rfind(b"\n") + 1])
num = json.load(open(O / "numbers.json"))
chrf = CHRF(word_order=2)
TINY = ("tiny-gemma3", "tiny-cohere2", "tiny-qwen3_5")
ok, diag = [], []
for cfg in TINY:
    for d_, recs in gen_lines(cfg).items():
        tgt = d_.split("-")[1]
        ok.append(abs(num["chrf"][d_][cfg]["score"] - chrf.corpus_score(
            [in22(r["hyp"], tgt) for r in recs], [[in22(items[r["id"]]["text"][tgt], tgt) for r in recs]]).score) < 1e-9)
        diag.append(num["diagnostics"][cfg][d_] == dict(
            outputs=len(recs), wrong_script=sum(ev.script_share(r["hyp"], man["languages"][tgt]["script"]) < 0.5
                                                for r in recs),
            length_stops=sum(r["stop"] == "length" for r in recs), special=sum(r["special"] > 0 for r in recs),
            added=sum(r["added"] > 0 for r in recs)))
md = (O / "tables.md").read_text()
mac = num["macro"]["chrf"]["English -> X"]
check("report: tables.md, numbers.json / .csv and chrf.svg written; chrF++ == sacrebleu's corpus_score on IN22's "
      "protocol (IndicNLP normalize + trivial_tokenize); signature and protocol printed",
      all(ok) and len(ok) == 12 and "nc:6|nw:2" in md and "version:2.6.0" in md and report.PROTOCOL in md
      and ET.parse(O / "chrf.svg").getroot().tag.endswith("svg") and num["protocol"] == report.PROTOCOL
      and len((O / "numbers.csv").read_text().splitlines()) == 1 + 3 * 4 * 3 + 3 * 2 * 3 * 3
      and "delta" in num["chrf"]["eng-hin"]["tiny-cohere2"] and num["ids"]["eng-hin"] == sorted(ev.sample_ids(items, 6, 0)))
check("report: macro rows over all languages (2), Tiny Aya's (1: Hindi) and one script (1: not Manipuri), each with "
      "paired CIs; bits per character with the end-of-answer token above those without",
      all(f"**macro, {n}**" in md for n in ("all (2)", "Tiny Aya's (1)", "one script (1)"))
      and mac["all"]["langs"] == ["hin", "mni"] and mac["Tiny Aya's"]["langs"] == mac["one script"]["langs"] == ["hin"]
      and "ci" in mac["all"]["configs"]["tiny-cohere2"] and "ci" not in mac["all"]["configs"]["tiny-gemma3"]
      and all(num["bpc_end"][d_][c]["score"] > num["bpc"][d_][c]["score"] for d_ in num["bpc"] for c in TINY))
check("report: per-direction diagnostics == a recount of the records (wrong script per output, length stops, special "
      "and other added tokens), the directions with any listed", all(diag) and len(diag) == 12
      and all((f"| {c} | {d_} |" in md) == any(list(v.values())[1:]) for c in TINY
              for d_, v in num["diagnostics"][c].items()))
check("report: code versions per config; a warning only when a config's segments differ",
      all(len(num["code"][c]) >= 1 for c in TINY)
      and ("**Warning:**" in md) == any(len(num["code"][c]) > 1 for c in TINY))
check("report: an unknown --baseline is refused", refused(lambda: ev.main(
    ["report", "--data", str(D), "--runs", str(R), "--out", str(T / "o3"), "--baseline", "nope"])))

# the prompt-swap control: its own run directory, a config of its own in the report
ev.main(["generate", "tiny-gemma3", "--data", str(D), "--runs", str(R), "--langs", "hin", "--n", "3", "--batch", "2",
         "--kv-blocks", "64", "--prompt", "neutral"])
ctl = R / "tiny-gemma3@neutral"
cmeta = json.load(open(ctl / "meta.json"))
ev.main(["report", "--data", str(D), "--runs", str(R), "--out", str(T / "o2"), "--boot", "50"])
num2 = json.load(open(T / "o2" / "numbers.json"))
mac2 = num2["macro"]["chrf"]["English -> X"]
check("control: --prompt neutral on the card model -> RUNS/tiny-gemma3@neutral (neutral user turn); the report "
      "(no --baseline: the first config) leaves it out of the macro over all languages and keeps it in Hindi's",
      cmeta["prompt"] == "neutral" and cmeta["messages"][0]["role"] == "user"
      and sorted(gen_lines("tiny-gemma3@neutral")) == ["eng-hin", "hin-eng"] and num2["baseline"] == "tiny-gemma3"
      and "tiny-gemma3@neutral" not in mac2["all"]["configs"] and "ci" in mac2["Tiny Aya's"]["configs"][
          "tiny-gemma3@neutral"] and "tiny-gemma3@neutral" in num2["chrf"]["eng-hin"])

# 6. the protocol on canonically equal text, the bootstrap
import unicodedata
nfc = unicodedata.normalize("NFC", "क़ानून और फ़ैसला सड़क पर हुआ।")
pre = nfc.replace("क़", "क़").replace("फ़", "फ़").replace("ड़", "ड़")
hi = report.in22_preprocess("hin", "Deva")
raw, prot = chrf.sentence_score(nfc, [pre]).score, chrf.sentence_score(hi(nfc), [hi(pre)]).score
danda = chrf.sentence_score(hi("सड़क पर हुआ।"), [hi("सड़क पर हुआ ।")]).score
urd = report.in22_preprocess("urd", "Arab")("یہ ایک کتاب ہے۔")
check(f"protocol: precomposed nukta letters vs their NFC form score 100 (raw {raw:.1f}); a danda attached or spaced "
      "100; Urdu through urduhack's normalizer (its TensorFlow package init skipped); English raw",
      pre != nfc and unicodedata.normalize("NFC", pre) == nfc and prot == 100 and raw < 60 and danda == 100
      and urd.endswith(" ۔") and report.in22_preprocess("eng", "Latn")(" A b. ") == " A b. ")

hyps, refs = [], []
for k in range(40):                                    # Indic and English pairs with word and character edits
    ref = items[k + 1]["text"]["hin" if k % 2 else "eng"]
    hyp = " ".join(ref.split()[::-1]) if k % 3 == 0 else ref[: max(1, len(ref) - k % 5)]
    hyps.append(hyp); refs.append(ref)
(T / "hyp.txt").write_text("\n".join(hyps) + "\n"); (T / "ref.txt").write_text("\n".join(refs) + "\n")
cli = subprocess.run([sys.executable, "-m", "sacrebleu", str(T / "ref.txt"), "-i", str(T / "hyp.txt"), "-m", "chrf",
                      "--chrf-word-order", "2", "-b", "-w", "6"], capture_output=True, text=True).stdout.strip()
fn = lambda s: chrf._compute_score_from_stats(list(s)).score
stats = np.array(chrf._extract_corpus_statistics(hyps, [refs]))
ids = list(range(40))
other = np.array(chrf._extract_corpus_statistics(refs[:20] + hyps[20:], [refs]))
per = {"a": {"d": (ids, stats)}, "b": {"d": (ids, other)}, "c": {"d": (ids, stats)}}
r1, r2, r3 = (report.compare(per, fn, 300, s, "a") for s in (7, 7, 8))
check(f"chrF++ of 40 pairs == sacrebleu's CLI ({cli})", cli and abs(r1["d"]["a"]["score"] - float(cli)) < 1e-6)
check("bootstrap: the same seed gives the same CIs, another seed others; a system against itself: delta 0, CI [0, 0]; "
      "lo <= delta <= hi", r1["d"]["b"]["ci"] == r2["d"]["b"]["ci"] and r1["d"]["b"]["ci"] != r3["d"]["b"]["ci"]
      and r1["d"]["c"]["ci"] == [0.0, 0.0] and r1["d"]["c"]["delta"] == 0
      and r1["d"]["b"]["ci"][0] <= r1["d"]["b"]["delta"] <= r1["d"]["b"]["ci"][1])


def correlated(rng, dirs=22, n=50):
    """Two configs' bits per character in `dirs` directions over the same n ids, each sentence's difficulty shared by
    every direction (and larger for b): the case IN22-Gen's n-way parallel sample makes."""
    chars, shared, per = rng.integers(60, 200, n), rng.normal(0, 1, n), {"a": {}, "b": {}}
    for d in range(dirs):
        a = 2.0 + .3 * shared + rng.normal(0, .3, n)
        b = a + .1 + .4 * shared + rng.normal(0, .3, n)
        per["a"][f"d{d:02d}"] = (list(range(n)), np.stack([a * chars, chars], 1))
        per["b"][f"d{d:02d}"] = (list(range(n)), np.stack([b * chars, chars], 1))
    return per


ratio = lambda s: float(s[0] / s[1])
rng, deltas, widths = np.random.default_rng(1), [], []
for t in range(150):
    m = report.macro(per := correlated(rng), ratio, sorted(per["a"]), 200, t, "a")
    deltas.append(m["b"]["delta"]); widths.append(m["b"]["ci"][1] - m["b"]["ci"][0])
spread = float(np.diff(np.percentile(deltas, [2.5, 97.5]))[0])
per["c"] = per["a"]
m, q = report.macro(per, ratio, sorted(per["a"]), 200, 0, "a"), report.compare(per, ratio, 200, 0, "a")
check(f"macro bootstrap: the CI of a macro delta ({np.mean(widths):.3f} wide on average) covers the spread of the delta "
      f"over 150 fresh samples ({spread:.3f}) when sentence difficulty is shared across directions; the mean of the "
      "directions' scores; a config against itself: CI [0, 0]",
      0.75 < np.mean(widths) / spread < 1.4 and m["c"]["ci"] == [0.0, 0.0]
      and abs(m["b"]["score"] - np.mean([q[d]["b"]["score"] for d in q])) < 1e-12)
check("script share: Hindi text is Devanagari, English is not, Meitei is Meetei Mayek",
      ev.script_share(TEXT["hin"], "Deva") == 1.0 and ev.script_share(TEXT["eng"], "Deva") == 0.0
      and ev.script_share(TEXT["mni"], "Mtei") == 1.0)

tmp.cleanup()
print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
