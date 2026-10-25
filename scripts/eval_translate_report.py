import importlib.metadata
import importlib.util
import json
import math
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_translate import CONFIGS, ENG, LANGS, NAMES, SCRIPT_PREF, SUPPORT, load_data, read_jsonl, script_share

# the normalizer's language for each target column: IndicTrans2's scripts/flores_codes_map_indic.py
ISO = {"asm_Beng": "as", "ben_Beng": "bn", "brx_Deva": "hi", "doi_Deva": "hi", "gom_Deva": "kK", "guj_Gujr": "gu",
       "hin_Deva": "hi", "kan_Knda": "kn", "kas_Arab": "ur", "kas_Deva": "hi", "mai_Deva": "hi", "mal_Mlym": "ml",
       "mar_Deva": "mr", "mni_Beng": "bn", "mni_Mtei": "hi", "npi_Deva": "ne", "ory_Orya": "or", "pan_Guru": "pa",
       "san_Deva": "hi", "sat_Olck": "or", "snd_Arab": "ur", "snd_Deva": "hi", "tam_Taml": "ta", "tel_Telu": "te",
       "urd_Arab": "ur"}
LIBS = {"indic-nlp-library": "0.92", "urduhack": "1.1.1"}      # Urdu's normalizer in IndicNLP is urduhack's
PROTOCOL = ("IN22-Gen / IndicTrans2 compute_metrics.sh: Indic targets IndicNLP normalize (default options) + "
            "trivial_tokenize, English raw; " + ", ".join(f"{k} {v}" for k, v in LIBS.items()))
MACROS = (("all", lambda x: True), ("Tiny Aya's", lambda x: x in SUPPORT["tiny-aya-global"]),
          ("one script", lambda x: x not in SCRIPT_PREF))
SIDES = (("English -> X", lambda x: f"{ENG}-{x}"), ("X -> English", lambda x: f"{x}-{ENG}"))


def in22_preprocess(lang: str, script: str):
    """-> text -> text as IN22-Gen's protocol scores a target in `lang`, written in `script`."""
    if lang == ENG:
        return lambda s: s
    have = {}
    for k in LIBS:
        try:
            have[k] = importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError:
            have[k] = None
    if have != LIBS:
        sys.exit(f"IN22's chrF++ protocol needs {LIBS}, found {have}: pip install --no-deps "
                 + " ".join(f"{k}=={v}" for k, v in LIBS.items()) + " (their other dependencies are not used)")
    iso = ISO[f"{lang}_{script}"]
    if iso == "ur" and "urduhack" not in sys.modules:
        # IndicNLP's Urdu normalizer imports urduhack's regex modules only, but urduhack's package __init__ imports
        # TensorFlow (for its taggers; no build for this Python): register the package without running it
        pkg = types.ModuleType("urduhack")
        pkg.__path__ = list(importlib.util.find_spec("urduhack").submodule_search_locations)
        sys.modules["urduhack"] = pkg
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize.indic_tokenize import trivial_tokenize
    norm = IndicNormalizerFactory().get_normalizer(iso)
    return lambda s: " ".join(trivial_tokenize(norm.normalize(s.strip()), iso))


def compare(per_cfg: dict, fn, n_boot: int, seed: int, base: str) -> dict:
    """per_cfg {config: {direction: (ids, per-item stats [n, k])}}, fn(summed stats) -> score ->
    {direction: {config: {score, n, boot[, delta, ci]}}} on the ids every config has; deltas against `base`, with one
    set of resampled ids per direction shared by every config (paired)."""
    out = {}
    for d in sorted({d for v in per_cfg.values() for d in v}):
        have = {c: v[d] for c, v in per_cfg.items() if d in v}
        if not (shared := sorted(set.intersection(*(set(ids) for ids, _ in have.values())))):
            continue
        idx = np.random.default_rng([seed, *d.encode()]).integers(0, len(shared), (n_boot, len(shared)))
        cell = {}
        for c, (ids, st) in have.items():
            st = st[[ids.index(i) for i in shared]]
            cell[c] = dict(score=fn(st.sum(axis=0)), n=len(shared), boot=np.array([fn(x) for x in st[idx].sum(1)]))
        out[d] = with_deltas(cell, base)
    return out


def with_deltas(cell: dict, base: str) -> dict:
    for c, v in cell.items():
        if base in cell and c != base:
            v |= dict(delta=v["score"] - cell[base]["score"],
                      ci=np.percentile(v["boot"] - cell[base]["boot"], [2.5, 97.5]).tolist())
    return cell


def macro(per_cfg: dict, fn, keys: list[str], n_boot: int, seed: int, base: str) -> dict:
    """The mean score over the directions `keys`, for the configs that have all of them, with paired bootstrap CIs on
    the differences from `base`. IN22-Gen is n-way parallel and the sample takes the same ids in every direction, so a
    sentence hard in one language tends to be hard in all (English -> X shares its sources, X -> English its
    references); resampling each direction alone would drop that covariance and make the CI far too narrow. So one
    resample of the ids (as counts per id) serves every direction and config, each direction keeping the ids it has."""
    cfgs = [c for c, v in per_cfg.items() if keys and all(k in v for k in keys)]
    if not cfgs:
        return {}
    shared = {k: sorted(set.intersection(*(set(per_cfg[c][k][0]) for c in cfgs))) for k in keys}
    ids = sorted(set().union(*shared.values()))
    col = {i: j for j, i in enumerate(ids)}
    w = np.random.default_rng(seed).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), n_boot)
    cell = {}
    for c in cfgs:
        scores, boots = [], []
        for k in keys:
            have, st = per_cfg[c][k]
            st = st[[have.index(i) for i in shared[k]]]
            scores.append(fn(st.sum(axis=0)))
            boots.append([fn(x) for x in w[:, [col[i] for i in shared[k]]] @ st])
        cell[c] = dict(score=float(np.mean(scores)), n=len(ids), boot=np.mean(boots, axis=0))
    return with_deltas(cell, base)


def svg_chart(table: dict, langs: list[str], cfgs: list[str], meta: dict) -> str:
    """chrF++ per language, a dot per config (hover: its value), one panel per direction. Color: the model, from the
    dataviz reference palette's first three slots (distinct for every pair, light and dark steps); hollow: INT8."""
    models = list(dict.fromkeys(meta[c]["model"] for c in cfgs))
    cls = {c: f"m{models.index(meta[c]['model']) % 3}{' h' if 'int8' in meta[c]['backend'] else ''}" for c in cfgs}
    dot = lambda c, x, y, tip="": f'<circle class="{cls[c]}" cx="{x:.1f}" cy="{y}" r="5">{tip}</circle>'
    W, top, row, lab, pw = 1000, 70, 22, 90, 400
    hi = math.ceil(max([10.0] + [v["score"] for t in table.values() for v in t.values()]) / 10) * 10
    H = top + row * len(langs) + 30
    css = lambda ink, grid, bg, *m: (f"text{{fill:{ink}}} .g{{stroke:{grid}}} .bg{{fill:{bg}}} " + " ".join(
        f".m{i}{{fill:{v};stroke:{v}}}" for i, v in enumerate(m)) + " .h{fill:none}")
    light = css("#1a1a19", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834", "#1baf7a")
    dark = css("#e8e7e3", "#383835", "#1a1a19", "#3987e5", "#d95926", "#199e70")
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" '
         f'font-size="12"><style>circle{{stroke-width:2}} {light} @media (prefers-color-scheme: dark){{{dark}}}</style>'
         f'<rect class="bg" width="{W}" height="{H}"/><text x="16" y="22" font-size="15">IN22-Gen chrF++ per language,'
         ' greedy</text>']
    s += [dot(c, 22 + 150 * j, 42) + f'<text x="{32 + 150 * j}" y="46">{c}</text>' for j, c in enumerate(cfgs)]
    for p, (name, key) in enumerate(SIDES):
        x0, name = lab + p * (pw + 70), name.replace("->", "→")
        s.append(f'<text x="{x0}" y="{top - 6}">{name}</text>')
        s += [f'<line class="g" x1="{x0 + pw * t / hi}" x2="{x0 + pw * t / hi}" y1="{top}" y2="{H - 24}"/><text '
              f'x="{x0 + pw * t / hi}" y="{H - 10}" text-anchor="middle">{t}</text>' for t in range(0, hi + 1, 10)]
        for r, x in enumerate(langs):
            y = top + row * r + row / 2
            if p == 0:
                s.append(f'<text x="{lab - 8}" y="{y + 4}" text-anchor="end">{NAMES[x]}</text>')
            cell = table.get(key(x), {})
            s += [dot(c, x0 + pw * cell[c]["score"] / hi, y, f"<title>{c}, {NAMES[x]}, {name}: "
                      f"{cell[c]['score']:.1f}</title>") for c in cfgs if c in cell]
    return "\n".join(s + ["</svg>"]) + "\n"


def load_results(runs: Path, cfgs: list[str], items: dict, manifest: dict, chrf) -> dict:
    """Every config's records -> per-item chrF++ stats (IN22's protocol) and bits-per-character stats (without and with
    the end-of-answer token), per-direction diagnostics, tokens per character (of each language's text, from the
    scored references), speed, counts and code versions."""
    r = {k: defaultdict(dict) for k in ("chrf", "bpc", "bpc_end", "diag", "tpc")} | dict(speed={}, counts={}, code={})
    prep = {}
    for c in cfgs:
        gen = {f.stem: read_jsonl(f) for f in sorted((runs / c / "gen").glob("*.jsonl"))}
        for d, rs in gen.items():
            tgt = d.split("-")[1]
            script = manifest["languages"][tgt]["script"]
            f = prep[tgt] = prep.get(tgt) or in22_preprocess(tgt, script)
            hyps, refs = [f(x["hyp"]) for x in rs], [f(items[x["id"]]["text"][tgt]) for x in rs]
            r["chrf"][c][d] = ([x["id"] for x in rs], np.array(chrf._extract_corpus_statistics(hyps, [refs])))
            r["diag"][c][d] = dict(outputs=len(rs), wrong_script=sum(script_share(x["hyp"], script) < 0.5 for x in rs),
                                   length_stops=sum(x["stop"] == "length" for x in rs),
                                   special=sum(x["special"] > 0 for x in rs), added=sum(x["added"] > 0 for x in rs))
        scored = {f.stem: read_jsonl(f) for f in sorted((runs / c / "score").glob("*.jsonl"))}
        by_lang = defaultdict(list)
        for d, rs in scored.items():
            ids = [x["id"] for x in rs]
            r["bpc"][c][d] = (ids, np.array([[x["nats"] / math.log(2), x["chars"]] for x in rs]))
            r["bpc_end"][c][d] = (ids, np.array([[(x["nats"] + x["end_nats"]) / math.log(2), x["chars"]] for x in rs]))
            by_lang[d.split("-")[1]] += rs
        r["tpc"][c] = {x: (sum(y["ref_tokens"] for y in rs) / sum(y["chars"] for y in rs), sum(y["chars"] for y in rs)
                           / len(rs)) for x, rs in by_lang.items()}
        flat = [x for rs in gen.values() for x in rs]
        r["counts"][c] = dict(translations=len(flat), references=sum(map(len, scored.values())),
                              not_roundtrip=sum(not x["roundtrip"] for rs in scored.values() for x in rs))
        r["code"][c] = sorted({f"{s['git']} {s['code_sha256']}" for s in read_jsonl(runs / c / "segments.jsonl")})
        if ss := read_jsonl(runs / c / "speed.jsonl"):
            tt, secs = sorted(x["ttft"] for x in flat if x["ttft"] is not None), sum(s["secs"] for s in ss)
            gpu = [s["gpu_peak_bytes"] for s in ss if s["gpu_peak_bytes"]]
            r["speed"][c] = dict(
                tok_s=sum(s["tokens"] for s in ss) / secs, chars_s=sum(s["chars"] for s in ss) / secs,
                ttft_p50=tt[len(tt) // 2] if tt else None, ttft_p90=tt[int(0.9 * (len(tt) - 1))] if tt else None,
                weights_gb=ss[-1]["weights_bytes"] / 1e9, gpu_peak_gib=max(gpu) / 2**30 if gpu else None,
                footprint_peak_gib=max(s["footprint_peak_mib"] for s in ss) / 1024,
                requests=sum(s["requests"] for s in ss), preempted=sum(s["preempted"] for s in ss),
                **{k: ss[-1][k] for k in ("prefix_cache", "batch", "kv_blocks")})
    return r


def md_table(head: list[str], rows: list[list]) -> list[str]:
    return [f"| {' | '.join(head)} |", "|---|" + "---:|" * (len(head) - 1)] + \
        [f"| {' | '.join(map(str, row))} |" for row in rows] + [""]


def cmd_report(a) -> None:
    from sacrebleu.metrics import CHRF
    items, manifest = load_data(a.data)
    runs, chrf = Path(a.runs), CHRF(word_order=2)
    found = [p.name for p in sorted(runs.iterdir()) if (p / "meta.json").exists()]
    rank = lambda c: (list(CONFIGS).index(c.split("@")[0]) if c.split("@")[0] in CONFIGS else len(CONFIGS), c)
    cfgs = a.configs or sorted(found, key=rank)
    if (missing := sorted(set(cfgs) - set(found))) or not cfgs:
        sys.exit(f"no results for {', '.join(missing) or 'any config'} in {runs} (found: {', '.join(found) or 'none'})")
    base = a.baseline or "sarvam-int8"
    if base not in cfgs:
        if a.baseline:
            sys.exit(f"--baseline {base} is not among the configs {cfgs}")
        print(f"note: no {base} among {cfgs}; deltas are against {cfgs[0]}")
        base = cfgs[0]
    meta = {c: json.load(open(runs / c / "meta.json")) for c in cfgs}
    res = load_results(runs, cfgs, items, manifest, chrf)
    fns = {"chrf": lambda s: chrf._compute_score_from_stats(list(s)).score, "bpc": lambda s: float(s[0] / s[1]),
           "bpc_end": lambda s: float(s[0] / s[1])}
    tab = {m: compare(res[m], fn, a.boot, a.seed, base) for m, fn in fns.items()}
    langs = [x for x in LANGS if any(k(x) in t for t in tab.values() for _, k in SIDES)]
    macros = defaultdict(dict)                       # metric -> side -> [(name, languages, cell)]
    for m, t in tab.items():
        for side, key in SIDES:
            present = [x for x in langs if key(x) in t]
            sets = [(name, [x for x in present if keep(x)]) for name, keep in MACROS]
            if (common := [x for x in present if set(cfgs) <= set(t[key(x)])]) and all(common != s for _, s in sets):
                sets.append(("every config's", common))
            macros[m][side] = [(name, xs, macro(res[m], fns[m], [key(x) for x in xs], a.boot, a.seed, base))
                               for name, xs in sets if xs]
    sig = str(chrf.get_signature()) if res["chrf"] else "n/a"           # known once references were read
    models = list(dict.fromkeys(meta[c]["model"] for c in cfgs))
    first = {m: next(c for c in cfgs if meta[c]["model"] == m) for m in models}     # one tokenizer per model
    fmt = lambda v, nd: "-" if v is None else f"{v['score']:.{nd}f}" + (
        f" ({v['delta']:+.{nd}f} [{v['ci'][0]:+.{nd}f}, {v['ci'][1]:+.{nd}f}])" if "delta" in v else "")
    g = lambda v, f=".2f": "n/a" if v is None else format(v, f)
    sup = lambda x: " / ".join("?" if m not in SUPPORT else "yes" if x in SUPPORT[m] else "no" for m in models)
    md = [f"chrF++: sacrebleu `{sig}`, corpus level over the sampled ids, on IN22's protocol ({PROTOCOL}). In "
          f"brackets: the difference from {base} and its paired bootstrap 95% CI ({a.boot} resamples, seed {a.seed}; "
          "a macro row resamples the shared ids jointly over its languages). Support claimed by "
          f"{' / '.join(models)}. Macro rows: all languages; Tiny Aya's (the languages its card claims); one script "
          f"(without {', '.join(NAMES[x] for x in SCRIPT_PREF)}, written in more than one); a config without every "
          "language of a row shows -.", ""]
    titles = {"chrf": ("chrF++ (higher is better)", 1), "bpc": ("bits per reference character (lower is better)", 3),
              "bpc_end": ("bits per reference character with the end-of-answer token (lower is better)", 3)}
    for m, (title, nd) in titles.items():
        t = tab[m]
        for side, key in SIDES:
            md += [f"### {title}, {side}", ""] + md_table(["language", "support", "n", *cfgs], [
                [NAMES[x], sup(x), next(iter(t[key(x)].values()))["n"], *(fmt(t[key(x)].get(c), nd) for c in cfgs)]
                for x in langs if key(x) in t] + [[f"**macro, {name} ({len(xs)})**", "", next(iter(cell.values()))[
                    "n"] if cell else "", *(fmt(cell.get(c), nd) for c in cfgs)] for name, xs, cell in macros[m][side]])
    rows = []                                       # a language's text is the source of one side, the other's reference
    for x in (x for x in NAMES if any(x in v for v in res["tpc"].values())):
        per_sentence = next(res["tpc"][c][x][1] for c in cfgs if x in res["tpc"][c])
        rows.append([NAMES[x], f"{per_sentence:.0f}", *(g(res["tpc"][first[m]].get(x, (None,))[0], ".3f")
                                                         for m in models)])
    md += ["### Tokens per character (the scored references, no BOS)", ""] + md_table(
        ["language", "characters per sentence", *models], rows)
    md += ["### Speed and memory (generation: greedy, a closed loop of `batch` requests in flight)", ""] + md_table(
        ["config", "weights (GB)", "output tok/s", "output chars/s", "TTFT p50 / p90 (s)", "peak GPU (GiB)",
         "peak footprint (GiB)", "requests", "preempted"],
        [[c, g(s["weights_gb"]), g(s["tok_s"], ".1f"), g(s["chars_s"], ".0f"), f"{g(s['ttft_p50'])} / "
          f"{g(s['ttft_p90'])}", g(s["gpu_peak_gib"]), g(s["footprint_peak_gib"]), s["requests"], s["preempted"]]
         for c, s in res["speed"].items()])
    diag = [[c, d, *v.values()] for c in cfgs for d, v in res["diag"][c].items()
            if any(v[k] for k in ("wrong_script", "length_stops", "special", "added"))]
    md += ["### Per-output diagnostics (the directions with any)", ""] + (md_table(
        ["config", "direction", "outputs", "under 50% in the target's script (letters)", "stopped by max_new_tokens",
         "with special tokens (removed before scoring)", "with other added tokens (kept)"], diag) if diag else
        ["none", ""])
    md += ["Scored references whose tokens decode to other text: " + ", ".join(
               f"{c} {n['not_roundtrip']} / {n['references']}" for c, n in res["counts"].items()) + " (Qwen3.5's "
           "tokenizer applies NFC: its bits are of the NFC text, over the reference's own characters).",
           "Code versions (git describe, SHA-256 of src/ and eval_translate.py) per config: " + "; ".join(
               f"{c} {', '.join(v) or 'n/a'}" for c, v in res["code"].items()) + "."]
    md += [f"**Warning:** {c}'s results come from {len(v)} code versions." for c, v in res["code"].items() if len(v) > 1]
    md += ["", "### Prompts (each through the model's own chat template; renderings in RUNS/CONFIG/meta.json)", ""]
    md += [f"- {c}: `{json.dumps(meta[c]['messages'], ensure_ascii=False)}`" for c in cfgs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text("\n".join(md) + "\n")
    nob = lambda cl: {c: {k: v for k, v in x.items() if k != "boot"} for c, x in cl.items()}
    mac = {m: {side: {name: dict(langs=xs, configs=nob(cell)) for name, xs, cell in v} for side, v in s.items()}
           for m, s in macros.items()}
    json.dump(dict(dataset=manifest["source"] | {k: manifest[k] for k in ("jsonl_sha256", "languages")},
                   chrf_signature=sig, protocol=PROTOCOL, baseline=base, boot=a.boot, seed=a.seed,
                   **{m: {d: nob(cl) for d, cl in t.items()} for m, t in tab.items()}, macro=mac,
                   configs={c: {k: v for k, v in m.items() if k != "example_prompt"} for c, m in meta.items()},
                   diagnostics=res["diag"], tokens_per_char=res["tpc"], speed=res["speed"], counts=res["counts"],
                   code=res["code"], ids={d: sorted(v[0]) for d, v in next(iter(res["chrf"].values()), {}).items()}),
              open(out / "numbers.json", "w"), indent=1, ensure_ascii=False)
    with open(out / "numbers.csv", "w") as f:
        f.write("metric,config,direction,value,n,delta,ci_lo,ci_hi\n")
        cells = [(m, d, cl) for m, t in tab.items() for d, cl in t.items()] + [
            (m, f"macro {side} {name} ({len(xs)})", cl) for m, s in macros.items() for side, v in s.items()
            for name, xs, cl in v]
        f.writelines(f"{m},{c},{d},{v['score']:.6g},{v['n']},{v.get('delta', '')},"
                     f"{','.join(map(str, v.get('ci', ['', ''])))}\n" for m, d, cl in cells for c, v in cl.items())
    (out / "chrf.svg").write_text(svg_chart(tab["chrf"], langs, [c for c in cfgs if "@" not in c], meta))
    print("\n".join(md) + f"\nchrF++ signature: {sig}\nwrote {out}/tables.md, numbers.json, numbers.csv, chrf.svg")
