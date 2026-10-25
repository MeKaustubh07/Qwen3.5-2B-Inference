import argparse
import collections
import json
import random
import sys
import time

import httpx

sys.path.insert(0, "src")
from decision import prepare, score
from engine import MODELS, ROOT, load_engine

API = "https://datasets-server.huggingface.co/rows"
DATASET = dict(dataset="avbiswas/bev-decision", config="default", split="test")
MAX_OPTIONS, MAX_OPTION_TOKENS = 16, 64                   # the server's limits


def fetch(rng: random.Random, pages: int) -> list[dict]:
    first = httpx.get(API, params={**DATASET, "offset": 0, "length": 1}, timeout=60).json()
    total = first["num_rows_total"]
    rows = []
    for off in rng.sample(range(0, total, 100), min(pages, total // 100)):
        r = httpx.get(API, params={**DATASET, "offset": off, "length": 100}, timeout=120)
        r.raise_for_status()
        rows += [x["row"] for x in r.json()["rows"]]
    return rows


def questions(rows: list[dict]):
    """-> (kind, state, instructions, options, label index or bool, domain) per question."""
    for row in rows:
        for spec in json.loads(row["questions_json"]).values():
            t = spec["type"]
            if t == "choice" and isinstance(spec.get("criteria"), dict):
                keys = list(spec["criteria"])
                if spec.get("label") in keys:
                    yield "choice", row["state"], spec["instructions"], list(spec["criteria"].values()), \
                        keys.index(spec["label"]), row["domain"]
            elif t == "noul" and isinstance(spec.get("label"), bool):
                yield "boolean", row["state"], spec["instructions"], None, spec["label"], row["domain"]
            elif t == "score" and isinstance(spec.get("criteria"), list) and isinstance(spec.get("label"), int):
                yield "score", row["state"], spec["instructions"], spec["criteria"], spec["label"], row["domain"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--backend", default="metal-int4")
    ap.add_argument("--per-type", type=int, default=100)
    ap.add_argument("--max-prompt-tokens", type=int, default=1536)
    ap.add_argument("--shuffle-check", type=int, default=40)
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    scheme = a.backend.removeprefix("metal-") if a.backend in ("metal-int8", "metal-int4") else None
    qt = ROOT / MODELS[a.model]["dir"] / f"model.{scheme}.qt" if scheme else None
    eng = load_engine(a.model, a.backend, str(qt) if qt and qt.exists() else None)

    rows = fetch(rng, a.pages)
    pool = list(questions(rows))
    rng.shuffle(pool)
    picked, skipped = collections.defaultdict(list), collections.Counter()
    for q in pool:
        kind, state, instr, options, label, domain = q
        if len(picked[kind]) >= a.per_type:
            continue
        options_, ctx, opt_ids = prepare(eng, kind, state, instr, options)
        reason = ("prompt too long" if len(ctx) > a.max_prompt_tokens else "too many options"
                  if len(options_) > MAX_OPTIONS else "option too long" if max(map(len, opt_ids)) > MAX_OPTION_TOKENS
                  else None)
        if reason:
            skipped[f"{kind}: {reason}"] += 1
            continue
        picked[kind].append((q, options_, ctx, opt_ids))

    res = collections.defaultdict(list)
    t0 = time.perf_counter()
    for kind, items in picked.items():
        for (q, options, ctx, opt_ids) in items:
            _, _, _, _, label, domain = q
            r = score(eng, kind, options, ctx, opt_ids)
            res[kind].append(dict(r=r, label=label, domain=domain, n=len(options), ctx=len(ctx)))
    elapsed = time.perf_counter() - t0

    shuffled_ok = shuffled_n = 0
    for kind in ("choice", "score"):
        for (q, options, ctx, opt_ids), rec in list(zip(picked[kind], res[kind]))[: a.shuffle_check // 2]:
            perm = list(range(len(options)))
            while perm == sorted(perm):                       # a real shuffle, never the identity
                rng.shuffle(perm)
            r2 = score(eng, kind, [options[i] for i in perm], ctx, [opt_ids[i] for i in perm])
            p1 = [o["probability"] for o in rec["r"]["options"]]
            same = perm[r2["decision"]] == rec["r"]["decision"]
            dp = max(abs(r2["options"][k]["probability"] - p1[i]) for k, i in enumerate(perm))
            shuffled_ok += same and dp == 0
            shuffled_n += 1
            if not (same and dp == 0):
                print(f"shuffle mismatch: {kind}, {len(options)} options, decision kept {same}, max |dp| {dp:.1e}")

    lines = [f"# Zero-shot decisions on avbiswas/bev-decision (test split): {eng.name} {eng.model.b.name}", "",
             f"Sample: {a.pages} random pages of 100 rows (seed {a.seed}), then up to {a.per_type} questions per type "
             f"with a prompt of at most {a.max_prompt_tokens} tokens, at most {MAX_OPTIONS} options and options of at most "
             f"{MAX_OPTION_TOKENS} tokens (questions skipped: {dict(skipped) or 'none'}). "
             f"Scoring: each option's log-probability per token from its own fork of the context's state "
             f"(src/decision.py); chat-formatted prompt, each option scored as a whole word at the start of the assistant's "
             f"answer (its tokens, then a token that cannot continue its last word or number).", "",
             "| type | questions | accuracy | baselines | notes |", "|---|---:|---:|---|---|"]
    if res["choice"]:
        c = res["choice"]
        acc = sum(x["r"]["decision"] == x["label"] for x in c) / len(c)
        rand = sum(1 / x["n"] for x in c) / len(c)
        first = sum(x["label"] == 0 for x in c) / len(c)
        lines.append(f"| choice | {len(c)} | **{acc:.1%}** | random {rand:.1%}; always the first option {first:.1%} | "
                     f"{min(x['n'] for x in c)}-{max(x['n'] for x in c)} options |")
    if res["boolean"]:
        b = res["boolean"]
        acc = sum(x["r"]["decision"] == x["label"] for x in b) / len(b)
        yes = sum(x["label"] for x in b) / len(b)
        lines.append(f"| boolean | {len(b)} | **{acc:.1%}** | majority class {max(yes, 1 - yes):.1%} | "
                     f"{yes:.0%} of labels are yes |")
    if res["score"]:
        s = res["score"]
        acc = sum(x["r"]["decision"] == x["label"] for x in s) / len(s)
        within = sum(abs(x["r"]["decision"] - x["label"]) <= 1 for x in s) / len(s)
        mae = sum(abs(x["r"]["expected"] - x["label"]) for x in s) / len(s)
        mid = sum(x["label"] == (x["n"] - 1) // 2 for x in s) / len(s)
        mid_mae = sum(abs((x["n"] - 1) / 2 - x["label"]) for x in s) / len(s)
        lines.append(f"| score | {len(s)} | **{acc:.1%}** exact, {within:.1%} within one | always the middle label "
                     f"{mid:.1%} exact, MAE {mid_mae:.2f} | MAE of the expected label {mae:.2f} |")
    n_all = sum(len(v) for v in res.values())
    lines += ["", f"Order invariance: {shuffled_ok}/{shuffled_n} shuffled choice/score questions gave the same "
              f"decision with exactly the same probabilities.",
              f"Time: {elapsed:.0f} s for {n_all} questions ({elapsed / max(n_all, 1):.2f} s each; mean prompt "
              f"{sum(x['ctx'] for v in res.values() for x in v) / max(n_all, 1):.0f} tokens).", "",
              "Accuracy by domain (all types):", "", "| domain | questions | accuracy |", "|---|---:|---:|"]
    by = collections.defaultdict(list)
    for kind, v in res.items():
        for x in v:
            by[x["domain"]].append(x["r"]["decision"] == x["label"])
    for d, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {d} | {len(v)} | {sum(v) / len(v):.0%} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
