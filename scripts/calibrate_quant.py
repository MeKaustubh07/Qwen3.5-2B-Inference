import argparse
import glob
import hashlib
import json
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from engine import MODELS, load_engine
from models.packing import pack
from ops import softmax
from quant import QtFile, QuantTensor, policy_group, quantize

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = {"qwen3.5-0.8b": "tests/golden_qwen35", "qwen3.5-2b": "tests/golden_qwen35_2b",
          "tiny-aya-global": "tests/golden_aya", "sarvam-translate": "models/sarvam-translate-calib/ref"}
MAKE = {"tiny-aya-global": "scripts/golden_aya.py", "sarvam-translate": "scripts/golden_gemma3.py --calib"}
PACK_TOKENS = 512                   # tokens per packed prefill of a rows reference
HEAD_ROWS, KL_ROWS = 128, 16        # rows through the head at a time; rows of 262K-wide KL temporaries at a time
CURVE = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112)    # prefixes of the per-byte ranking measured
BOOTSTRAP = 2000                    # resamples of the prompts behind the rule's choice


def kl_scores(model, goldens, vocab):
    """Mean KL at position 0 (over the references that keep it) and at the later positions."""
    pos0, later = [], []
    for g in goldens:
        start = g.get("logits_from", 0)
        p = softmax(g["logits"][:, :vocab].float()); q = softmax(model.forward(g["ids"]).cpu()[start:, :vocab])
        kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(-1)
        if start == 0:
            pos0.append(kl[0].item()); later.append(kl[1:].mean().item())
        else:
            later.append(kl.mean().item())
    return sum(pos0) / max(len(pos0), 1), sum(later) / len(later)


# ---------------------------------------------------------------- rows reference

def kl_rows(logits: torch.Tensor, rows: torch.Tensor, vocab: int) -> torch.Tensor:
    """llama.cpp's KL(base || ours) per row (lang_kl.score; here in fp32 on the logits' device): logits [R, >= vocab],
    rows: the reference's uint16 rows as int16 [R, nv] on that device (a float32 scale and min log-probability, then
    each log-probability as a uint16 step)."""
    head = rows[:, :4].contiguous().view(torch.float32)
    plb = head[:, :1] * (rows[:, 4:4 + vocab].int() & 0xFFFF).float() + head[:, 1:2]
    logq = torch.log_softmax(logits[:, :vocab].float(), -1)
    return torch.where(plb > -16.0, plb.exp() * (plb - logq), 0.0).sum(-1)


def packs(prompts: list[dict]) -> list[list[dict]]:
    """Consecutive prompts in groups of at most PACK_TOKENS tokens (a longer prompt alone)."""
    out, cur = [], []
    for p in prompts:
        if cur and sum(len(q["ids"]) for q in cur) + len(p["ids"]) > PACK_TOKENS:
            out.append(cur); cur = []
        cur.append(p)
    return out + [cur] if cur else out


class Mixed:
    """The tensors of an INT8 and a flat INT4 .qt of one checkpoint: the groups in `int4` (quant.policy_group) from
    the INT4 file, the rest from the INT8 one: the weights quantize.py writes with a policy, without writing them."""

    def __init__(self, q8: QtFile, q4: QtFile):
        self.q8, self.q4, self.int4 = q8, q4, frozenset()

    def tensor_names(self) -> list[str]:
        return self.q8.tensor_names()

    def get(self, name: str):
        return (self.q4 if policy_group(name) in self.int4 else self.q8).get(name)


class Streamed:
    """Prompts in packs, run one layer at a time: every pack goes through layer i, then the layer's weights leave the
    cache, so the GPU holds one layer and the embedding. Each pack's input to layer `at` is kept (made with the weights
    of advance()'s time), so passes may start there. built: every 2-D weight a layer built -> the bytes INT8 costs over
    INT4 for it, in the model's order."""

    def __init__(self, model, prompts: list[dict]):
        self.m, self.groups = model, packs(prompts)
        self.args = [pack([(torch.tensor(p["ids"]), None) for p in g], model.b.device) for g in self.groups]
        self.at, self.h, self.built = 0, [model.embed(ids) for ids, _, _ in self.args], {}

    def _layer(self, i: int, hs: list[torch.Tensor]) -> list[torch.Tensor]:
        out = [self.m.block(i, h, partial(self.m._attend, positions=pos, segs=segs))
               for h, (_, pos, segs) in zip(hs, self.args)]
        for k in [k for k in self.m._cache if k.startswith(f"layers.{i}.")]:
            if isinstance(w := self.m._cache.pop(k), QuantTensor):
                self.built[k] = extra_bytes(w)
        return out

    def advance(self, layer: int) -> None:
        while self.at < layer:
            self.h, self.at = self._layer(self.at, self.h), self.at + 1

    def hidden(self) -> list[torch.Tensor]:
        """Each pack's final-norm hidden states, from layer `at` on with the weights as they are now."""
        hs = self.h
        for i in range(self.at, self.m.config.num_hidden_layers):
            hs = self._layer(i, hs)
        return [self.m._final_norm(h) for h in hs]


def rows_kl(model, streamed: Streamed, ref_dir: Path, vocab: int) -> dict[int, tuple[float, int]]:
    """Prompt index -> (sum of KL over its reference rows, number of rows)."""
    out, dev = {}, model.b.device
    for group, h in zip(streamed.groups, streamed.hidden()):
        lo = 0
        for p in group:
            hp, lo = h[lo + p["logits_from"]:lo + len(p["ids"]) - 1], lo + len(p["ids"])
            rows = torch.from_numpy(np.load(ref_dir / p["file"]).view(np.int16)).to(dev)
            assert len(rows) == len(hp), (p["file"], len(rows), len(hp))
            total = torch.zeros((), device=dev)
            for r in range(0, len(hp), HEAD_ROWS):
                lg = model.head(hp[r:r + HEAD_ROWS])
                for s in range(0, len(lg), KL_ROWS):
                    total += kl_rows(lg[s:s + KL_ROWS], rows[r + s:r + s + KL_ROWS], vocab).sum()
                del lg
            out[p["i"]] = (total.item(), len(hp))
    return out


def mean_kl(per_prompt: dict, prompts: list[dict]) -> float:
    s = [per_prompt[p["i"]] for p in prompts]
    return sum(a for a, _ in s) / sum(n for _, n in s)


def breakdown(per_prompt: dict, prompts: list[dict]) -> dict:
    """Mean KL per split, per split and direction (into English / out of it), and per split and language."""
    groups = defaultdict(list)
    for p in prompts:
        d = "x2eng" if p["tgt"] == "eng" else "eng2x"
        for k in (p["split"], f"{p['split']}/{d}", f"{p['split']}/{p['lang']}"):
            groups[k].append(p)
    return {k: round(mean_kl(per_prompt, v), 5) for k, v in sorted(groups.items())}


def rule_choice(points: list[dict], kls: list[float]) -> int:
    """The rule: keep the next prefix of the ranking while its KL drop per extra byte is at least the average of going
    all INT8 (points: the curve, its last the all-INT8 endpoint; kls: their KL) -> the prefix length kept."""
    rate = (kls[0] - kls[-1]) / points[-1]["extra_bytes"]
    chosen = 0
    for (prev, k0), (nxt, k1) in zip(zip(points, kls), zip(points[1:], kls[1:])):
        if (k0 - k1) / (nxt["extra_bytes"] - prev["extra_bytes"]) < rate:
            break
        chosen = nxt["n"]
    return chosen


def choice_spread(points: list[dict], per_point: list[dict], prompts: list[dict], seed: int = 0) -> dict:
    """How often the rule picks each prefix when the prompts are resampled with replacement -> {n: share}."""
    g, counts = np.random.default_rng(seed), defaultdict(int)
    for _ in range(BOOTSTRAP):
        sample = [prompts[i] for i in g.integers(0, len(prompts), len(prompts))]
        counts[rule_choice(points, [mean_kl(r, sample) for r in per_point])] += 1
    return {n: round(c / BOOTSTRAP, 3) for n, c in sorted(counts.items())}


def extra_bytes(w: QuantTensor) -> int:
    """What keeping w INT8 costs over INT4: INT8 1 B + an fp16 scale per 32 weights, INT4 0.5 B + scale and min."""
    n = w.shape[0] * w.shape[1]
    return n + n // 16 - (n // 2 + n // 8)


def qt_files(model: str) -> tuple[str, str]:
    """The model's INT8 and flat INT4 .qt, checked: every quantized tensor of the first INT8, of the second INT4
    (the tied embedding INT8 in both)."""
    d = ROOT / MODELS[model]["dir"]
    paths = (d / "model.int8.qt", d / "model.int4-flat.qt")
    for path, scheme in zip(paths, ("int8", "int4")):
        if not path.exists():
            sys.exit(f"no {path}: scripts/quantize.py {d / MODELS[model]['weights']} {path} --scheme {scheme} "
                     f"(--prefix for a multimodal checkpoint)")
        q = QtFile(str(path))
        wrong = [n for n in q.tensor_names() if isinstance(w := q.get(n), QuantTensor)
                 and w.scheme != ("int8" if "embed_tokens" in n else scheme)]
        if wrong:
            sys.exit(f"{path} is not a flat {scheme} file: {wrong[:3]} ...")
    return tuple(str(p) for p in paths)


def calibrate_rows(a, ref_dir: Path, out: str) -> None:
    q8_path, q4_path = qt_files(a.model)
    eng = load_engine(a.model, "metal-int8", q8_path)
    m, vocab = eng.model, eng.tokenizer.vocab_size()
    eng.tokenizer = None                            # ~0.4 GB of Python objects the rest never needs (8 GB)
    m.weights = mixed = Mixed(m.weights, QtFile(q4_path))
    index = json.load(open(ref_dir / "index.json", encoding="utf-8"))
    prompts = [r | {"i": i} for i, r in enumerate(index)]
    calib = [p for p in prompts if p["split"] == a.split]
    if not calib:
        sys.exit(f"no prompts of split {a.split!r} in {ref_dir}/index.json")
    ref_sha = hashlib.sha256((ref_dir / "index.json").read_bytes()).hexdigest()
    state_path = ROOT / "build" / "calibrate" / f"{a.model}.json"
    state = json.load(open(state_path)) if state_path.exists() else {}
    calib_sha = hashlib.sha256(json.dumps([{k: v for k, v in p.items() if k != "i"} for p in calib],
                                          ensure_ascii=False).encode()).hexdigest()
    run = {"calib_sha256": calib_sha, "split": a.split, "pack_tokens": PACK_TOKENS}   # what the damage depends on
    if any(state.get(k) != v for k, v in run.items()):             # damage measured otherwise: start again
        state = {"reference": str(ref_dir)} | run | {"damage": {}}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    save = lambda: json.dump(state, open(state_path, "w"), indent=1)

    t0 = time.perf_counter()
    stream = Streamed(m, prompts)
    base = rows_kl(m, stream, ref_dir, vocab)
    base_kl, size = mean_kl(base, calib), stream.built
    keys = list(size)
    print(f"INT8 baseline: KL {base_kl:.5f} on {sum(n for _, n in (base[p['i']] for p in calib))} rows of "
          f"{len(calib)} {a.split} prompts; {len(keys)} tensors, {len(keys) - len(state['damage'])} to measure  "
          f"[{time.perf_counter() - t0:.0f} s]", flush=True)
    stream, checked = Streamed(m, calib), False
    for n, key in enumerate(keys):
        if key in state["damage"]:
            continue
        layer = int(key.split(".")[1])
        stream.advance(layer)                       # the baseline's input to this tensor's layer
        mixed.int4 = frozenset({policy_group(key)})
        try:
            kl = mean_kl(rows_kl(m, stream, ref_dir, vocab), calib)
            if layer and not checked:               # once: the same number from layer 0 on
                full = mean_kl(rows_kl(m, Streamed(m, calib), ref_dir, vocab), calib)
                print(f"check: {key} from layer {layer}: KL {kl!r}, from layer 0: {full!r}", flush=True)
                if full != kl:
                    sys.exit("starting at a layer changed the KL")
                checked = True
        finally:
            mixed.int4 = frozenset()
        state["damage"][key] = round(kl - base_kl, 7)
        save()
        print(f"{n + 1}/{len(keys)} {key}: damage {kl - base_kl:+.5f} ({size[key] / 1e6:.1f} MB)  "
              f"[{time.perf_counter() - t0:.0f} s]", flush=True)
    damage = state["damage"]
    ranked = sorted(keys, key=lambda k: -damage[k] / size[k])
    every = frozenset(policy_group(k) for k in keys)
    curve, per_point = [], []
    for n in [0] + [n for n in CURVE if n < len(keys)]:
        mixed.int4 = every - {policy_group(k) for k in ranked[:n]}
        per_point.append(rows_kl(m, Streamed(m, prompts), ref_dir, vocab))
        curve.append(dict(n=n, extra_bytes=sum(size[k] for k in ranked[:n]), kl=round(mean_kl(per_point[-1], calib), 5),
                          by=breakdown(per_point[-1], prompts)))
        print(f"top {n} INT8 (+{curve[-1]['extra_bytes'] / 1e6:.0f} MB): KL {mean_kl(per_point[-1], calib):.5f}  "
              f"[{time.perf_counter() - t0:.0f} s]", flush=True)
    per_point.append(base)                                        # every tensor INT8: the baseline
    curve.append(dict(n=len(keys), extra_bytes=sum(size.values()), kl=round(base_kl, 5), by=breakdown(base, prompts)))
    rate = (mean_kl(per_point[0], calib) - base_kl) / sum(size.values())    # the average KL per byte of all INT8
    chosen = rule_choice(curve, [mean_kl(r, calib) for r in per_point])
    splits = sorted({p["split"] for p in prompts})
    choice = {"chosen": {sp: rule_choice(curve, [mean_kl(r, [p for p in prompts if p["split"] == sp])
                                                 for r in per_point]) for sp in splits},
              "bootstrap": {sp: choice_spread(curve, per_point, [p for p in prompts if p["split"] == sp])
                            for sp in splits},
              "resamples": BOOTSTRAP}
    picks = {n for spread in choice["bootstrap"].values() for n, share in spread.items() if share >= 0.1}
    if len(set(choice["chosen"].values()) | picks) > 1:
        choice["note"] = (f"not resolved by this data: the rule keeps {choice['chosen']} tensors by split, and "
                          f"{sorted(picks)} each in 10%+ of the resamples; {a.split} decides")
    print(f"the rule on each split: {choice['chosen']}; over {BOOTSTRAP} resamples of the prompts: "
          f"{choice['bootstrap']}", flush=True)
    policy = {"model": a.model, "scheme": "int4", "rule": "KL per extra byte", "reference": str(ref_dir),
              "reference_sha256": ref_sha, "split": a.split, "baseline_kl_int8": round(base_kl, 5),
              "average_kl_per_mb": round(rate * 1e6, 7),
              "keep_int8": sorted(ranked[:chosen]), "always_int8": ["embed_tokens.weight"],
              "choice": choice, "curve": curve,
              "damage": {k: {"kl": damage[k], "extra_bytes": size[k], "kl_per_mb": round(damage[k] / size[k] * 1e6, 7)}
                         for k in ranked}}
    json.dump(policy, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"{chosen} of {len(keys)} tensors stay INT8 (+{sum(size[k] for k in ranked[:chosen]) / 1e6:.0f} MB)\n"
          f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--baseline", choices=["bf16", "int8"], default="bf16",
                    help="precision of every other tensor while one is measured in INT4 (int8: bf16 does not fit)")
    ap.add_argument("--golden-dir", default=None, help="the reference (default: the model's)")
    ap.add_argument("--threshold", type=float, default=0.005, help="KL damage (nats) above which a tensor stays INT8")
    ap.add_argument("--split", default="calib", help="rows reference: the prompts measured (the rest are held out)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"configs/quant/{a.model}.json"
    golden_dir = a.golden_dir or GOLDEN[a.model]
    rows_ref = (Path(golden_dir) / "index.json").exists()
    files = sorted(glob.glob(f"{golden_dir}/*.pt"))
    if not rows_ref and not files:                                     # else every KL is 0 and nothing is kept INT8
        sys.exit(f"no reference in {golden_dir}; run scripts/{MAKE.get(a.model, 'golden_qwen35.py')}")
    if rows_ref and a.baseline != "int8":
        sys.exit("a rows reference is measured against the int8 baseline (--baseline int8)")
    torch.set_grad_enabled(False)
    if rows_ref:
        return calibrate_rows(a, Path(golden_dir), out)
    eng = load_engine(a.model, "mps" if a.baseline == "bf16" else "metal-int8")   # int8: quantized at load
    goldens = [torch.load(f) for f in files]
    m, vocab = eng.model, eng.tokenizer.vocab_size()
    base0, base_later = kl_scores(m, goldens, vocab)                      # also fills the resident cache
    if a.baseline == "bf16":
        keys = [k for k, w in m._cache.items() if isinstance(w, torch.Tensor) and w.ndim == 2
                and w.dtype == torch.bfloat16 and "embed" not in k]
    else:
        keys = [k for k, w in m._cache.items() if isinstance(w, QuantTensor) and "embed" not in k]
    damage = {}
    for n, key in enumerate(keys):
        orig = m._cache[key]
        if a.baseline == "bf16":
            m._cache[key] = quantize(orig.float().cpu(), "int4").dequantize().to(orig.device, torch.bfloat16)
        else:                                       # the next forward rebuilds only this tensor, in INT4, from the
            del m._cache[key]                       # checkpoint (every other tensor is still cached in INT8)
            m.b.scheme = "int4"
        try:
            k0, kl = kl_scores(m, goldens, vocab)
        finally:                                    # put the baseline back even if a forward fails
            if a.baseline == "int8":
                m.b.scheme = "int8"
            m._cache[key] = orig
        damage[key] = {"pos0": round(k0 - base0, 4), "later": round(kl - base_later, 4)}
        print(f"\r{n + 1}/{len(keys)} tensors measured", end="", flush=True)
    keep = sorted(k for k, v in damage.items() if max(v["pos0"], v["later"]) > a.threshold)
    policy = {"model": a.model, "scheme": "int4", "threshold_nats": a.threshold, "reference": golden_dir,
              f"baseline_kl_{a.baseline}": {"pos0": round(base0, 4), "later": round(base_later, 4)},
              "keep_int8": keep, "always_int8": ["embed_tokens.weight"],
              "damage": dict(sorted(damage.items(), key=lambda kv: -max(kv[1].values())))}
    json.dump(policy, open(out, "w"), indent=1)
    print(f"\n{len(keep)} of {len(damage)} tensors stay INT8\nwrote {out}")


if __name__ == "__main__":
    main()
