import argparse
import glob
import json
import sys

import torch

sys.path.insert(0, "src")
from engine import load_engine
from ops import softmax
from quant import quantize

GOLDEN = {"qwen3.5-0.8b": "tests/golden_qwen35", "qwen3.5-2b": "tests/golden_qwen35_2b"}


def kl_scores(model, goldens, vocab):
    """Mean KL at position 0 and at the later positions."""
    pos0, later = [], []
    for g in goldens:
        p = softmax(g["logits"][:, :vocab].float()); q = softmax(model.forward(g["ids"]).cpu()[:, :vocab])
        kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(-1)
        pos0.append(kl[0].item()); later.append(kl[1:].mean().item())
    return sum(pos0) / max(len(pos0), 1), sum(later) / len(later)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--golden-dir", default=None, help="the reference (default: the model's)")
    ap.add_argument("--threshold", type=float, default=0.005, help="KL damage (nats) above which a tensor stays INT8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"configs/quant/{a.model}.json"
    golden_dir = a.golden_dir or GOLDEN[a.model]
    files = sorted(glob.glob(f"{golden_dir}/*.pt"))
    if not files:                                                      # else every KL is 0 and nothing is kept INT8
        sys.exit(f"no reference in {golden_dir}; run scripts/golden_qwen35.py")
    torch.set_grad_enabled(False)
    eng = load_engine(a.model, "mps")
    goldens = [torch.load(f) for f in files]
    m, vocab = eng.model, eng.tokenizer.vocab_size()
    base0, base_later = kl_scores(m, goldens, vocab)                      # also fills the resident cache
    keys = [k for k, w in m._cache.items() if isinstance(w, torch.Tensor) and w.ndim == 2
            and w.dtype == torch.bfloat16 and "embed" not in k]
    damage = {}
    for n, key in enumerate(keys):
        orig = m._cache[key]
        m._cache[key] = quantize(orig.float().cpu(), "int4").dequantize().to(orig.device, torch.bfloat16)
        try:
            k0, kl = kl_scores(m, goldens, vocab)
        finally:                                    # put the baseline back even if a forward fails
            m._cache[key] = orig
        damage[key] = {"pos0": round(k0 - base0, 4), "later": round(kl - base_later, 4)}
        print(f"\r{n + 1}/{len(keys)} tensors measured", end="", flush=True)
    keep = sorted(k for k, v in damage.items() if max(v["pos0"], v["later"]) > a.threshold)
    policy = {"model": a.model, "scheme": "int4", "threshold_nats": a.threshold, "reference": golden_dir,
              "baseline_kl_bf16": {"pos0": round(base0, 4), "later": round(base_later, 4)},
              "keep_int8": keep, "always_int8": ["embed_tokens.weight"],
              "damage": dict(sorted(damage.items(), key=lambda kv: -max(kv[1].values())))}
    json.dump(policy, open(out, "w"), indent=1)
    print(f"\n{len(keep)} of {len(damage)} tensors stay INT8\nwrote {out}")


if __name__ == "__main__":
    main()
