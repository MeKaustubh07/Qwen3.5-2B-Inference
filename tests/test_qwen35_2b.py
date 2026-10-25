import gc
import glob
import os
import sys
import time
import torch

sys.path.insert(0, "src")
from engine import load_engine
from generate import generate_greedy
from ops import softmax
from quant import QuantTensor

D = "models/qwen3.5-2b"
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")

def greedy_vs_hf(m, tok, g, eos):
    """-> (tokens equal to HF before the first divergence, divergence ok, gap). A divergence is acceptable only at
    a near-tie: HF's token must be our runner-up within 0.5 logits (HF rounds activations to bf16, we keep fp32)."""
    out, ref = generate_greedy(m, tok, g["text"], 10, eos), g["greedy"].tolist()
    j = next((j for j, (x, y) in enumerate(zip(out, ref)) if x != y), None)
    if j is None:
        return min(len(out), len(ref)), True, None
    lg = m.forward(torch.cat([g["ids"], torch.tensor(ref[:j], dtype=g["ids"].dtype)]), last_only=True)[0].float().cpu()
    top = lg.topk(2)
    gap = (top.values[0] - top.values[1]).item()
    return j, int(top.indices[1]) == ref[j] and gap < 0.5, gap


goldens = [torch.load(f) for f in sorted(glob.glob("tests/golden_qwen35_2b/*.pt"), key=lambda f: int(os.path.basename(f)[:-3]))]
variants = [("bf16", "metal", None), ("int8", "metal-int8", f"{D}/model.int8.qt"), ("int4", "metal-int4", f"{D}/model.int4.qt")]
summary = {}
for label, backend, qt in variants:
    if qt and not os.path.exists(qt):
        check(f"{label}: {qt} exists (run scripts/quantize.py first)", False); continue
    eng = load_engine("qwen3.5-2b", backend, qt)
    m, tok, vocab = eng.model, eng.tokenizer, eng.tokenizer.vocab_size()
    match = tot = flips = clear_n = 0; kl = 0.0; ties_ok = True; gaps = []
    for g in goldens:
        lg = m.forward(g["ids"]).cpu()[:, :vocab]; ref = g["logits"][:, :vocab]
        p, q = softmax(ref), softmax(lg)
        kl += (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(-1).mean().item() / len(goldens)
        top2 = ref.topk(2, -1).values; clear = (top2[:, 0] - top2[:, 1]) > 0.5
        flips += int(((lg.argmax(-1) != ref.argmax(-1)) & clear).sum()); clear_n += int(clear.sum())
        j, ok, gap = greedy_vs_hf(m, tok, g, eng.eos_ids)
        match += j; tot += 10; ties_ok &= ok
        if gap is not None:
            gaps.append(round(gap, 3))
    gb = sum(w.nbytes if isinstance(w, QuantTensor) else w.numel() * w.element_size() for w in m._cache.values()) / 1e9
    st = m.new_state(128); m.forward(torch.tensor(tok.encode("The capital of France is")), state=st, last_only=True)
    x = torch.tensor([12])
    for _ in range(3): m.forward(x, state=st, last_only=True)
    m.b.sync(); t0 = time.perf_counter()
    for _ in range(20): int(m.forward(x, state=st, last_only=True)[0].argmax())
    tps = 20 / (time.perf_counter() - t0)
    answer = tok.decode(generate_greedy(m, tok, "The capital of France is", 6, eng.eos_ids))
    summary[label] = dict(match=match, tot=tot, flips=flips, clear=clear_n, kl=kl, gb=gb, tps=tps, answer=answer,
                          ties_ok=ties_ok, gaps=gaps)
    print(f"      {label:5s}: weights {gb:.2f} GB | greedy = HF for {match}/{tot} tokens | top-1 flips {flips}/{clear_n} "
          f"non-tied positions | KL vs HF bf16 {kl:.4f} | decode {tps:.1f} tok/s | {answer!r} | divergence gaps {gaps}")
    del eng, m; gc.collect(); torch.mps.empty_cache()

s = summary
if "bf16" in s:
    check(f"bf16: greedy == HF bf16 ({s['bf16']['match']}/{s['bf16']['tot']} tokens) except at near-ties "
          f"(HF's token is our runner-up within 0.5 logits; gaps {s['bf16']['gaps']})",
          s["bf16"]["ties_ok"] and s["bf16"]["match"] >= 0.8 * s["bf16"]["tot"])
if "int8" in s:
    check(f"int8: KL vs HF bf16 < 0.05 ({s['int8']['kl']:.4f}) and top-1 flips <= 2% of non-tied positions", s["int8"]["kl"] < 0.05 and s["int8"]["flips"] <= 0.02 * s["int8"]["clear"])
if "int4" in s:
    check(f"int4 (calibrated policy): KL vs HF bf16 < 0.15 ({s['int4']['kl']:.4f}) and top-1 flips <= 10% of non-tied positions",
          s["int4"]["kl"] < 0.15 and s["int4"]["flips"] <= 0.10 * s["int4"]["clear"])
    check(f"int4 fits comfortably in 8 GB: {s['int4']['gb']:.2f} GB of weights", s["int4"]["gb"] < 1.6)
    check("int4 still answers Paris", "Paris" in s["int4"]["answer"])
if {"bf16", "int8", "int4"} <= s.keys():
    check(f"quantization speeds up decode: bf16 {s['bf16']['tps']:.1f} < int8 {s['int8']['tps']:.1f} and int4 {s['int4']['tps']:.1f} tok/s",
          s["bf16"]["tps"] < s["int8"]["tps"] and s["bf16"]["tps"] < s["int4"]["tps"])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
