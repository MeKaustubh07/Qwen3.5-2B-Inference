import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, "src")
from engine import load_engine
from serving_targets import free_gpu, gpu_peak, kl, metal_aborts, top2_gap

D, LONG, SHORT = "models/sarvam-translate", "tests/golden_gemma3_long/0.pt", "tests/golden_gemma3/0.pt"
BUILDS = {"int8": ("metal-int8", "model.int8.qt"), "int4": ("metal-int4", "model.int4.qt")}
# KL (over the rows, the decode steps and the short sequence), top-1 flips: margins over what each build measured on
# this run (INT8 0.0016 / 0 of 34, the policy INT4 0.0516 / 1); the short key's gates are tests/test_gemma3_quant.py's
GATES = {"int8": (0.005, 0.03), "int4": (0.065, 0.06)}
CHUNK, BS = 300, 16                                            # chunks 0-300, ..., 900-1200 (across 1024), 1200-1416
if not os.path.exists(f"{D}/model.safetensors.index.json") or not torch.backends.mps.is_available():
    print(f"SKIP: needs the checkpoint in {D} and a Metal GPU")
    sys.exit(0)
if not os.path.exists(LONG) or not os.path.exists(SHORT):
    print("FAIL: no answer keys: run scripts/.venv/bin/python scripts/golden_gemma3.py and golden_gemma3.py --long")
    sys.exit(1)

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


t0, started = time.perf_counter(), time.strftime("%Y-%m-%d %H:%M:%S")
key, short = torch.load(LONG), torch.load(SHORT)
ids, rows, g = key["ids"], key["rows"], key["greedy"].tolist()
n, n_steps = len(ids), len(g) - 1
s_ids, s_g = short["ids"], short["greedy"].tolist()
next_ref = torch.cat([key["logits"][-1:], key["step_logits"]])     # the key's logits predicting each greedy token


def prefill(m, st):
    return torch.cat([m.packed_hidden([(ids[i:i + CHUNK], st)]) for i in range(0, n, CHUNK)])


summary, peaks = {}, {}
for build, (backend, file) in BUILDS.items():
    qt = f"{D}/{file}"
    if not os.path.exists(qt):
        check(f"{build}: {qt} exists (scripts/quantize.py; see tests/test_gemma3_quant.py)", False)
        continue
    tb = time.perf_counter()
    eng = load_engine("sarvam-translate", backend, qt)
    m, V, W = eng.model, eng.tokenizer.vocab_size(), eng.model.config.sliding_window
    free_gpu()
    torch.accelerator.reset_peak_memory_stats()

    st = m.new_state(n + n_steps)                                   # 1. contiguous, the same chunks
    h_c = prefill(m, st).cpu()
    steps_c = torch.cat([m.forward(torch.tensor([t]), state=st) for t in g[:-1]]).float().cpu()
    del st
    free_gpu()

    probe = m.new_paged_pool(1, BS, max_seqs=1, max_chunk=CHUNK).kv  # 2. the paged pool with rings, poisoned
    units = probe.blocks_for(n + n_steps) + probe.blocks_for(len(s_ids) + n_steps)
    pool = m.new_paged_pool(-(-units // probe.n_groups), BS, max_seqs=2, max_chunk=CHUNK, poison=True)
    del probe
    kv, (long_st, short_st) = pool.kv, (m.new_paged_state(pool), m.new_paged_state(pool))
    h_p = prefill(m, long_st)
    lg = m.head(h_p[rows]).float().cpu()
    h_p = h_p.cpu()
    m.forward(s_ids, state=short_st, last_only=True)
    out = [m.decode_batch([a, b], [long_st, short_st]).float().cpu() for a, b in zip(g[:-1], s_g)]
    steps_p, steps_s = torch.cat([o[:1] for o in out]), torch.cat([o[1:] for o in out])
    R, tags = kv.ring, [t for t, sl in zip(long_st.kv.tags, kv.sliding) if sl]
    reused = sorted({sum(b >= R for b in t) for t in tags})        # slots now holding block R or later
    held = sorted({len(t) for t, sl in zip(long_st.kv.tables, kv.sliding) if not sl})
    rings_full = all(len(t) == R for t in tags)
    del long_st, short_st, pool, kv
    peaks[build] = gpu_peak()
    del eng, m
    free_gpu()

    ours, ref = torch.cat([lg, steps_p]), torch.cat([key["logits"], key["step_logits"]])
    k_all, clear = kl(ref, ours, V), top2_gap(ref, V) > 0.5
    preds = torch.cat([lg[-1:], steps_p])[:, :V].argmax(-1)
    miss = (preds != key["greedy"]).nonzero()
    free = int(miss[0]) if len(miss) else len(g)
    nr = len(rows)
    s_ref = short["step_logits"][:n_steps]
    summary[build] = x = dict(
        same=torch.equal(h_p, h_c), finite=bool(all(torch.isfinite(t).all() for t in (h_p, lg, steps_p, steps_s))),
        reused=reused, held=held, ring=R, rings_full=rings_full,
        before=k_all[:nr][rows < W].mean().item(), after=k_all[:nr][rows >= W].mean().item(),
        steps=k_all[nr:].mean().item(), all=k_all.mean().item(), max=k_all.max().item(),
        flips=int(((ours[:, :V].argmax(-1) != ref[:, :V].argmax(-1)) & clear).sum()), clear=int(clear.sum()),
        agree=int((preds == key["greedy"]).sum()), free=free,
        free_gap=top2_gap(next_ref[free:free + 1], V).item() if free < len(g) else None,
        short=kl(s_ref, steps_s, V).mean().item(),
        short_agree=int((steps_s[:, :V].argmax(-1) == s_ref[:, :V].argmax(-1)).sum()),
        vs_contig=kl(steps_c, steps_p, V), steps_c=kl(key["step_logits"], steps_c, V).mean().item())
    print(f"      {build}, bf16 KV, paged ({x['ring']}-block rings, {x['reused']} slots reused per sliding layer; full "
          f"layers {x['held']} blocks): KL vs the fp32 key, rows before {W} {x['before']:.4f}, from it on "
          f"{x['after']:.4f}, decode steps {x['steps']:.4f} (contiguous {x['steps_c']:.4f}; max {x['max']:.3f}) | "
          f"top-1 flips {x['flips']}/{x['clear']} | greedy: {x['agree']}/{len(g)} the argmax on the key's history, "
          f"free run == key for {x['free']}" + (f" (gap {x['free_gap']:.2f})" if x["free_gap"] is not None else "")
          + f" | short sequence in the batch: KL {x['short']:.4f}, argmax == key {x['short_agree']}/{n_steps} | "
          f"paged batched vs contiguous decode: KL mean {x['vs_contig'].mean().item():.1e}, max "
          f"{x['vs_contig'].max().item():.1e} | GPU {peaks[build] / 2**30:.2f} GiB | {time.perf_counter() - tb:.0f} s",
          flush=True)

for build, x in summary.items():                                   # every build: a broken one must not hide
    T, F = GATES[build]
    check(f"{build}: prefill in {CHUNK}-token chunks (one across position 1024) into {x['ring']}-block rings, "
          f"{x['reused']} slots of every ring reused and NaN-poisoned first: every logit finite, hidden states of all "
          f"{n} positions bit-identical to the contiguous cache's ({x['same']})",
          x["same"] and x["finite"] and x["rings_full"] and min(x["reused"]) > 0)
    check(f"{build}: KL vs the fp32 key < {T} over its {len(rows)} rows and {n_steps} batched paged decode steps "
          f"({x['all']:.4f}; from position {W} {x['after']:.4f}), over the decode steps alone ({x['steps']:.4f}) and "
          f"for the short sequence decoded beside it ({x['short']:.4f}); top-1 flips <= {F:.0%} of non-tied positions "
          f"({x['flips']}/{x['clear']})",
          max(x["all"], x["steps"], x["short"]) < T and x["flips"] <= F * x["clear"])
    check(f"{build}: batched paged decode == contiguous decode within fp32 rounding: KL(contiguous || paged) max "
          f"{x['vs_contig'].max().item():.1e} < 1e-3", x["vs_contig"].max().item() < 1e-3)
if "int8" in summary:                                               # INT8 may leave the key only at a near-tie
    x = summary["int8"]
    check(f"int8: free greedy == the key for {x['free']}/{len(g)} tokens"
          + (f"; it leaves the key where the key's top two are {x['free_gap']:.2f} apart (<= 0.5)"
             if x["free_gap"] is not None else ""), x["free_gap"] is None or x["free_gap"] <= 0.5)
if peaks:                                                           # Metal aborts command buffers it cannot fit,
    rec = torch.mps.recommended_max_memory()                       # and torch 2.14 then returns garbage silently
    worst = max(peaks, key=peaks.get)
    check(f"GPU memory: every build stays under 0.95 x Metal's recommended {rec / 2**30:.2f} GiB (highest: {worst}, "
          f"{peaks[worst] / 2**30:.2f} GiB)", all(p <= 0.95 * rec for p in peaks.values()))
ok, what = metal_aborts(started)                                    # torch 2.14 never reads a command buffer's status
check(f"Metal aborted no command buffer of this process ({what})", ok)
print(f"      {time.perf_counter() - t0:.0f} s in all")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
