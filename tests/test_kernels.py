"""Weeks 9-10: every Metal kernel vs the torch reference op at Qwen3.5-0.8B shapes (and Tiny Aya's: LayerNorm,
16/4 x 128 attention; Gemma 3's: GeGLU, RoPE with linear scaling, both against transformers), micro-benchmarks,
weight locking."""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import Gemma3TextConfig
from transformers.activations import ACT2FN
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3RotaryEmbedding, apply_rotary_pos_emb

sys.path.insert(0, "src")
import ops
from backend.metal import MetalBackend
from backend.torch_ref import TorchBackend
from quant import quantize

if not torch.backends.mps.is_available():
    print("SKIP: no MPS device"); sys.exit(0)

mb = MetalBackend()
dev = torch.device("mps")
torch.manual_seed(0)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")
def maxdiff(a, b):
    return (a.float().cpu() - b.float().cpu()).abs().max().item()
def bench(fn, iters=200):
    for _ in range(10): fn()
    torch.mps.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.mps.synchronize(); return (time.perf_counter() - t0) / iters * 1e6   # microseconds

# silu_mul over [gate | up] rows (the MLP width of Qwen3.5-0.8B): a decode row, a batch, a prefill block
for M in (1, 5, 64):
    gu = torch.randn(M, 2 * 3584, device=dev)
    d = maxdiff(mb.silu_mul(gu), ops.silu_mul(gu[:, :3584], gu[:, 3584:]))
    check(f"silu_mul M={M} max|diff|={d:.1e}", d < 1e-5)

# rms_norm with fp32 weights (Qwen3.5 stores 1 + w in fp32): hidden rows (decode row and prefill blocks) and the
# per-head q/k norms; plus one row through the bf16-weight kernel
for shape in ((1, 1024), (7, 1024), (64, 1024), (7, 8, 256), (7, 2, 256)):
    w = 1 + torch.randn(shape[-1], device=dev) * 0.1
    x = torch.randn(*shape, device=dev) * 3
    d = maxdiff(mb.rms_norm(x, w, 1e-6), ops.rms_norm(x, w, 1e-6))
    check(f"rms_norm (fp32 weights) x{list(shape)} max|diff|={d:.1e}", d < 1e-4)
w = (torch.randn(1024) * 0.1).to(torch.bfloat16).to(dev)
x = torch.randn(7, 1024, device=dev) * 3
d = maxdiff(mb.rms_norm(x, w, 1e-6), ops.rms_norm(x, w, 1e-6))
check(f"rms_norm (bf16 weights) T=7 max|diff|={d:.1e}", d < 1e-4)

# layer_norm (Tiny Aya): fp32 weights over hidden rows of 2048; inputs with a large common offset, where a wrong mean
# subtraction would show
for shape, offset in (((1, 2048), 0.0), ((7, 2048), 5.0), ((64, 2048), 100.0)):
    w = 1 + torch.randn(shape[-1], device=dev) * 0.3
    x = torch.randn(*shape, device=dev) * 3 + offset
    d = maxdiff(mb.layer_norm(x, w, 1e-5), ops.layer_norm(x, w, 1e-5))
    check(f"layer_norm x{list(shape)} (offset {offset:g}) max|diff|={d:.1e}", d < 1e-4)

# rope vs the CPU fp32 reference (transformers' frequencies): Qwen3.5's rotary slice (8 query + 2 key heads, first 64
# of 256 dims, theta 1e7) and Tiny Aya's 16 + 4 heads of 128 (theta 5e4), at positions up to 8191 and beyond. The
# angle is position x frequency, so a frequency off in its last bit gives an error that grows with the position
# (the kernel computed its own pow() until M5: 2e-4 relative at 4095, 4e-4 at 8191); now it must stay flat
for H, d, theta, pos in ((10, 64, 1e7, torch.arange(5)), (10, 64, 1e7, torch.tensor([0, 1, 700, 4095, 31999])),
                         (20, 128, 5e4, torch.tensor([0, 386, 4095, 4096, 8191]))):
    x = torch.randn(len(pos), H, d) * 50
    ours, ref = mb.rope(x.to(dev), pos.to(dev), theta), ops.rope(x, pos, theta)
    rel = maxdiff(ours, ref) / ref.abs().max().item()
    check(f"rope {H} heads d={d} theta {theta:g}, positions up to {int(pos.max())}: relative max|diff| vs the CPU "
          f"reference {rel:.1e}", rel < 1e-6)

# rope with linear scaling (Gemma 3): transformers' own Gemma3RotaryEmbedding, built from a tiny Gemma3TextConfig that
# carries the real model's two tables (sliding: theta 1e4; full: theta 1e6, every frequency / 8), and its
# apply_rotary_pos_emb, at every position 0..8191, vs ops.rope (CPU) and the Metal kernel. The Metal backend caches
# frequencies per (theta, d, factor): the unscaled table at theta 1e6 is cached first, so a key without the factor
# would hand the full-attention call the wrong one
gcfg = Gemma3TextConfig(vocab_size=128, hidden_size=256, intermediate_size=1024, num_hidden_layers=6,
                        num_attention_heads=8, num_key_value_heads=4, head_dim=256,
                        rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                                         "full_attention": {"rope_type": "linear", "factor": 8.0,
                                                            "rope_theta": 1_000_000.0}})
rot = Gemma3RotaryEmbedding(gcfg)
pos = torch.arange(8192)
x = torch.randn(len(pos), 2, 256) * 50
xd, pd = x.to(dev), pos.to(dev)
unscaled = mb.rope(xd, pd, 1e6)
for layer_type, theta, factor in (("sliding_attention", 1e4, 1.0), ("full_attention", 1e6, 8.0)):
    same_table = torch.equal(ops.rope_inv_freq(theta, 256, factor), getattr(rot, f"{layer_type}_inv_freq"))
    cos, sin = rot(x, pos[None], layer_type)
    ref = apply_rotary_pos_emb(x.transpose(0, 1)[None], x.transpose(0, 1)[None], cos, sin)[0][0].transpose(0, 1)
    rel_cpu = maxdiff(ops.rope(x, pos, theta, factor), ref) / ref.abs().max().item()
    rel_metal = maxdiff(mb.rope(xd, pd, theta, factor), ref) / ref.abs().max().item()
    check(f"rope {layer_type} (theta {theta:g}, factor {factor:g}) vs transformers' Gemma3RotaryEmbedding + "
          f"apply_rotary_pos_emb, d=256, positions 0..8191: frequencies bit-identical {same_table}, attention scaling "
          f"{getattr(rot, f'{layer_type}_attention_scaling')}; relative max|diff| CPU {rel_cpu:.1e}, Metal "
          f"{rel_metal:.1e}", same_table and getattr(rot, f"{layer_type}_attention_scaling") == 1.0
          and rel_cpu < 1e-6 and rel_metal < 1e-6)
legacy = all(torch.equal(ops.rope_inv_freq(theta, d), 1.0 / (theta ** (torch.arange(0, d // 2, dtype=torch.float32)
                                                                       / (d // 2))))
             for theta in (1e4, 5e4, 1e6, 1e7) for d in (64, 128, 256))
check(f"rope factor 1.0 (the default) leaves results bit-identical: tables == the unscaled formula {legacy}, Metal "
      f"theta 1e6 before and after a factor-8 call identical {torch.equal(unscaled, mb.rope(xd, pd, 1e6, 1.0))}",
      legacy and torch.equal(unscaled, mb.rope(xd, pd, 1e6, 1.0)))
del x, xd, cos, sin, ref, unscaled

# matvec: every projection shape in Qwen3.5-0.8B (qkvg, o_proj/out_proj, DeltaNet in_proj, gate_up, down) plus the
# 248,320-row output head, and one synthetic 896x896 +bias row (the bias path, and K % 256 = 128: every Qwen3.5 K is
# a multiple of 256). Weights are made on the GPU in bf16: an fp32 CPU copy of the head would be 1 GB of temporaries.
for N, K, bias in ((5120, 1024, False), (1024, 2048, False), (8224, 1024, False), (7168, 1024, False),
                   (1024, 3584, False), (248320, 1024, False), (896, 896, True)):
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    b = (torch.randn(N) * 0.1).to(torch.bfloat16).to(dev) if bias else None
    x = torch.randn(1, K, device=dev)
    ref = (x @ W.float().T) + (b.float() if bias else 0)
    rel = maxdiff(mb.linear(x, W, b), ref) / ref.abs().max().item()
    # Timing with COLD weights: cycle through distinct copies totalling >= 64 MB so reads come from DRAM, as in
    # decode (re-reading one small matrix would be served from the on-chip cache and overstate bandwidth).
    Ws = [W] + [W.clone() for _ in range(max(1, (64 << 20) // (N * K * 2)) - 1)]
    it = iter(range(10 ** 9))
    t_ours = bench(lambda: mb.linear(x, Ws[next(it) % len(Ws)], b), 2 * len(Ws))
    it = iter(range(10 ** 9))
    torch_lin = lambda w: (x.to(torch.bfloat16) @ w.T).float() + b.float() if bias else (x.to(torch.bfloat16) @ w.T).float()
    t_torch = bench(lambda: torch_lin(Ws[next(it) % len(Ws)]), 2 * len(Ws))                 # same work as TorchBackend.linear
    gbps = N * K * 2 / (t_ours * 1e-6) / 1e9
    check(f"matvec {N}x{K}{' +bias' if bias else ''}: rel diff {rel:.1e} | cold weights: ours {t_ours:.0f} us ({gbps:.0f} GB/s) vs torch {t_torch:.0f} us", rel < 1e-3)
del Ws, W, ref; torch.mps.empty_cache()

# tiled GEMM (prefill) vs PyTorch's tuned GEMM, including ragged sizes not divisible by the tile
for T, N, K in ((7, 5120, 1024), (64, 7168, 1024), (256, 1024, 3584), (1024, 8224, 1024)):
    W = (torch.randn(N, K) * 0.05).to(torch.bfloat16).to(dev); x = torch.randn(T, K, device=dev)
    ref = x @ W.float().T
    rel = maxdiff(mb.gemm(x, W), ref) / ref.abs().max().item()
    t_ours = bench(lambda: mb.gemm(x, W), 50); t_torch = bench(lambda: (x.to(torch.bfloat16) @ W.T).float(), 50)
    tflops = 2 * T * N * K / (t_ours * 1e-6) / 1e12
    check(f"gemm T={T} {N}x{K}: rel diff {rel:.1e} | tiled {t_ours:.0f} us ({tflops:.2f} TFLOP/s) vs torch {t_torch:.0f} us", rel < 1e-3)

# fused kernels: matvec + residual (down_proj), and SwiGLU (gate and up rows read in one pass)
W = (torch.randn(1024, 3584) * 0.05).to(torch.bfloat16).to(dev); x = torch.randn(1, 3584, device=dev); r = torch.randn(1, 1024, device=dev)
ref = x @ W.float().T + r
check(f"matvec + fused residual: rel diff {maxdiff(mb.linear(x, W, residual=r), ref) / ref.abs().max().item():.1e}",
      maxdiff(mb.linear(x, W, residual=r), ref) / ref.abs().max().item() < 1e-3)
Wgu = (torch.randn(2 * 3584, 1024) * 0.05).to(torch.bfloat16).to(dev); x = torch.randn(1, 1024, device=dev)
yg = x @ Wgu.float().T; ref = ops.silu_mul(yg[:, :3584], yg[:, 3584:])
rel = maxdiff(mb.swiglu(x, Wgu), ref) / ref.abs().max().item()
t_fused = bench(lambda: mb.swiglu(x, Wgu)); t_split = bench(lambda: mb.silu_mul(mb.linear(x, Wgu)))
check(f"fused swiglu: rel diff {rel:.1e} | fused {t_fused:.0f} us vs linear + silu_mul {t_split:.0f} us", rel < 1e-3)

# GeGLU (Gemma 3's MLP): gelu_tanh(gate) * up vs transformers' "gelu_pytorch_tanh" (torch's gelu(approximate="tanh")),
# errors in units of |up| * max(1, |gate|). Torch's own kernel is not one function of its input: the SIMD body of each
# thread's chunk computes gate + 0.044715 gate^3 in two roundings, the scalar loop at the chunk's end in one (FMA), up to
# 1.2e-7 apart. ops.gelu_tanh_mul is the SIMD body's operations, so on the CPU it may differ only in those tails. The
# Metal kernel's tanh is not torch's (near gate = -5, 1 + tanh cancels and shows it); where tanh saturates
# (|gate| >= 10, also where gate^3 overflows) it must be exact
edge = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 1e-3, -1e-3, 0.5, -0.5, 1, -1, 3, -3, 5, -5, 9, -9, 10, -10, 50, -50,
                     1e4, -1e4, 1e13, -1e13, 1e30, -1e30])
g = torch.cat([edge, torch.linspace(-12, 12, 100001), torch.randn(1 << 20) * 3])
u = torch.randn_like(g)
ref = ACT2FN["gelu_pytorch_tanh"](g) * u
unit = (u.abs() * g.abs().clamp_min(1)).clamp_min(torch.finfo(torch.float32).tiny)        # randn can give up = 0
cpu, ours = ops.gelu_tanh_mul(g, u), mb.gelu_tanh_mul(torch.cat([g, u])[None].to(dev))[0].cpu()
err_cpu, err = ((cpu - ref).abs() / unit).max().item(), ((ours - ref).abs() / unit).max().item()
sat = g.abs() >= 10
check(f"gelu_tanh_mul on {len(g)} values (edge values, -12..12, random) vs transformers: CPU identical in "
      f"{int((cpu == ref).sum())}, max|diff| {err_cpu:.1e}; Metal max|diff| {err:.1e}, saturated values exact "
      f"{torch.equal(ours[sat], ref[sat])}", err_cpu < 3e-7 and err < 1e-6 and torch.equal(ours[sat], ref[sat]))
tb = TorchBackend()
mlp = Gemma3MLP(gcfg)
xm = torch.randn(9, gcfg.hidden_size) * 3
with torch.no_grad():
    want = mlp(xm)
    got = tb.linear(tb.geglu(xm, torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight])), mlp.down_proj.weight)
check(f"TorchBackend (CPU fp32) geglu + down == transformers' Gemma3MLP, T=9: max|diff| {maxdiff(got, want):.1e}",
      maxdiff(got, want) <= 1e-6 * want.abs().max().item())
del g, u, unit, ref, cpu, ours, mlp

# geglu at Sarvam-Translate's shape (K 2560, F 10240) on every path, vs fp32 on the CPU (bf16 or dequantized weights):
# the fused decode kernel (T=1), batched matvec kernels (2..8), groups of them (20), prefill GEMM (512); INT8 / INT4
F, K = 10240, 2560
Wgu = (torch.randn(2 * F, K) * 0.05).to(torch.bfloat16)
xg = torch.randn(512, K)
for scheme in ("bf16", "int8", "int4"):
    w = Wgu.to(dev) if scheme == "bf16" else quantize(Wgu.float(), scheme)
    y = xg @ (Wgu.float() if scheme == "bf16" else w.dequantize()).T
    want = ops.gelu_tanh_mul(y[:, :F], y[:, F:])
    w = w.to(dev)
    worst = {T: maxdiff(mb.geglu(xg[:T].to(dev), w), want[:T]) / want[:T].abs().max().item()
             for T in (1, 2, 3, 5, 8, 20, 512)}
    timing = ""
    if scheme == "bf16":
        x1 = xg[:1].to(dev)
        t_fused = bench(lambda: mb.geglu(x1, w))
        t_split = bench(lambda: mb.gelu_tanh_mul(mb.linear(x1, w)))
        timing = (f" | decode: fused {t_fused:.0f} us ({2 * F * K * 2 / (t_fused * 1e-6) / 1e9:.0f} GB/s) vs "
                  f"linear + gelu_tanh_mul {t_split:.0f} us")
    check(f"geglu {scheme} {2 * F}x{K}: relative max|diff| vs fp32 reference "
          + ", ".join(f"T={T} {e:.1e}" for T, e in worst.items()) + timing, max(worst.values()) < 1e-5)
del Wgu, w, xg, y, want; torch.mps.empty_cache()

# decode attention: 1 query, S cached positions, 8 query heads sharing 2 KV heads of 256 dims. The kernel splits its
# value pass into 256/d groups plus a reduction: 1 group at d=256, so one d=64 case (14/2 heads) keeps 4 covered.
for S, scale, Hq, d in ((1, 1, 8, 256), (5, 1, 8, 256), (300, 1, 8, 256), (2048, 1, 8, 256), (300, 30, 8, 256),
                        (2048, 30, 8, 256), (300, 30, 14, 64)):
    # scale 30 pushes scores past exp()'s fp32 range: only a correct max-subtracting softmax stays finite
    q = torch.randn(1, Hq, d, device=dev) * scale; k = torch.randn(S, 2, d, device=dev); v = torch.randn(S, 2, d, device=dev)
    diff = maxdiff(mb.attention(q, k, v), ops.attention(q, k, v, causal=True))
    t_ours = bench(lambda: mb.attention(q, k, v), 100); t_torch = bench(lambda: ops.attention(q, k, v, causal=True), 100)
    check(f"attention_decode S={S} q-scale {scale} d={d}: max|diff|={diff:.1e} | ours {t_ours:.0f} us vs torch ops {t_torch:.0f} us", diff < 1e-4)

# Tiny Aya's decode attention: 16 query heads sharing 4 KV heads of 128 dims, up to its 4096-token window
for S in (1, 300, 4096):
    q = torch.randn(1, 16, 128, device=dev); k = torch.randn(S, 4, 128, device=dev); v = torch.randn(S, 4, 128, device=dev)
    diff = maxdiff(mb.attention(q, k, v), ops.attention(q, k, v, causal=True))
    check(f"attention_decode Tiny Aya shape S={S} 16/4 heads d=128: max|diff|={diff:.1e}", diff < 1e-4)

# paged decode attention: 5 sequences of different lengths in one dispatch, each reading its keys/values in place
# through a scrambled block table, vs the reference gather-then-attend (TorchBackend on MPS, fp32)
rb = TorchBackend("mps", torch.float32)
lens_l, bs = [1, 7, 16, 33, 300], 16
perm = torch.randperm(32).tolist()
nbs = [-(-n // bs) for n in lens_l]
tabs = [perm[sum(nbs[:i]):sum(nbs[:i + 1])] for i in range(len(lens_l))]
tables = torch.tensor([t + [0] * (max(nbs) - len(t)) for t in tabs], dtype=torch.int32, device=dev)
lens = torch.tensor(lens_l, dtype=torch.int32, device=dev)
for Hq, Hkv, d in ((8, 2, 256), (14, 2, 64), (16, 4, 128)):                     # the last: Tiny Aya
    kp, vp = torch.randn(32, bs, Hkv, d, device=dev), torch.randn(32, bs, Hkv, d, device=dev)
    q = torch.randn(len(lens_l), Hq, d, device=dev)
    diff = maxdiff(mb.paged_attention(q, kp, vp, tables, lens, bs), rb.paged_attention(q, kp, vp, tables, lens, bs))
    check(f"paged_attention_decode B=5 lengths {lens_l}, block size {bs}, scrambled tables, {Hq}/{Hkv} heads d={d}: "
          f"max|diff|={diff:.1e}", diff < 1e-4)

# bf16 KV cache (half the bytes): the _bf16 kernels widen each element to fp32 as they read it, so they must give
# EXACTLY what the fp32 kernels give on the widened copy, and stay within fp32 rounding of the reference
for S, scale, Hq, Hkv, d in ((1, 1, 16, 4, 128), (300, 1, 16, 4, 128), (4096, 1, 16, 4, 128), (8192, 1, 16, 4, 128),
                             (300, 30, 8, 2, 256),
                             (300, 1, 14, 2, 64)):
    q = torch.randn(1, Hq, d, device=dev) * scale
    k, v = (torch.randn(S, Hkv, d, device=dev).bfloat16() for _ in range(2))
    ours = mb.attention(q, k, v)
    same = torch.equal(ours, mb.attention(q, k.float(), v.float()))
    diff = maxdiff(ours, ops.attention(q, k.float(), v.float(), causal=True))
    kf, vf = k.float(), v.float()                                   # widened outside the timed call
    t16, t32 = bench(lambda: mb.attention(q, k, v), 100), bench(lambda: mb.attention(q, kf, vf), 100)
    check(f"attention_decode_bf16 S={S} q-scale {scale} {Hq}/{Hkv} heads d={d}: == fp32 kernel on the widened cache: "
          f"{same}, max|diff| vs reference {diff:.1e} | bf16 {t16:.0f} us vs fp32 {t32:.0f} us", same and diff < 1e-4)
for Hq, Hkv, d in ((8, 2, 256), (16, 4, 128)):
    kp, vp = (torch.randn(32, bs, Hkv, d, device=dev).bfloat16() for _ in range(2))
    q = torch.randn(len(lens_l), Hq, d, device=dev)
    ours = mb.paged_attention(q, kp, vp, tables, lens, bs)
    same = torch.equal(ours, mb.paged_attention(q, kp.float(), vp.float(), tables, lens, bs))
    diff = maxdiff(ours, rb.paged_attention(q, kp, vp, tables, lens, bs))
    check(f"paged_attention_decode_bf16 B=5, {Hq}/{Hkv} heads d={d}: == fp32 kernel on the widened pool: {same}, "
          f"max|diff| vs reference {diff:.1e}", same and diff < 1e-4)
# sliding window (Tiny Aya's 27 sliding layers): the newest query sees only the last W keys. Decode at the window's
# edges (S = W - 1, W, W + 1, 2W + 3) and at Tiny Aya's 4096 up to 8192, both KV dtypes, vs ops.attention(window=W)
for kvdt in (torch.float32, torch.bfloat16):
    worst, cases = 0.0, [(S, W) for W in (1, 16) for S in (1, W - 1, W, W + 1, 2 * W + 3) if S >= 1]
    for S, W in cases + [(4095, 4096), (4096, 4096), (4097, 4096), (8192, 4096)]:
        q = torch.randn(1, 16, 128, device=dev)
        k, v = (torch.randn(S, 4, 128, device=dev).to(kvdt) for _ in range(2))
        worst = max(worst, maxdiff(mb.attention(q, k, v, window=W), ops.attention(q, k.float(), v.float(), window=W)))
    check(f"attention_decode{'_bf16' if kvdt == torch.bfloat16 else ''} with a window, 16/4 heads d=128, "
          f"{len(cases) + 4} cases at W - 1, W, W + 1, 2W + 3 (W 1, 16) and 4095-8192 (W 4096): max|diff| {worst:.1e}",
          worst < 1e-4)
# paged: blocks wholly before each sequence's window hold NaN (as if freed or reused): the kernel must never read them
for kvdt in (torch.float32, torch.bfloat16):
    for W in (16, 20, 128):
        lens_w = [1, W, W + 1, 2 * W + 3, 300]
        nbs_w = [-(-n // bs) for n in lens_w]
        perm_w = torch.randperm(sum(nbs_w)).tolist()
        tabs_w = [perm_w[sum(nbs_w[:i]):sum(nbs_w[:i + 1])] for i in range(len(lens_w))]
        tables_w = torch.tensor([t + [0] * (max(nbs_w) - len(t)) for t in tabs_w], dtype=torch.int32, device=dev)
        lens_t = torch.tensor(lens_w, dtype=torch.int32, device=dev)
        kp, vp = (torch.randn(sum(nbs_w), bs, 4, 128, device=dev).to(kvdt) for _ in range(2))
        q = torch.randn(len(lens_w), 16, 128, device=dev)
        want = rb.paged_attention(q, kp, vp, tables_w, lens_t, bs, window=W)       # before any poisoning
        for t, n in zip(tabs_w, lens_w):
            for b in t[:max(0, n - W) // bs]:
                kp[b], vp[b] = float("nan"), float("nan")
        got = mb.paged_attention(q, kp, vp, tables_w, lens_t, bs, window=W)
        check(f"paged_attention_decode{'_bf16' if kvdt == torch.bfloat16 else ''} window {W}, lengths {lens_w}, blocks "
              f"before the window NaN: finite, max|diff| {maxdiff(got, want):.1e}",
              bool(torch.isfinite(got).all()) and maxdiff(got, want) < 1e-4)

# a bf16 tensor bound to a float* kernel reads as garbage without an error: any other dtype must be refused
for kd, vd in ((torch.float16, torch.float16), (torch.bfloat16, torch.float32)):
    try:
        mb.paged_attention(q, kp.to(kd), vp.to(vd), tables, lens, bs); refused = False
    except TypeError:
        refused = True
    check(f"paged_attention refuses a {str(kd)[6:]} / {str(vd)[6:]} pool (TypeError)", refused)

# weights locked in RAM (backend/pinning.py): mlock of the MTLBuffer pages behind MPS tensors. The kernel's user wire
# count on the buffers' memory must go to 1 and back to 0 after unlock, and the GPU must still read the same data.
# (The system-wide wired-page count is no test: the GPU driver itself wires buffers it has just used, and lets go of
# idle ones, which is when the user wire matters.)
import ctypes
from backend.pinning import _ranges, lock_in_memory, unlock
class SubmapInfo64(ctypes.Structure):                    # <mach/vm_region.h> vm_region_submap_info_64, pack(4)
    _pack_ = 4
    _fields_ = [(n, t) for n, t in [
        ("protection", ctypes.c_int), ("max_protection", ctypes.c_int), ("inheritance", ctypes.c_uint),
        ("offset", ctypes.c_uint64), ("user_tag", ctypes.c_uint), ("pages_resident", ctypes.c_uint),
        ("pages_shared_now_private", ctypes.c_uint), ("pages_swapped_out", ctypes.c_uint), ("pages_dirtied", ctypes.c_uint),
        ("ref_count", ctypes.c_uint), ("shadow_depth", ctypes.c_ushort), ("external_pager", ctypes.c_ubyte),
        ("share_mode", ctypes.c_ubyte), ("is_submap", ctypes.c_int), ("behavior", ctypes.c_int),
        ("object_id", ctypes.c_uint32), ("user_wired_count", ctypes.c_ushort), ("pages_reusable", ctypes.c_uint)]]
libc = ctypes.CDLL(None)
def user_wired(addr: int) -> int:
    """user_wired_count of the VM map entry holding addr (descending into submaps)."""
    a, size, depth, info = ctypes.c_uint64(addr), ctypes.c_uint64(0), ctypes.c_uint(0), SubmapInfo64()
    while True:
        cnt = ctypes.c_uint(ctypes.sizeof(SubmapInfo64) // 4)
        kr = libc.mach_vm_region_recurse(ctypes.c_uint.in_dll(libc, "mach_task_self_"), ctypes.byref(a),
                                         ctypes.byref(size), ctypes.byref(depth), ctypes.byref(info), ctypes.byref(cnt))
        if kr != 0 or not info.is_submap:
            return -1 if kr != 0 else info.user_wired_count
        depth.value += 1
ws = [torch.randn(4096, 4096, device=dev) for _ in range(4)]            # 4 x 64 MiB
xv = torch.randn(4096, device=dev)
ref = [w @ xv for w in ws]
torch.mps.synchronize()
need = sum(w.untyped_storage().nbytes() for w in ws)
addrs = [p for p, _ in _ranges(ws)]
before = [user_wired(p) for p in addrs]
locked = lock_in_memory(ws + [ws[0][:10]])                               # a view: its buffer counted once
during = [user_wired(p) for p in addrs]
same = all(maxdiff(w @ xv, r) == 0 for w, r in zip(ws, ref))
unlock(ws)
after = [user_wired(p) for p in addrs]
check(f"lock_in_memory: {locked / 2**20:.0f} MiB locked for 4 x 64 MiB MPS tensors (+ a view of one); user wire "
      f"count of their memory {before} -> {during} -> {after} after unlock; GPU results unchanged",
      need <= locked < need + 4 * 2**20 and before == [0] * 4 and during == [1] * 4 and after == [0] * 4 and same)
del ws, ref

print(f"\n{sum(results)}/{len(results)} kernel checks passed")
sys.exit(0 if all(results) else 1)
