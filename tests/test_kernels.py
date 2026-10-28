import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, "src")
import ops
from backend.metal import MetalBackend
from backend.torch_ref import TorchBackend

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

# rope vs the CPU fp32 reference (transformers' frequencies): Qwen3.5's rotary slice (8 query + 2 key heads, first 64
# of 256 dims, theta 1e7), at positions up to 31999. The
# angle is position x frequency, so a frequency off in its last bit gives an error that grows with the position
# (the kernel computed its own pow() until M5: 2e-4 relative at 4095, 4e-4 at 8191); now it must stay flat
for H, d, theta, pos in ((10, 64, 1e7, torch.arange(5)), (10, 64, 1e7, torch.tensor([0, 1, 700, 4095, 31999]))):
    x = torch.randn(len(pos), H, d) * 50
    ours, ref = mb.rope(x.to(dev), pos.to(dev), theta), ops.rope(x, pos, theta)
    rel = maxdiff(ours, ref) / ref.abs().max().item()
    check(f"rope {H} heads d={d} theta {theta:g}, positions up to {int(pos.max())}: relative max|diff| vs the CPU "
          f"reference {rel:.1e}", rel < 1e-6)

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

# decode attention: 1 query, S cached positions, 8 query heads sharing 2 KV heads of 256 dims. The kernel splits its
# value pass into 256/d groups plus a reduction: 1 group at d=256, so one d=64 case (14/2 heads) keeps 4 covered.
for S, scale, Hq, d in ((1, 1, 8, 256), (5, 1, 8, 256), (300, 1, 8, 256), (2048, 1, 8, 256), (300, 30, 8, 256),
                        (2048, 30, 8, 256), (300, 30, 14, 64)):
    # scale 30 pushes scores past exp()'s fp32 range: only a correct max-subtracting softmax stays finite
    q = torch.randn(1, Hq, d, device=dev) * scale; k = torch.randn(S, 2, d, device=dev); v = torch.randn(S, 2, d, device=dev)
    diff = maxdiff(mb.attention(q, k, v), ops.attention(q, k, v, causal=True))
    t_ours = bench(lambda: mb.attention(q, k, v), 100); t_torch = bench(lambda: ops.attention(q, k, v, causal=True), 100)
    check(f"attention_decode S={S} q-scale {scale} d={d}: max|diff|={diff:.1e} | ours {t_ours:.0f} us vs torch ops {t_torch:.0f} us", diff < 1e-4)

# paged decode attention: 5 sequences of different lengths in one dispatch, each reading its keys/values in place
# through a scrambled block table, vs the reference gather-then-attend (TorchBackend on MPS, fp32)
rb = TorchBackend("mps", torch.float32)
lens_l, bs = [1, 7, 16, 33, 300], 16
perm = torch.randperm(32).tolist()
nbs = [-(-n // bs) for n in lens_l]
tabs = [perm[sum(nbs[:i]):sum(nbs[:i + 1])] for i in range(len(lens_l))]
tables = torch.tensor([t + [0] * (max(nbs) - len(t)) for t in tabs], dtype=torch.int32, device=dev)
lens = torch.tensor(lens_l, dtype=torch.int32, device=dev)
for Hq, Hkv, d in ((8, 2, 256), (14, 2, 64)):
    kp, vp = torch.randn(32, bs, Hkv, d, device=dev), torch.randn(32, bs, Hkv, d, device=dev)
    q = torch.randn(len(lens_l), Hq, d, device=dev)
    diff = maxdiff(mb.paged_attention(q, kp, vp, tables, lens, bs), rb.paged_attention(q, kp, vp, tables, lens, bs))
    check(f"paged_attention_decode B=5 lengths {lens_l}, block size {bs}, scrambled tables, {Hq}/{Hkv} heads d={d}: "
          f"max|diff|={diff:.1e}", diff < 1e-4)

# a bf16 tensor bound to a float* kernel reads as garbage without an error: any dtype but fp32 must be refused
for kd, vd in ((torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)):
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
