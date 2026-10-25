import sys
import time
import torch

sys.path.insert(0, "src")
from native.metal_native import NativeMetal

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")

nm = NativeMetal()
print(f"      device: {nm.device_name}")
torch.manual_seed(0)
# synthetic shapes (Qwen2.5-0.5B's), kept so the GB/s stay comparable with docs/bench/decode.md
for N, K in ((896, 896), (4864, 896), (151936, 896)):
    W = (torch.randn(N, K) * 0.05).to(torch.bfloat16)
    x = torch.randn(K)
    w_h, x_h, y_h = nm.upload(W), nm.upload(x), nm.alloc(N * 4)
    nm.matvec(w_h, x_h, y_h, N, K, 1)
    y = nm.view(y_h, N).clone()
    ref = W.float() @ x
    rel = (y - ref).abs().max().item() / ref.abs().max().item()
    # cold timing: rotate over enough distinct copies that the set (>= 64 MB) can't live in the on-chip cache
    copies = max(1, (64 << 20) // (N * K * 2))
    ws = [w_h] + [nm.upload(W) for _ in range(copies - 1)]
    t0 = time.perf_counter(); gpu_s = nm.matvec_rotate(ws, x_h, y_h, N, K, 2 * len(ws)); wall = (time.perf_counter() - t0) / (2 * len(ws))
    check(f"native matvec {N}x{K}: rel diff {rel:.1e} | cold reads over {len(ws)} copies: {gpu_s * 1e6:.0f} us GPU "
          f"({N * K * 2 / gpu_s / 1e9:.0f} GB/s), {wall * 1e6:.0f} us wall per dispatch", rel < 1e-3)

# the shared buffer really is shared: writing through the torch view is visible to the GPU
x = torch.zeros(896); x_h = nm.upload(x); nm.view(x_h, 896)[:] = 1.0
W = torch.ones(8, 896, dtype=torch.bfloat16); w_h, y_h = nm.upload(W), nm.alloc(8 * 4)
nm.matvec(w_h, x_h, y_h, 8, 896, 1)
check("unified memory: a CPU write through the torch view is seen by the GPU kernel (8 x 896 ones -> 896)",
      torch.allclose(nm.view(y_h, 8), torch.full((8,), 896.0)))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
