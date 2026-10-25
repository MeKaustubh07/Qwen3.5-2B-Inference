import gc
import sys
import time
import torch

sys.path.insert(0, "src")
from backend.torch_ref import TorchBackend
from config import Qwen35Config
from generate import generate_greedy
from models.qwen3_5 import Qwen35Model
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile

D = "models/qwen3.5-0.8b"
EOS = {248046, 248044}
cfg = Qwen35Config.from_json(f"{D}/config.json")
weights = SafetensorsFile(f"{D}/model.safetensors-00001-of-00001.safetensors")
tok = Tokenizer(f"{D}/tokenizer.json")
model = Qwen35Model(cfg, weights)                                  # fp32 CPU reference backend

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")

# 1. Cached logits == uncached logits at every position (prefill part of the prompt, then one token at a time).
#    Prompt 1 prefills a single token from an empty DeltaNet state; prompt 2 prefills 2 tokens, fewer than the conv's
#    K-1 = 3, so the conv tail is only partly filled when decoding starts.
for i in range(6):
    g = torch.load(f"tests/golden_qwen35/{i}.pt")
    ids = g["ids"]
    full = model.forward(ids)                                      # [T, vocab], no cache
    state = model.new_state(len(ids))
    k = max(1, len(ids) // 2)
    parts = [model.forward(ids[:k], state=state)]                  # prefill k tokens
    for t in range(k, len(ids)):
        parts.append(model.forward(ids[t:t + 1], state=state))     # decode the rest one by one
    inc = torch.cat(parts)
    d = (inc - full).abs().max().item()
    check(f"prompt {i} ({len(ids)} tokens, prefill {k}): cached vs uncached logits max|diff|={d:.1e}, same argmax everywhere",
          d < 1e-3 and torch.equal(inc.argmax(-1), full.argmax(-1)) and state.length == len(ids))

# 1b. A multi-token chunk on a non-empty state (S > T > 1): prefill 3, then a chunk of 4, then single tokens
ids = torch.load("tests/golden_qwen35/4.pt")["ids"]
full = model.forward(ids)
state = model.new_state(len(ids))
inc = torch.cat([model.forward(ids[:3], state=state), model.forward(ids[3:7], state=state)] +
                [model.forward(ids[t:t + 1], state=state) for t in range(7, len(ids))])
d = (inc - full).abs().max().item()
check(f"chunked prefill on a non-empty state (3 + 4 + 1...) matches uncached: max|diff|={d:.1e}", d < 1e-3)

# 2. Speed: 32 new tokens with and without the cache (the only test of the uncached generate path)
prompt = "The capital of France is"
t0 = time.perf_counter(); a = generate_greedy(model, tok, prompt, 32, set(), use_cache=False); t_nc = time.perf_counter() - t0
t0 = time.perf_counter(); b = generate_greedy(model, tok, prompt, 32, set(), use_cache=True); t_c = time.perf_counter() - t0
print(f"      CPU fp32, 32 tokens: no cache {t_nc:.2f}s ({32/t_nc:.1f} tok/s)  |  cache {t_c:.2f}s ({32/t_c:.1f} tok/s)  -> {t_nc/t_c:.1f}x")
check("cached and uncached produce the same 32 tokens", a == b)
check("cache is faster", t_c < t_nc)

# 3. State memory matches the formula. Attention layers: layers x 2 (K,V) x kv_heads x head_dim x bytes per token.
#    DeltaNet layers: a fixed S [H, dk, dv] plus the last K-1 conv inputs per layer, whatever the length.
per_tok = 2 * len(model.attn_layers) * cfg.num_key_value_heads * cfg.head_dim * 4
fixed = len(model.linear_layers) * (cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim
                                    + (cfg.linear_conv_kernel_dim - 1) * model.conv_dim) * 4
state = model.new_state(100); model.forward(torch.tensor(tok.encode(prompt)), state=state)
L, before = state.length, state.bytes_used()
check(f"state bytes = {before} for {L} tokens: KV {per_tok} B/token in fp32 ({per_tok // 2} in bf16) + fixed DeltaNet {fixed}",
      state.kv.bytes_used() == per_tok * L and before == per_tok * L + fixed)
model.forward(torch.tensor(b[:7]), state=state)
check(f"7 more tokens grow the state by exactly 7 x {per_tok} B (the DeltaNet state does not grow)",
      state.bytes_used() - before == 7 * per_tok and state.length == L + 7)

# 4. GPU backends.
#    (a) Cache CORRECTNESS on the GPU, isolated from rounding: with fp32 weights, cached must equal uncached tightly.
#    (b) bf16 backends: prefill (matrix-matrix) and decode (matrix-vector) round differently, so cached vs uncached may
#        differ by as much as each path differs from the fp32 reference. Hold them to that budget, and require the same
#        argmax wherever the top-2 logits are not a near-tie (bf16 can produce exact ties).
del model; gc.collect()                                            # never hold the CPU and MPS fp32 copies together
if torch.backends.mps.is_available():
    from backend.metal import MetalBackend

    def cached_vs_uncached(gpu, st, ids):
        full = gpu.forward(ids).cpu()
        inc = torch.cat([gpu.forward(ids[:2], state=st).cpu()] + [gpu.forward(ids[t:t + 1], state=st).cpu() for t in range(2, len(ids))])
        return full, inc

    def states(gpu, T):
        return (("contiguous", gpu.new_state(T)), ("paged", gpu.new_paged_state(gpu.new_paged_pool(16, 4, max_seqs=1))))

    g0 = torch.load("tests/golden_qwen35/3.pt"); ids = g0["ids"]
    gpu = Qwen35Model(cfg, weights, TorchBackend("mps", torch.float32))
    for kind, st in states(gpu, len(ids)):
        full, inc = cached_vs_uncached(gpu, st, ids)
        d = (inc - full).abs().max().item()
        check(f"MPS fp32 weights: cached ({kind}) == uncached at every position, max|diff|={d:.1e}", d < 1e-3)
    del gpu; gc.collect(); torch.mps.empty_cache()

    # prompt 3 (10 tokens) on both bf16 backends; on Metal also prompt 5 (57 tokens), whose uncached prefill takes
    # the GEMM path (more than 32 rows) while the cached decode takes the fused conv_step + gdn_decode kernels
    for name, be, prompts in (("MPS bf16", TorchBackend("mps", torch.bfloat16), [3]), ("Metal kernels", MetalBackend(), [3, 5])):
        gpu = Qwen35Model(cfg, weights, be)
        for p in prompts:
            g = torch.load(f"tests/golden_qwen35/{p}.pt"); ids, ref = g["ids"], g["logits"]
            for kind, st in states(gpu, len(ids)):
                full, inc = cached_vs_uncached(gpu, st, ids)
                gap, budget = (inc - full).abs().max().item(), 1.5 * (full - ref).abs().max().item()
                top2 = full.topk(2, -1).values
                clear = (top2[:, 0] - top2[:, 1]) > 0.5
                same = torch.equal(inc.argmax(-1)[clear], full.argmax(-1)[clear])
                check(f"{name} ({kind}, {len(ids)} tokens): cached vs uncached {gap:.2f} within bf16 budget {budget:.2f}; "
                      f"argmax equal at {int(clear.sum())}/{len(clear)} non-tied positions", gap <= budget and same)
        if name == "MPS bf16":                                     # Metal greedy vs HF is covered by test_qwen35
            out = generate_greedy(gpu, tok, g0["text"], 10, EOS)
            n = next((j for j, (a, b) in enumerate(zip(out, g0["greedy"].tolist())) if a != b), len(out))
            check(f"{name}: 10-token cached greedy matches HF fp32 for {n}/10 tokens", n >= 8)
        del gpu; gc.collect(); torch.mps.empty_cache()

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
