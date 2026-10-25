import gc
import glob
import sys
import time
import torch

sys.path.insert(0, "src")
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

check("loader: the header's tensor catalog accounts for every byte of the file", weights.integrity_check())
print(f"layer types: {''.join('A' if t == 'full_attention' else 'L' for t in cfg.layer_types)}  (A = gated attention, L = Gated DeltaNet)")
N_GOLD = len(glob.glob("tests/golden_qwen35/*.pt"))              # 7: includes prompts longer than a 64-token chunk
for n in range(N_GOLD):
    g = torch.load(f"tests/golden_qwen35/{n}.pt")
    ids = torch.tensor(tok.encode(g["text"]))
    check(f"prompt {n}: tokenizer ids == HF ids ({len(ids)} tokens)", torch.equal(ids, g["ids"]))
    cap = {}
    logits = model.forward(g["ids"], capture=cap)
    rel = [((cap[f"l{i}_out"] - g[f"l{i}_out"]).abs().max() / g[f"l{i}_out"].abs().max()).item() for i in range(24)]
    worst = max(range(24), key=lambda i: rel[i])
    d_logits = (logits - g["logits"]).abs().max().item()
    d_norm = ((cap["final_norm"] - g["final_norm"]).abs().max() / g["final_norm"].abs().max()).item()
    agree = (logits.argmax(-1) == g["logits"].argmax(-1)).float().mean().item()
    check(f"prompt {n} ({len(ids)} tokens): embed exact, worst layer rel diff {rel[worst]:.1e} (layer {worst}, {cfg.layer_types[worst]}), "
          f"final norm rel {d_norm:.1e}, logits max|diff| {d_logits:.1e}, top-1 {agree:.0%}",
          torch.equal(cap["embed"], g["embed"]) and max(rel) < 1e-4 and d_norm < 1e-4 and d_logits < 1e-3 and agree == 1.0)

# greedy decoding with the hybrid cache vs HF generate()
t0 = time.perf_counter()
for n in range(N_GOLD):
    g = torch.load(f"tests/golden_qwen35/{n}.pt")
    ours = generate_greedy(model, tok, g["text"], 10, EOS)
    hf = g["greedy"].tolist()
    check(f"prompt {n}: cached greedy == HF: {tok.decode(ours)!r}", ours == hf)
print(f"      greedy decoding, {N_GOLD} prompts x 10 tokens: {time.perf_counter() - t0:.1f}s")

# cached decode == uncached, with contiguous and paged attention caches (and the recurrent state carried)
g = torch.load(f"tests/golden_qwen35/{N_GOLD - 1}.pt"); ids = g["ids"]                     # the multi-chunk prompt
full = model.forward(ids)
T = len(ids)
for kind, st in (("contiguous", model.new_state(T)), ("paged", model.new_paged_state(model.new_paged_pool(64, 4)))):
    # prefill 70 (crosses a chunk boundary), continue with a 65-token chunk from the carried state, then single tokens
    parts = [model.forward(ids[:70], state=st), model.forward(ids[70:135], state=st)] + [model.forward(ids[t:t + 1], state=st) for t in range(135, T)]
    d = (torch.cat(parts) - full).abs().max().item()
    check(f"hybrid state ({kind} KV + recurrent/conv): {T} tokens as 70 + 65 + singles == uncached, max|diff| {d:.1e}", d < 1e-3)

# atomic forward: running out of KV blocks fails before ANY layer mutates the recurrent state; free() clears it
from state import OutOfBlocks
st = model.new_paged_state(model.new_paged_pool(2, 4))                  # room for 8 tokens only
model.forward(ids[:6], state=st)
S_before, tail_before = st.S.clone(), st.conv_tail.clone()
try:
    model.forward(ids[6:12], state=st); raised = False
except OutOfBlocks:
    raised = True
check("KV exhaustion raises before any layer runs: recurrent and conv state unchanged, length unchanged",
      raised and torch.equal(st.S, S_before) and torch.equal(st.conv_tail, tail_before) and st.length == 6)
pool, seq = st.pool, st.seq
st.free()
check("free() returns the KV blocks and the state slot, zeroed, and detaches the state from it",
      pool.allocator.num_free == 2 and seq in pool.free_seqs and pool.S[seq].abs().sum() == 0
      and pool.conv[seq].abs().sum() == 0 and st.S is None)

# special tokens declared only in tokenizer_config.json (audio/TTS markers) are single ids, as in HF
check("tokenizer: '<|audio_start|>' -> [248070], '<tts_pad>' -> [248072], decode round-trips",
      tok.encode("<|audio_start|>") == [248070] and tok.encode("<tts_pad>") == [248072] and tok.decode([248070]) == "<|audio_start|>")

# Qwen3.5 chat template matches HF's renderer (no default system; empty think block when thinking is off)
from chat import format_chat
from transformers import AutoTokenizer
hf_tok = AutoTokenizer.from_pretrained(D)
convs = [[{"role": "user", "content": "What is the capital of France?"}],
         [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}],
         [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "and 3+3?"}],
         [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "<think>\nreasoning\n</think>\n\nhello"}, {"role": "user", "content": "bye"}],
         # what the model really produces in thinking mode: no opening tag (the prompt opened it), trailing newline
         [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "The user greets me.\n</think>\n\nHello! How can I help?\n"}, {"role": "user", "content": "bye"}],
         # surrounding whitespace is trimmed; an assistant turn AFTER the last user query keeps its reasoning
         [{"role": "system", "content": "  Be brief. \n"}, {"role": "user", "content": "  hi \n"}, {"role": "assistant", "content": "thinking\n</think>\n\nhello"}]]
for c in convs:
    for think in (False, True):
        ok = format_chat(c, style="qwen3.5", enable_thinking=think) == hf_tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        check(f"qwen3.5 chat template matches HF ({len(c)} messages, thinking {'on' if think else 'off'})", ok)

# fused Metal DeltaNet decode kernels (conv_step + gdn_decode) == the reference composition, over several steps
if torch.backends.mps.is_available():
    from backend.metal import MetalBackend
    from backend.torch_ref import TorchBackend
    mb, rb = MetalBackend(), TorchBackend("mps", torch.float32)
    H, dk, dv, KD, C = 16, 128, 128, 2048, 6144
    gen = torch.Generator().manual_seed(0)
    r = lambda *s_, scale=1.0: (torch.randn(*s_, generator=gen) * scale).to("mps")
    conv_w, A_log, dt_bias, norm_w = r(C, 4, scale=0.5), r(H), r(H), 1 + r(dv, scale=0.1)
    class St: pass
    sm, sr = St(), St()
    sm.conv_tail, sm.S = r(3, 3, C), r(3, H, dk, dv, scale=0.05)
    sr.conv_tail, sr.S = sm.conv_tail.clone(), sm.S.clone()
    worst = 0.0
    for step in range(4):                                     # state carried across steps, slot 1 of 3
        qkv, z, bl, a = r(1, C), r(1, H * dv), r(1, H), r(1, H, scale=3)
        om = mb.deltanet_decode(qkv, z, bl, a, sm, 1, conv_w, A_log, dt_bias, norm_w, 1e-6, (H, dk, dv, KD))
        orr = rb.deltanet_decode(qkv, z, bl, a, sr, 1, conv_w, A_log, dt_bias, norm_w, 1e-6, (H, dk, dv, KD))
        worst = max(worst, (om - orr).abs().max().item() / orr.abs().max().item(),
                    (sm.S - sr.S).abs().max().item(), (sm.conv_tail - sr.conv_tail).abs().max().item())
    check(f"fused Metal DeltaNet decode == reference over 4 steps (output rel / state / conv-tail worst {worst:.1e}); other slots untouched",
          worst < 1e-4 and torch.equal(sm.S[0], sr.S[0]) and torch.equal(sm.S[2], sr.S[2]))

    del model; gc.collect()                                   # the fp32 CPU copy is done: keep one model in RAM
    gm = Qwen35Model(cfg, weights, MetalBackend())
    V = tok.vocab_size()
    match, flips, n_clear = 0, [], 0
    for n in range(N_GOLD):
        g = torch.load(f"tests/golden_qwen35/{n}.pt"); out = generate_greedy(gm, tok, g["text"], 10, EOS)
        match += next((j for j, (x_, y_) in enumerate(zip(out, g["greedy"].tolist())) if x_ != y_), min(len(out), len(g["greedy"])))
        lg, ref = gm.forward(g["ids"]).cpu()[:, :V], g["logits"][:, :V]
        top2 = ref.topk(2, -1).values
        clear = (top2[:, 0] - top2[:, 1]) > 0.5                  # bf16 may flip a near-tie; a clear winner must hold
        flips.append(int((lg.argmax(-1) != ref.argmax(-1))[clear].sum())); n_clear += int(clear.sum())
    check(f"Metal backend (bf16 weights): prefill top-1 == HF fp32 at every non-tied position ({n_clear} positions, "
          f"flips per prompt: {flips})", sum(flips) == 0)
    check(f"Metal backend (bf16 weights): greedy matches HF fp32 for {match}/{10 * N_GOLD} tokens", match >= 9 * N_GOLD)
    del gm

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
