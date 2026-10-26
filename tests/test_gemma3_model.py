import glob
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import Gemma3Config as HFGemma3Config, Gemma3ForConditionalGeneration
from transformers.cache_utils import DynamicCache

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import golden_gemma3
from chat import TemplateChat, format_chat
from config import Gemma3Config
from models.gemma3 import Gemma3Model
from tiny_gemma3 import build, rel
from tokenizer import Tokenizer
from weight_loader import open_weights

D, G = "models/sarvam-translate", "tests/golden_gemma3"
torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def rel_rows(a, b):
    """Each row's worst error over that row's largest value; the worst row."""
    return ((a - b).abs().amax(-1) / b.abs().amax(-1)).max().item()


def hf_decode(hf, text_model, ids: list[int], n: int, stop=()):
    """transformers' own model with its own DynamicCache: (logits of every prompt position, the captures of the
    prefill (embedding, every layer, final norm), up to n greedy tokens to a stop id, the logits of each decode
    step)."""
    cap, hooks = {}, [text_model.embed_tokens.register_forward_hook(lambda _m, _i, o: cap.__setitem__("embed", o[0]))]
    for i, layer in enumerate(text_model.layers):
        hooks.append(layer.register_forward_hook(lambda _m, _i, o, i=i: cap.__setitem__(f"l{i}_out", o[0])))
    hooks.append(text_model.norm.register_forward_hook(lambda _m, _i, o: cap.__setitem__("final_norm", o[0])))
    cache = DynamicCache(config=hf.config.get_text_config())
    logits = hf(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True).logits[0]
    for h in hooks:
        h.remove()
    tokens, steps = [int(logits[-1].argmax())], []
    while len(tokens) < n and tokens[-1] not in stop:
        steps.append(hf(input_ids=torch.tensor([[tokens[-1]]]), past_key_values=cache, use_cache=True).logits[0, -1])
        tokens.append(int(steps[-1].argmax()))
    return logits, cap, tokens, torch.stack(steps)


TINY_SIGLIP = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=28,
                   patch_size=14)


def checkpoint_like(tmp: str, d: str, window: int, seed: int):
    """A random tiny Gemma3ForConditionalGeneration shaped like the real checkpoint: the text model of
    tests/tiny_gemma3.py (norms and embedding randomised) beside a small SigLIP tower, saved in bf16 in shards with an
    index -> transformers' own fp32 model of those bf16 weights, loaded from them."""
    text, _ = build(tmp, window=window, seed=seed)
    mm = Gemma3ForConditionalGeneration._from_config(
        HFGemma3Config(text_config=text.config.to_dict(), vision_config=TINY_SIGLIP, mm_tokens_per_image=4),
        attn_implementation="eager")
    mm.model.language_model.load_state_dict(text.model.state_dict())
    mm.to(torch.bfloat16).save_pretrained(d, max_shard_size="300KB")
    return Gemma3ForConditionalGeneration.from_pretrained(d, dtype=torch.float32, attn_implementation="eager").eval()


# ---------------------------------------------------------------- 0. the answer-key generator == transformers
t0 = time.perf_counter()
prompt = torch.randint(3, 512, (30,), generator=torch.Generator().manual_seed(0)).tolist()
with tempfile.TemporaryDirectory() as tmp:                     # a Gemma3ForCausalLM as transformers saves it
    hf, _ = build(tmp, window=8)
    want_logits, want_cap, want, want_steps = hf_decode(hf, hf.model, prompt, 12)
    cap = {}
    ref = golden_gemma3.StreamedGemma3(tmp)
    (logits,), (got,), (steps,) = ref.greedy_all([prompt], 12, [cap])
    check(f"answer-key generator on a Gemma3ForCausalLM (text under {ref.prefix!r}), 30 tokens past an 8-token window: "
          f"the scaled embedding, all 12 layers, the final norm, the logits of every position and of 11 decode steps "
          f"bit-identical to transformers' own model and cache; the same 12 greedy tokens",
          ref.prefix == "model." and set(cap) == set(want_cap) and all(torch.equal(cap[k], want_cap[k]) for k in cap)
          and torch.equal(logits, want_logits) and torch.equal(steps, want_steps) and got == want)

with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as d:
    hf = checkpoint_like(tmp, d, window=8, seed=3)
    shards = len(glob.glob(f"{d}/*.safetensors"))
    short = prompt[20:27]                                      # its greedy tokens do not repeat early
    free_long, free_short = (hf_decode(hf, hf.model.language_model, p, 8)[2] for p in (prompt, short))
    at = next(k for k in range(2, 8) if free_short[k] not in free_short[:k] + free_long)
    stop = (free_short[at],)                    # the short prompt stops at its token `at`, the long one never
    refs = [hf_decode(hf, hf.model.language_model, p, 8, stop) for p in (prompt, short)]
    ref = golden_gemma3.StreamedGemma3(d, head_rows=100)
    caps = [{}, {}]
    logits, got, steps = ref.greedy_all([prompt, short], 8, caps, stop=stop)
    exact = all(torch.equal(lg, r[0]) and set(c) == set(r[1]) and all(torch.equal(c[k], r[1][k]) for k in c)
                for lg, c, r in zip(logits, caps, refs))
    step_err = max(rel(s, r[3]) for s, r in zip(steps, refs) if len(s))
    check(f"answer-key generator on a Gemma3ForConditionalGeneration shaped like the checkpoint (bf16 in {shards} "
          f"shards, text under {ref.prefix!r}, a vision tower beside it, the head in 100-row slices), prompts of 30 "
          f"and 7 tokens in lockstep: every capture and the logits of every position bit-identical to transformers'",
          ref.prefix == "language_model.model." and shards > 2 and exact)
    check(f"... decode: the same greedy tokens, the short prompt leaving the lockstep at its stop id "
          f"({len(got[1])} tokens, ending {got[1][-1]}) while the long one runs to 8; step logits within "
          f"{step_err:.1e} (the decode steps' hidden states are exact; a single-row product through the head's "
          f"100-row slices rounds differently from the whole table)",
          all(g == r[2] for g, r in zip(got, refs)) and got[1] == free_short[:at + 1] and len(got[0]) == 8
          and step_err < 1e-6)
    long = torch.randint(3, 512, (40,), generator=torch.Generator().manual_seed(4)).tolist()
    want_logits, want_cap, want, want_steps = hf_decode(hf, hf.model.language_model, long, 6, golden_gemma3.STOP)
    key = golden_gemma3.gemma3_long_key(ref, long, 6)
    r = key["rows"]
    check(f"answer-key generator --long, 40 tokens past the 8-token window: the final norm of every position and every "
          f"layer at the {len(r)} kept rows ({r.tolist()}) bit-identical; logits there within "
          f"{rel(key['logits'], want_logits[r]):.1e}, decode steps {rel(key['step_logits'], want_steps):.1e}; "
          f"the same greedy tokens",
          torch.equal(key["final_norm"], want_cap["final_norm"])
          and all(torch.equal(v, want_cap[k][r]) for k, v in key["layers"].items()) and len(key["layers"]) == 13
          and rel(key["logits"], want_logits[r]) < 1e-6 and rel(key["step_logits"], want_steps) < 1e-6
          and key["greedy"].tolist() == want and set(range(5, 14)) <= set(r.tolist()))
    made, path = golden_gemma3.provenance(d), Path(tmp) / "0.pt"
    stale = [made | {"generator": made["generator"] + 1}, made | {"transformers": "0"},
             golden_gemma3.provenance(tmp), None]
    torch.save({"ids": torch.tensor(long), "n_new": 6, "provenance": made}, path)
    reused = golden_gemma3.done(path, long, 6, made, force=False)
    redone = [not golden_gemma3.done(path, long, 6, p, force=False) for p in stale]
    redone.append(not golden_gemma3.done(path, long, 6, made, force=True))
    torch.save({"ids": torch.tensor(long), "n_new": 6}, path)
    redone.append(not golden_gemma3.done(path, long, 6, made, force=False))
    check(f"answer-key resume: a saved record is reused only when its provenance matches (generator "
          f"{made['generator']}, transformers {made['transformers']}, torch {made['torch']}, checkpoint "
          f"{made['checkpoint'][:12]}); a newer generator, another transformers or checkpoint, a record without "
          f"provenance or --force regenerates it ({sum(redone)}/{len(redone)})", reused and all(redone))
print(f"      {time.perf_counter() - t0:.0f} s")

# ---------------------------------------------------------------- the real model
if not os.path.exists(f"{D}/model.safetensors.index.json"):
    print(f"SKIP  the real model: needs the checkpoint in {D}")
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
goldens = sorted(glob.glob(f"{G}/*.pt"), key=lambda f: int(os.path.basename(f)[:-3]))
if not goldens:
    print("FAIL: no answer key: run scripts/.venv/bin/python scripts/golden_gemma3.py")
    sys.exit(1)

t0 = time.perf_counter()
recs = [torch.load(f) for f in goldens]
tok, tmpl = Tokenizer(f"{D}/tokenizer.json"), TemplateChat(f"{D}/tokenizer_config.json")
cfg, weights = Gemma3Config.from_json(f"{D}/config.json"), open_weights(f"{D}/model.safetensors.index.json")
model = Gemma3Model(cfg, weights, stream=True)
STOP = set(golden_gemma3.STOP)


# 1. the engine's tokenizer and chat template give the answer key's ids (made with transformers' tokenizer)
def engine_ids(r):
    if r["chat"]:
        return tok.encode(format_chat(golden_gemma3.messages(r["target"], r["text"]), style=tmpl))
    return tok.encode(r["text"], add_bos=True)
targets = sorted({r["target"] for r in recs if r["chat"]})
check(f"ids from the engine's tokenizer and chat template == the answer key's, all {len(recs)} prompts "
      f"({sum(len(r['ids']) for r in recs)} tokens; targets {', '.join(targets)}; "
      f"{sum(not r['chat'] for r in recs)} raw)",
      all(engine_ids(r) == r["ids"].tolist() for r in recs))

# 2. one packed prefill for all prompts, each into its own state: every layer and the logits of every position
states = [model.new_state(len(r["ids"]) + len(r["greedy"]) + 1) for r in recs]
cap = {}
h = model.packed_hidden([(r["ids"], st) for r, st in zip(recs, states)], capture=cap)
lens = [len(r["ids"]) for r in recs]
caps = {k: v.split(lens) for k, v in cap.items()}
KEYS = ["embed"] + [f"l{i}_out" for i in range(cfg.num_hidden_layers)] + ["final_norm"]
per_layer = {k: max(rel(caps[k][j], r[k]) for j, r in enumerate(recs)) for k in KEYS}
per_row = {k: max(rel_rows(caps[k][j], r[k]) for j, r in enumerate(recs)) for k in KEYS}
embed_exact = all(torch.equal(caps["embed"][j], r["embed"]) for j, r in enumerate(recs))
del cap, caps
logit_err, first = [], []
for hj, r in zip(h.split(lens), recs):                     # the head one prompt at a time (~1 MB per row)
    lg = model.head(hj)
    logit_err.append((rel(lg, r["logits"]), rel_rows(lg, r["logits"])))
    first.append(int(lg[-1].argmax()))
del h
worst, worst_row = max(per_layer, key=per_layer.get), max(per_row, key=per_row.get)
check(f"the scaled embedding rows of all {len(recs)} prompts are bit-identical to the answer key's", embed_exact)
check(f"all {len(KEYS)} captured tensors (embedding, {cfg.num_hidden_layers} layers, final norm) of every position "
      f"== the answer key: worst relative error {per_layer[worst]:.1e} ({worst}); worst single token "
      f"{per_row[worst_row]:.1e} ({worst_row})", per_layer[worst] < 3e-5 and per_row[worst_row] < 5e-5)
check(f"logits of every position ({sum(lens)}) == the answer key: worst {max(e for e, _ in logit_err):.1e}, worst "
      f"single token {max(r for _, r in logit_err):.1e}",
      max(e for e, _ in logit_err) < 3e-5 and max(r for _, r in logit_err) < 5e-5)

# 3. greedy: one batched decode step at a time for the prompts not yet stopped, each step's logits compared with the
# answer key's (the same tokens are fed to both, as long as the greedy tokens agree)
tokens, step_err, n_steps = [[f] for f in first], 0.0, 0
while live := [j for j, t in enumerate(tokens) if t[-1] not in STOP and len(t) < len(recs[j]["greedy"])]:
    out = model.decode_batch([tokens[j][-1] for j in live], [states[j] for j in live])
    for j, row in zip(live, out):
        k = len(tokens[j]) - 1
        tokens[j].append(int(row.argmax()))
        step_err = max(step_err, rel_rows(row[None], recs[j]["step_logits"][k][None]))
        n_steps += 1
same = [t == r["greedy"].tolist() for t, r in zip(tokens, recs)]
for t, r, ok in zip(tokens, recs, same):
    print(f"      {'ok  ' if ok else 'DIFF'} {(r['target'] or 'raw'):8s} {len(t):2d} tokens"
          f"{' + stop' if t[-1] in STOP else ''}  {tok.decode([x for x in t if x not in STOP])!r}")
stopped = sum(t[-1] in STOP for t in tokens)
check(f"greedy: identical to the answer key on {sum(same)}/{len(same)} prompts ({stopped} end at a stop id, "
      f"{sum(map(len, tokens))} tokens); the logits of all {n_steps} decode steps == the answer key's "
      f"(worst {step_err:.1e})", all(same) and step_err < 7e-5)
del states
print(f"      {time.perf_counter() - t0:.0f} s")


# 4. the embedding scale rounded to bf16 (50.5) instead of fp32: the same tokens (prompt, then the key's greedy ones),
# KL from the key at every position the key has logits for (prompt positions and decode steps)
def kl_rows(p_logits, q_logits):
    p, q = p_logits.log_softmax(-1), q_logits.log_softmax(-1)
    return (p.exp() * (p - q)).sum(-1)


bf = Gemma3Model(cfg, weights, stream=True, embed_scale_dtype=torch.bfloat16)
seqs = [torch.cat([r["ids"], r["greedy"][:-1]]) for r in recs]
hb = bf.packed_hidden([(s, bf.new_state(len(s))) for s in seqs])
kl, agree, path_same = [], 0, True
for hj, r in zip(hb.split([len(s) for s in seqs]), recs):
    key = torch.cat([r["logits"], r["step_logits"]])
    lg = bf.head(hj)
    kl.append(kl_rows(key, lg))
    agree += (lg.argmax(-1) == key.argmax(-1)).sum().item()
    path_same &= torch.equal(lg[len(r["ids"]) - 1:].argmax(-1), r["greedy"])       # bf16 scale's own greedy == key's
kl = torch.cat(kl)
check(f"embedding scale {bf.embed_scale} (bf16) vs {model.embed_scale:.4f} (fp32) over {len(kl)} positions: KL from "
      f"the key mean {kl.mean().item():.2e}, median {kl.median().item():.2e}, max {kl.max().item():.2e}; same top "
      f"token at {agree}/{len(kl)}; greedy {'identical' if path_same else 'DIFFERENT'} on every prompt",
      bf.embed_scale == 50.5 and 0 < kl.mean().item() < 1e-2)
print(f"      {time.perf_counter() - t0:.0f} s")

print(f"\n{sum(results)}/{len(results)} passed ({time.perf_counter() - t0:.0f} s)")
sys.exit(0 if all(results) else 1)
