"""Sarvam-Translate port, M1-M2: Gemma3Config == transformers' parsing, and Gemma3Model == transformers'
Gemma3ForCausalLM on small random models, fp32 on the CPU (tests/tiny_gemma3.py: 12 layers, full attention at 5 and 11,
8 query heads per 2 KV heads of 32 dims, linear factor 8 on the full layers, local theta 1e4, global 1e6).

Config: the real checkpoint's config.json (a Gemma3ForConditionalGeneration with legacy RoPE keys), a tiny model's
(transformers 5's gemma3_text), legacy files with other thetas, and a text_config left to the defaults, field by field
and RoPE table by table against transformers; every setting the engine does not implement is refused.
Model: every layer and the logits, and bit-identical once transformers' RMSNorm and attention formulas are swapped into
the reference backend (what is left is those ops' rounding); that the test notices a missing embedding scale, a missing
q/k norm, a missing RoPE factor and swapped thetas; greedy tokens; cached == uncached; paged == contiguous; batched
decode == one at a time; packed prefill == separate, with a continuing chunk; stream=True; forks; bf16 KV;
embed_scale_dtype=bfloat16 == transformers' bf16 scale; the checkpoint's layout (text tensors under
"language_model.model.", vision tensors never read, an untied head refused); and with an 8-token window, every path
past it, rings reused and NaN-poisoned, and the length cap. On a Metal GPU: the bf16, INT8 and INT4 backends == the
fp32 CPU model on the same rounded weights. The real checkpoint is read by name only (its index): no weight is loaded.
"""
import dataclasses
import json
import os
import re
import sys
import tempfile
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, Gemma3ForCausalLM
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

sys.path.insert(0, "src")
import ops
from backend.torch_ref import TorchBackend
from config import Gemma3Config
from models.gemma3 import Gemma3Model
from tiny_gemma3 import FACTOR, FULL_LAYERS, GLOBAL_THETA, LOCAL_THETA, build, hf_config, hf_greedy, hf_run, rel
from weight_loader import SafetensorsFile

D = "models/sarvam-translate"
REAL = os.path.exists(f"{D}/config.json") and os.path.exists(f"{D}/model.safetensors.index.json")
torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


# 1. the config against transformers' own parsing: every field, and per layer type the RoPE table that transformers'
# Gemma3RotaryEmbedding builds from it (and its attention scaling, 1.0)
FIELDS = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "head_dim", "rms_norm_eps", "sliding_window", "layer_types", "max_position_embeddings",
          "tie_word_embeddings")


def differences(d: str) -> list[str]:
    ours, hf = Gemma3Config.from_json(f"{d}/config.json"), AutoConfig.from_pretrained(d).get_text_config()
    diff = [f for f in FIELDS if getattr(ours, f) != getattr(hf, f)]
    rot = Gemma3RotaryEmbedding(hf)
    for t in set(hf.layer_types):
        if not torch.equal(ops.rope_inv_freq(ours.rope[t][0], ours.head_dim, ours.rope[t][1]),
                           getattr(rot, f"{t}_inv_freq")) or getattr(rot, f"{t}_attention_scaling") != 1.0:
            diff.append(f"RoPE of {t}")
    return diff


with tempfile.TemporaryDirectory() as tmp:
    def write(cfg: dict, name: str) -> str:
        os.makedirs(f"{tmp}/{name}", exist_ok=True)
        json.dump(cfg, open(f"{tmp}/{name}/config.json", "w"))
        return f"{tmp}/{name}"
    hf_config(window=8).save_pretrained(f"{tmp}/tiny")
    tiny = json.load(open(f"{tmp}/tiny/config.json"))
    cases = {"a tiny model's (gemma3_text, rope_parameters per layer type)": f"{tmp}/tiny",
             "a text_config left to transformers' defaults": write({"model_type": "gemma3",
                                                                    "text_config": {"num_hidden_layers": 7}}, "dflt")}
    if REAL:
        real = json.load(open(f"{D}/config.json"))
        legacy = dict(real["text_config"], num_hidden_layers=8, sliding_window_pattern=3, rope_theta=777.0,
                      rope_local_base_freq=55.0, rope_scaling={"rope_type": "linear", "factor": 4.0})
        cases |= {"the real checkpoint's (text_config, legacy rope_theta / rope_local_base_freq / rope_scaling)": D,
                  "legacy keys with other thetas, factor 4, sliding_window_pattern 3":
                      write(dict(real, text_config=legacy), "legacy"),
                  "legacy rope_scaling {'type': 'linear'}, which transformers leaves at rope_type 'default'":
                      write(dict(real, text_config=dict(legacy, rope_scaling={"type": "linear", "factor": 4.0})), "q")}
    for name, d in cases.items():
        diff = differences(d)
        check(f"config == transformers on all {len(FIELDS)} fields and both RoPE tables: {name}"
              + (f" (differ: {diff})" if diff else ""), not diff)
    if REAL:
        c = Gemma3Config.from_json(f"{D}/config.json")
        full = [i for i in range(c.num_hidden_layers) if not c.is_sliding(i)]
        check(f"Sarvam-Translate: 34 layers, full attention at {full}, 8/4 heads x 256, window 1024, RoPE sliding "
              f"{c.rope['sliding_attention']} full {c.rope['full_attention']}, {c.max_position_embeddings} positions",
              (c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.sliding_window)
              == (34, 8, 4, 256, 1024) and full == [5, 11, 17, 23, 29]
              and c.rope == {"sliding_attention": (1e4, 1.0), "full_attention": (1e6, 8.0)})
    else:
        print(f"SKIP  the real checkpoint's config: needs {D}/config.json")
    rope = tiny["rope_parameters"]
    bad = {"final_logit_softcapping": dict(tiny, final_logit_softcapping=30.0),
           "attn_logit_softcapping": dict(tiny, attn_logit_softcapping=50.0),
           "use_bidirectional_attention": dict(tiny, use_bidirectional_attention=True),
           "attention_bias": dict(tiny, attention_bias=True),
           "hidden_activation gelu": dict(tiny, hidden_activation="gelu"),
           "query_pre_attn_scalar != head_dim": dict(tiny, query_pre_attn_scalar=64),
           "an untied head": dict(tiny, tie_word_embeddings=False),
           "an untied head (top level)": {"model_type": "gemma3", "text_config": tiny, "tie_word_embeddings": False},
           "yarn RoPE": dict(tiny, rope_parameters=rope | {"full_attention": {"rope_type": "yarn", "factor": 8.0,
                                                                               "rope_theta": 1e6}}),
           "partial rotary": dict(tiny, rope_parameters=rope | {"sliding_attention": rope["sliding_attention"]
                                                                | {"partial_rotary_factor": 0.5}}),
           "a linear factor of 0": dict(tiny, rope_parameters=rope | {"full_attention": rope["full_attention"]
                                                                       | {"factor": 0}}),
           "layer_types one short": dict(tiny, layer_types=tiny["layer_types"][:-1]),
           "a chunked layer type": dict(tiny, layer_types=["chunked_attention"] + tiny["layer_types"][1:]),
           "sliding_window_pattern 0": {"model_type": "gemma3", "text_config": dict(
               {k: v for k, v in tiny.items() if k != "layer_types"}, sliding_window_pattern=0)},
           "no sliding window": dict(tiny, sliding_window=None), "model_type gemma2": dict(tiny, model_type="gemma2")}
    accepted = []
    for name, cfg in bad.items():
        try:
            Gemma3Config.from_json(f"{write(cfg, 'bad')}/config.json"); accepted.append(name)
        except ValueError:
            pass
    check(f"refused, before any weight is read: {', '.join(bad)}" + (f" (accepted: {accepted})" if accepted else ""),
          not accepted)


class HFFormulas(TorchBackend):
    """The reference backend with transformers' formulas: Gemma3RMSNorm (modeling_gemma3.py:142-150) and eager
    attention, scores * query_pre_attn_scalar^-0.5 (== head_dim here) with a causal and sliding mask."""
    def rms_norm(self, x, w, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def attention(self, q, k, v, causal=True, window=None):
        (T, Hq, d), S = q.shape, k.shape[0]
        kk, vv = (t.repeat_interleave(Hq // t.shape[1], dim=1).transpose(0, 1) for t in (k, v))
        scores = q.transpose(0, 1) @ kk.transpose(1, 2) * d ** -0.5
        qp, kp = torch.arange(S - T, S)[:, None], torch.arange(S)[None]
        sees = kp <= qp
        if window:
            sees &= kp > qp - window
        return (torch.softmax(scores.masked_fill(~sees, float("-inf")), -1) @ vv).transpose(0, 1)


# 2. the model, window 4096 (above every test's length)
with tempfile.TemporaryDirectory() as tmp:
    hf, make = build(tmp, window=4096)
    model = make()
    prompt = torch.randint(3, 512, (23,), generator=torch.Generator().manual_seed(1))
    ref_logits, ref = hf_run(hf, prompt)
    cap = {}
    logits = model.forward(prompt, capture=cap)
    worst = max(rel(cap[k], ref[k]) for k in ref)
    check(f"all {len(ref)} captured tensors (scaled embedding, 12 layers, final norm) == HF, worst relative error "
          f"{worst:.1e}; logits {rel(logits, ref_logits):.1e}",
          len(ref) == 14 and worst < 1e-5 and rel(logits, ref_logits) < 1e-5)
    cap = {}
    same = torch.equal(make(backend=HFFormulas()).forward(prompt, capture=cap), ref_logits)
    check("with transformers' RMSNorm and attention formulas in the reference backend, every captured tensor and the "
          "logits are bit-identical to HF", same and all(torch.equal(cap[k], ref[k]) for k in ref))

    no_scale, no_qk, no_factor, swapped = make(), make(), make(), make()
    no_scale.embed_scale = 1.0
    norm = no_qk._norm
    no_qk._norm = lambda x, name: x if name.endswith(("q_norm.weight", "k_norm.weight")) else norm(x, name)
    no_factor.config = dataclasses.replace(model.config, rope=model.config.rope | {"full_attention": (GLOBAL_THETA, 1)})
    swapped.config = dataclasses.replace(model.config, rope={"sliding_attention": (GLOBAL_THETA, 1.0),
                                                             "full_attention": (LOCAL_THETA, FACTOR)})
    errs = {k: rel(m.forward(prompt), ref_logits) for k, m in (("no embedding scale", no_scale),
            ("no q/k norm", no_qk), ("no RoPE factor", no_factor), ("local/global theta swapped", swapped))}
    check(f"sensitivity: the comparison catches {', '.join(f'{k} ({v:.1e})' for k, v in errs.items())}",
          min(errs.values()) > 1e-2)

    want = hf_greedy(hf, prompt, 12)
    state = model.new_state(64)
    step = model.forward(prompt, state=state, last_only=True)[0]
    got, worst = [], 0.0
    for _ in range(12):
        nxt = int(step.argmax())
        got.append(nxt)
        full = model.forward(torch.tensor(prompt.tolist() + got))[-1]          # no cache: the whole sequence again
        step = model.forward(torch.tensor([nxt]), state=state)[0]
        worst = max(worst, rel(step, full))
    check(f"greedy: 12 tokens ({len(set(got))} distinct) identical to HF's", got == want and len(set(got)) > 3)
    check(f"cached decode == recomputing the whole sequence, relative error {worst:.1e}", worst < 1e-5)

    def run(st):
        out = [model.forward(prompt[:9], state=st), model.forward(prompt[9:], state=st)]
        return torch.cat(out + [model.forward(torch.tensor([t]), state=st) for t in got[:6]])
    pool = model.new_paged_pool(num_blocks=64, block_size=4, max_seqs=8)
    paged_state = model.new_paged_state(pool)
    a, b = run(model.new_state(64)), run(paged_state)
    check(f"paged KV (4-token blocks) == contiguous, max difference {(a - b).abs().max().item():.1e}",
          torch.allclose(a, b, rtol=0, atol=1e-6) and len(paged_state.kv.block_table) == -(-(23 + 6) // 4))
    paged_state.free()

    prompts = [prompt[:5], prompt[:17], prompt[3:14]]
    for kind in ("paged", "contiguous"):
        new = (lambda: model.new_paged_state(pool)) if kind == "paged" else (lambda: model.new_state(64))
        alone, batched = [new() for _ in prompts], [new() for _ in prompts]
        for p, s in zip(prompts, alone):
            model.forward(p, state=s)
        packed = model.forward_packed([(p, s) for p, s in zip(prompts, batched)])
        worst = rel(packed, torch.stack([model.forward(p)[-1] for p in prompts]))
        for t in (7, 11, 13):
            one = torch.cat([model.forward(torch.tensor([t]), state=s) for s in alone])
            worst = max(worst, rel(model.decode_batch([t] * 3, batched), one))
        for s in alone + batched:
            s.free()
        cont, fresh = new(), new()
        model.forward(prompt[:9], state=cont)
        packed = model.forward_packed([(prompt[:5], fresh), (prompt[9:], cont)])
        worst = max(worst, rel(packed, torch.stack([model.forward(prompt[:5])[-1], model.forward(prompt)[-1]])))
        cont.free(); fresh.free()
        check(f"{kind}: packed prefill == separate (also a chunk continuing a state), batched decode == one at a time, "
              f"relative error {worst:.1e}", worst < 1e-5)
    pc = {}
    h = model.packed_hidden([(prompt, None), (prompt[:5], None)], capture=pc)
    check("packed_hidden: the head of its rows == forward; its capture (the prompt and a 5-token prefix in one pack) "
          "== HF at every layer", rel(model.head(h[:23]), ref_logits) < 1e-5 and set(pc) == set(ref)
          and max(max(rel(pc[k][:23], ref[k]), rel(pc[k][23:], ref[k][:5])) for k in ref) < 1e-5)

    class Bf16:                                        # bf16 weights, so that stream=True's widening is real
        def get(self, name):
            return model.weights.get(name).to(torch.bfloat16)

        def tensor_names(self):
            return model.weights.tensor_names()
    streamed, normal = Gemma3Model(model.config, Bf16(), stream=True), Gemma3Model(model.config, Bf16())
    streamed.head_rows = 100
    widened, prepare = [], streamed.b.prepare
    streamed.b.prepare = lambda w, name=None: (widened.append(tuple(w.shape)), prepare(w, name))[1]
    s_state, n_state = streamed.new_state(64), normal.new_state(64)
    a = torch.cat([streamed.forward(prompt, state=s_state), streamed.decode_batch([7], [s_state])])
    b = torch.cat([normal.forward(prompt, state=n_state), normal.decode_batch([7], [n_state])])
    kept = [k for k, v in streamed._cache.items() if v.dim() > 1]
    check(f"stream=True on bf16 weights: the same logits ({rel(a, b):.1e}), no widened weight kept, the 512-row table "
          "only ever widened in slices of 100 rows",
          rel(a, b) < 1e-6 and not kept and (512, 128) not in widened and (100, 128) in widened)

    base = model.new_state(64)
    model.forward(prompt, state=base)
    fork = base.fork()
    x, y = model.forward(torch.tensor([7]), state=base), model.forward(torch.tensor([9]), state=fork)
    w = model.forward(torch.tensor([5]), state=base)[0]
    check("fork: copy and original diverge at the same position, and the original stays exact",
          rel(w, model.forward(torch.tensor(prompt.tolist() + [7, 5]))[-1]) < 1e-5
          and rel(y[0], model.forward(torch.tensor(prompt.tolist() + [9]))[-1]) < 1e-5)

    m16 = make(kv_dtype=torch.bfloat16)
    def run16(st):
        out = [m16.forward(prompt[:9], state=st), m16.forward(prompt[9:], state=st)]
        return torch.cat(out + [m16.forward(torch.tensor([t]), state=st) for t in got[:6]])
    c16, g16 = run16(m16.new_state(64)), run16(m16.new_paged_state(m16.new_paged_pool(64, 4, 8)))
    s16 = m16.new_state(64)
    p16 = m16.forward(prompt, state=s16)
    lp, lq = torch.log_softmax(ref_logits, -1), torch.log_softmax(p16, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    agree = int((p16.argmax(-1) == ref_logits.argmax(-1)).sum())
    check(f"bf16 KV (the registry's): stored as bfloat16, paged == contiguous (max difference "
          f"{(c16 - g16).abs().max().item():.1e}); vs HF's fp32 K/V: same argmax at {agree}/{len(prompt)} positions, "
          f"KL mean {kl.mean():.1e} max {kl.max():.1e}", s16.kv.k.dtype == torch.bfloat16
          and torch.allclose(c16, g16, rtol=0, atol=1e-6) and agree == len(prompt))

    # embed_scale_dtype: transformers multiplies by its fp32 sqrt(hidden) buffer cast to the weights' dtype; with bf16
    # weights that is the bf16 value. HF here (fp32) is given that value, so only the scale changes
    m_bf = make(embed_scale_dtype=torch.bfloat16)
    buf = hf.model.embed_tokens.embed_scale
    fp32_scale, bf16_scale = buf.item(), buf.to(torch.bfloat16).item()
    buf.fill_(bf16_scale)
    want_bf, _ = hf_run(hf, prompt)
    buf.fill_(fp32_scale)
    got_bf = m_bf.forward(prompt)
    check(f"embed_scale_dtype: float32 {model.embed_scale:.7f} and bfloat16 {m_bf.embed_scale} == transformers' with "
          f"fp32 and bf16 weights; bf16: logits == HF given that scale ({rel(got_bf, want_bf):.1e}), "
          f"{rel(got_bf, ref_logits):.1e} from the fp32 scale's",
          (model.embed_scale, m_bf.embed_scale) == (fp32_scale, bf16_scale) and fp32_scale != bf16_scale
          and rel(got_bf, want_bf) < 1e-5 and rel(got_bf, ref_logits) > 1e-5)

    # the checkpoint's layout: a Gemma3ForConditionalGeneration's text tensors under "language_model.model.", next to
    # vision tensors (NaN here: a read would show), the config under text_config
    with tempfile.TemporaryDirectory() as ckpt:
        tensors = {"language_model." + n: model.weights.get(n) for n in model.weights.tensor_names()}
        tensors |= {n: torch.full((4, 4), float("nan")) for n in ("multi_modal_projector.mm_soft_emb_norm.weight",
                                                                  "vision_tower.vision_model.post_layernorm.weight")}
        save_file(tensors, f"{ckpt}/model.safetensors")
        json.dump({"model_type": "gemma3", "text_config": json.load(open(f"{tmp}/config.json"))},
                  open(f"{ckpt}/config.json", "w"))
        read = set()

        class Recording(SafetensorsFile):
            def get(self, name):
                read.add(name)
                return super().get(name)
        mm = Gemma3Model(Gemma3Config.from_json(f"{ckpt}/config.json"), Recording(f"{ckpt}/model.safetensors"))
        same = torch.equal(mm.forward(prompt), logits)
        check(f"the checkpoint's layout (text under {mm.prefix!r}, vision tensors beside it, config under "
              f"text_config): the same logits bit for bit; {len(read)} tensors read, none of the vision ones",
              mm.prefix == "language_model.model." and same and all(n.startswith(mm.prefix) for n in read))
        head = tensors["language_model.model.embed_tokens.weight"].clone()
        save_file(tensors | {"language_model.lm_head.weight": head}, f"{ckpt}/untied.safetensors")
        try:
            Gemma3Model(mm.config, SafetensorsFile(f"{ckpt}/untied.safetensors")); refused = False
        except ValueError:
            refused = True
        check("a checkpoint with its own language_model.lm_head.weight is refused (the head is tied)", refused)
    kinds = {re.sub(r"layers\.\d+\.", "layers.N.", n.removeprefix(mm.prefix)) for n in read}

if REAL:                                               # by name only: the index, never the weights
    index = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]

    class Names:
        @staticmethod
        def tensor_names():
            return list(index)
    cfg = Gemma3Config.from_json(f"{D}/config.json")
    fp32, bf16 = Gemma3Model(cfg, Names()), Gemma3Model(cfg, Names(), embed_scale_dtype=torch.bfloat16)
    text = [n for n in index if n.startswith(fp32.prefix)]
    real_kinds = {re.sub(r"layers\.\d+\.", "layers.N.", n.removeprefix(fp32.prefix)) for n in text}
    check(f"the real checkpoint's index: text model under {fp32.prefix!r}, {len(text)} tensors (34 x 13 + embedding + "
          f"final norm), exactly the {len(kinds)} kinds the engine reads; {len(index) - len(text)} others never read; "
          f"embedding scale {fp32.embed_scale:.6f} (fp32), {bf16.embed_scale} (bf16, the registry's)",
          fp32.prefix == "language_model.model." and len(text) == 34 * 13 + 2 and real_kinds == kinds
          and bf16.embed_scale == 50.5 and fp32.embed_scale == torch.tensor(2560 ** 0.5).item())
else:
    print(f"SKIP  the real checkpoint's names: needs {D}/model.safetensors.index.json")

# 3. an 8-token window over 40 positions (5 windows), on every path: stateless; chunked prefill on contiguous and paged
# states (chunks ending at the window, longer than it, token by token; blocks of 4 and 3, which do not align with it);
# greedy decode; batched decode; packed prefill; forks; and the grouped pool's rings, every slot reused and filled with
# NaN whenever it takes a new block, so a stale read shows
with tempfile.TemporaryDirectory() as tmp:
    hf, make = build(tmp, window=8, seed=5)
    model = make()
    ids = torch.randint(3, 512, (40,), generator=torch.Generator().manual_seed(5))
    ref_logits, ref = hf_run(hf, ids)
    cap = {}
    worst = max(rel(model.forward(ids, capture=cap), ref_logits), max(rel(cap[k], ref[k]) for k in ref))
    hf_full = Gemma3ForCausalLM._from_config(hf_config(window=4096), attn_implementation="eager").eval()
    hf_full.load_state_dict(hf.state_dict())
    binds = rel(hf_full(ids[None]).logits[0], ref_logits)
    check(f"window 8, 40 positions, stateless: every layer and the logits == HF, worst {worst:.1e} (HF's window moves "
          f"its logits by {binds:.1e})", worst < 1e-5 and binds > 1e-2)

    def chunked(st, sched):
        out, pos = [], 0
        for n in sched:
            out.append(model.forward(ids[pos:pos + n], state=st)); pos += n
        return torch.cat(out)
    schedules = [[40], [8, 32], [5, 3, 32], [3, 17, 20], [9, 7, 8, 16], [1] * 40]
    for kind, bs in (("contiguous", None), ("paged", 4), ("paged", 3)):
        new = (lambda: model.new_state(64)) if bs is None else \
            (lambda bs=bs: model.new_paged_state(model.new_paged_pool(num_blocks=64, block_size=bs, max_seqs=2)))
        worst = max(rel(chunked(new(), sch), ref_logits) for sch in schedules)
        check(f"window 8, {kind}{f' (blocks of {bs})' if bs else ''}: chunked prefill == HF for {len(schedules)} "
              f"chunk schedules, worst {worst:.1e}", worst < 1e-5)
        want = hf_greedy(hf, ids[:6], 20)
        st, got = new(), []
        nxt = int(model.forward(ids[:6], state=st, last_only=True)[0].argmax())
        for _ in range(20):
            got.append(nxt)
            nxt = int(model.forward(torch.tensor([nxt]), state=st)[0].argmax())
        check(f"window 8, {kind}: greedy 20 tokens ({len(set(got))} distinct) from a 6-token prompt == HF", got == want)

    pool = model.new_paged_pool(num_blocks=64, block_size=4, max_seqs=8)
    lens, toks = (5, 13, 30), (7, 9, 11)
    paged, alone = [model.new_paged_state(pool) for _ in lens], [model.new_state(64) for _ in lens]
    for st, a, n in zip(paged, alone, lens):
        model.forward(ids[:n], state=st); model.forward(ids[:n], state=a)
    worst = 0.0
    for step in range(3):
        b = model.decode_batch([t + step for t in toks], paged)
        worst = max(worst, rel(b, torch.cat([model.forward(torch.tensor([t + step]), state=a)
                                             for t, a in zip(toks, alone)])))
    a, b = model.new_paged_state(pool), model.new_paged_state(pool)
    model.forward(ids[:11], state=b)
    packed = model.forward_packed([(ids[:7], a), (ids[11:25], b)])          # b's chunk crosses its window again
    worst = max(worst, rel(packed, torch.stack([model.forward(ids[:7])[-1], model.forward(ids[:25])[-1]])))
    base = model.new_state(64)
    model.forward(ids[:30], state=base)
    fork = base.fork()
    x, y = model.forward(torch.tensor([7]), state=base)[0], model.forward(torch.tensor([9]), state=fork)[0]
    worst = max(worst, rel(x, model.forward(torch.cat([ids[:30], torch.tensor([7])]))[-1]),
                rel(y, model.forward(torch.cat([ids[:30], torch.tensor([9])]))[-1]))
    check(f"window 8: batched paged decode (5, 13, 30 positions, 3 steps) == each alone, a packed chunk continuing "
          f"past the window == separate, a fork at 30 and the original exact; worst {worst:.1e}", worst < 1e-5)

    # rings: chunks of at most 3 tokens, so the sliding groups keep R = ceil((8 + 3 + 4 - 2) / 4) = 4 blocks of 4 and
    # every slot is reused 2-3 times over 40 positions; the full group (layers 5 and 11) keeps every block
    ring_pool = model.new_paged_pool(num_blocks=64, block_size=4, max_seqs=4, max_chunk=3, poison=True)
    st, pos, out = model.new_paged_state(ring_pool), 0, []
    for n in [3, 1, 2, 3, 3, 2, 1, 3, 3, 3, 2, 3, 3, 3, 3, 2]:
        out.append(model.forward(ids[pos:pos + n], state=st)); pos += n
    got, held = torch.cat(out), [len(t) for t in st.kv.tables]
    nxt, gen = int(got[-1].argmax()), []
    for _ in range(10):
        gen.append(nxt)
        nxt = int(model.forward(torch.tensor([nxt]), state=st)[0].argmax())
    check(f"ring: 40 positions in chunks <= 3, every slot reused and NaN-poisoned: == HF {rel(got, ref_logits):.1e}, "
          f"then 10 greedy tokens == HF's; units held per group {held} (full layers {FULL_LAYERS}: 10 blocks; "
          f"sliding: rings of {ring_pool.kv.ring})", rel(got, ref_logits) < 1e-5 and bool(torch.isfinite(got).all())
          and held == [10] + [4] * 5 and ring_pool.kv.groups[0] == FULL_LAYERS and gen == hf_greedy(hf, ids, 10))
    ring_pool = model.new_paged_pool(num_blocks=64, block_size=4, max_seqs=4, max_chunk=3, poison=True)
    rings, alone = [model.new_paged_state(ring_pool) for _ in range(3)], [model.new_state(64) for _ in range(3)]
    for r, a, n in zip(rings, alone, (5, 19, 30)):
        for i in range(0, n, 3):
            model.forward(ids[i:min(n, i + 3)], state=r)
        model.forward(ids[:n], state=a)
    worst = 0.0
    for step in range(6):
        b = model.decode_batch([7 + step, 9 + step, 11 + step], rings)
        worst = max(worst, rel(b, torch.cat([model.forward(torch.tensor([t + step]), state=a)
                                             for t, a in zip((7, 9, 11), alone)])))
    free, length = ring_pool.allocator.num_free, rings[1].length
    try:
        model.forward(ids[:20], state=rings[1]); refused = False
    except ValueError:
        refused = True
    check(f"ring: batched decode of 3 sequences (5, 19, 30 positions, 6 steps) == each alone, worst {worst:.1e}; a "
          "20-token chunk (the ring holds chunks of 3) refused, nothing moved", worst < 1e-5 and refused
          and ring_pool.allocator.num_free == free and rings[1].length == length)

# 4. the length cap (max_position_embeddings): a model with an 8-token window and 16 positions
with tempfile.TemporaryDirectory() as tmp:
    hf, make = build(tmp, window=8, seed=2, max_pos=16)
    model = make()
    ids = torch.randint(3, 512, (17,), generator=torch.Generator().manual_seed(3))
    ref16, _ = hf_run(hf, ids[:16])
    exact = rel(model.forward(ids[:16]), ref16)
    try:
        model.forward(ids); refused = False
    except ValueError:
        refused = True
    check(f"cap 16: 16 positions exact with the window binding ({exact:.1e}), 17 refused", exact < 1e-5 and refused)
    for kind in ("contiguous", "paged"):
        pool = model.new_paged_pool(num_blocks=16, block_size=4, max_seqs=8)
        st = model.new_state(32) if kind == "contiguous" else model.new_paged_state(pool)
        model.forward(ids[:9], state=st)
        free = pool.allocator.num_free
        try:
            model.forward(ids[9:17], state=st); refused = False
        except ValueError:
            refused = True
        untouched = st.length == 9 and pool.allocator.num_free == free
        rest = model.forward(ids[9:16], state=st)[-1]
        check(f"cap 16, {kind}: a continuing chunk crossing it is refused, nothing moved, then exact",
              refused and untouched and rel(rest, ref16[-1]) < 1e-5)
    pool = model.new_paged_pool(num_blocks=16, block_size=4, max_seqs=8)
    a, b = model.new_paged_state(pool), model.new_paged_state(pool)
    model.forward(ids[:5], state=a)
    model.forward(ids[:16], state=b)
    free, refused = pool.allocator.num_free, []
    for attempt in (lambda: model.forward_packed([(ids[:3], model.new_paged_state(pool)), (ids[:1], b)]),
                    lambda: model.decode_batch([5, 5], [a, b])):
        try:
            attempt(); refused.append(False)
        except ValueError:
            refused.append(True)
    check("cap 16: a pack and a batched decode with one sequence at it are refused whole, no block taken, no state "
          "moved", refused == [True, True] and (a.length, b.length) == (5, 16) and pool.allocator.num_free == free)

# 5. on a Metal GPU, the registry's backends: bf16, INT8 and INT4 weights through the kernels (prefill, batched paged
# and single-row decode past the window, the fused GeGLU, both RoPE tables) == the fp32 CPU model on the same rounded
# weights
if torch.backends.mps.is_available():
    from backend.metal import MetalBackend
    from quant import quantize, scheme_for
    with tempfile.TemporaryDirectory() as tmp:
        _, make = build(tmp, window=8, seed=5)
        cpu = make()
        ids = torch.randint(3, 512, (40,), generator=torch.Generator().manual_seed(5))

        def rounded(scheme):
            """The weights as MetalBackend(scheme) holds them, widened back to fp32."""
            def get(name):
                w = cpu.weights.get(name)
                if w.ndim == 2 and scheme and w.shape[1] % 32 == 0:
                    return quantize(w, scheme_for(name, scheme)).dequantize().float()
                return w.to(torch.bfloat16).float() if w.ndim == 2 else w
            return types.SimpleNamespace(get=get, tensor_names=cpu.weights.tensor_names)
        for scheme in (None, "int8", "int4"):
            want = Gemma3Model(cpu.config, rounded(scheme)).forward(ids)
            m = make(backend=MetalBackend(scheme))
            pool = m.new_paged_pool(num_blocks=64, block_size=4, max_seqs=4, max_chunk=8)
            a, b, one = m.new_paged_state(pool), m.new_paged_state(pool), m.new_state(64)
            pre = m.forward(ids[:20], state=a)
            m.forward(ids[:12], state=b)
            m.forward(ids[:20], state=one)
            batched = torch.stack([m.decode_batch([int(ids[t]), int(ids[t - 8])], [a, b]) for t in range(20, 40)]).cpu()
            single = torch.cat([m.forward(ids[t:t + 1], state=one) for t in range(20, 40)]).cpu()
            worst = max(rel(pre.cpu(), want[:20]), rel(batched[:, 0], want[20:]), rel(batched[:, 1], want[12:32]),
                        rel(single, want[20:]))
            check(f"Metal {scheme or 'bf16'}: prefill, batched paged decode (two sequences) and single-row decode to "
                  f"position 40 == the fp32 CPU model on the same rounded weights, worst {worst:.1e}", worst < 1e-4)
else:
    print("SKIP  the Metal backends: no Metal GPU")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
