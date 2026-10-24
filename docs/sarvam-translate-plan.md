# Porting Sarvam-Translate (Gemma 3) to this engine: plan

Written 2026-10-23, before any code. Results and changes go to [`results-log.md`](results-log.md) as milestones land.

Every fact below was checked against a primary source; anything not yet confirmed is marked UNVERIFIED. Sources:
- `transformers` 5.16.1 `models/gemma3/`, `masking_utils.py` and `modeling_rope_utils.py`, all local;
- the official small files of `sarvamai/sarvam-translate` at revision `e0a7a6bc21665b568c0c624986d422f6d7f58326`
  (config.json, model.safetensors.index.json, tokenizer.json, tokenizer_config.json, chat_template.jinja,
  generation_config.json), fetched 2026-10-23;
- the model card, Sarvam's API docs and the HF API;
- this engine's code and docs.

## 1. Sarvam-Translate in one table

| item | value | source |
|---|---|---|
| what | Sarvam's translation model, built with AI4Bharat; the hosted version is the API model `sarvam-translate:v1`; fine-tuned from `google/gemma-3-4b-it` | model card; docs.sarvam.ai models/sarvam-translate |
| class | `Gemma3ForConditionalGeneration` (model_type `gemma3`). Text tower = `Gemma3TextModel`. The checkpoint also holds a SigLIP vision tower and projector: 439 of its 883 tensors, unused here | config.json; index |
| parameters (text) | 3,880,263,168 = embedding 671,252,480 + 34 x 94,382,592 + final norm 2,560 (computed from config; recount at M1) | config.json |
| layers / hidden / FFN | 34 / 2560 / 10240, GeGLU with `gelu_pytorch_tanh`: `down(gelu_tanh(gate(x)) * up(x))` | config; modeling_gemma3.py:120-133 |
| heads | 8 query, 4 KV (GQA), head_dim 256 (explicit). Attention scale `query_pre_attn_scalar ** -0.5` = 1/16, which equals 1/sqrt(head_dim) here | config; modeling_gemma3.py:318 |
| attention pattern | `(i + 1) % 6 != 0` -> sliding, else full: full = layers 5, 11, 17, 23, 29 (5 full, 29 sliding; layers 30-33 sliding) | configuration_gemma3.py:109-113 |
| sliding window | 1024: a query sees itself and the 1023 keys before it (`kv_idx > q_idx - sliding_window`, the same predicate as Cohere2) | config; masking_utils.py:98-99 |
| positions | RoPE, half-split (`rotate_half`), all 256 dims. Sliding layers: theta 10,000 (`rope_local_base_freq`), default. Full layers: theta 1,000,000 with linear scaling factor 8 (`inv_freq /= 8`), attention scaling 1.0 | configuration_gemma3.py:137-158; modeling_rope_utils.py:189; modeling_gemma3.py:158-213 |
| norms | RMSNorm with `(1 + w)`, computed in fp32, eps 1e-6. Per layer: input, post_attention, pre_feedforward, post_feedforward, plus `q_norm` / `k_norm` over each head before RoPE. One final norm | modeling_gemma3.py:136-150, 338-357 |
| block | "sandwich": `h = h + post_attn_norm(attn(input_norm(h)))`, then `h = h + post_ff_norm(mlp(pre_ff_norm(h)))`. The norm sits between each projection and its residual add | modeling_gemma3.py:386-430 |
| embedding | rows multiplied by `sqrt(2560)` cast to the weight dtype first: 50.596 in fp32, **50.5 in bf16** | modeling_gemma3.py:106-117, 496 |
| head | tied to the embedding (no `lm_head` tensor in the index); no final-logit and no attention softcapping (both null) | index; config |
| vocabulary | 262,208 embedding rows; the tokenizer defines 262,145 contiguous ids (0..262144), so 63 rows have no token | config; tokenizer.json |
| tokenizer | BPE over **characters** with `byte_fallback` (256 `<0xXX>` tokens), `fuse_unk`, `<unk>`, 514,906 merges. Normalizer: Replace `" "` -> `"▁"`. Pre-tokenizer: Split `" "` MergedWithPrevious, which finds nothing once the normalizer has run, so a whole text between special tokens is one BPE chunk. Post-processor: `<bos>` (2) in front. Decoder: `▁` -> space, ByteFallback, Fuse. 6,415 added tokens, none with lstrip / rstrip / normalized | tokenizer.json |
| stop ids | 1 `<eos>`, 106 `<end_of_turn>` | generation_config.json |
| chat | Gemma template in `chat_template.jinja`: `{{ bos_token }}`; a system message is not a turn but is prepended to the first user turn; roles `user` / `model`; `<start_of_turn>` ... `<end_of_turn>` | chat_template.jinja |
| prompt (card) | system "Translate the text below to {tgt_lang}.", user = the source text | model card |
| directions | the open checkpoint does English -> Indic and Indic -> English only; Indic <-> Indic pivots through English | model card |
| context | card: 8K max context (its vLLM command uses `--max-model-len 8192`). config's 131072 is Gemma 3's, so cap at 8192 | model card; config |
| sampling | card: temperature 0.01 (in effect greedy). generation_config: do_sample, top_k 64, top_p 0.95 | model card; generation_config.json |
| languages | 22 scheduled Indian languages + English | card; API docs |
| licence | HF tag `gpl-3.0`; the base model is Gemma 3. How the two combine is UNVERIFIED (section 9) | HF API |

## 2. New ideas compared with Qwen3.5 and Tiny Aya (one line each)

- **Sandwich norms.** Each sub-block's output is normalised again before it joins the residual stream, so no branch can blow up the stream's scale.
- **Embedding scale.** Input rows are multiplied by sqrt(hidden), about 50.6, so tokens start at the size the residual stream works at.
- **GeGLU.** The same gate x up shape as SwiGLU, but the gate uses GELU (tanh form) instead of SiLU.
- **Local + global attention, 5 : 1.** Most layers see only the last 1024 tokens. One layer in six sees everything, with RoPE stretched 8x so long distances still look new.
- **Two RoPE tables.** Local layers rotate at base 10,000; global layers at base 1,000,000 with every frequency divided by 8.
- **SentencePiece-style BPE with byte fallback.** Merges start from characters, not bytes. A character missing from the vocabulary is spelled as its UTF-8 bytes (`<0xEA>`, ...).
- **Multimodal checkpoint, text-only port.** The vision tensors are never loaded.

## 3. Engine gaps (component, today, change, size, risk)

Size: S = under a day; M = a few days; L = a week or more.

| component | engine today | change | size | risk |
|---|---|---|---|---|
| tokenizer | byte-level BPE only; refuses `byte_fallback`, a Replace normalizer and a non-ByteLevel pre-tokenizer | a SentencePiece mode: Replace normalizer, Split(MergedWithPrevious), character-level BPE through the existing heap merge (a whole text is one chunk, so n log n matters), byte fallback for characters with no vocab entry, decode `▁` -> space + byte fusion. Answer key against HF `tokenizers` for all 22 scripts. Meitei Mayek falls back to bytes: `ꯃꯤꯇꯩꯂꯣꯟ ꯑꯁꯤ` is 17 tokens, 9 of them bytes (measured) | M | med |
| config | `Qwen35Config`, `Cohere2Config` | `Gemma3Config` from `text_config`, refusing: softcapping, bidirectional attention, `query_pre_attn_scalar != head_dim`, an untied head, rope types other than default / linear, layer_types out of pattern | S | low |
| loader | sharded index + prefix (`quantize.py --prefix`) | prefix `language_model.model.`; vision tensors never touched | S | low |
| RMSNorm (1 + w), QK-norm | done for Qwen3.5 (`one_plus=True`, `rms_norm_f32w`, q/k norm before RoPE) | reuse | - | - |
| sandwich norms | the residual is fused into `o_proj` / `down_proj` | norm, then add (a norm sits in between). Optional fused `rms_norm + add` kernel for decode, measured | S (M with kernel) | low |
| embedding scale | none | multiply the looked-up rows; the value is a decision (section 8) | S | low |
| GeGLU | SwiGLU only (`silu_mul`, `matvec_swiglu_bf16`, quantized = linear + `silu_mul`) | `gelu_tanh_mul` kernel, fused `matvec_geglu_bf16`, backend `geglu`, Torch reference | S | low |
| RoPE | one theta per call; frequencies from (theta, d) on the host | frequencies per layer type, incl. `/ 8` for the full layers, computed as transformers computes them (the Metal kernel already reads a host table) | S | low |
| attention | windowed kernels (Tiny Aya), head_dim 256 (Qwen3.5) | reuse with window 1024 | - | low |
| KV pool | `GroupedKVPool`: groups of gcd(n_full, n_sliding) layers | gcd(5, 29) = 1, so 34 one-layer groups and 34 block tables per decode step. It works; measure the cost, and if it matters allow unequal groups (1 full group of 5, 1 sliding group of 29) | S-M | med (speed) |
| model | `qwen3_5.py`, `cohere2.py` | `src/models/gemma3.py`, the readable spec, ~250 lines like `cohere2.py` | M | low |
| chat | `TemplateChat` reads `chat_template.jinja` | reuse; check == `apply_chat_template` incl. a system message and the role-alternation error (400) | S | low |
| registry | Qwen3.5, Tiny Aya | `sarvam-translate`: stops {1, 106}, cap 8192, Metal INT8 / INT4 only, bf16 KV, sampling from the card | S | low |
| quantization | INT8 / INT4 + calibrated policy, incremental writer | reuse. Calibrate on FLORES+ dev text in the 22 languages, never on the eval set | M | med |
| correctness key | `golden_aya.py` (layer-streamed fp32 HF key) | `golden_gemma3.py` with `Gemma3DecoderLayer`, a rotary table per layer type and a mask per layer type | M | med |
| translation eval | none | `scripts/eval_translate.py`: chrF++ (sacrebleu, signature printed), paired bootstrap 95% CIs, per language and direction; `sacrebleu` pinned in requirements-dev | M | low |
| serving | chat + completions | optional `/v1/translate`: sentence chunking under a length budget, Indic <-> Indic pivot, the instruction prefix-cached | S-M | low |

## 4. Budgets

**Disk:** 47 GiB free (measured 2026-10-23).
1. Download: 8.64 GB, two shards that mix text and vision tensors, so both are needed.
2. INT8 .qt ~4.12 GB; INT4 .qt ~2.72 GB.
3. Baselines: a Q4_K_M GGUF and `mlx-community/sarvam-translate-mlx-4bit`, ~5 GB (sizes UNVERIFIED).
4. Total ~20 GB.

**Weights in RAM:**
- INT4 body: 3,209,008,128 x 0.625 B = 2.01 GB, plus an INT8 tied head (0.71 GB) = **~2.72 GB**. The head is ~26% of the bytes read per token.
- INT8: ~4.12 GB, close to the 5.33 GiB Metal recommends on this Mac. Plan INT8 as in Tiny Aya: capped context and a smaller pool, and gate peak GPU memory in the tests.

**KV cache, bf16 (K + V, 4 heads x 256):** 4 KiB per token per layer.
- All 34 layers: 136 KiB per token, so an 8K sequence would be 1,088 MiB.
- With rings: the 5 full layers take 20 KiB per token (160 MiB at 8K). The 29 sliding layers hold at most a ring of ceil((1024 + 512 + 14) / 16) = 97 blocks per sequence (176 MiB).
- So an 8K sequence holds **~336 MiB**.

**Decode speed (estimate, UNVERIFIED):** ~2.72 GB read per token. At the ~70 GB/s effective that Tiny Aya's INT4 reached, that is **~26 tok/s** single stream.

## 5. Correctness on 8 GB

HF needs 15.5 GB for the text model in fp32 and 7.8 GB in bf16, so neither fits. The plan is the same as Tiny Aya's:

1. **Tiny random Gemma 3 text models** built with HF from `Gemma3TextConfig` (hidden 64, 12 layers so the 5 : 1 pattern repeats, head_dim 16, linear scaling on). Require engine == HF in fp32 on CPU.
   - **tiny-full**: window at least the longest test.
   - **tiny-window**: window 8.
   - Embedding scale and both RoPE tables are exercised.
2. **Real-model fp32 key, one layer at a time:**
   - transformers' own `Gemma3DecoderLayer`, its tensors loaded with `safe_open`, fed the matching rotary table and mask for its layer type;
   - the tied head in row chunks;
   - prompts in at least 8 languages and scripts, incl. Meitei (byte fallback), Ol Chiki and Urdu, in both directions.
3. **INT8 / INT4 on Metal:** KL and top-1 flips against that key; greedy match.

## 6. Milestones (each with an exit test)

| # | milestone | exit test | size |
|---|---|---|---|
| M0 | this plan; download the pinned revision; licence reading; access to IN22-Gen and FLORES+ (both gated with automatic approval: accept the terms yourself) | SHA-256 of both shards == their HF LFS hashes; IN22-Gen's Manipuri script known (Meitei or Bengali: UNVERIFIED); prior-work check on the ~25 public conversions (does any report per-language quality?) | S |
| M1 | tokenizer SentencePiece mode, `Gemma3Config`, loader prefix, chat, registry entry | tokenizer == HF on the existing answer key (Qwen and Tiny Aya unchanged) plus a new 22-script key (byte fallback, whitespace runs, tabs, newlines, emoji, special tokens); chat == `apply_chat_template`; `tests/test_gemma3.py` | M |
| M2 | `Gemma3Model` (fp32, CPU) on tiny-full and tiny-window; ops `geglu` and per-type RoPE | every layer and the logits within ~1e-6 of HF; greedy identical; cached == uncached; paged == contiguous; batched and packed == alone; forks; past the window on every path | M |
| M3 | real-model layer-streamed fp32 key + `Gemma3Model(stream=True)` | every layer, logits and greedy tokens against the key, both directions, 8+ languages | M |
| M4 | Metal: GeGLU kernels, INT8 / INT4 `.qt`, a calibrated INT4 policy | KL / flips / greedy vs the M3 key; peak GPU memory <= 0.95x recommended; no Metal aborts in macOS's log; tok/s | M |
| M5 | the 1024 window with rings on the real model; the cost of 34 KV groups | long-text key past the window; decode step time with 34 vs (if built) 2 groups | S-M |
| M6 | serving: registry defaults, server suites on the new target, optional `/v1/translate` (chunking, pivot, cached instruction) | the server, prefill and decision suites pass on `--target sarvam-translate-int4/int8` | M |
| M7 | the study and write-up (section 7) | `docs/bench/sarvam-translate.md` with every number reproducible from one command per table | M |

Time in weeks comes after M1-M2, once the pace on this model is known.

## 7. Results to aim for (what makes this more than a port)

1. **Correctness:** every layer and the logits against HF at fp32 (layer-streamed); KL for INT8 / INT4 and for bf16 KV; greedy match in both directions.
2. **Quality in every language** (the headline):
   - chrF++ on IN22-Gen for all 22 languages, English -> X and X -> English;
   - configurations: INT8, INT4, INT4 with the calibrated policy, llama.cpp Q4_K_M, MLX 4-bit;
   - greedy decoding, the card's prompt, paired bootstrap 95% CIs on each delta against INT8;
   - optional: a true bf16 reference with the card's `vllm serve` on a rented GPU or Colab;
   - KL per language as the diagnostic beside it;
   - cite Marchisio et al. 2024 (arXiv 2407.03211) on why automatic metrics understate quantization damage in non-Latin scripts.
3. **A policy aimed at the worst languages.** If 4-bit hurts some languages most, show which tensors carry it and what keeping them at INT8 costs in bytes and buys back in chrF++.
4. **Tokenizer cost per language:**
   - tokens per character and characters per second;
   - Meitei Mayek is spelled in bytes, about 3 tokens per character, and Ol Chiki gets almost no merges (measured on two sample phrases; measure on IN22);
   - this sets how fast each language is served, independent of the engine.
5. **Speed on the same M2:** single stream, batch 8, prefill and TTFT against llama.cpp and MLX, with the same interleaved protocol as Tiny Aya M7.
6. **Memory:** concurrent 8K sequences in a fixed budget, with rings vs one flat table.

## 8. Decisions (to take before M1)

1. **Embedding scale.**
   - HF multiplies by sqrt(2560) cast to the weight dtype: 50.5 with bf16 weights (how the card's bf16 vLLM serving runs, UNVERIFIED for vLLM's code), 50.596 in fp32.
   - Proposal: the fp32 reference mode uses 50.596 (to match the fp32 key exactly); the quantized Metal path uses 50.5, the value the model was served and probably fine-tuned with; report the KL between the two settings.
2. **Data.** IN22-Gen (CC-BY-4.0) for evaluation; FLORES+ dev (CC-BY-SA-4.0) for calibration only. Publish scores and IDs, never the text.
3. **Direction order.** English -> X first (generating Indic text is where 4-bit should hurt most), then X -> English.
4. **Decoding for the study.** Greedy (the card's temperature 0.01 is effectively greedy, and greedy is reproducible).
5. **Repository.** This repository, on the checked-out branch; same venv: transformers 5.16.1 has Gemma 3, and tokenizers 0.23.1 has byte-fallback BPE for the answer key.
6. **Context.** Cap 8192 (the card's).

## 9. Licence and access (short)

- **Download:** ungated: `hf download sarvamai/sarvam-translate --revision e0a7a6bc21665b568c0c624986d422f6d7f58326 --local-dir models/sarvam-translate`.
- **Licence:** the HF tag is `gpl-3.0`; the base is Gemma 3, released under Google's Gemma terms. Which terms bind this checkpoint, and what they say about redistributing derived weights, is UNVERIFIED (read both at M0; not legal advice).
- **Until then:** publish code, numbers, correctness results and labelled sample outputs. Never publish `.qt` files. Never commit copies of tokenizer.json, config or the template: load them from `models/sarvam-translate/`.
- **No Sarvam API comparisons:** never use the hosted `sarvam-translate:v1` as a reference. Sarvam's ToS v2.0 section 10.5(a) bars using its services to "test" ML models. The open weights and human references avoid that entirely.
- **README:** "Sarvam-Translate by Sarvam AI, in partnership with AI4Bharat", model link, licence tag, "independent; not affiliated with or endorsed by Sarvam AI", no logos.
