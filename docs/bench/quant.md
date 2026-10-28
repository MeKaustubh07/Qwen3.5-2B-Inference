# Weight quantization (INT8 / INT4) on Qwen2.5-0.5B

Measured on Qwen2.5-0.5B, the bring-up model, removed 2026-09-30 (last present at `4f9581b`); kept as history.

### metal-kernels-bf16 — Qwen2.5-0.5B, 32 new tokens, M2 8 GB

Weights resident: 0.99 GB → bandwidth ceiling ≈ 101 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 762 MiB, Metal driver 2065 MiB. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 35 | 15.4 | 456 | 64.9 | 64% |
| 256 | yes | 209 | 16.4 | 1223 | 61.1 | 60% |
| 1024 | yes | 1130 | 16.6 | 906 | 60.3 | 60% |

### metal-kernels-int8 — Qwen2.5-0.5B, 32 new tokens, M2 8 GB

Weights resident: 0.52 GB → bandwidth ceiling ≈ 190 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 2405 MiB, Metal driver 1105 MiB. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 164 | 11.8 | 98 | 85.0 | 45% |
| 256 | yes | 252 | 10.8 | 1015 | 92.9 | 49% |
| 1024 | yes | 907 | 11.3 | 1129 | 88.8 | 47% |

### metal-kernels-int4 — Qwen2.5-0.5B, 32 new tokens, M2 8 GB

Weights resident: 0.41 GB → bandwidth ceiling ≈ 245 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 2405 MiB, Metal driver 1105 MiB. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 333 | 10.1 | 48 | 99.3 | 40% |
| 256 | yes | 445 | 10.9 | 576 | 92.0 | 37% |
| 1024 | yes | 1083 | 10.0 | 946 | 99.8 | 41% |

## Accuracy (tests/test_quant.py)

Reference: HF fp32 goldens. Perplexity on a fixed 70-token paragraph; KL and top-1 agreement over 5 golden prompts.

| weights | resident | perplexity | KL vs fp32 | top-1 agreement | greedy after "The capital of France is" |
|---|---:|---:|---:|---:|---|
| bf16 | 0.99 GB | 11.34 | 0.0009 | 98% | Paris. It is the largest city in Europe and |
| INT8 (symmetric, block 32) | 0.52 GB | 11.30 | 0.019 | 98% | *identical to bf16* |
| INT4 symmetric (first attempt) | 0.37 GB | 14.54 | 2.26 | 62% | Paris. It is the largest city in Europe, |
| INT4 asymmetric + INT8 embedding | 0.37 GB | 11.50 | 1.85 | 64% | Paris. Paris is the capital of which country? |
| **INT4 asymmetric + calibrated policy** | **0.41 GB** | **11.20** | **0.114** | **94%** | Paris. Paris is the capital of France. Paris |

## How INT4 got there

1. **Symmetric INT4 costs +28% perplexity.** Scale = absmax / 7 spends half the 16 levels on the side of zero a block barely uses. Switching to **asymmetric** (scale = (max − min) / 15, plus a per-block fp16 minimum) brought perplexity to within 1.4% of bf16, and keeping the tied embedding / output head at INT8 (it scores all 151,936 tokens) helped further. Inside the kernel the minimum folds in as `scale·dot(q, x) + min·sum(x)`.
2. **Perplexity hid a first-token problem.** Per-position KL showed 6.8–8.2 nats at position 0 and 0.01–0.9 elsewhere. Small LLMs route huge activations through the first token (an "attention sink"), which makes a handful of weights extremely sensitive.
3. **Measured sensitivity instead of a guess.** `scripts/calibrate_quant.py` quantizes one tensor at a time to INT4 (the rest bf16) and records the KL damage. Four tensors (layer 21's MLP, `down_proj` of layers 2–3) each added 3–5 nats; the median tensor added 0.0004. This matches the massive-activation pattern described for small transformers.
4. **Threshold chosen at the knee of a sweep:**

| keep INT8 if damage > | tensors INT8 | resident | KL vs fp32 | top-1 agreement |
|---:|---:|---:|---:|---:|
| 0.1 | 4 | 0.378 GB | 0.169 | 86% |
| 0.01 | 9 | 0.395 GB | 0.124 | 87% |
| **0.005 (default)** | **19** | **0.408 GB** | **0.114** | **94%** |
| 0.002 | 37 | 0.437 GB | 0.084 | 94% |
| 0.001 | 62 | 0.497 GB | 0.035 | 98% |

The policy was `configs/quant/qwen2.5-0.5b.json` (names are fused tensors: `qkv`, `gate_up`; every component of a fused tensor gets the same scheme); the file, with its calibration data, is now `docs/bench/raw/quant_policy_qwen2.5-0.5b.json`.

## Speed

- **Decode gets faster as bytes shrink:** ~61–65 tok/s (bf16) → ~85–93 (INT8) → ~92–100 (INT4). Fewer bytes per token is the whole point: decode reads every weight once per token.
- **The quantized kernels sit further below their (higher) ceilings** (40–49% vs 60–64%): unpacking nibbles and applying scales adds ALU work per byte, and the fixed per-token overhead is a larger share of a shorter step.
- **INT4 prefill is slower** (TTFT 333 ms at 16 tokens vs 35 ms for bf16) because prefill dequantizes each weight matrix on the fly before PyTorch's GEMM. A quantized GEMM kernel would remove this; it is the main open item.
- `.qt` files (`scripts/quantize.py`, streamed one tensor at a time): 0.41 GB INT4 / 0.52 GB INT8 vs 0.99 GB safetensors. Loading the INT4 file takes 1.4 s vs 3.4 s for quantizing at startup, and produces bit-identical logits.

## Qwen3.5-2B INT4 per language, on long text (2026-10-10)

How far INT8 and INT4 move Qwen3.5-2B's next-token distribution, language by language, on long text. Raw output:
[`raw/qwen_lang_kl/`](raw/qwen_lang_kl/); script: `scripts/lang_kl.py`.

- **Text:** two 512-token windows per language (en fr de es hi ar zh ja), cut at 25% and 60% of one public-domain work
  each (`scripts/long_texts.py`: Austen, Hugo, Kafka, Cervantes, Bharatendu Harishchandra, the Nights, Journey to the
  West, Akutagawa), past their famous openings. One work per language, not parallel text.
- **Scored:** positions 256-510 of each window (510 per language, each with 256+ tokens of context).
- **Reference:** Qwen3.5-2B HF in bf16 (fp32 does not fit in 8 GB).
- **Metric:** KL(reference || build) in nats per token, with llama.cpp's formula (tokens whose reference log
  probability is above -16).

| build | en | fr | de | es | hi | ar | zh | ja | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 | 0.0009 | 0.0007 | 0.0008 | 0.0009 | 0.0007 | 0.0010 | 0.0010 | 0.0010 | 0.0009 |
| INT4, nothing kept INT8 but the embedding | 0.0679 | 0.0716 | 0.0708 | 0.0690 | 0.0698 | 0.1198 | 0.1004 | 0.1076 | 0.0846 |
| INT4 as shipped (1.41 GB) | 0.0547 | 0.0565 | 0.0628 | 0.0577 | 0.0509 | 0.0906 | 0.0786 | 0.0832 | 0.0669 |

- The 0.045 in the Qwen3.5-2B accuracy table is on short prompts; on this long text the shipped INT4 is at 0.067.
- Calibration puts 83% of the position-0 damage in 3 tensors (layer 14 `down_proj`, layer 15 `o_proj`, layer 6
  DeltaNet `out_proj`); keeping the policy's 5 tensors INT8 cuts the long-text KL by 21%.
- INT8's floor (0.0009) is the bf16 reference's own rounding.

Tokens per 1,000 characters (`lang_kl.py tpc`: the first 20,000 characters of each work, or all of it where shorter)
and bits per character, log2(reference perplexity) x tokens per character on the scored positions (`lang_kl.py bpc`),
which compares across tokenizers where per-token perplexity does not:

| language | characters | tokens / 1,000 chars | reference perplexity | tokens/char (scored) | bits/char |
|---|---:|---:|---:|---:|---:|
| en | 20,000 | 254.8 | 14.79 | 0.238 | 0.927 |
| fr | 20,000 | 287.4 | 10.03 | 0.301 | 1.000 |
| de | 20,000 | 264.7 | 16.34 | 0.265 | 1.068 |
| es | 20,000 | 281.2 | 22.01 | 0.290 | 1.294 |
| hi | 16,300 | 511.7 | 9.27 | 0.551 | 1.771 |
| ar | 20,000 | 337.4 | 30.46 | 0.358 | 1.764 |
| zh | 20,000 | 875.0 | 40.34 | 0.882 | 4.707 |
| ja | 6,128 | 676.6 | 40.24 | 0.687 | 3.664 |

These are famous public-domain works, so the numbers measure familiarity with those texts more than quality; 510
scored positions per language; no accuracy or human evaluation.

```bash
scripts/.venv/bin/python scripts/lang_kl.py ref qwen REF_QWEN           # ~8.5 min CPU (HF bf16)
scripts/.venv/bin/python scripts/lang_kl.py ours qwen3.5-2b metal-int4 models/qwen3.5-2b/model.int4.qt REF_QWEN qwen_ours_int4.json
scripts/.venv/bin/python scripts/lang_kl.py table RESULTS_DIR
scripts/.venv/bin/python scripts/lang_kl.py tpc
scripts/.venv/bin/python scripts/lang_kl.py bpc docs/bench/raw/qwen_lang_kl
```
