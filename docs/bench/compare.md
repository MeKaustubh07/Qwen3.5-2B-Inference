# This engine vs llama.cpp vs MLX-LM on an M2 Air: Qwen3.5-2B

All numbers measured on one MacBook Air M2 (Mac14,2: 8-core GPU, 4P+4E CPU, 8 GB, ~100 GB/s), one engine in
memory at a time, VS Code and other heavy apps closed. Raw tool output: `docs/bench/raw/`. Qwen3.5-2B's GGUF
and MLX files are no longer kept locally (removed 2026-09-30); to reproduce, download them again from the repositories
named below.

| engine | version | weights | language-model weights (decimal GB) |
|---|---|---|---:|
| this engine, INT4 | commit `f9280fa` | asymmetric INT4, block 32, fp16 scale + min; tied embedding/head and 5 calibrated tensors INT8 | 1.41 |
| this engine, INT8 | same | symmetric INT8, block 32 | 2.00 |
| llama.cpp | build 11146 (`brew install llama.cpp`) | GGUF from `bartowski/Qwen_Qwen3.5-2B-GGUF`: Q4_1 (same block-32 scale + min scheme as ours; the file is imatrix-calibrated and keeps some tensors at other types), Q4_K_M, Q8_0 | 1.38 / 1.40 / 2.08 (file size; includes an unused multi-token-prediction layer) |
| MLX-LM | 0.31.3 (mlx 0.32.2) | `mlx-community/Qwen3.5-2B-MLX-4bit` (4-bit, group 64, embedding/head 4-bit too) | 1.06 (1.72 with the vision tower, which it does not load; peak process memory 1.14) |

## Results

| metric | ours INT4 | ours INT8 | llama.cpp Q4_1 | llama.cpp Q4_K_M | llama.cpp Q8_0 | MLX-LM 4-bit |
|---|---:|---:|---:|---:|---:|---:|
| decode, 1 stream (tok/s) | **50.1** | 39.2 | 42.3 (bench) / 53.3 (batched-bench) | 32.5 | 28.6 / 38.6 | **69.9** |
| prefill, 256-token prompt (tok/s) | 398 | 389 | **601** | 445 | 513 | 499 |
| prefill, 1024-token prompt (tok/s) | 450 | 443 | **452** | 375 | 438 | 448 |
| batch 2, aggregate decode (tok/s) | 53.6 | 54.3 | 55.3 | | 57.5 | 58.8 |
| batch 4, aggregate decode (tok/s) | 85.4 | 93.7 | 63.8 | | 66.0 | 62.0 |
| batch 8, aggregate decode (tok/s) | **115.8** | **126.4** | 61.9 | | 66.1 | 73.3 |

How each number was produced:
- **ours:** `scripts/bench.py --model qwen3.5-2b --backend metal-int4 --weights models/qwen3.5-2b/model.int4.qt
  --prompt-lens 16,256,1024 --new 32 --batch 1,2,4,8` (and `metal-int8`). Decode, 1 stream = `decode_batch` with
  B = 1, the step the server runs (timed alone: no sampling or HTTP; the standalone forward path measured
  42.7–50.0 tok/s across runs); prefill = prompt tokens / time to first token, median of 3 runs; batch rows =
  median `decode_batch` step over 32 steps with B sequences of 16-token prompts.
- **llama.cpp:** `llama-bench -p 256,1024 -n 32 -ngl 99 -r 3` (tg32 = decode from an empty context) and
  `llama-batched-bench -npp 16 -ntg 32 -npl 1,2,4,8 -ngl 99` (S_TG = aggregate generation speed over B sequences).
  The two tools disagree on single-stream decode (42 vs 53 tok/s); both are listed.
- **MLX-LM:** `python -m mlx_lm.benchmark -p 16 -g 32 -b B -n 3` and `-p 256/1024 -b 1`, mean of 3 trials. For
  B > 1 its `generation_tps` is generated tokens of all sequences / generation time, i.e. aggregate (checked in
  `mlx_lm/generate.py`, `BatchGenerator.stats`). For B = 1 it is (n + 1) tokens / time since the first token, which
  reads ~3% high against a per-step median like ours.

Run-to-run spread on this fanless machine: our batched path repeated within ~3% (INT4 B = 8: 115.8 and 112.7 tok/s
in two runs); the standalone single-sequence path varied by up to ~20% (45.6 vs 36.9 tok/s after a 1024-token
prompt); llama.cpp's `±` columns reach 8–11% (e.g. Q4_K_M pp256 445 ± 50, tg32 32.5 ± 2.7).

## Reading the comparison

**Single-stream decode: MLX-LM leads clearly (70 tok/s); ours (50) and llama.cpp Q4_1 (42–53) are level within
measurement spread.** In INT8 we match llama.cpp's Q8_0 (39 vs 29–39). Decode reads every weight once per token,
so bytes and overhead decide it:
- *Bytes.* MLX's 4-bit model quantizes the tied embedding / output head (248k × 2048) to 4 bits too. We keep it at
  INT8 by a fixed rule carried over from the Qwen2.5 work, where keeping it INT8 improved INT4 perplexity
  (`docs/bench/quant.md`); the Qwen3.5 calibration does not test the head. Our 1.41 GB at 50.1 tok/s is ~71 GB/s
  effective; MLX's language-model weights are 1.06 GB, and at our effective rate that would give ~67 tok/s, so
  bytes account for most (~80%) of the gap.
- *Overhead.* Our step is roughly 300 GPU dispatches (estimated from the ops per layer) issued from Python through
  PyTorch's MPS layer, each costing CPU time; MLX records a lazy graph and encodes it with fused kernels.

**Batched decode: ours scales furthest on this hybrid model.** From 1 to 8 concurrent sequences, aggregate decode
grows 2.3× for our INT4 (50 → 116 tok/s) and 3.2× for our INT8 (39 → 126); llama.cpp grows 1.2× for Q4_1 (53 → 62)
and 1.7× for Q8_0 (39 → 66, batched-bench); MLX-LM stays flat (70 → 73). INT8 is ahead of INT4 from batch 2
(54.3 vs 53.6, within noise) and clearly at 4–8; a likely reason (not profiled) is that sharing each weight read
makes the step compute-bound, where INT4's nibble unpacking costs more than INT8's extra bytes. Ours scales because
a batched step reads each weight once for all sequences (2-rows-per-SIMD-group matvec kernels), attention for the
whole batch is one paged-attention dispatch per layer, and every sequence's DeltaNet state sits in one pooled
tensor updated by two dispatches per layer (conv step + delta-rule update). Our first version did not scale
either: it looped over sequences for attention and DeltaNet, and its batch-8 step took 186–189 ms, i.e. ~43 tok/s
aggregate vs ~38 at batch 1 (through the HTTP server: 30 → 31 tok/s from 1 to 8 clients; `docs/bench/serving.md`). Why the other two scale less on Qwen3.5 was not investigated; how they handle the
recurrent DeltaNet state of a hybrid model is a plausible place to look.

**Prefill: llama.cpp leads at 256 tokens; all three are level at 1024.** Prefill is compute-bound (a
[T × K] × [K × N] GEMM per layer). llama.cpp multiplies quantized blocks directly with simdgroup-matrix kernels.
We expand each quantized matrix to fp32 with one kernel pass and use MPS's fp32 GEMM (2.5 TFLOPS on the M2;
bf16 runs at only 1.4 because the M2 has no native bf16 math). The three engines are within ~1% at 1024 tokens
(450 vs 452 vs 448; ours is the median-TTFT run of 3, llama.cpp a mean, MLX a mean including one slow trial of
407), which suggests the GEMM dominates there; the per-phase split of our prefill after the expansion kernel was
not profiled.

## What would close the remaining gaps

1. Quantized GEMM with simdgroup matrices for prefill (skip the fp32 expansion): the 256-token gap.
2. Fewer, larger dispatches per decode step (fuse norm + projection, command-buffer reuse, or a C++ step loop):
   the single-stream gap to MLX that bytes do not explain.
3. An INT4 head option, after measuring its accuracy cost on Qwen3.5 (not done yet).
