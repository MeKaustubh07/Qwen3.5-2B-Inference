# This engine vs llama.cpp vs MLX-LM on an M2 Air: Qwen3.5-2B and Tiny Aya

Two models: Qwen3.5-2B (below, 2026-09-28) and [Tiny Aya Global](#tiny-aya-global-335b-2026-10-11) (2026-10-11).
Quality per language for Tiny Aya's 4-bit builds is in [`aya-4bit.md`](aya-4bit.md).

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

## Tiny Aya Global (3.35B), 2026-10-11

Same Mac, one engine in memory at a time. Two full passes in opposite order (ours INT4, Q4_K_M, MLX, ours INT8, Q4_0, then
back), 90 s of idle between engines; the table gives both passes (A / B), measured 00:35-01:09 on 2026-10-11. A first pass under memory pressure and heat (it
ran next to other work; llama.cpp read 18 tok/s single-stream) was discarded. Raw output and the run script:
[`raw/aya_speed/`](raw/aya_speed/).

| engine | version | weights (decimal GB) |
|---|---|---:|
| this engine, INT4 / INT8 | engine code of `ecc1c71` (`src/` unchanged through `d4de954`) | 2.34 / 3.56 (`.qt`; tied embedding/head INT8 in both) |
| llama.cpp | build 11146 (7fe450e19) | Q4_0 2.03, Q4_K_M 2.14 (Cohere's own GGUFs, `CohereLabs/tiny-aya-global-GGUF`) |
| MLX-LM | 0.31.3 (mlx 0.32.2) | 1.88 (`mlx_lm convert -q` from the official weights: 4 bits, group 64, the tied head too; no MLX 4-bit Global is published) |

| metric (tok/s) | ours INT4 | ours INT8 | llama.cpp Q4_0 | llama.cpp Q4_K_M | MLX-LM 4-bit |
|---|---:|---:|---:|---:|---:|
| decode, 1 stream | 30.0 / 29.5 | 20.0 / 22.9 | 41.8 / 39.7 (bench) · 39.9 / 41.6 (batched) | 34.6 / 36.8 · 36.4 / 37.7 | **44.6 / 44.8** |
| prefill, 100-token prompt | 172 / 179 | 177 / 169 | **303 / 301** | 265 / 269 | 191 / 188 |
| time to first token, 100-token prompt (ms) | 580 / 558 | 566 / 590 | **330 / 332**\* | 377 / 372\* | 523 / 533\* |
| prefill, 256 | 254 / 264 | 231 / 256 | **390 / 388** | 343 / 347 | 261 / 261 |
| prefill, 1024 | 266 / 279 | 185 / 267 | **379 / 378** | 333 / 339 | 272 / 266 |
| batch 2, aggregate decode | 29.8 / 30.3 | 26.8 / 30.0 | **49.6 / 51.2** | 41.3 / 41.8 | 44.5 / 45.1 |
| batch 4 | 47.7 / 45.1 | 43.8 / 50.3 | **74.1 / 73.9** | 58.9 / 59.4 | 46.0 / 45.2 |
| batch 8 | 62.8 / 64.7 | 68.5 / 67.1 | **73.9 / 70.2** | 62.7 / 63.2 | 48.8 / 49.6 |
| 100 in / 100 out, decode | 28.7 / 28.3 | 22.0 / 21.0 | 39.2 / 38.1 | 36.4 / 37.8 | **44.5 / 39.4** |

How each number was produced: as for Qwen3.5 above (ours: `scripts/bench.py --model tiny-aya-global --backend
metal-int4|int8 --weights models/tiny-aya-global/model.int4|int8.qt --prompt-lens 16,100,256,1024 --new 32 --batch
1,2,4,8`, plus `--prompt-lens 100 --new 100`; decode, 1 stream = `decode_batch` at B = 1, 100 in / 100 out = its decode
row); llama.cpp: `llama-bench -p 100,256,1024 -n 32,100 -ngl 99 -r 5` and `llama-batched-bench -c 2048 -npp 16 -ntg 32
-npl 1,2,4,8 -ngl 99`; MLX-LM: `mlx_lm.benchmark -p 16 -g 32 -b B -n 5`, `-p 256|1024 -g 32 -b 1`, `-p 100 -g 100 -b 1`.
llama.cpp's 100 in / 100 out row is `llama-bench` tg100, i.e. decode from an empty context (ours and MLX-LM decode
after the 100-token prompt). As in the Qwen3.5 section, MLX-LM's B = 1 `generation_tps` reads ~3% high against a
per-step median like ours.

\* The time to first token is 100 / the 100-token prefill rate for every engine: measured by `bench.py` for ours (its
prefill rate is computed from it), derived from llama-bench's pp100 and MLX-LM's prompt_tps for the other two.

Swap grew from 1.7 to 2.7 GiB over the run: INT8 (a ~5 GB footprint on an 8 GB machine) is the most exposed, e.g. its
1024-token prefill at 185 vs 267 tok/s between passes.

**Single stream: MLX-LM leads (45 tok/s, ~43 corrected for its timing), then llama.cpp Q4_0 (40–42) and Q4_K_M
(35–38); ours is last (30).** The effective bandwidth (weights x tok/s) is ~82–84 GB/s for MLX-LM, ~83 for Q4_0, ~78
for Q4_K_M and ~70 for ours: we read the most bytes (2.34 GB; the 262K-row tied head at INT8 with its scales is 0.57 GB
of it) and read them less efficiently (the per-step dispatch overhead discussed above). At our effective rate MLX-LM's
1.88 GB would give ~37 tok/s, so bytes explain about half of the gap. Against MLX-LM's default the extra bytes buy
quality (its KL is 1.9x ours), but llama.cpp's Q4_K_M gets 0.64x our KL from 0.20 GB fewer (`aya-4bit.md`): our
bytes are not spent well.

**Batched decode: llama.cpp Q4_0 leads at every batch size.** It gains 1.7–1.85x by batch 4 (40–42 → 74) and then
plateaus (70–74 at batch 8); Q4_K_M gains 1.7x (36–38 → 63). Ours INT4 gains nothing at batch 2 (0.99–1.03x: the step
goes from 33 to 66–67 ms) and then reaches 63–65 at batch 8 (2.1–2.2x); INT8 goes 20–23 → 67–69 (2.9–3.4x) and is
again ahead of INT4 at batch 8; MLX-LM stays nearly flat (45 → 49–50). At batch 8 Q4_0 leads, ours INT8 is close
behind (67–69 vs 70–74) and ours INT4 ties Q4_K_M (63–65 vs 63). On Qwen3.5 llama.cpp gained 1.2x with Q4_1 and 1.7x
with Q8_0, so only the Q4_1 contrast is clear; and MLX-LM is flat on both models, which the DeltaNet guess in the
Qwen3.5 section does not explain. Neither was investigated.

**Prefill: llama.cpp leads at every length here** (Q4_0 388–390 vs ours 254–264 at 256 tokens, 378–379 vs 266–279 at
1024), unlike Qwen3.5 where the three engines were level at 1024. Ours and MLX-LM are level at 256 and 1024 tokens
(254–279); at 100 tokens MLX-LM is ~7% ahead (188–191 vs 172–179). Tiny Aya's FFN is wider than Qwen3.5-2B's (11008
vs 6144; the head is not involved, prefill computes only the last position's logits); our path expands each quantized
matrix to fp32 and runs MPS's fp32 GEMM, which item 1 of "What would close the remaining gaps" above would replace.
The split of our prefill time was not profiled on Tiny Aya.

For scale only (a different device and runtime): Cohere's report gives Q4_K_M at 446 tok/s prefill and 32.4 tok/s
decode on an iPhone 17 Pro for 100 tokens in and 100 out (Tiny Aya report §6, Fig. 13); llama.cpp's Q4_K_M on this
fanless M2 does 265–269 tok/s prefill on 100 tokens and 36–38 tok/s decode (tg100, from an empty context).
