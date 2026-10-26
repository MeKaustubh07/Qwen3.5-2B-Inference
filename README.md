# Inference engine for Qwen3.5, from scratch, on an 8 GB Mac

An LLM inference engine written from first principles in Python, with PyTorch used for tensors, devices and a
few primitive ops (prefill runs PyTorch's GEMM), plus hand-written Metal kernels for decode. It serves **Qwen3.5-2B** (hybrid Gated DeltaNet + gated attention) in INT4 through
an OpenAI-compatible API with continuous batching, on an Apple M2 with 8 GB of memory.

Everything a model needs is implemented here: safetensors parsing, byte-level BPE, RMSNorm, RoPE, GQA attention,
the chunked and recurrent gated delta rule, paged KV cache, sampling, chat templates, quantization, kernels,
scheduler and server. The model path is checked against Hugging Face `transformers`; the serving path is checked
against the engine's own one-request-at-a-time decoding (batched, preempted and streamed outputs must match it).

- Design and trade-offs: [`docs/design.md`](docs/design.md)
- Benchmarks: [`docs/bench/`](docs/bench/); every measurement in order: [`docs/results-log.md`](docs/results-log.md)

## Results (MacBook Air M2, 8-core GPU, 8 GB)

| Qwen3.5-2B | this engine INT4 (1.41 GB) | llama.cpp Q4_1 | MLX-LM 4-bit |
|---|---:|---:|---:|
| accuracy vs HF bf16 | KL 0.045, 3/190 top-1 flips | — | — |
| single-stream decode (engine step) | 50 tok/s | 42–53 tok/s | 70 tok/s |
| batch-8 aggregate decode (engine step) | **116 tok/s** | 62 tok/s | 73 tok/s |
| prefill, 256 / 1024 tokens | 398 / 450 tok/s | 601 / 452 tok/s | 499 / 448 tok/s |
| through the HTTP server, 8 clients | 94 tok/s (TTFT p50 0.84 s) | — | — |

bf16 on our engine: KL 0.0004 with 0/190 flips; INT8: KL 0.0007 with greedy output identical to HF on all 70 test
tokens. The server went from 77 to 94 tok/s at 8 clients (measured side by side) after profiling showed where the
gap to the engine step was: mostly prompts prefilled one at a time while every running request waited, plus ~9 ms
per step of per-request sampling. Prompts are
now packed into shared, chunked prefill passes, and sampling is batched (on the CPU, where top-k over the 248k
vocabulary is 4× faster than on MPS). What remains is mostly the prefill pass itself (~15% of the time in this
load test). Details and methodology: [`docs/bench/compare.md`](docs/bench/compare.md),
[`docs/bench/qwen35.md`](docs/bench/qwen35.md), [`docs/bench/serving.md`](docs/bench/serving.md).

## What is in it

| layer | highlights |
|---|---|
| models | Qwen3.5-2B (served, INT4) and Qwen3.5-0.8B (same-architecture fp32 reference), 18 Gated DeltaNet + 6 gated-attention layers; [Tiny Aya Global](#tiny-aya) (Cohere2, 3.35B); [Sarvam-Translate](#sarvam-translate) (Gemma 3 4B text model); Qwen2.5-0.5B was the bring-up model (removed 2026-09-30) |
| correctness | vs HF `transformers`: bring-up (Qwen2.5-0.5B, removed): layer-0 block ≤ 3.3e-5 and logits ≤ 3e-4 (fp32), greedy == HF on 5 prompts; Qwen3.5-0.8B every one of 24 layers ≤ 3.8e-6 relative (fp32), greedy identical on 7 prompts incl. Hindi; Qwen3.5-2B vs HF bf16 (fp32 does not fit in 8 GB): KL 0.0004 / 0.0007 / 0.045 for bf16 / INT8 / INT4, greedy 64 / 70 / 47 of 70 tokens, every divergence at a near-tie |
| state | paged KV cache (block allocator, block tables) + fixed-size DeltaNet recurrent/conv state per sequence |
| kernels (MSL) | bf16/INT8/INT4 matvec (+ batched), paged attention, batched DeltaNet step, RMSNorm, RoPE, fused SwiGLU, INT4/INT8 → fp32 expansion |
| quantization | block-32 INT8, asymmetric INT4 with a calibrated per-tensor mixed-precision policy; `.qt` mmap format |
| serving | continuous batching, packed + chunked prefill, batched sampling, recompute preemption, 429 backpressure, SSE streaming, cancellation, graceful drain; an order-invariant decision endpoint (choice / boolean / score) |
| operations | `/health`, `/ready`, Prometheus `/metrics` (TTFT/TPOT/e2e histograms), JSON request logs, weights locked in RAM (so macOS cannot page out an idle server's weights), Dockerfile |

## Quick start

```bash
python3.14 -m venv scripts/.venv && scripts/.venv/bin/pip install -r requirements-dev.txt
scripts/.venv/bin/hf download Qwen/Qwen3.5-2B --local-dir models/qwen3.5-2b
```

Quantize once (reads one tensor at a time, keeps the quantized output in memory and writes the file at the end;
measured peak 3.5 GB for the 2B; uses the calibrated policy in `configs/quant/`):

```bash
scripts/.venv/bin/python scripts/quantize.py models/qwen3.5-2b/model.safetensors-00001-of-00001.safetensors models/qwen3.5-2b/model.int4.qt --scheme int4 --policy configs/quant/qwen3.5-2b.json --prefix model.language_model.
```

Chat in the terminal:

```bash
scripts/.venv/bin/python scripts/repl.py --model qwen3.5-2b --backend metal-int4 --weights models/qwen3.5-2b/model.int4.qt
```

Serve the OpenAI-compatible API:

```bash
scripts/.venv/bin/python scripts/serve.py --model qwen3.5-2b --backend metal-int4 --weights models/qwen3.5-2b/model.int4.qt
```

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Explain KV caching in two sentences."}],"stream":true}'
```

Any OpenAI client works with `base_url="http://127.0.0.1:8000/v1"`. Sampling fields left out or `null` take the
model's own values from its registry entry (Tiny Aya: temperature 0.1, top-p 0.95, top-k 50; Qwen3.5: 0.7, 0.8, 20);
`top_k` and `repetition_penalty` are extensions to the OpenAI API. A request without `max_tokens` gets the endpoint's
default, cut to the room its prompt leaves.

Decisions: each option is scored as a continuation of the context, from a fork of the context's state or, when only one
option fits a group (long contexts), on the context's own state, which is then rewound exactly; so the answer cannot
depend on the order the options are listed in (`docs/design.md`, `docs/bench/decision.md`):

```bash
curl http://127.0.0.1:8000/v1/decide -H 'content-type: application/json' -d '{"type":"choice","question":"What is the capital of France?","options":["Lyon","Paris","Nice"]}'
```

`type` is `choice` (pick one), `boolean` (yes / no; no options needed) or `score` (ordinal labels, lowest first).

## Tests

The suites need more than the Quick start: the Qwen3.5-0.8B checkpoint, the 2B INT8 file, the native runtime and
the HF golden references (gitignored; generating them needs `transformers`). Once:

```bash
scripts/.venv/bin/hf download Qwen/Qwen3.5-0.8B --local-dir models/qwen3.5-0.8b
```

```bash
scripts/.venv/bin/python scripts/quantize.py models/qwen3.5-2b/model.safetensors-00001-of-00001.safetensors models/qwen3.5-2b/model.int8.qt --scheme int8 --prefix model.language_model.
```

```bash
scripts/build_native.sh && scripts/.venv/bin/python scripts/golden_qwen35.py && scripts/.venv/bin/python scripts/golden_qwen35.py models/qwen3.5-2b tests/golden_qwen35_2b bf16
```

The Tiny Aya checks need its gated files (accept the terms on the model page, then download a pinned revision; see
[`docs/tiny-aya-plan.md`](docs/tiny-aya-plan.md)) and three local answer keys (the tokenizer's, the model's, and the
long one past the window). Without the files they print SKIP; with
the files but without a key they fail and name the command that makes it:

```bash
scripts/.venv/bin/hf download CohereLabs/tiny-aya-global --revision af89d219b53ed9b13b8a4645f9c8028973510324 --local-dir models/tiny-aya-global
```

```bash
scripts/.venv/bin/python scripts/golden_tokens.py models/tiny-aya-global tests/golden_tokens_aya.json
```

```bash
scripts/.venv/bin/python scripts/golden_aya.py
```

```bash
scripts/.venv/bin/python scripts/golden_aya.py --long
```

(`golden_tokens.py` makes the tokenizer's answer key; `golden_aya.py` the real model's: transformers' own layers one
at a time in fp32, ~3.5 min and ~2.7 GB; `--long`, a 4,802-token key past the 4096-token sliding window, ~6 min and
2.4 GB, from public-domain texts in 8 languages that `scripts/long_texts.py` downloads, pinned by SHA-256, into the
gitignored `models/long_texts/`). The quantized files (~25 s each; INT4 with the calibrated policy):

```bash
scripts/.venv/bin/python scripts/quantize.py models/tiny-aya-global/model.safetensors.index.json models/tiny-aya-global/model.int8.qt --scheme int8
```

```bash
scripts/.venv/bin/python scripts/quantize.py models/tiny-aya-global/model.safetensors.index.json models/tiny-aya-global/model.int4.qt --scheme int4 --policy configs/quant/tiny-aya-global.json
```

Then:

```bash
scripts/.venv/bin/python scripts/run_tests.py
```

`--quick` skips the cache, quantization, Qwen3.5, Tiny Aya model / long / quant suites and the serving, prefill and
decision suites on their real models: Qwen3.5-0.8B, and Tiny Aya INT4 and INT8 on Metal (`--target aya-int4`,
`aya-int8`: batched replies are compared with greedy decoding alone under a near-tie rule, and GPU memory and
Metal's aborted work are gated). It runs tokenizer, aya, cohere2, window, sampling, paged, kernels and native, and
the serving, prefill, decision and prefix suites on a tiny target: a random Cohere2 with a sliding window of 8, a tokenizer
trained at test time and a chat template of its own (`--target tiny`, see `tests/serving_targets.py`). The 20 suites
cover the tokenizer (both models vs HF, the BPE merge vs HF on synthetic tables),
Tiny Aya's files (config, sharded weights, chat template, registry, server routing), the Cohere2 model vs HF on small
random models (every layer, greedy, caches, batching, bf16 KV, the length cap, a decode step reading its token id when called on Metal), the sliding window (exhaustively vs a
float64 oracle, the model vs HF past the window on every path, the sliding layers' per-sequence ring of blocks with every slot reused and poisoned, the scheduler on it with preemption) and the real 3.35B Tiny Aya in fp32
streamed one layer at a time (every layer, logits, decode steps and greedy vs the answer key; 5 languages, code and
chat) and past its 4096-token window on a 4,802-token text in 8 languages (every position, logits, decode, greedy,
and a window-off sensitivity check), Tiny Aya in INT8 and INT4 on Metal vs both keys (KL, top-1 flips, fp32 and bf16
KV, the bf16 KV cache out to 7.6K tokens, GPU memory headroom and Metal's own log of aborted work), sampling (incl. batched == per-request), the hybrid cache
(cached == uncached, KV and DeltaNet byte accounting, MPS fp32/bf16 and Metal), paged state (isolation, running out
of blocks or state slots), kernels, native runtime, quantization, Qwen3.5-0.8B vs HF fp32 (every layer),
Qwen3.5-2B (bf16/INT8/INT4), the server (batched == sequential, preemption, streaming, 429, cancellation, drain)
packed/chunked prefill (packed == alone, chunked == sequential, batching window) and decisions (forked == alone,
shuffled options give bit-identical scores, stepwise jobs between decode steps) and the pinned chat preamble
(a borrowed prefix == a chunk boundary at it, bit for bit; served == uncached; accounting, preemption, never-fit). `--save docs/bench/raw/tests`
archives each suite's output and a summary with timings: the last full run (733 checks in 32 suite runs, every suite on every target, 2,675 s) is in
[`docs/bench/raw/tests/`](docs/bench/raw/tests/); the last run with Qwen2.5 is in
[`docs/bench/raw/tests/2026-09-30-with-qwen2.5/`](docs/bench/raw/tests/2026-09-30-with-qwen2.5/).

## Benchmarks

`scripts/bench.py` measures TTFT, TPOT, prefill/decode throughput and batched decode. `scripts/loadgen.py` drives
a running server at several concurrency levels. Results and methodology are in [`docs/bench/`](docs/bench/).

## Deployment

| target | how |
|---|---|
| Mac, native (GPU) | `scripts/serve.py` as a launchd service (definition in `docs/design.md`) behind a reverse proxy or tunnel; Metal INT4 |
| cloud Apple silicon | AWS EC2 Mac or Scaleway Apple silicon, same setup |
| Linux / any Docker host | `docker build -t inference-engine .` then `docker run --stop-timeout 30 -p 8000:8000 -v "$PWD/models:/app/models:ro" inference-engine` (CPU backend, Qwen3.5-0.8B in fp32 by default; for Linux hosts and CI with RAM to spare, not for a Mac) |

Docker on macOS cannot reach the Apple GPU (containers run in a Linux VM), so on a Mac the deployment is the native
process. The CPU image (1.37 GB) is for Linux hosts and CI. On this 8 GB Mac under Docker Desktop, even with a 5 GB VM,
the Qwen3.5-0.8B default was ready in 51 s and passed the healthcheck but decoded at ~0.1 tok/s, because the fp32
weights do not stay resident in the VM (`docs/bench/raw/docker_2026-10-01.md`; natively, the same model on the CPU
does 11.7 tok/s). With the previous default, Qwen2.5-0.5B (since removed), the image was ready in ~12 s, used 2.4 GB,
decoded ~4 tok/s, and `docker stop` drained an in-flight stream before exiting. See
[`docs/design.md`](docs/design.md#9-deployment).

## Layout

```
src/weight_loader.py   safetensors mmap, sharded index  src/state.py        KV caches, block allocator, HybridState
src/tokenizer.py       byte-level / SentencePiece BPE   src/quant.py        INT8/INT4, policy, .qt files
src/chat.py            ChatML, model Jinja templates    src/sampler.py      temperature/top-k/top-p/repetition
src/config.py          Qwen35Config, Cohere2Config, Gemma3Config
src/models/            Qwen3.5, AttentionDecoder (Cohere2, Gemma 3), packed prefill
src/engine.py          model registry, load_engine
src/backend/           protocol, torch reference, Metal src/kernels/        *.metal kernels
src/native/            Objective-C++ Metal runtime      src/server/         scheduler, API, metrics
scripts/               goldens, bench, quantize, calibrate, serve, loadgen, repl, run_tests
```

## Tiny Aya

The engine also runs [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (Cohere2, 3.35B
parameters, 70 languages): plan in [`docs/tiny-aya-plan.md`](docs/tiny-aya-plan.md), results in
[`docs/results-log.md`](docs/results-log.md). On an 8 GB Mac it serves up to 8,192 tokens in INT4 (6,144 for
`/v1/decide`) and 4,096 in INT8, so as to stay within the GPU memory Metal recommends. With `serve.py --prefix-cache`
its chat template's 352-token preamble is prefilled once and kept in the KV pool, so a one-line chat prefills only the
~20 tokens after it: on a real server the first token comes 4x sooner (INT4 1.88 s -> 0.45 s; a burst of 8 chats p95
15.3 s -> 2.2 s).

| Tiny Aya Global | this engine INT4 (2.34 GB) | llama.cpp Q4_K_M (2.14 GB) | llama.cpp Q4_0 (2.03 GB) | MLX-LM 4-bit (1.88 GB) |
|---|---:|---:|---:|---:|
| KL from fp32, mean of 8 languages (lower is better) | 0.080 | **0.051** | 0.102 | 0.154 |
| single-stream decode | 30 tok/s | 35–38 tok/s | 40–42 tok/s | **45 tok/s** |
| batch-8 aggregate decode | 63–65 tok/s (INT8: 67–69) | 63 tok/s | **70–74 tok/s** | 49–50 tok/s |
| prefill, 256 tokens | 254–264 tok/s | 343–347 tok/s | **388–390 tok/s** | 261 tok/s |

On Tiny Aya this engine is behind llama.cpp: Q4_K_M does the least 4-bit damage in all 8 languages from a smaller file,
llama.cpp decodes faster at one stream, and Q4_0 is faster at every batch size (Q4_K_M ties our INT4 at batch 8). On the
same long text Qwen3.5-2B's INT4 is at KL 0.067 (the 0.045 in the Qwen table above is on short prompts, against HF
bf16). Same Mac, same text, one engine in memory at a time; details in
[`docs/bench/aya-4bit.md`](docs/bench/aya-4bit.md) (quality per language, tokens per character) and
[`docs/bench/compare.md`](docs/bench/compare.md) (speed).

Tiny Aya is by Cohere and
Cohere Labs. Its weights are not part of this repository: they are gated on Hugging Face, licensed
[CC-BY-NC 4.0 with an Acceptable Use Addendum](https://cohere.com/cohere-labs-cc-by-nc-license), subject to the
[Cohere Labs Acceptable Use Policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy), and provided as
is, without warranty. This is an independent, non-commercial project, not affiliated with or endorsed by Cohere; sample
outputs quoted in `docs/` are model-generated.

```bibtex
@misc{salamanca2026tinyayabridgingscale,
      title={Tiny Aya: Bridging Scale and Multilingual Depth},
      author={Alejandro R. Salamanca and Diana Abagyan and Daniel D'souza and Ammar Khairi and David Mora and Saurabh Dash and Viraat Aryabumi and Sara Rajaee and Mehrnaz Mofakhami and Ananya Sahu and Thomas Euyang and Brittawnya Prince and Madeline Smith and Hangyu Lin and Acyr Locatelli and Sara Hooker and Tom Kocmi and Aidan Gomez and Ivan Zhang and Phil Blunsom and Nick Frosst and Joelle Pineau and Beyza Ermis and Ahmet Üstün and Julia Kreutzer and Marzieh Fadaee},
      year={2026},
      eprint={2603.11510},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.11510},
}
```

## Sarvam-Translate

The engine also runs [Sarvam-Translate](https://huggingface.co/sarvamai/sarvam-translate) (Gemma 3 4B text model,
fine-tuned for translation between English and the 22 scheduled Indian languages): plan in
[`docs/sarvam-translate-plan.md`](docs/sarvam-translate-plan.md), results in [`docs/results-log.md`](docs/results-log.md),
the 22-language study in [`docs/bench/sarvam-translate.md`](docs/bench/sarvam-translate.md). Its text model in fp32,
streamed one layer at a time, matches transformers' own layers within 1.3e-5 on the logits, with greedy output
identical on 10 of 10 prompts and past its 1024-token sliding window on a 1,416-token text. On an 8 GB Mac it serves up
to 8,192 tokens in INT4 (4,096 for `/v1/decide`) and 4,096 in INT8 (1,024), budgets measured in one long-running
process (`scripts/bench.py --serve-mix`).

| IN22-Gen, 50 sentences per direction, all INT4 | Sarvam-Translate (2.99 GB) | Tiny Aya Global (2.34 GB) | Qwen3.5-2B (1.41 GB) |
|---|---:|---:|---:|
| chrF++, English -> 22 languages (macro) | **50.4** | 26.7 | 15.4 |
| chrF++, 22 languages -> English (macro) | **52.2** | 40.5 | 33.8 |
| chrF++, Tiny Aya's 9 languages -> English | 54.9 | **56.7** | 45.4 |
| KL from fp32 (the M3 answer key) | 0.041 (INT8: 0.0005) | — | — |

Sarvam-Translate leads into every language, and only it writes Manipuri (Meitei Mayek), Santali (Ol Chiki), Bodo and
Dogri (39.7-59.5 vs 0.4-8.3); into English Tiny Aya beats it on Hindi, Marathi and Urdu (+3.1 to +4.7, CIs above or
touching zero). Its tokenizer spells Meitei Mayek mostly in bytes (1.83 tokens per character vs Hindi's 0.28), so a
Manipuri translation takes ~6.5x the decode steps. Details, CIs and threats to validity in
[`docs/bench/sarvam-translate.md`](docs/bench/sarvam-translate.md). Setup (ungated download, pinned revision; the
answer keys are gitignored):

```bash
scripts/.venv/bin/hf download sarvamai/sarvam-translate --revision e0a7a6bc21665b568c0c624986d422f6d7f58326 --local-dir models/sarvam-translate
```

```bash
scripts/.venv/bin/python scripts/quantize.py models/sarvam-translate/model.safetensors.index.json models/sarvam-translate/model.int4.qt --scheme int4 --prefix language_model.model. --policy configs/quant/sarvam-translate.json
```

```bash
scripts/.venv/bin/python scripts/golden_gemma3.py && scripts/.venv/bin/python scripts/golden_gemma3.py --long
```

Sarvam-Translate is by Sarvam AI, in partnership with AI4Bharat, and is built on Google's Gemma 3. Its weights are not
part of this repository: they are on Hugging Face under the licence tag `gpl-3.0`, and Gemma 3 has its own terms of
use. This is an independent project, not affiliated with or endorsed by Sarvam AI; the hosted Sarvam API was never
used. Sample outputs quoted in `docs/` are model-generated. IN22-Gen is by AI4Bharat (CC-BY-4.0); only scores and
sentence ids are published here, never its text.

## License

[Apache-2.0](LICENSE). Model weights are not part of this repository; each model keeps its own licence.
