### metal-kernels-int4 — tiny-aya-global, 32 new tokens, M2 8 GB

Weights resident: 2.34 GB → bandwidth ceiling ≈ 43 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 560 MiB, Metal driver 3681 MiB, physical footprint 4035 MiB now and 4084 MiB at its peak. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 215 | 34.0 | 74 | 29.4 | 69% |
| 100 | yes | 580 | 33.5 | 172 | 29.8 | 70% |
| 256 | yes | 1010 | 38.5 | 254 | 26.0 | 61% |
| 1024 | yes | 3856 | 41.6 | 266 | 24.1 | 56% |

| batch | step (ms) | per-sequence tok/s | aggregate tok/s | aggregate vs batch 1 |
|---:|---:|---:|---:|---:|
| 1 | 33.4 | 30.0 | 30.0 | 1.00x |
| 2 | 67.2 | 14.9 | 29.8 | 0.99x |
| 4 | 83.9 | 11.9 | 47.7 | 1.59x |
| 8 | 127.4 | 7.9 | 62.8 | 2.10x |

