### metal-kernels-int4 — tiny-aya-global, 32 new tokens, M2 8 GB

Weights resident: 2.34 GB → bandwidth ceiling ≈ 43 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 617 MiB, Metal driver 3681 MiB, physical footprint 4033 MiB now and 4092 MiB at its peak. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 210 | 32.6 | 76 | 30.6 | 72% |
| 100 | yes | 558 | 39.1 | 179 | 25.6 | 60% |
| 256 | yes | 970 | 34.4 | 264 | 29.1 | 68% |
| 1024 | yes | 3676 | 41.0 | 279 | 24.4 | 57% |

| batch | step (ms) | per-sequence tok/s | aggregate tok/s | aggregate vs batch 1 |
|---:|---:|---:|---:|---:|
| 1 | 33.8 | 29.5 | 29.5 | 1.00x |
| 2 | 66.0 | 15.1 | 30.3 | 1.03x |
| 4 | 88.7 | 11.3 | 45.1 | 1.53x |
| 8 | 123.6 | 8.1 | 64.7 | 2.19x |

