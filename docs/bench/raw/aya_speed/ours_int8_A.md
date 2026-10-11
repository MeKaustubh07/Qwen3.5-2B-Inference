### metal-kernels-int8 — tiny-aya-global, 32 new tokens, M2 8 GB

Weights resident: 3.56 GB → bandwidth ceiling ≈ 28 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 668 MiB, Metal driver 4649 MiB, physical footprint 4986 MiB now and 5005 MiB at its peak. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 195 | 42.3 | 82 | 23.7 | 84% |
| 100 | yes | 566 | 44.5 | 177 | 22.5 | 80% |
| 256 | yes | 1108 | 48.3 | 231 | 20.7 | 74% |
| 1024 | yes | 5533 | 53.8 | 185 | 18.6 | 66% |

| batch | step (ms) | per-sequence tok/s | aggregate tok/s | aggregate vs batch 1 |
|---:|---:|---:|---:|---:|
| 1 | 50.0 | 20.0 | 20.0 | 1.00x |
| 2 | 74.5 | 13.4 | 26.8 | 1.34x |
| 4 | 91.4 | 10.9 | 43.8 | 2.19x |
| 8 | 116.8 | 8.6 | 68.5 | 3.42x |

