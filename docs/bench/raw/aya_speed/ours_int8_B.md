### metal-kernels-int8 — tiny-aya-global, 32 new tokens, M2 8 GB

Weights resident: 3.56 GB → bandwidth ceiling ≈ 28 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 561 MiB, Metal driver 4649 MiB, physical footprint 4991 MiB now and 5009 MiB at its peak. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 | yes | 208 | 44.6 | 77 | 22.4 | 80% |
| 100 | yes | 590 | 47.3 | 169 | 21.1 | 75% |
| 256 | yes | 1000 | 47.4 | 256 | 21.1 | 75% |
| 1024 | yes | 3831 | 53.6 | 267 | 18.7 | 66% |

| batch | step (ms) | per-sequence tok/s | aggregate tok/s | aggregate vs batch 1 |
|---:|---:|---:|---:|---:|
| 1 | 43.7 | 22.9 | 22.9 | 1.00x |
| 2 | 66.6 | 15.0 | 30.0 | 1.31x |
| 4 | 79.5 | 12.6 | 50.3 | 2.20x |
| 8 | 119.2 | 8.4 | 67.1 | 2.93x |

