### metal-kernels-int8 — tiny-aya-global, 100 new tokens, M2 8 GB

Weights resident: 3.56 GB → bandwidth ceiling ≈ 28 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 1033 MiB, Metal driver 3627 MiB, physical footprint 3952 MiB now and 4023 MiB at its peak. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 100 | yes | 631 | 47.6 | 159 | 21.0 | 75% |

