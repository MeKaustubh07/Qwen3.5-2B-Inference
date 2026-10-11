### metal-kernels-int4 — tiny-aya-global, 100 new tokens, M2 8 GB

Weights resident: 2.34 GB → bandwidth ceiling ≈ 43 tok/s (every decode step reads every weight once at ~100 GB/s).
Memory: peak CPU RSS 1003 MiB, Metal driver 2659 MiB, physical footprint 3029 MiB now and 3097 MiB at its peak. Each row: median of 3 runs after a warm-up at that prompt length.

| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |
|---:|:---:|---:|---:|---:|---:|---:|
| 100 | yes | 568 | 35.4 | 176 | 28.3 | 66% |

