| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| cohere2 ?B Q4_0                |   1.88 GiB |     3.35 B | BLAS,MTL   |       4 |           pp100 |        302.80 ± 0.44 |
| cohere2 ?B Q4_0                |   1.88 GiB |     3.35 B | BLAS,MTL   |       4 |           pp256 |        389.89 ± 0.18 |
| cohere2 ?B Q4_0                |   1.88 GiB |     3.35 B | BLAS,MTL   |       4 |          pp1024 |        378.84 ± 2.17 |
| cohere2 ?B Q4_0                |   1.88 GiB |     3.35 B | BLAS,MTL   |       4 |            tg32 |         41.84 ± 0.32 |
| cohere2 ?B Q4_0                |   1.88 GiB |     3.35 B | BLAS,MTL   |       4 |           tg100 |         39.19 ± 1.93 |

build: 7fe450e19 (11146)
