| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| cohere2 ?B Q4_0                |   1.88 GiB |     3.35 B | BLAS,MTL   |       4 |           pp100 |        301.23 ± 0.43 |
| cohere2 ?B Q4_0                |   1.88 GiB |     3.35 B | BLAS,MTL   |       4 |           pp256 |        388.34 ± 0.21 |
| cohere2 ?B Q4_0                |   1.88 GiB |     3.35 B | BLAS,MTL   |       4 |          pp1024 |        378.36 ± 1.46 |
| cohere2 ?B Q4_0                |   1.88 GiB |     3.35 B | BLAS,MTL   |       4 |            tg32 |         39.73 ± 0.44 |
| cohere2 ?B Q4_0                |   1.88 GiB |     3.35 B | BLAS,MTL   |       4 |           tg100 |         38.12 ± 1.59 |

build: 7fe450e19 (11146)
