| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| cohere2 ?B Q4_K - Medium       |   1.99 GiB |     3.35 B | BLAS,MTL   |       4 |           pp100 |        268.54 ± 1.03 |
| cohere2 ?B Q4_K - Medium       |   1.99 GiB |     3.35 B | BLAS,MTL   |       4 |           pp256 |        346.66 ± 0.47 |
| cohere2 ?B Q4_K - Medium       |   1.99 GiB |     3.35 B | BLAS,MTL   |       4 |          pp1024 |        339.10 ± 0.66 |
| cohere2 ?B Q4_K - Medium       |   1.99 GiB |     3.35 B | BLAS,MTL   |       4 |            tg32 |         36.83 ± 0.28 |
| cohere2 ?B Q4_K - Medium       |   1.99 GiB |     3.35 B | BLAS,MTL   |       4 |           tg100 |         37.78 ± 0.21 |

build: 7fe450e19 (11146)
