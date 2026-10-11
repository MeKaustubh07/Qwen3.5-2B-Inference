| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| cohere2 ?B Q4_K - Medium       |   1.99 GiB |     3.35 B | BLAS,MTL   |       4 |           pp100 |        265.31 ± 0.20 |
| cohere2 ?B Q4_K - Medium       |   1.99 GiB |     3.35 B | BLAS,MTL   |       4 |           pp256 |        342.94 ± 0.15 |
| cohere2 ?B Q4_K - Medium       |   1.99 GiB |     3.35 B | BLAS,MTL   |       4 |          pp1024 |        332.95 ± 5.33 |
| cohere2 ?B Q4_K - Medium       |   1.99 GiB |     3.35 B | BLAS,MTL   |       4 |            tg32 |         34.55 ± 1.89 |
| cohere2 ?B Q4_K - Medium       |   1.99 GiB |     3.35 B | BLAS,MTL   |       4 |           tg100 |         36.43 ± 1.13 |

build: 7fe450e19 (11146)
