| Target | Model | Conc. | Requests | Stream | OK / Err | p50 ms | p95 ms | p99 ms | TTFT p50 ms | req/s | tok/s |
|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| in-process:gateway.fake.yaml | fake-baseline | 1 | 300 | no | 300 / 0 | 1 | 1 | 1 | - | 1111.7 | 10005 |
| in-process:gateway.fake.yaml | fake-baseline | 8 | 300 | no | 300 / 0 | 5 | 8 | 18 | - | 1428.4 | 12855 |
| in-process:gateway.fake.yaml | fake-baseline | 32 | 300 | no | 300 / 0 | 18 | 40 | 40 | - | 1540.9 | 13868 |
