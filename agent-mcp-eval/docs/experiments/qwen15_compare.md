# supervisor/Qwen/Qwen2.5-1.5B-Instruct vs single/Qwen/Qwen2.5-1.5B-Instruct

- Paired tasks: 72
- Decision: **hold**
- Reason: inconclusive: the 95% interval [-0.0694, +0.0972] straddles the margin of -0.0500, so 72 observations cannot separate an acceptable loss from an unacceptable one
- Discordant: 6 to supervisor/Qwen/Qwen2.5-1.5B-Instruct, 5 to single/Qwen/Qwen2.5-1.5B-Instruct (McNemar p = 1.0000)

| Metric | supervisor/Qwen/Qwen2.5-1.5B-Instruct | single/Qwen/Qwen2.5-1.5B-Instruct | Difference | Low | High |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Success | 0.1528 | 0.1389 | 0.0139 | -0.0694 | 0.0972 |
| Answer score | 0.1852 | 0.1620 | 0.0231 | -0.0579 | 0.1089 |
| Call precision | 0.4947 | 0.5109 | -0.0162 | -0.1197 | 0.0850 |
| Call recall | 0.4676 | 0.5266 | -0.0590 | -0.1597 | 0.0347 |
| Call F1 | 0.4681 | 0.5096 | -0.0415 | -0.1367 | 0.0482 |
| Step efficiency | 0.8405 | 0.8538 | -0.0133 | -0.0610 | 0.0312 |
| Redundant calls | 0.04 | 0.07 | -0.03 | -0.12 | 0.06 |
| Forbidden violations | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Steps | 3.01 | 2.62 | 0.39 | -0.18 | 0.97 |
| Tokens | 10651.6 | 9938.2 | 713.4 | -1293.9 | 2752.9 |
| Wall ms | 8582.0 | 7345.6 | 1236.4 | -1532.8 | 4098.2 |
