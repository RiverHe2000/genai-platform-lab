# supervisor/scripted vs single/scripted

- Paired tasks: 72
- Decision: **hold**
- Reason: inconclusive: the 95% interval [-0.0694, +0.0694] straddles the margin of -0.0500, so 72 observations cannot separate an acceptable loss from an unacceptable one
- Discordant: 3 to supervisor/scripted, 3 to single/scripted (McNemar p = 1.0000)

| Metric | supervisor/scripted | single/scripted | Difference | Low | High |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Success | 0.1806 | 0.1806 | 0.0000 | -0.0694 | 0.0694 |
| Answer score | 0.2662 | 0.2569 | 0.0093 | -0.0509 | 0.0694 |
| Call precision | 0.6806 | 0.7222 | -0.0417 | -0.0972 | 0.0000 |
| Call recall | 0.4988 | 0.5405 | -0.0417 | -0.0972 | 0.0000 |
| Call F1 | 0.5486 | 0.5903 | -0.0417 | -0.0972 | 0.0000 |
| Step efficiency | 0.3932 | 0.9375 | -0.5443 | -0.5896 | -0.4993 |
| Redundant calls | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Forbidden violations | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Steps | 6.82 | 1.82 | 5.00 | 5.00 | 5.00 |
| Tokens | 17673.3 | 6964.8 | 10708.6 | 10612.1 | 10812.0 |
| Wall ms | 35.9 | 9.1 | 26.7 | 24.0 | 29.2 |
