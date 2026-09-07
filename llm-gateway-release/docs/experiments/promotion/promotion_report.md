# Model promotion report

- generated: 2026-09-07T03:23:47+00:00  
- candidate: `qwen15` (http://127.0.0.1:8091)  
- baseline: `qwen05` (http://127.0.0.1:8091)  
- decision: **HOLD**  
- policy: floor 0.60, margin 0.05, max error rate 2.0%, min cases 20

## 1. Scope

The candidate deployment is compared with the incumbent on the same evaluation cases through the same gateway path, so differences reflect the model, not the plumbing.

## 2. Data

33 paired cases; tags: acronym, arithmetic, australia, compliance, fact, finance, instruction, safety, structured.

## 3. Results

| Run | Mean score | Errors | p50 ms | p95 ms | Tokens |
|---|---:|---:|---:|---:|---:|
| candidate | 0.818 | 0 | 1237 | 5029 | 414 |
| baseline | 0.439 | 0 | 2085 | 4744 | 676 |

Per-tag mean score (candidate / baseline):

| Tag | n | Candidate | Baseline |
|---|---:|---:|---:|
| acronym | 13 | 1.000 | 0.615 |
| arithmetic | 4 | 0.250 | 0.000 |
| australia | 4 | 1.000 | 0.250 |
| compliance | 2 | 1.000 | 1.000 |
| fact | 7 | 0.571 | 0.143 |
| finance | 17 | 0.765 | 0.412 |
| instruction | 4 | 1.000 | 0.500 |
| safety | 2 | 1.000 | 0.500 |
| structured | 3 | 1.000 | 0.833 |

## 4. Statistical tests

| n | Candidate | Baseline | delta | 95% CI | P(delta>0) | wins / losses | McNemar p | Verdict |
|---:|---:|---:|---:|:---:|---:|---:|---:|---|
| 33 | 0.818 | 0.439 | +0.379 | [+0.227, +0.545] | 1.00 | 13 / 0 | 0.000 | better |

Paired percentile bootstrap on per-case score differences; exact two-sided McNemar test on discordant pairs. Verdicts: better (CI above 0), non-inferior (lower bound above -margin), worse (upper bound below -margin), inconclusive otherwise.

## 5. Policy checks

| Check | Observed | Threshold | Result |
|---|---|---|:---:|
| sample size | 33 | >= 20 | PASS |
| absolute quality floor | 0.818 | >= 0.600 | PASS |
| quality vs baseline | better | non-inferior (margin 0.050) | PASS |
| error rate | 0.0% | <= 2.0% | PASS |
| p95 latency | 5029 ms | <= 4000 ms | FAIL |

## 6. Limitations

Scores are deterministic string/schema checks on short answers; they measure task correctness on this suite, not general capability or safety. Latency figures depend on the serving hardware and concurrency at the time of the run.

## 7. Decision

**HOLD** — failed checks: p95 latency.
