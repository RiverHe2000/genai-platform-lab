# Evaluation report

- created: 2026-09-05T11:51:13+00:00  
- retriever: `hybrid[rrf](bm25+dense[st[BAAI/bge-small-en-v1.5]])+rerank`  
- generator: `hf[Qwen/Qwen2.5-1.5B-Instruct]`  
- judge: `hf[Qwen/Qwen2.5-1.5B-Instruct]`  
- samples: 60, k = 5, bootstrap = 1000 resamples, CI level = 0.95  
- judge calls: 488, unparseable after retry: 0

## Metrics

| Metric | Mean | 95% CI | n | missing |
|---|---:|:---:|---:|---:|
| retrieval/hit_rate@1 | 1.000 | [1.000, 1.000] | 54 | 6 |
| retrieval/hit_rate@5 | 1.000 | [1.000, 1.000] | 54 | 6 |
| retrieval/recall@5 | 1.000 | [1.000, 1.000] | 54 | 6 |
| retrieval/precision@5 | 0.219 | [0.204, 0.233] | 54 | 6 |
| retrieval/mrr | 1.000 | [1.000, 1.000] | 54 | 6 |
| retrieval/map | 1.000 | [1.000, 1.000] | 54 | 6 |
| retrieval/ndcg@5 | 1.000 | [1.000, 1.000] | 54 | 6 |
| abstained | 0.083 | [0.017, 0.167] | 60 | 0 |
| false_abstention | 0.019 | [0.000, 0.056] | 54 | 6 |
| lexical/f1 | 0.453 | [0.402, 0.500] | 54 | 6 |
| lexical/exact_match | 0.000 | [0.000, 0.000] | 54 | 6 |
| lexical/reference_coverage | 0.649 | [0.577, 0.717] | 54 | 6 |
| faithfulness | 0.885 | [0.819, 0.942] | 55 | 5 |
| answer_relevancy | 0.694 | [0.620, 0.757] | 54 | 6 |
| context_precision | 0.975 | [0.954, 0.992] | 54 | 6 |
| context_recall | 0.867 | [0.802, 0.926] | 54 | 6 |
| correct_abstention | 0.667 | [0.333, 1.000] | 6 | 54 |

## By tag

| Tag | n | retrieval/hit_rate@1 | retrieval/hit_rate@5 | retrieval/recall@5 | retrieval/precision@5 | retrieval/mrr | retrieval/map | retrieval/ndcg@5 | abstained | false_abstention | lexical/f1 | lexical/exact_match | lexical/reference_coverage | faithfulness | answer_relevancy | context_precision | context_recall | correct_abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aml | 3 | 1.000 | 1.000 | 1.000 | 0.267 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.553 | 0.000 | 0.688 | 0.833 | 0.790 | 1.000 | 0.833 | n/a |
| capital | 5 | 1.000 | 1.000 | 1.000 | 0.240 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.506 | 0.000 | 0.672 | 0.933 | 0.787 | 1.000 | 0.700 | n/a |
| climate | 1 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.564 | 0.000 | 0.458 | 1.000 | 0.899 | 1.000 | 1.000 | n/a |
| conduct | 4 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.291 | 0.000 | 0.622 | 1.000 | 0.822 | 1.000 | 1.000 | n/a |
| credit | 14 | 1.000 | 1.000 | 1.000 | 0.215 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.464 | 0.000 | 0.559 | 0.883 | 0.657 | 0.958 | 0.795 | 0.000 |
| genai | 4 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.453 | 0.000 | 0.757 | 1.000 | 0.681 | 0.958 | 0.875 | n/a |
| governance | 9 | 1.000 | 1.000 | 1.000 | 0.222 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.478 | 0.000 | 0.697 | 1.000 | 0.635 | 0.981 | 0.889 | n/a |
| liquidity | 3 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.333 | 0.333 | 0.444 | 0.000 | 0.581 | 0.333 | 0.544 | 1.000 | 1.000 | n/a |
| market | 4 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.563 | 0.000 | 0.722 | 0.875 | 0.375 | 1.000 | 0.750 | n/a |
| multi-hop | 5 | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.456 | 0.000 | 0.651 | 0.833 | 0.782 | 0.983 | 0.900 | n/a |
| numeric | 46 | 1.000 | 1.000 | 1.000 | 0.222 | 1.000 | 1.000 | 1.000 | 0.022 | 0.022 | 0.441 | 0.000 | 0.630 | 0.889 | 0.680 | 0.974 | 0.855 | n/a |
| operational | 6 | 1.000 | 1.000 | 1.000 | 0.233 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.420 | 0.000 | 0.723 | 0.917 | 0.813 | 0.986 | 0.917 | n/a |
| paraphrase | 12 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.083 | 0.083 | 0.258 | 0.000 | 0.369 | 0.909 | 0.542 | 0.965 | 0.833 | n/a |
| privacy | 4 | 1.000 | 1.000 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.343 | 0.000 | 0.674 | 0.750 | 0.822 | 0.863 | 1.000 | n/a |
| provisioning | 4 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.405 | 0.000 | 0.720 | 0.675 | 0.764 | 1.000 | 1.000 | n/a |
| single-hop | 37 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.516 | 0.000 | 0.739 | 0.893 | 0.732 | 0.977 | 0.874 | n/a |
| unanswerable | 6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.667 | n/a | n/a | n/a | n/a | 0.750 | n/a | n/a | n/a | 0.667 |

## Lowest faithfulness

- **q17** (0.000): The internal trigger for the liquidity coverage ratio (LCR) is 110%, and the management target is also 115%.
- **q08** (0.200): An exposure is moved from Stage 1 to Stage 2 if any of the following conditions apply: (a) the account is 30 days past due; (b) the lifetime probability of defa
- **q07** (0.500): Yes, there is a cap on construction lending within the SME portfolio. Construction lending is capped at 10% of the SME portfolio. [1]
- **q22** (0.500): APRA must be informed immediately upon becoming aware of such a disruption, but no later than 72 hours afterwards. This requirement is outlined in [1], specific
- **q27** (0.500): According to the provided context, 10 or more exceptions in the red zone raise the regulatory capital multiplier to 4.0 and trigger a model review. Therefore, 1
