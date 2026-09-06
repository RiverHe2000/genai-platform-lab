# Evaluation report

- created: 2026-09-05T12:15:23+00:00  
- retriever: `bm25`  
- generator: `hf[Qwen/Qwen2.5-1.5B-Instruct]`  
- judge: `hf[Qwen/Qwen2.5-1.5B-Instruct]`  
- samples: 60, k = 5, bootstrap = 1000 resamples, CI level = 0.95  
- judge calls: 475, unparseable after retry: 0

## Metrics

| Metric | Mean | 95% CI | n | missing |
|---|---:|:---:|---:|---:|
| retrieval/hit_rate@1 | 0.852 | [0.759, 0.944] | 54 | 6 |
| retrieval/hit_rate@5 | 1.000 | [1.000, 1.000] | 54 | 6 |
| retrieval/recall@5 | 1.000 | [1.000, 1.000] | 54 | 6 |
| retrieval/precision@5 | 0.219 | [0.204, 0.233] | 54 | 6 |
| retrieval/mrr | 0.911 | [0.853, 0.967] | 54 | 6 |
| retrieval/map | 0.911 | [0.853, 0.967] | 54 | 6 |
| retrieval/ndcg@5 | 0.934 | [0.890, 0.975] | 54 | 6 |
| abstained | 0.100 | [0.033, 0.183] | 60 | 0 |
| false_abstention | 0.019 | [0.000, 0.056] | 54 | 6 |
| lexical/f1 | 0.472 | [0.421, 0.523] | 54 | 6 |
| lexical/exact_match | 0.019 | [0.000, 0.056] | 54 | 6 |
| lexical/reference_coverage | 0.666 | [0.587, 0.739] | 54 | 6 |
| faithfulness | 0.903 | [0.828, 0.965] | 53 | 7 |
| answer_relevancy | 0.689 | [0.619, 0.753] | 54 | 6 |
| context_precision | 0.884 | [0.825, 0.937] | 54 | 6 |
| context_recall | 0.863 | [0.796, 0.918] | 54 | 6 |
| correct_abstention | 0.833 | [0.500, 1.000] | 6 | 54 |

## By tag

| Tag | n | retrieval/hit_rate@1 | retrieval/hit_rate@5 | retrieval/recall@5 | retrieval/precision@5 | retrieval/mrr | retrieval/map | retrieval/ndcg@5 | abstained | false_abstention | lexical/f1 | lexical/exact_match | lexical/reference_coverage | faithfulness | answer_relevancy | context_precision | context_recall | correct_abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aml | 3 | 1.000 | 1.000 | 1.000 | 0.267 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.457 | 0.000 | 0.542 | 0.833 | 0.753 | 1.000 | 0.833 | n/a |
| capital | 5 | 0.800 | 1.000 | 1.000 | 0.240 | 0.900 | 0.900 | 0.926 | 0.000 | 0.000 | 0.518 | 0.000 | 0.666 | 1.000 | 0.803 | 0.867 | 0.900 | n/a |
| climate | 1 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.579 | 0.000 | 0.458 | 1.000 | 0.899 | 1.000 | 1.000 | n/a |
| conduct | 4 | 0.500 | 1.000 | 1.000 | 0.200 | 0.633 | 0.633 | 0.722 | 0.000 | 0.000 | 0.357 | 0.000 | 0.653 | 1.000 | 0.845 | 0.738 | 0.625 | n/a |
| credit | 14 | 0.846 | 1.000 | 1.000 | 0.215 | 0.910 | 0.910 | 0.933 | 0.000 | 0.000 | 0.436 | 0.000 | 0.578 | 0.900 | 0.689 | 0.877 | 0.872 | 0.000 |
| genai | 4 | 0.750 | 1.000 | 1.000 | 0.200 | 0.875 | 0.875 | 0.908 | 0.000 | 0.000 | 0.549 | 0.000 | 0.907 | 1.000 | 0.659 | 0.646 | 1.000 | n/a |
| governance | 9 | 0.778 | 1.000 | 1.000 | 0.222 | 0.870 | 0.870 | 0.903 | 0.000 | 0.000 | 0.517 | 0.000 | 0.727 | 0.889 | 0.557 | 0.735 | 0.944 | n/a |
| liquidity | 3 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.333 | 0.333 | 0.411 | 0.000 | 0.548 | 0.333 | 0.563 | 1.000 | 0.833 | n/a |
| market | 4 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.570 | 0.000 | 0.737 | 0.875 | 0.404 | 0.925 | 0.875 | n/a |
| multi-hop | 5 | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.484 | 0.000 | 0.600 | 0.900 | 0.795 | 0.990 | 0.900 | n/a |
| numeric | 46 | 0.870 | 1.000 | 1.000 | 0.222 | 0.921 | 0.921 | 0.941 | 0.022 | 0.022 | 0.469 | 0.022 | 0.657 | 0.924 | 0.689 | 0.910 | 0.839 | n/a |
| operational | 6 | 1.000 | 1.000 | 1.000 | 0.233 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.444 | 0.000 | 0.811 | 0.917 | 0.759 | 1.000 | 0.833 | n/a |
| paraphrase | 12 | 0.417 | 1.000 | 1.000 | 0.200 | 0.642 | 0.642 | 0.732 | 0.083 | 0.083 | 0.342 | 0.083 | 0.513 | 0.864 | 0.540 | 0.663 | 0.812 | n/a |
| privacy | 4 | 0.750 | 1.000 | 1.000 | 0.250 | 0.875 | 0.875 | 0.908 | 0.000 | 0.000 | 0.599 | 0.250 | 0.874 | 0.875 | 0.812 | 0.951 | 1.000 | n/a |
| provisioning | 4 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.388 | 0.000 | 0.578 | 0.800 | 0.798 | 0.988 | 0.812 | n/a |
| single-hop | 37 | 0.973 | 1.000 | 1.000 | 0.200 | 0.986 | 0.986 | 0.990 | 0.000 | 0.000 | 0.513 | 0.000 | 0.724 | 0.913 | 0.724 | 0.942 | 0.874 | n/a |
| unanswerable | 6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.833 | n/a | n/a | n/a | n/a | 1.000 | n/a | n/a | n/a | 0.833 |

## Lowest faithfulness

- **q17** (0.000): The internal trigger for the liquidity coverage ratio (LCR) is 110%, and the management target is 115%. [1]
- **q54** (0.000): Every two years. [2]
- **q08** (0.200): An exposure is moved from Stage 1 to Stage 2 if any of the following conditions apply: (a) the account is 30 days past due; (b) the lifetime probability of defa
- **q22** (0.500): APRA must be informed immediately upon becoming aware of such a disruption, but no later than 72 hours afterwards. This requirement is outlined in [1], specific
- **q27** (0.500): According to the provided context, 10 or more exceptions in the red zone raise the regulatory capital multiplier to 4.0 and trigger a model review. Therefore, 1
