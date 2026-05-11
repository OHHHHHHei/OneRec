# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.06706 | 0.06199 | -0.00507 | 0.26715 | 0.14141 | 0.13192 | 0.09773 | 54 | 77 |
| 3 | 0.09839 | 0.09067 | -0.00772 | 0.33245 | 0.20274 | 0.16656 | 0.14097 | 90 | 125 |
| 5 | 0.11824 | 0.10589 | -0.01235 | 0.36819 | 0.23318 | 0.18266 | 0.15795 | 90 | 146 |
| 10 | 0.15089 | 0.12972 | -0.02118 | 0.43062 | 0.28965 | 0.21090 | 0.18398 | 106 | 202 |
| 20 | 0.18972 | 0.17009 | -0.01963 | 0.51335 | 0.36775 | 0.24796 | 0.22127 | 167 | 256 |
| 50 | 0.24531 | 0.22458 | -0.02074 | 0.63402 | 0.48555 | 0.31480 | 0.28524 | 220 | 314 |

## Reading

- The hierarchy run underperforms the baseline at every tracked `k`; the losses are present on `top1`, `top3`, `top5`, `top10`, `top20`, `top50`.
- Hierarchy changes same-prefix retention in the beam at almost every `k`, so the comparison is not only about exact hits; it also shows whether the model keeps the target inside the correct local neighborhood.
- Even on crowded targets (`l2>=4`), hierarchy does not show a stable advantage (`top1` delta = -0.00286, `top3` delta = -0.00714, `top10` delta = -0.04214).
- Sparse or easier targets also degrade at `top10` (`l2<=2` delta = -0.01127), so the current tokenizer-to-SFT transfer is not only failing on a tiny corner case.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 227 | 30 | 14 | 9 | 7 | 9 | 8 |
| 2-3 | 26 | 38 | 21 | 10 | 16 | 12 | 19 |
| 4-5 | 8 | 20 | 6 | 11 | 13 | 12 | 20 |
| 6-10 | 10 | 20 | 8 | 24 | 26 | 19 | 41 |
| 11-20 | 2 | 5 | 7 | 12 | 34 | 37 | 79 |
| 21-50 | 3 | 6 | 7 | 21 | 34 | 34 | 147 |
| >50 | 5 | 11 | 6 | 21 | 53 | 124 | 3201 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2661 | 0.07027 | 0.06652 | -0.00376 | 19 | 29 |
| 3 | 2661 | 0.09207 | 0.08418 | -0.00789 | 27 | 48 |
| 5 | 2661 | 0.09996 | 0.09207 | -0.00789 | 33 | 54 |
| 10 | 2661 | 0.11612 | 0.10485 | -0.01127 | 47 | 77 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 472 | 0.05297 | 0.03390 | -0.01907 | 1 | 10 |
| 3 | 472 | 0.05932 | 0.05085 | -0.00847 | 6 | 10 |
| 5 | 472 | 0.06992 | 0.05932 | -0.01059 | 7 | 12 |
| 10 | 472 | 0.09534 | 0.08051 | -0.01483 | 11 | 18 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1400 | 0.06571 | 0.06286 | -0.00286 | 34 | 38 |
| 3 | 1400 | 0.12357 | 0.11643 | -0.00714 | 57 | 67 |
| 5 | 1400 | 0.16929 | 0.14786 | -0.02143 | 50 | 80 |
| 10 | 1400 | 0.23571 | 0.19357 | -0.04214 | 48 | 107 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 54 | 0.57407 | 0.44444 | 18.29630 |
| worsened by hierarchy | 77 | 1.00000 | 1.00000 | 11.16883 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 90 | 0.70000 | 0.41111 | 16.44444 |
| worsened by hierarchy | 125 | 1.00000 | 1.00000 | 11.77600 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 90 | 0.67778 | 0.36667 | 14.90000 |
| worsened by hierarchy | 146 | 1.00000 | 1.00000 | 11.69178 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 106 | 0.69811 | 0.29245 | 11.15094 |
| worsened by hierarchy | 202 | 1.00000 | 1.00000 | 10.07426 |

## Main Takeaways

- The hierarchy effect is negative in this comparison: tokenizer-side changes do not convert into a net top-k ranking gain.
- There is no clear crowded-case rescue in this comparison; the losses are broad enough that local disambiguation is not compensating.
- The clearest failure mode is rank-drop on examples that the baseline already kept in the correct local neighborhood, which is a beam-retention problem rather than a simple local collision problem.
