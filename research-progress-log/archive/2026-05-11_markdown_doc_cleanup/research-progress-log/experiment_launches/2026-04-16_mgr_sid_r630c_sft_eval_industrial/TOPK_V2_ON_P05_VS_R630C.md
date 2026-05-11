# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.07059 | 0.06199 | -0.00860 | 0.15266 | 0.14141 | 0.10104 | 0.09773 | 52 | 91 |
| 3 | 0.09508 | 0.09067 | -0.00441 | 0.23075 | 0.20274 | 0.15553 | 0.14097 | 97 | 117 |
| 5 | 0.11471 | 0.10589 | -0.00882 | 0.25789 | 0.23318 | 0.17472 | 0.15795 | 89 | 129 |
| 10 | 0.14626 | 0.12972 | -0.01655 | 0.31216 | 0.28965 | 0.20296 | 0.18398 | 118 | 193 |
| 20 | 0.18244 | 0.17009 | -0.01235 | 0.38143 | 0.36775 | 0.24112 | 0.22127 | 172 | 228 |
| 50 | 0.24818 | 0.22458 | -0.02360 | 0.49327 | 0.48555 | 0.31149 | 0.28524 | 226 | 333 |

## Reading

- The hierarchy run underperforms the baseline at every tracked `k`; the losses are present on `top1`, `top3`, `top5`, `top10`, `top20`, `top50`.
- Hierarchy changes same-prefix retention in the beam at almost every `k`, so the comparison is not only about exact hits; it also shows whether the model keeps the target inside the correct local neighborhood.
- The clearest surviving upside is on crowded targets (`l2>=4`): `top1` delta = +0.00990, `top3` delta = +0.00990. But that advantage does not survive to `top10` (delta = -0.02871), which points to better head disambiguation but weaker beam retention.
- Sparse or easier targets also degrade at `top10` (`l2<=2` delta = -0.01369), so the current tokenizer-to-SFT transfer is not only failing on a tiny corner case.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 229 | 42 | 16 | 7 | 5 | 5 | 16 |
| 2-3 | 17 | 26 | 20 | 8 | 11 | 10 | 19 |
| 4-5 | 17 | 18 | 6 | 15 | 16 | 6 | 11 |
| 6-10 | 7 | 11 | 14 | 17 | 27 | 23 | 44 |
| 11-20 | 2 | 9 | 5 | 19 | 35 | 36 | 58 |
| 21-50 | 7 | 7 | 3 | 14 | 36 | 46 | 185 |
| >50 | 2 | 17 | 5 | 28 | 53 | 121 | 3182 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3141 | 0.07673 | 0.06686 | -0.00987 | 24 | 55 |
| 3 | 3141 | 0.09551 | 0.08883 | -0.00669 | 43 | 64 |
| 5 | 3141 | 0.10602 | 0.09901 | -0.00700 | 46 | 68 |
| 10 | 3141 | 0.12639 | 0.11270 | -0.01369 | 61 | 104 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 382 | 0.09686 | 0.04974 | -0.04712 | 2 | 20 |
| 3 | 382 | 0.11518 | 0.09162 | -0.02356 | 2 | 11 |
| 5 | 382 | 0.12565 | 0.11257 | -0.01309 | 3 | 8 |
| 10 | 382 | 0.14398 | 0.13613 | -0.00785 | 7 | 10 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1010 | 0.04158 | 0.05149 | +0.00990 | 26 | 16 |
| 3 | 1010 | 0.08614 | 0.09604 | +0.00990 | 52 | 42 |
| 5 | 1010 | 0.13762 | 0.12475 | -0.01287 | 40 | 53 |
| 10 | 1010 | 0.20891 | 0.18020 | -0.02871 | 50 | 79 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 52 | 0.17308 | 0.09615 | 10.76923 |
| worsened by hierarchy | 91 | 1.00000 | 1.00000 | 3.98901 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 97 | 0.49485 | 0.37113 | 11.03093 |
| worsened by hierarchy | 117 | 1.00000 | 1.00000 | 7.91453 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 89 | 0.57303 | 0.34831 | 8.93258 |
| worsened by hierarchy | 129 | 1.00000 | 1.00000 | 7.84496 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 118 | 0.58475 | 0.33898 | 8.07627 |
| worsened by hierarchy | 193 | 1.00000 | 1.00000 | 8.38860 |

## Main Takeaways

- The hierarchy effect is negative in this comparison: tokenizer-side changes do not convert into a net top-k ranking gain.
- The remaining positive signal, if any, is concentrated on crowded local structures and short-range disambiguation.
- The clearest failure mode is rank-drop on examples that the baseline already kept in the correct local neighborhood, which is a beam-retention problem rather than a simple local collision problem.
