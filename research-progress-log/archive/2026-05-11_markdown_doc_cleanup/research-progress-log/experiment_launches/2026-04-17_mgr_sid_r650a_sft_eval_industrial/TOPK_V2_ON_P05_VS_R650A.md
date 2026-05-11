# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.07059 | 0.06530 | -0.00529 | 0.15266 | 0.15178 | 0.10104 | 0.10854 | 55 | 79 |
| 3 | 0.09508 | 0.09354 | -0.00154 | 0.23075 | 0.20053 | 0.15553 | 0.13744 | 109 | 116 |
| 5 | 0.11471 | 0.10920 | -0.00552 | 0.25789 | 0.22965 | 0.17472 | 0.15310 | 120 | 145 |
| 10 | 0.14626 | 0.13236 | -0.01390 | 0.31216 | 0.28039 | 0.20296 | 0.17957 | 143 | 206 |
| 20 | 0.18244 | 0.17075 | -0.01169 | 0.38143 | 0.34591 | 0.24112 | 0.21707 | 168 | 221 |
| 50 | 0.24818 | 0.22987 | -0.01831 | 0.49327 | 0.46945 | 0.31149 | 0.27951 | 220 | 303 |

## Reading

- The hierarchy run underperforms the baseline at every tracked `k`; the losses are present on `top1`, `top3`, `top5`, `top10`, `top20`, `top50`.
- Hierarchy changes same-prefix retention in the beam at almost every `k`, so the comparison is not only about exact hits; it also shows whether the model keeps the target inside the correct local neighborhood.
- The clearest surviving upside is on crowded targets (`l2>=4`): `top1` delta = +0.01089, `top3` delta = +0.03762. But that advantage does not survive to `top10` (delta = -0.00099), which points to better head disambiguation but weaker beam retention.
- Sparse or easier targets also degrade at `top10` (`l2<=2` delta = -0.01878), so the current tokenizer-to-SFT transfer is not only failing on a tiny corner case.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 241 | 31 | 11 | 10 | 11 | 8 | 8 |
| 2-3 | 22 | 21 | 12 | 16 | 17 | 10 | 13 |
| 4-5 | 7 | 23 | 7 | 12 | 15 | 9 | 16 |
| 6-10 | 4 | 17 | 10 | 13 | 30 | 30 | 39 |
| 11-20 | 7 | 8 | 16 | 19 | 26 | 32 | 56 |
| 21-50 | 6 | 14 | 8 | 14 | 26 | 59 | 171 |
| >50 | 9 | 14 | 7 | 21 | 49 | 120 | 3188 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3141 | 0.07673 | 0.06909 | -0.00764 | 25 | 49 |
| 3 | 3141 | 0.09551 | 0.08500 | -0.01051 | 39 | 72 |
| 5 | 3141 | 0.10602 | 0.09137 | -0.01465 | 49 | 95 |
| 10 | 3141 | 0.12639 | 0.10761 | -0.01878 | 64 | 123 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 382 | 0.09686 | 0.06806 | -0.02880 | 6 | 17 |
| 3 | 382 | 0.11518 | 0.08377 | -0.03141 | 5 | 17 |
| 5 | 382 | 0.12565 | 0.11257 | -0.01309 | 7 | 12 |
| 10 | 382 | 0.14398 | 0.13613 | -0.00785 | 11 | 14 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1010 | 0.04158 | 0.05248 | +0.01089 | 24 | 13 |
| 3 | 1010 | 0.08614 | 0.12376 | +0.03762 | 65 | 27 |
| 5 | 1010 | 0.13762 | 0.16337 | +0.02574 | 64 | 38 |
| 10 | 1010 | 0.20891 | 0.20792 | -0.00099 | 68 | 69 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 55 | 0.21818 | 0.09091 | 10.41818 |
| worsened by hierarchy | 79 | 1.00000 | 1.00000 | 4.20253 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 109 | 0.55963 | 0.42202 | 11.49541 |
| worsened by hierarchy | 116 | 1.00000 | 1.00000 | 5.75862 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 120 | 0.50000 | 0.35000 | 10.04167 |
| worsened by hierarchy | 145 | 1.00000 | 1.00000 | 5.50345 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 143 | 0.56643 | 0.37063 | 8.90210 |
| worsened by hierarchy | 206 | 1.00000 | 1.00000 | 6.81553 |

## Main Takeaways

- The hierarchy effect is negative in this comparison: tokenizer-side changes do not convert into a net top-k ranking gain.
- The remaining positive signal, if any, is concentrated on crowded local structures and short-range disambiguation.
- The clearest failure mode is rank-drop on examples that the baseline already kept in the correct local neighborhood, which is a beam-retention problem rather than a simple local collision problem.
