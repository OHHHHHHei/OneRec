# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.07037 | 0.07059 | +0.00022 | 0.13964 | 0.15266 | 0.10390 | 0.10104 | 62 | 61 |
| 3 | 0.09420 | 0.09508 | +0.00088 | 0.20913 | 0.23075 | 0.13898 | 0.15553 | 79 | 75 |
| 5 | 0.11030 | 0.11471 | +0.00441 | 0.24840 | 0.25789 | 0.16192 | 0.17472 | 96 | 76 |
| 10 | 0.14251 | 0.14626 | +0.00375 | 0.30532 | 0.31216 | 0.19590 | 0.20296 | 112 | 95 |
| 20 | 0.18045 | 0.18244 | +0.00199 | 0.38120 | 0.38143 | 0.23450 | 0.24112 | 136 | 127 |
| 50 | 0.24289 | 0.24818 | +0.00529 | 0.49967 | 0.49327 | 0.30333 | 0.31149 | 189 | 165 |

## Reading

- `hierarchy` is strongest on short-range metrics: `top3` improves most clearly, and `top5` remains slightly positive.
- `hierarchy` reduces same-prefix presence in the beam at almost every `k`, which means its gain does not come from retaining more nearby candidates; it comes from cleaning up some hard local neighborhoods while losing others.
- This already suggests the core tradeoff: better local disambiguation on difficult cases, but weaker neighborhood retention on some examples that the baseline already handled well.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 258 | 27 | 21 | 5 | 4 | 2 | 2 |
| 2-3 | 36 | 31 | 12 | 16 | 5 | 5 | 3 |
| 4-5 | 7 | 19 | 13 | 18 | 6 | 5 | 5 |
| 6-10 | 8 | 17 | 23 | 40 | 23 | 26 | 9 |
| 11-20 | 4 | 8 | 12 | 28 | 50 | 45 | 25 |
| 21-50 | 3 | 4 | 3 | 18 | 46 | 88 | 121 |
| >50 | 4 | 5 | 5 | 18 | 30 | 127 | 3243 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3141 | 0.07800 | 0.07673 | -0.00127 | 26 | 30 |
| 3 | 3141 | 0.09169 | 0.09551 | +0.00382 | 50 | 38 |
| 5 | 3141 | 0.10474 | 0.10602 | +0.00127 | 56 | 52 |
| 10 | 3141 | 0.12671 | 0.12639 | -0.00032 | 53 | 54 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 382 | 0.06021 | 0.09686 | +0.03665 | 16 | 2 |
| 3 | 382 | 0.10209 | 0.11518 | +0.01309 | 9 | 4 |
| 5 | 382 | 0.11780 | 0.12565 | +0.00785 | 7 | 4 |
| 10 | 382 | 0.14921 | 0.14398 | -0.00524 | 10 | 12 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1010 | 0.05050 | 0.04158 | -0.00891 | 20 | 29 |
| 3 | 1010 | 0.09901 | 0.08614 | -0.01287 | 20 | 33 |
| 5 | 1010 | 0.12475 | 0.13762 | +0.01287 | 33 | 20 |
| 10 | 1010 | 0.18911 | 0.20891 | +0.01980 | 49 | 29 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 62 | 0.27419 | 0.24194 | 5.90323 |
| worsened by hierarchy | 61 | 1.00000 | 1.00000 | 10.49180 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 79 | 0.29114 | 0.12658 | 5.67089 |
| worsened by hierarchy | 75 | 1.00000 | 1.00000 | 8.82667 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 96 | 0.41667 | 0.17708 | 6.65625 |
| worsened by hierarchy | 76 | 1.00000 | 1.00000 | 4.76316 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 112 | 0.50893 | 0.29464 | 8.15179 |
| worsened by hierarchy | 95 | 1.00000 | 1.00000 | 5.97895 |

## Main Takeaways

- The positive effect of `hierarchy` is not a uniform top-k gain. It is concentrated in short-range ranking and crowded local structures.
- The strongest positive evidence remains on high-fanout `same_l2`-like cases, where moving the target upward inside a hard local neighborhood matters most.
- The strongest negative evidence is rank-drop on examples that the baseline already kept in the correct local neighborhood. This is the clearest sign that the current tokenizer-to-SFT transfer is still imperfect.
