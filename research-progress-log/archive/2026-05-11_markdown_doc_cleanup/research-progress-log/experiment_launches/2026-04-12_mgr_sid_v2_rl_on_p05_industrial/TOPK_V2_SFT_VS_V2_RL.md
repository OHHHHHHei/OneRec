# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.07059 | 0.07434 | +0.00375 | 0.15266 | 0.17737 | 0.10104 | 0.12861 | 84 | 67 |
| 3 | 0.09508 | 0.10280 | +0.00772 | 0.23075 | 0.21663 | 0.15553 | 0.14869 | 120 | 85 |
| 5 | 0.11471 | 0.11692 | +0.00221 | 0.25789 | 0.23759 | 0.17472 | 0.16038 | 119 | 109 |
| 10 | 0.14626 | 0.14185 | -0.00441 | 0.31216 | 0.27796 | 0.20296 | 0.18068 | 153 | 173 |
| 20 | 0.18244 | 0.17516 | -0.00728 | 0.38143 | 0.32319 | 0.24112 | 0.20318 | 197 | 230 |
| 50 | 0.24818 | 0.23737 | -0.01081 | 0.49327 | 0.39157 | 0.31149 | 0.26473 | 238 | 287 |

## Reading

- `hierarchy` is strongest on short-range metrics: `top3` improves most clearly, and `top5` remains slightly positive.
- `hierarchy` reduces same-prefix presence in the beam at almost every `k`, which means its gain does not come from retaining more nearby candidates; it comes from cleaning up some hard local neighborhoods while losing others.
- This already suggests the core tradeoff: better local disambiguation on difficult cases, but weaker neighborhood retention on some examples that the baseline already handled well.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 253 | 33 | 4 | 1 | 1 | 16 | 12 |
| 2-3 | 25 | 35 | 12 | 7 | 5 | 12 | 15 |
| 4-5 | 19 | 22 | 8 | 11 | 7 | 15 | 7 |
| 6-10 | 13 | 12 | 13 | 22 | 22 | 20 | 41 |
| 11-20 | 9 | 7 | 16 | 19 | 21 | 40 | 52 |
| 21-50 | 12 | 9 | 6 | 23 | 37 | 51 | 160 |
| >50 | 6 | 11 | 5 | 30 | 58 | 128 | 3170 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3141 | 0.07673 | 0.07736 | +0.00064 | 32 | 30 |
| 3 | 3141 | 0.09551 | 0.09328 | -0.00223 | 39 | 46 |
| 5 | 3141 | 0.10602 | 0.10092 | -0.00509 | 41 | 57 |
| 10 | 3141 | 0.12639 | 0.11079 | -0.01560 | 51 | 100 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 382 | 0.09686 | 0.06545 | -0.03141 | 8 | 20 |
| 3 | 382 | 0.11518 | 0.07853 | -0.03665 | 5 | 19 |
| 5 | 382 | 0.12565 | 0.08115 | -0.04450 | 3 | 20 |
| 10 | 382 | 0.14398 | 0.09424 | -0.04974 | 7 | 26 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1010 | 0.04158 | 0.06832 | +0.02673 | 44 | 17 |
| 3 | 1010 | 0.08614 | 0.14158 | +0.05545 | 76 | 20 |
| 5 | 1010 | 0.13762 | 0.18020 | +0.04257 | 75 | 32 |
| 10 | 1010 | 0.20891 | 0.25644 | +0.04752 | 95 | 47 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 84 | 0.27381 | 0.16667 | 11.71429 |
| worsened by hierarchy | 67 | 1.00000 | 1.00000 | 5.32836 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 120 | 0.68333 | 0.47500 | 13.42500 |
| worsened by hierarchy | 85 | 1.00000 | 1.00000 | 5.37647 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 119 | 0.67227 | 0.51261 | 12.83193 |
| worsened by hierarchy | 109 | 1.00000 | 1.00000 | 5.53211 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 153 | 0.76471 | 0.52941 | 12.06536 |
| worsened by hierarchy | 173 | 1.00000 | 1.00000 | 5.50867 |

## Main Takeaways

- The positive effect of `hierarchy` is not a uniform top-k gain. It is concentrated in short-range ranking and crowded local structures.
- The strongest positive evidence remains on high-fanout `same_l2`-like cases, where moving the target upward inside a hard local neighborhood matters most.
- The strongest negative evidence is rank-drop on examples that the baseline already kept in the correct local neighborhood. This is the clearest sign that the current tokenizer-to-SFT transfer is still imperfect.
