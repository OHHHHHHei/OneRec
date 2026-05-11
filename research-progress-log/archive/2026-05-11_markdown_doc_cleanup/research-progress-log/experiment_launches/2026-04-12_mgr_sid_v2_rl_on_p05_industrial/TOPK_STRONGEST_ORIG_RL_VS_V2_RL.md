# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.07324 | 0.07434 | +0.00110 | 0.27267 | 0.17737 | 0.14472 | 0.12861 | 63 | 58 |
| 3 | 0.10038 | 0.10280 | +0.00243 | 0.33201 | 0.21663 | 0.16700 | 0.14869 | 92 | 81 |
| 5 | 0.11979 | 0.11692 | -0.00287 | 0.35297 | 0.23759 | 0.17869 | 0.16038 | 95 | 108 |
| 10 | 0.15133 | 0.14185 | -0.00949 | 0.38010 | 0.27796 | 0.19854 | 0.18068 | 125 | 168 |
| 20 | 0.17670 | 0.17516 | -0.00154 | 0.39929 | 0.32319 | 0.21840 | 0.20318 | 161 | 168 |
| 50 | 0.21994 | 0.23737 | +0.01743 | 0.41805 | 0.39157 | 0.25347 | 0.26473 | 265 | 186 |

## Reading

- `hierarchy` is strongest on short-range metrics: `top3` improves most clearly, and `top5` remains slightly positive.
- `hierarchy` reduces same-prefix presence in the beam at almost every `k`, which means its gain does not come from retaining more nearby candidates; it comes from cleaning up some hard local neighborhoods while losing others.
- This already suggests the core tradeoff: better local disambiguation on difficult cases, but weaker neighborhood retention on some examples that the baseline already handled well.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 274 | 22 | 5 | 7 | 1 | 18 | 5 |
| 2-3 | 27 | 51 | 10 | 10 | 6 | 10 | 9 |
| 4-5 | 21 | 19 | 6 | 8 | 6 | 13 | 15 |
| 6-10 | 8 | 17 | 10 | 23 | 29 | 24 | 32 |
| 11-20 | 2 | 9 | 13 | 21 | 28 | 18 | 24 |
| 21-50 | 0 | 4 | 2 | 17 | 29 | 43 | 101 |
| >50 | 5 | 7 | 18 | 27 | 52 | 156 | 3271 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2661 | 0.07779 | 0.07854 | +0.00075 | 17 | 15 |
| 3 | 2661 | 0.09057 | 0.09583 | +0.00526 | 35 | 21 |
| 5 | 2661 | 0.09771 | 0.10372 | +0.00601 | 42 | 26 |
| 10 | 2661 | 0.10748 | 0.11086 | +0.00338 | 44 | 35 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 472 | 0.05297 | 0.04873 | -0.00424 | 2 | 4 |
| 3 | 472 | 0.07203 | 0.05932 | -0.01271 | 2 | 8 |
| 5 | 472 | 0.08475 | 0.06144 | -0.02331 | 0 | 11 |
| 10 | 472 | 0.09746 | 0.07627 | -0.02119 | 4 | 14 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1400 | 0.07143 | 0.07500 | +0.00357 | 44 | 39 |
| 3 | 1400 | 0.12857 | 0.13071 | +0.00214 | 55 | 52 |
| 5 | 1400 | 0.17357 | 0.16071 | -0.01286 | 53 | 71 |
| 10 | 1400 | 0.25286 | 0.22286 | -0.03000 | 77 | 119 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 63 | 0.60317 | 0.57143 | 18.34921 |
| worsened by hierarchy | 58 | 1.00000 | 1.00000 | 17.24138 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 92 | 0.71739 | 0.54348 | 12.97826 |
| worsened by hierarchy | 81 | 1.00000 | 1.00000 | 13.13580 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 95 | 0.70526 | 0.45263 | 11.01053 |
| worsened by hierarchy | 108 | 1.00000 | 1.00000 | 13.77778 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 125 | 0.79200 | 0.46400 | 11.91200 |
| worsened by hierarchy | 168 | 1.00000 | 1.00000 | 14.17262 |

## Main Takeaways

- The positive effect of `hierarchy` is not a uniform top-k gain. It is concentrated in short-range ranking and crowded local structures.
- The strongest positive evidence remains on high-fanout `same_l2`-like cases, where moving the target upward inside a hard local neighborhood matters most.
- The strongest negative evidence is rank-drop on examples that the baseline already kept in the correct local neighborhood. This is the clearest sign that the current tokenizer-to-SFT transfer is still imperfect.
