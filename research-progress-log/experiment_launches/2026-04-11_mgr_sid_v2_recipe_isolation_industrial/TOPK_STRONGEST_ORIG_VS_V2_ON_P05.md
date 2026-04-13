# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.06706 | 0.07059 | +0.00353 | 0.26715 | 0.15266 | 0.13192 | 0.10104 | 73 | 57 |
| 3 | 0.09839 | 0.09508 | -0.00331 | 0.33245 | 0.23075 | 0.16656 | 0.15553 | 102 | 117 |
| 5 | 0.11824 | 0.11471 | -0.00353 | 0.36819 | 0.25789 | 0.18266 | 0.17472 | 122 | 138 |
| 10 | 0.15089 | 0.14626 | -0.00463 | 0.43062 | 0.31216 | 0.21090 | 0.20296 | 162 | 183 |
| 20 | 0.18972 | 0.18244 | -0.00728 | 0.51335 | 0.38143 | 0.24796 | 0.24112 | 176 | 209 |
| 50 | 0.24531 | 0.24818 | +0.00287 | 0.63402 | 0.49327 | 0.31480 | 0.31149 | 264 | 251 |

## Reading

- `hierarchy` is strongest on short-range metrics: `top3` improves most clearly, and `top5` remains slightly positive.
- `hierarchy` reduces same-prefix presence in the beam at almost every `k`, which means its gain does not come from retaining more nearby candidates; it comes from cleaning up some hard local neighborhoods while losing others.
- This already suggests the core tradeoff: better local disambiguation on difficult cases, but weaker neighborhood retention on some examples that the baseline already handled well.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 247 | 17 | 11 | 8 | 9 | 4 | 8 |
| 2-3 | 30 | 35 | 17 | 14 | 13 | 18 | 15 |
| 4-5 | 12 | 16 | 13 | 14 | 14 | 7 | 14 |
| 6-10 | 12 | 12 | 18 | 25 | 32 | 18 | 31 |
| 11-20 | 6 | 14 | 9 | 29 | 24 | 45 | 49 |
| 21-50 | 5 | 3 | 11 | 21 | 24 | 54 | 134 |
| >50 | 8 | 14 | 10 | 32 | 48 | 152 | 3157 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2661 | 0.07027 | 0.07779 | +0.00752 | 36 | 16 |
| 3 | 2661 | 0.09207 | 0.09169 | -0.00038 | 37 | 38 |
| 5 | 2661 | 0.09996 | 0.10071 | +0.00075 | 44 | 42 |
| 10 | 2661 | 0.11612 | 0.11950 | +0.00338 | 66 | 57 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 472 | 0.05297 | 0.04873 | -0.00424 | 4 | 6 |
| 3 | 472 | 0.05932 | 0.06144 | +0.00212 | 7 | 6 |
| 5 | 472 | 0.06992 | 0.06992 | +0.00000 | 9 | 9 |
| 10 | 472 | 0.09534 | 0.09322 | -0.00212 | 14 | 15 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1400 | 0.06571 | 0.06429 | -0.00143 | 33 | 35 |
| 3 | 1400 | 0.12357 | 0.11286 | -0.01071 | 58 | 73 |
| 5 | 1400 | 0.16929 | 0.15643 | -0.01286 | 69 | 87 |
| 10 | 1400 | 0.23571 | 0.21500 | -0.02071 | 82 | 111 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 73 | 0.47945 | 0.12329 | 6.06849 |
| worsened by hierarchy | 57 | 1.00000 | 1.00000 | 14.22807 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 102 | 0.63725 | 0.34314 | 11.59804 |
| worsened by hierarchy | 117 | 1.00000 | 1.00000 | 14.13675 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 122 | 0.64754 | 0.36066 | 13.41803 |
| worsened by hierarchy | 138 | 1.00000 | 1.00000 | 14.50725 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 162 | 0.68519 | 0.33951 | 11.36420 |
| worsened by hierarchy | 183 | 1.00000 | 1.00000 | 12.40437 |

## Main Takeaways

- The positive effect of `hierarchy` is not a uniform top-k gain. It is concentrated in short-range ranking and crowded local structures.
- The strongest positive evidence remains on high-fanout `same_l2`-like cases, where moving the target upward inside a hard local neighborhood matters most.
- The strongest negative evidence is rank-drop on examples that the baseline already kept in the correct local neighborhood. This is the clearest sign that the current tokenizer-to-SFT transfer is still imperfect.
