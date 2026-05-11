# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.06706 | 0.05912 | -0.00794 | 0.26715 | 0.13457 | 0.13192 | 0.09398 | 63 | 99 |
| 3 | 0.09839 | 0.08427 | -0.01412 | 0.33245 | 0.20803 | 0.16656 | 0.14273 | 81 | 145 |
| 5 | 0.11824 | 0.10015 | -0.01809 | 0.36819 | 0.24377 | 0.18266 | 0.16148 | 97 | 179 |
| 10 | 0.15089 | 0.13082 | -0.02008 | 0.43062 | 0.29539 | 0.21090 | 0.19060 | 135 | 226 |
| 20 | 0.18972 | 0.16391 | -0.02581 | 0.51335 | 0.35672 | 0.24796 | 0.22259 | 168 | 285 |
| 50 | 0.24531 | 0.21597 | -0.02934 | 0.63402 | 0.46636 | 0.31480 | 0.28392 | 228 | 361 |

## Reading

- `hierarchy` is strongest on short-range metrics: `top3` improves most clearly, and `top5` remains slightly positive.
- `hierarchy` reduces same-prefix presence in the beam at almost every `k`, which means its gain does not come from retaining more nearby candidates; it comes from cleaning up some hard local neighborhoods while losing others.
- This already suggests the core tradeoff: better local disambiguation on difficult cases, but weaker neighborhood retention on some examples that the baseline already handled well.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 205 | 32 | 14 | 8 | 8 | 11 | 26 |
| 2-3 | 38 | 26 | 11 | 18 | 8 | 13 | 28 |
| 4-5 | 6 | 14 | 11 | 16 | 9 | 13 | 21 |
| 6-10 | 9 | 16 | 14 | 20 | 24 | 10 | 55 |
| 11-20 | 3 | 9 | 6 | 20 | 30 | 37 | 71 |
| 21-50 | 2 | 4 | 8 | 21 | 22 | 35 | 160 |
| >50 | 5 | 13 | 8 | 36 | 49 | 117 | 3193 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2661 | 0.07027 | 0.06840 | -0.00188 | 34 | 39 |
| 3 | 2661 | 0.09207 | 0.08643 | -0.00564 | 37 | 52 |
| 5 | 2661 | 0.09996 | 0.09583 | -0.00413 | 45 | 56 |
| 10 | 2661 | 0.11612 | 0.11575 | -0.00038 | 73 | 74 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 472 | 0.05297 | 0.04449 | -0.00847 | 2 | 6 |
| 3 | 472 | 0.05932 | 0.05508 | -0.00424 | 2 | 4 |
| 5 | 472 | 0.06992 | 0.06568 | -0.00424 | 7 | 9 |
| 10 | 472 | 0.09534 | 0.07627 | -0.01907 | 8 | 17 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1400 | 0.06571 | 0.04643 | -0.01929 | 27 | 54 |
| 3 | 1400 | 0.12357 | 0.09000 | -0.03357 | 42 | 89 |
| 5 | 1400 | 0.16929 | 0.12000 | -0.04929 | 45 | 114 |
| 10 | 1400 | 0.23571 | 0.17786 | -0.05786 | 54 | 135 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 63 | 0.42857 | 0.33333 | 11.44444 |
| worsened by hierarchy | 99 | 1.00000 | 1.00000 | 10.40404 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 81 | 0.55556 | 0.33333 | 14.16049 |
| worsened by hierarchy | 145 | 1.00000 | 1.00000 | 12.87586 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 97 | 0.54639 | 0.29897 | 12.38144 |
| worsened by hierarchy | 179 | 1.00000 | 1.00000 | 13.30726 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 135 | 0.59259 | 0.25185 | 9.36296 |
| worsened by hierarchy | 226 | 1.00000 | 1.00000 | 11.82743 |

## Main Takeaways

- The positive effect of `hierarchy` is not a uniform top-k gain. It is concentrated in short-range ranking and crowded local structures.
- The strongest positive evidence remains on high-fanout `same_l2`-like cases, where moving the target upward inside a hard local neighborhood matters most.
- The strongest negative evidence is rank-drop on examples that the baseline already kept in the correct local neighborhood. This is the clearest sign that the current tokenizer-to-SFT transfer is still imperfect.
