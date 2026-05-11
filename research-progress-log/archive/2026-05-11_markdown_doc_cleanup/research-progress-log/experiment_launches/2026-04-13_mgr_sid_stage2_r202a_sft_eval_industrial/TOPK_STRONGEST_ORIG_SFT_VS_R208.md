# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.06706 | 0.06552 | -0.00154 | 0.26715 | 0.16987 | 0.13192 | 0.10015 | 51 | 58 |
| 3 | 0.09839 | 0.09729 | -0.00110 | 0.33245 | 0.22171 | 0.16656 | 0.14361 | 105 | 110 |
| 5 | 0.11824 | 0.11383 | -0.00441 | 0.36819 | 0.25480 | 0.18266 | 0.15972 | 118 | 138 |
| 10 | 0.15089 | 0.14251 | -0.00838 | 0.43062 | 0.31171 | 0.21090 | 0.18906 | 133 | 171 |
| 20 | 0.18972 | 0.18001 | -0.00971 | 0.51335 | 0.39135 | 0.24796 | 0.22943 | 175 | 219 |
| 50 | 0.24531 | 0.23737 | -0.00794 | 0.63402 | 0.53430 | 0.31480 | 0.29495 | 234 | 270 |

## Reading

- `hierarchy` is strongest on short-range metrics: `top3` improves most clearly, and `top5` remains slightly positive.
- `hierarchy` reduces same-prefix presence in the beam at almost every `k`, which means its gain does not come from retaining more nearby candidates; it comes from cleaning up some hard local neighborhoods while losing others.
- This already suggests the core tradeoff: better local disambiguation on difficult cases, but weaker neighborhood retention on some examples that the baseline already handled well.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 246 | 21 | 6 | 10 | 6 | 8 | 7 |
| 2-3 | 27 | 42 | 17 | 13 | 12 | 10 | 21 |
| 4-5 | 8 | 24 | 7 | 14 | 11 | 12 | 14 |
| 6-10 | 4 | 28 | 16 | 30 | 19 | 17 | 34 |
| 11-20 | 4 | 10 | 11 | 20 | 35 | 36 | 60 |
| 21-50 | 4 | 8 | 7 | 21 | 31 | 47 | 134 |
| >50 | 4 | 11 | 11 | 22 | 56 | 130 | 3187 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2661 | 0.07027 | 0.06877 | -0.00150 | 23 | 27 |
| 3 | 2661 | 0.09207 | 0.08230 | -0.00977 | 23 | 49 |
| 5 | 2661 | 0.09996 | 0.09094 | -0.00902 | 31 | 55 |
| 10 | 2661 | 0.11612 | 0.10823 | -0.00789 | 54 | 75 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 472 | 0.05297 | 0.05297 | +0.00000 | 3 | 3 |
| 3 | 472 | 0.05932 | 0.06568 | +0.00636 | 7 | 4 |
| 5 | 472 | 0.06992 | 0.07839 | +0.00847 | 11 | 7 |
| 10 | 472 | 0.09534 | 0.10381 | +0.00847 | 17 | 13 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1400 | 0.06571 | 0.06357 | -0.00214 | 25 | 28 |
| 3 | 1400 | 0.12357 | 0.13643 | +0.01286 | 75 | 57 |
| 5 | 1400 | 0.16929 | 0.16929 | +0.00000 | 76 | 76 |
| 10 | 1400 | 0.23571 | 0.22071 | -0.01500 | 62 | 83 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 51 | 0.49020 | 0.29412 | 10.72549 |
| worsened by hierarchy | 58 | 1.00000 | 1.00000 | 9.75862 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 105 | 0.76190 | 0.52381 | 15.32381 |
| worsened by hierarchy | 110 | 1.00000 | 1.00000 | 10.90000 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 118 | 0.71186 | 0.47458 | 13.75424 |
| worsened by hierarchy | 138 | 1.00000 | 1.00000 | 12.21014 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 133 | 0.65414 | 0.36842 | 9.59398 |
| worsened by hierarchy | 171 | 1.00000 | 1.00000 | 10.84795 |

## Main Takeaways

- The positive effect of `hierarchy` is not a uniform top-k gain. It is concentrated in short-range ranking and crowded local structures.
- The strongest positive evidence remains on high-fanout `same_l2`-like cases, where moving the target upward inside a hard local neighborhood matters most.
- The strongest negative evidence is rank-drop on examples that the baseline already kept in the correct local neighborhood. This is the clearest sign that the current tokenizer-to-SFT transfer is still imperfect.
