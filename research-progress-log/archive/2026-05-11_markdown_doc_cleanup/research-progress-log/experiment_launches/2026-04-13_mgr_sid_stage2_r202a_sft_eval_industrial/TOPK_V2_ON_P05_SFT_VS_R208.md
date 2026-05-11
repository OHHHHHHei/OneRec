# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.07059 | 0.06552 | -0.00507 | 0.15266 | 0.16987 | 0.10104 | 0.10015 | 48 | 71 |
| 3 | 0.09508 | 0.09729 | +0.00221 | 0.23075 | 0.22171 | 0.15553 | 0.14361 | 102 | 92 |
| 5 | 0.11471 | 0.11383 | -0.00088 | 0.25789 | 0.25480 | 0.17472 | 0.15972 | 111 | 115 |
| 10 | 0.14626 | 0.14251 | -0.00375 | 0.31216 | 0.31171 | 0.20296 | 0.18906 | 130 | 147 |
| 20 | 0.18244 | 0.18001 | -0.00243 | 0.38143 | 0.39135 | 0.24112 | 0.22943 | 176 | 187 |
| 50 | 0.24818 | 0.23737 | -0.01081 | 0.49327 | 0.53430 | 0.31149 | 0.29495 | 227 | 276 |

## Reading

- `hierarchy` is strongest on short-range metrics: `top3` improves most clearly, and `top5` remains slightly positive.
- `hierarchy` reduces same-prefix presence in the beam at almost every `k`, which means its gain does not come from retaining more nearby candidates; it comes from cleaning up some hard local neighborhoods while losing others.
- This already suggests the core tradeoff: better local disambiguation on difficult cases, but weaker neighborhood retention on some examples that the baseline already handled well.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 249 | 37 | 8 | 8 | 4 | 8 | 6 |
| 2-3 | 22 | 31 | 16 | 10 | 12 | 10 | 10 |
| 4-5 | 9 | 22 | 11 | 19 | 7 | 6 | 15 |
| 6-10 | 7 | 17 | 12 | 38 | 21 | 13 | 35 |
| 11-20 | 4 | 12 | 13 | 16 | 35 | 35 | 49 |
| 21-50 | 1 | 12 | 7 | 19 | 40 | 58 | 161 |
| >50 | 5 | 13 | 8 | 20 | 51 | 130 | 3181 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3141 | 0.07673 | 0.06909 | -0.00764 | 19 | 43 |
| 3 | 3141 | 0.09551 | 0.08819 | -0.00732 | 34 | 57 |
| 5 | 3141 | 0.10602 | 0.09869 | -0.00732 | 45 | 68 |
| 10 | 3141 | 0.12639 | 0.12003 | -0.00637 | 68 | 88 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 382 | 0.09686 | 0.06806 | -0.02880 | 4 | 15 |
| 3 | 382 | 0.11518 | 0.10733 | -0.00785 | 5 | 8 |
| 5 | 382 | 0.12565 | 0.12304 | -0.00262 | 7 | 8 |
| 10 | 382 | 0.14398 | 0.14660 | +0.00262 | 10 | 9 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1010 | 0.04158 | 0.05347 | +0.01188 | 25 | 13 |
| 3 | 1010 | 0.08614 | 0.12178 | +0.03564 | 63 | 27 |
| 5 | 1010 | 0.13762 | 0.15743 | +0.01980 | 59 | 39 |
| 10 | 1010 | 0.20891 | 0.21089 | +0.00198 | 52 | 50 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 48 | 0.22917 | 0.16667 | 10.41667 |
| worsened by hierarchy | 71 | 1.00000 | 1.00000 | 4.57746 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 102 | 0.64706 | 0.47059 | 11.71569 |
| worsened by hierarchy | 92 | 1.00000 | 1.00000 | 6.93478 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 111 | 0.60360 | 0.44144 | 9.95495 |
| worsened by hierarchy | 115 | 1.00000 | 1.00000 | 7.17391 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 130 | 0.53846 | 0.35385 | 7.71538 |
| worsened by hierarchy | 147 | 1.00000 | 1.00000 | 6.97959 |

## Main Takeaways

- The positive effect of `hierarchy` is not a uniform top-k gain. It is concentrated in short-range ranking and crowded local structures.
- The strongest positive evidence remains on high-fanout `same_l2`-like cases, where moving the target upward inside a hard local neighborhood matters most.
- The strongest negative evidence is rank-drop on examples that the baseline already kept in the correct local neighborhood. This is the clearest sign that the current tokenizer-to-SFT transfer is still imperfect.
