# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.06309 | 0.06265 | -0.00044 | 0.23318 | 0.17847 | 0.11229 | 0.09729 | 63 | 65 |
| 3 | 0.08824 | 0.09287 | +0.00463 | 0.31480 | 0.25149 | 0.15619 | 0.14604 | 119 | 98 |
| 5 | 0.10743 | 0.10788 | +0.00044 | 0.35429 | 0.28657 | 0.17648 | 0.16942 | 125 | 123 |
| 10 | 0.13435 | 0.13038 | -0.00397 | 0.42841 | 0.34128 | 0.20781 | 0.19921 | 154 | 172 |
| 20 | 0.17362 | 0.16744 | -0.00618 | 0.51555 | 0.40922 | 0.24619 | 0.23781 | 202 | 230 |
| 50 | 0.24244 | 0.23605 | -0.00640 | 0.67240 | 0.52482 | 0.32716 | 0.30774 | 263 | 292 |

## Reading

- `hierarchy` is strongest on short-range metrics: `top3` improves most clearly, and `top5` remains slightly positive.
- `hierarchy` reduces same-prefix presence in the beam at almost every `k`, which means its gain does not come from retaining more nearby candidates; it comes from cleaning up some hard local neighborhoods while losing others.
- This already suggests the core tradeoff: better local disambiguation on difficult cases, but weaker neighborhood retention on some examples that the baseline already handled well.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 221 | 30 | 7 | 6 | 5 | 9 | 8 |
| 2-3 | 24 | 27 | 11 | 9 | 10 | 18 | 15 |
| 4-5 | 13 | 19 | 12 | 12 | 10 | 12 | 9 |
| 6-10 | 4 | 22 | 7 | 13 | 20 | 26 | 30 |
| 11-20 | 5 | 12 | 12 | 12 | 34 | 45 | 58 |
| 21-50 | 7 | 16 | 6 | 20 | 32 | 59 | 172 |
| >50 | 10 | 11 | 13 | 30 | 57 | 142 | 3171 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2259 | 0.02435 | 0.02302 | -0.00133 | 15 | 18 |
| 3 | 2259 | 0.03630 | 0.03453 | -0.00177 | 22 | 26 |
| 5 | 2259 | 0.04382 | 0.04338 | -0.00044 | 33 | 34 |
| 10 | 2259 | 0.05578 | 0.05799 | +0.00221 | 53 | 48 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 778 | 0.20566 | 0.19794 | -0.00771 | 18 | 24 |
| 3 | 778 | 0.24036 | 0.24936 | +0.00900 | 27 | 20 |
| 5 | 778 | 0.26093 | 0.27121 | +0.01028 | 28 | 20 |
| 10 | 778 | 0.29177 | 0.29177 | +0.00000 | 28 | 28 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1496 | 0.04746 | 0.05214 | +0.00468 | 30 | 23 |
| 3 | 1496 | 0.08757 | 0.09960 | +0.01203 | 70 | 52 |
| 5 | 1496 | 0.12366 | 0.12032 | -0.00334 | 64 | 69 |
| 10 | 1496 | 0.17112 | 0.15575 | -0.01537 | 73 | 96 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 63 | 0.44444 | 0.19048 | 7.31746 |
| worsened by hierarchy | 65 | 1.00000 | 1.00000 | 4.80000 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 119 | 0.66387 | 0.36134 | 10.32773 |
| worsened by hierarchy | 98 | 1.00000 | 1.00000 | 8.27551 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 125 | 0.69600 | 0.32800 | 9.16800 |
| worsened by hierarchy | 123 | 1.00000 | 1.00000 | 8.58537 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 154 | 0.68831 | 0.33766 | 8.08442 |
| worsened by hierarchy | 172 | 1.00000 | 1.00000 | 8.82558 |

## Main Takeaways

- The positive effect of `hierarchy` is not a uniform top-k gain. It is concentrated in short-range ranking and crowded local structures.
- The strongest positive evidence remains on high-fanout `same_l2`-like cases, where moving the target upward inside a hard local neighborhood matters most.
- The strongest negative evidence is rank-drop on examples that the baseline already kept in the correct local neighborhood. This is the clearest sign that the current tokenizer-to-SFT transfer is still imperfect.
