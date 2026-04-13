# Top-k Structural Error Analysis

## Scope

This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.

The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.

## Top-k Summary

| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.07037 | 0.05912 | -0.01125 | 0.13964 | 0.13457 | 0.10390 | 0.09398 | 60 | 111 |
| 3 | 0.09420 | 0.08427 | -0.00993 | 0.20913 | 0.20803 | 0.13898 | 0.14273 | 103 | 148 |
| 5 | 0.11030 | 0.10015 | -0.01015 | 0.24840 | 0.24377 | 0.16192 | 0.16148 | 122 | 168 |
| 10 | 0.14251 | 0.13082 | -0.01169 | 0.30532 | 0.29539 | 0.19590 | 0.19060 | 158 | 211 |
| 20 | 0.18045 | 0.16391 | -0.01655 | 0.38120 | 0.35672 | 0.23450 | 0.22259 | 157 | 232 |
| 50 | 0.24289 | 0.21597 | -0.02691 | 0.49967 | 0.46636 | 0.30333 | 0.28392 | 205 | 327 |

## Reading

- `hierarchy` is strongest on short-range metrics: `top3` improves most clearly, and `top5` remains slightly positive.
- `hierarchy` reduces same-prefix presence in the beam at almost every `k`, which means its gain does not come from retaining more nearby candidates; it comes from cleaning up some hard local neighborhoods while losing others.
- This already suggests the core tradeoff: better local disambiguation on difficult cases, but weaker neighborhood retention on some examples that the baseline already handled well.

## Rank Transition Matrix

Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.

| baseline \\ hierarchy | 1 | 2-3 | 4-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 208 | 39 | 18 | 15 | 13 | 4 | 22 |
| 2-3 | 18 | 14 | 13 | 23 | 14 | 10 | 16 |
| 4-5 | 5 | 11 | 6 | 10 | 9 | 16 | 16 |
| 6-10 | 23 | 16 | 3 | 13 | 28 | 24 | 39 |
| 11-20 | 6 | 17 | 11 | 29 | 24 | 25 | 60 |
| 21-50 | 3 | 8 | 11 | 16 | 26 | 45 | 174 |
| >50 | 5 | 9 | 10 | 33 | 36 | 112 | 3227 |

## Fanout Bucket Analysis

Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.

### l2<=2

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3141 | 0.07800 | 0.06940 | -0.00860 | 36 | 63 |
| 3 | 3141 | 0.09169 | 0.09137 | -0.00032 | 63 | 64 |
| 5 | 3141 | 0.10474 | 0.10188 | -0.00287 | 67 | 76 |
| 10 | 3141 | 0.12671 | 0.12225 | -0.00446 | 79 | 93 |

### l2=3

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 382 | 0.06021 | 0.04712 | -0.01309 | 3 | 8 |
| 3 | 382 | 0.10209 | 0.07068 | -0.03141 | 4 | 16 |
| 5 | 382 | 0.11780 | 0.08639 | -0.03141 | 6 | 18 |
| 10 | 382 | 0.14921 | 0.12042 | -0.02880 | 10 | 21 |

### l2>=4

| k | count | baseline hit | hierarchy hit | delta | improved | worsened |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1010 | 0.05050 | 0.03168 | -0.01881 | 21 | 40 |
| 3 | 1010 | 0.09901 | 0.06733 | -0.03168 | 36 | 68 |
| 5 | 1010 | 0.12475 | 0.10000 | -0.02475 | 49 | 74 |
| 10 | 1010 | 0.18911 | 0.16139 | -0.02772 | 69 | 97 |

## Improved vs Worsened Sets

### top1

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 60 | 0.21667 | 0.20000 | 6.91667 |
| worsened by hierarchy | 111 | 1.00000 | 1.00000 | 7.36036 |

### top3

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 103 | 0.36893 | 0.22330 | 7.85437 |
| worsened by hierarchy | 148 | 1.00000 | 1.00000 | 9.18243 |

### top5

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 122 | 0.45082 | 0.28689 | 8.73770 |
| worsened by hierarchy | 168 | 1.00000 | 1.00000 | 8.42262 |

### top10

| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 158 | 0.53797 | 0.36076 | 9.27848 |
| worsened by hierarchy | 211 | 1.00000 | 1.00000 | 8.72986 |

## Main Takeaways

- The positive effect of `hierarchy` is not a uniform top-k gain. It is concentrated in short-range ranking and crowded local structures.
- The strongest positive evidence remains on high-fanout `same_l2`-like cases, where moving the target upward inside a hard local neighborhood matters most.
- The strongest negative evidence is rank-drop on examples that the baseline already kept in the correct local neighborhood. This is the clearest sign that the current tokenizer-to-SFT transfer is still imperfect.
