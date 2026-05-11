# R693a vs Original MiniOneRec Primary Error Analysis

这份报告把原版 MiniOneRec strongest SFT（最强监督微调）作为第一优先级 baseline（基线）。v2/R680a 等内部实验只用于机制参考，不作为主要胜负判断。

## 1. Main Metrics
| variant   |    n |   NDCG@1 |   NDCG@3 |   NDCG@5 |   NDCG@10 |     HR@1 |     HR@3 |     HR@5 |    HR@10 |    HR@50 |   top10_has_l1 |   top10_has_l2 |   top1_same_family |   top1_history_repeat |   target_l1_in_history |   target_l2_in_history |
|:----------|-----:|---------:|---------:|---------:|----------:|---------:|---------:|---------:|---------:|---------:|---------------:|---------------:|-------------------:|----------------------:|-----------------------:|-----------------------:|
| original  | 4533 | 0.067064 | 0.085008 | 0.093153 |  0.103720 | 0.067064 | 0.098390 | 0.118244 | 0.150893 | 0.245312 |       0.430620 |       0.210898 |           0.515773 |              0.429517 |               0.401941 |               0.172292 |
| r693a     | 4533 | 0.063093 | 0.079719 | 0.086782 |  0.097308 | 0.063093 | 0.091551 | 0.108758 | 0.141628 | 0.243106 |       0.385396 |       0.201632 |           0.506949 |              0.401059 |               0.344584 |               0.162144 |

## 2. Failure Decomposition
| variant   | failure_type                   |   count |     rate |
|:----------|:-------------------------------|--------:|---------:|
| original  | hit@10                         |     684 | 0.150893 |
| original  | L1_miss                        |    2581 | 0.569380 |
| original  | L2_miss_after_L1_hit           |     996 | 0.219722 |
| original  | leaf_or_rank_miss_after_L2_hit |     272 | 0.060004 |
| r693a     | hit@10                         |     642 | 0.141628 |
| r693a     | L1_miss                        |    2786 | 0.614604 |
| r693a     | L2_miss_after_L1_hit           |     833 | 0.183764 |
| r693a     | leaf_or_rank_miss_after_L2_hit |     272 | 0.060004 |

## 3. Hit Overlap
|   k |   both_hit |   r693_only_hit |   original_only_hit |   both_miss |   net_r693_minus_original_hits |
|----:|-----------:|----------------:|--------------------:|------------:|-------------------------------:|
|   1 |        230 |              56 |                  74 |        4173 |                            -18 |
|   3 |        323 |              92 |                 123 |        3995 |                            -31 |
|   5 |        386 |             107 |                 150 |        3890 |                            -43 |
|  10 |        504 |             138 |                 180 |        3711 |                            -42 |
|  50 |        849 |             253 |                 263 |        3168 |                            -10 |

## 4. Family Compare
| family            |    n |   orig_NDCG@10 |   r693_NDCG@10 |   delta_NDCG@10 |   orig_HR@1 |   r693_HR@1 |   delta_HR@1 |   orig_HR@10 |   r693_HR@10 |   delta_HR@10 |   r693_only_hit10_count |   original_only_hit10_count |
|:------------------|-----:|---------------:|---------------:|----------------:|------------:|------------:|-------------:|-------------:|-------------:|--------------:|------------------------:|----------------------------:|
| other             | 1873 |       0.057588 |       0.049228 |       -0.008360 |    0.042712 |    0.036305 |    -0.006407 |     0.075280 |     0.067272 |     -0.008009 |                      30 |                          45 |
| 3d_filament       | 1101 |       0.103567 |       0.102749 |       -0.000818 |    0.024523 |    0.038147 |     0.013624 |     0.212534 |     0.188919 |     -0.023615 |                      53 |                          79 |
| gauge_meter       |  513 |       0.373789 |       0.361723 |       -0.012066 |    0.327485 |    0.302144 |    -0.025341 |     0.424951 |     0.423002 |     -0.001949 |                      21 |                          22 |
| tape              |  301 |       0.060580 |       0.054332 |       -0.006248 |    0.029900 |    0.023256 |    -0.006645 |     0.099668 |     0.093023 |     -0.006645 |                       8 |                          10 |
| connector_fitting |  273 |       0.078706 |       0.067894 |       -0.010812 |    0.043956 |    0.040293 |    -0.003663 |     0.120879 |     0.113553 |     -0.007326 |                       9 |                          11 |
| adhesive_epoxy    |  224 |       0.034700 |       0.027124 |       -0.007576 |    0.017857 |    0.008929 |    -0.008929 |     0.058036 |     0.053571 |     -0.004464 |                       8 |                           9 |
| fastener          |  142 |       0.015258 |       0.007476 |       -0.007782 |    0.007042 |    0.000000 |    -0.007042 |     0.028169 |     0.014085 |     -0.014085 |                       1 |                           3 |

## 5. Primary Interpretation

- Compared with original, R693a loses at every important cutoff: @1, @3, @5, and @10. The gap is largest in ranking-sensitive early positions, especially HR@1 / NDCG@1.
- R693a has more correct L1-prefix coverage than v2/R680a in internal comparisons, but this does not matter enough against the original baseline because original still has stronger exact ranking and better top10 hit count.
- R693a loses 42 net hit@10 cases to original: original-only hit@10 = 180, R693a-only hit@10 = 138.
- The main absolute failure remains L1 routing miss（第一层路由失败）, but the original comparison shows another critical issue: even when R693a reaches the right broad family, it often fails at exact item ranking（精确物品排序） within similar items.

## 6. Artifacts
- `R693A_VS_ORIGINAL_PRIMARY_DIAGNOSIS_SUMMARY.csv`
- `R693A_VS_ORIGINAL_PRIMARY_DIAGNOSIS_FAILURE_TYPES.csv`
- `R693A_VS_ORIGINAL_PRIMARY_DIAGNOSIS_OVERLAP.csv`
- `R693A_VS_ORIGINAL_PRIMARY_DIAGNOSIS_FAMILY_COMPARE.csv`
- `R693A_VS_ORIGINAL_PRIMARY_DIAGNOSIS_CASES.csv`