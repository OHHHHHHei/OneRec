# R693a Focus Family SID Analysis（重点物品族 SID 分配分析）

Status（状态）: `diagnostic（诊断）`

## Overall（整体）

| variant   |   active_l1 |   unique_l2_pairs |   collision_count |   weighted_top_family_purity |   l1_max_bucket_size |
|:----------|------------:|------------------:|------------------:|-----------------------------:|---------------------:|
| original  |          48 |              2295 |                16 |                       0.7477 |                  247 |
| v2        |         203 |              2680 |                13 |                       0.7968 |                   97 |
| R650a     |         199 |              2782 |                11 |                       0.7789 |                   74 |
| R680a     |         226 |              2833 |                11 |                       0.8060 |                  117 |
| R690a     |         118 |              2527 |                11 |                       0.7821 |                  214 |
| R690b     |          33 |              1989 |                14 |                       0.7157 |                  233 |
| R693a     |          90 |              2274 |                12 |                       0.7778 |                  212 |

## Active L1 by family（各物品族覆盖的第一层码数量）

| family            |   original |   R680a |   R690a |   R690b |   R693a |
|:------------------|-----------:|--------:|--------:|--------:|--------:|
| 3d_filament       |         21 |      32 |      31 |      15 |      33 |
| adhesive_epoxy    |         14 |      26 |      18 |      10 |      16 |
| connector_fitting |         19 |      47 |      24 |      15 |      24 |
| fastener          |         27 |      46 |      33 |      19 |      29 |
| gauge_meter       |         29 |      80 |      55 |      23 |      44 |
| tape              |         22 |      39 |      28 |      17 |      23 |
| test_strip        |          3 |       2 |       2 |       3 |       2 |
| ventilation_fan   |         24 |      33 |      29 |      18 |      26 |

## Top L1 coverage（最大第一层桶覆盖该族比例）

| family            |   original |   R680a |   R690a |   R690b |   R693a |
|:------------------|-----------:|--------:|--------:|--------:|--------:|
| 3d_filament       |      0.593 |   0.301 |   0.508 |   0.570 |   0.544 |
| adhesive_epoxy    |      0.472 |   0.152 |   0.304 |   0.640 |   0.272 |
| connector_fitting |      0.343 |   0.091 |   0.318 |   0.343 |   0.358 |
| fastener          |      0.626 |   0.102 |   0.380 |   0.626 |   0.422 |
| gauge_meter       |      0.124 |   0.083 |   0.121 |   0.139 |   0.136 |
| tape              |      0.530 |   0.124 |   0.479 |   0.760 |   0.714 |
| test_strip        |      0.556 |   0.778 |   0.778 |   0.667 |   0.778 |
| ventilation_fan   |      0.155 |   0.121 |   0.155 |   0.172 |   0.190 |

## Top L1 bucket purity（该最大桶自身纯度）

| family            |   original |   R680a |   R690a |   R690b |   R693a |
|:------------------|-----------:|--------:|--------:|--------:|--------:|
| 3d_filament       |      0.927 |   0.991 |   0.916 |   0.944 |   0.991 |
| adhesive_epoxy    |      0.819 |   1.000 |   0.844 |   0.485 |   0.872 |
| connector_fitting |      0.790 |   1.000 |   0.784 |   0.662 |   0.907 |
| fastener          |      0.613 |   1.000 |   0.780 |   0.722 |   0.929 |
| gauge_meter       |      0.420 |   0.966 |   0.446 |   0.331 |   0.548 |
| tape              |      0.983 |   1.000 |   0.990 |   0.907 |   0.969 |
| test_strip        |      0.045 |   0.206 |   0.226 |   0.055 |   0.212 |
| ventilation_fan   |      0.095 |   0.333 |   0.290 |   0.130 |   0.141 |

## R693a Top Buckets（R693a 重点族主桶）

### 3d_filament
|   rank | l1_code   |   family_items_in_bucket |   family_coverage |   bucket_size |   bucket_family_purity |   unique_l2_for_family_inside_bucket | bucket_family_composition_top    |
|-------:|:----------|-------------------------:|------------------:|--------------:|-----------------------:|-------------------------------------:|:---------------------------------|
|      1 | <a_178>   |                      210 |             0.544 |           212 |                  0.991 |                                   46 | 3d_filament:210; other:2         |
|      2 | <a_220>   |                       55 |             0.142 |            58 |                  0.948 |                                   20 | 3d_filament:55; other:3          |
|      3 | <a_212>   |                       31 |             0.080 |            44 |                  0.705 |                                   24 | 3d_filament:31; other:12; tape:1 |

### connector_fitting
|   rank | l1_code   |   family_items_in_bucket |   family_coverage |   bucket_size |   bucket_family_purity |   unique_l2_for_family_inside_bucket | bucket_family_composition_top                               |
|-------:|:----------|-------------------------:|------------------:|--------------:|-----------------------:|-------------------------------------:|:------------------------------------------------------------|
|      1 | <a_40>    |                       98 |             0.358 |           108 |                  0.907 |                                   46 | connector_fitting:98; gauge_meter:5; other:4; 3d_filament:1 |
|      2 | <a_166>   |                       46 |             0.168 |            46 |                  1.000 |                                   16 | connector_fitting:46                                        |
|      3 | <a_231>   |                       42 |             0.153 |            59 |                  0.712 |                                   26 | connector_fitting:42; other:17                              |

### gauge_meter
|   rank | l1_code   |   family_items_in_bucket |   family_coverage |   bucket_size |   bucket_family_purity |   unique_l2_for_family_inside_bucket | bucket_family_composition_top                              |
|-------:|:----------|-------------------------:|------------------:|--------------:|-----------------------:|-------------------------------------:|:-----------------------------------------------------------|
|      1 | <a_69>    |                       46 |             0.136 |            84 |                  0.548 |                                   20 | gauge_meter:46; other:37; connector_fitting:1              |
|      2 | <a_197>   |                       35 |             0.104 |            48 |                  0.729 |                                   25 | gauge_meter:35; other:13                                   |
|      3 | <a_190>   |                       31 |             0.092 |            64 |                  0.484 |                                   16 | gauge_meter:31; other:30; 3d_filament:2; ventilation_fan:1 |

### tape
|   rank | l1_code   |   family_items_in_bucket |   family_coverage |   bucket_size |   bucket_family_purity |   unique_l2_for_family_inside_bucket | bucket_family_composition_top                                                        |
|-------:|:----------|-------------------------:|------------------:|--------------:|-----------------------:|-------------------------------------:|:-------------------------------------------------------------------------------------|
|      1 | <a_31>    |                      155 |             0.714 |           160 |                  0.969 |                                   55 | tape:155; other:3; 3d_filament:2                                                     |
|      2 | <a_157>   |                       12 |             0.055 |            51 |                  0.235 |                                    9 | other:25; tape:12; ventilation_fan:5; gauge_meter:4; adhesive_epoxy:3; 3d_filament:2 |
|      3 | <a_109>   |                        6 |             0.028 |            33 |                  0.182 |                                    6 | other:20; tape:6; connector_fitting:3; gauge_meter:2; 3d_filament:1; fastener:1      |

### adhesive_epoxy
|   rank | l1_code   |   family_items_in_bucket |   family_coverage |   bucket_size |   bucket_family_purity |   unique_l2_for_family_inside_bucket | bucket_family_composition_top                  |
|-------:|:----------|-------------------------:|------------------:|--------------:|-----------------------:|-------------------------------------:|:-----------------------------------------------|
|      1 | <a_124>   |                       34 |             0.272 |            39 |                  0.872 |                                   25 | adhesive_epoxy:34; other:5                     |
|      2 | <a_78>    |                       26 |             0.208 |            26 |                  1.000 |                                   18 | adhesive_epoxy:26                              |
|      3 | <a_180>   |                       18 |             0.144 |            30 |                  0.600 |                                   16 | adhesive_epoxy:18; other:7; fastener:4; tape:1 |

### fastener
|   rank | l1_code   |   family_items_in_bucket |   family_coverage |   bucket_size |   bucket_family_purity |   unique_l2_for_family_inside_bucket | bucket_family_composition_top                                                              |
|-------:|:----------|-------------------------:|------------------:|--------------:|-----------------------:|-------------------------------------:|:-------------------------------------------------------------------------------------------|
|      1 | <a_235>   |                       79 |             0.422 |            85 |                  0.929 |                                   38 | fastener:79; other:6                                                                       |
|      2 | <a_68>    |                       34 |             0.182 |            72 |                  0.472 |                                   24 | fastener:34; other:33; gauge_meter:5                                                       |
|      3 | <a_209>   |                        9 |             0.048 |            75 |                  0.120 |                                    7 | other:43; connector_fitting:14; fastener:9; gauge_meter:5; 3d_filament:3; adhesive_epoxy:1 |

### ventilation_fan
|   rank | l1_code   |   family_items_in_bucket |   family_coverage |   bucket_size |   bucket_family_purity |   unique_l2_for_family_inside_bucket | bucket_family_composition_top                                                        |
|-------:|:----------|-------------------------:|------------------:|--------------:|-----------------------:|-------------------------------------:|:-------------------------------------------------------------------------------------|
|      1 | <a_90>    |                       11 |             0.190 |            78 |                  0.141 |                                    6 | other:60; ventilation_fan:11; gauge_meter:4; tape:1; fastener:1; connector_fitting:1 |
|      2 | <a_35>    |                        8 |             0.138 |            23 |                  0.348 |                                    5 | other:13; ventilation_fan:8; connector_fitting:2                                     |
|      3 | <a_157>   |                        5 |             0.086 |            51 |                  0.098 |                                    4 | other:25; tape:12; ventilation_fan:5; gauge_meter:4; adhesive_epoxy:3; 3d_filament:2 |

### test_strip
|   rank | l1_code   |   family_items_in_bucket |   family_coverage |   bucket_size |   bucket_family_purity |   unique_l2_for_family_inside_bucket | bucket_family_composition_top                                        |
|-------:|:----------|-------------------------:|------------------:|--------------:|-----------------------:|-------------------------------------:|:---------------------------------------------------------------------|
|      1 | <a_153>   |                        7 |             0.778 |            33 |                  0.212 |                                    6 | other:16; gauge_meter:9; test_strip:7; ventilation_fan:1             |
|      2 | <a_159>   |                        2 |             0.222 |            25 |                  0.080 |                                    2 | other:20; test_strip:2; ventilation_fan:1; fastener:1; gauge_meter:1 |
