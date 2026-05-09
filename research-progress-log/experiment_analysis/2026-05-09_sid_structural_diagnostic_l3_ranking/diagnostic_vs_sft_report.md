# SID Structural Diagnostic（语义标识结构诊断）

## Pre-registered Rules（预注册规则）

- 诊断只使用 tokenizer（分词器）、semantic embedding（语义嵌入）和 train interaction（训练交互）构造的 pair（物品对）。
- SFT（监督微调）指标只用于事后对照，不参与四轴 verdict（裁决）。
- 这不是单指标 predictor（预测器），而是 multi-axis diagnostic（多轴诊断）。

- `l1_stability`: fail（失败） if active_l1 > 150 or top5_l1_cover < 300 or S-near C-near same_l1 < 80; warn（警告） if active_l1 > 100 or top5_l1_cover < 500 or S-near C-near same_l1 < 86.
- `selective_separation`: fail（失败） if S-near C-far same_l1 < 80 or split_after_l1 < 55 or same_l12 > 32.5; warn（警告） if split_after_l1 < 60 or same_l12 > 30.5.
- `collaborative_preservation`: fail（失败） if S-near C-near same_l1 < 80 or same_l12 < 27; warn（警告） if S-near C-near same_l1 < 86 or same_l12 < 31.
- `learnability`: fail（失败） if active_l1 > 180 or unique_l12 > 2750 or top5_l1_cover < 300 or catalog_l12_zero_train_pct > 50 or catalog_sid_zero_train_pct > 80; warn（警告） if active_l1 > 100 or unique_l12 > 2550 or top5_l1_cover < 500 or l12_singletons > 2000 or catalog_l12_zero_train_pct > 25 or catalog_sid_zero_train_pct > 55.

## Diagnostic vs SFT（诊断对监督微调）

| label                        | split       | diagnostic_profile         | predicted_sft_band   | actual_sft_band   | diagnostic_consistency   |   ndcg_at_10 |   hr_at_10 | l1_axis   | separation_axis   | preservation_axis   | learnability_axis   |
|:-----------------------------|:------------|:---------------------------|:---------------------|:------------------|:-------------------------|-------------:|-----------:|:----------|:------------------|:--------------------|:--------------------|
| R690b L2=0.010 main          | calibration | balanced-positive          | high                 | high              | match                    |       0.1044 |     0.1469 | pass      | pass              | pass                | pass                |
| R690b L2=0.003 weak          | calibration | under-separated            | medium               | low               | partial                  |       0.0957 |     0.1372 | pass      | warn              | pass                | pass                |
| R690b L2=0.005 weak          | calibration | over-separated-unstable    | low                  | low               | match                    |       0.0944 |     0.1315 | fail      | pass              | fail                | warn                |
| R690b L2=0.015 fragmented    | calibration | separating-but-risky       | low                  | low               | match                    |       0.0941 |     0.1350 | warn      | pass              | warn                | warn                |
| R690b no L1 semantic         | calibration | over-separated-unstable    | low                  | low               | match                    |       0.0938 |     0.1319 | warn      | pass              | fail                | warn                |
| R690b L3=0.010 pending       | prospective | balanced-positive          | high                 | medium            | partial                  |       0.0975 |     0.1427 | pass      | pass              | pass                | pass                |
| R690b L3=0.005 pending       | prospective | separating-but-risky       | low                  | pending           | pending                  |     nan      |   nan      | pass      | pass              | warn                | pass                |
| R690b L3=0.015 gate-failed   | prospective | structurally-risky         | low                  | pending           | pending                  |     nan      |   nan      | warn      | fail              | warn                | pass                |
| R690b L3 ranking             | prospective | borderline                 | medium               | pending           | pending                  |     nan      |   nan      | warn      | warn              | warn                | warn                |
| Original semantic            | validation  | semantic-stable-baseline   | high                 | high              | match                    |       0.1037 |     0.1509 | pass      | pass              | pass                | pass                |
| V2 offline                   | validation  | out-of-family-flat-routing | unknown              | high              | out-of-scope             |       0.1027 |     0.1463 | fail      | fail              | fail                | fail                |
| Original L2 multihop ranking | validation  | borderline                 | medium               | medium            | match                    |       0.1017 |     0.1474 | warn      | warn              | warn                | pass                |
| QCR L2 conflict ranking      | validation  | separating-but-risky       | low                  | medium            | partial                  |       0.0998 |     0.1388 | warn      | pass              | warn                | warn                |
| Stage3 prefix retained       | validation  | fragmented-routing         | low                  | medium            | partial                  |       0.0991 |     0.1385 | fail      | fail              | fail                | fail                |
| TAGCF attr mid               | validation  | fragmented-routing         | low                  | medium            | partial                  |       0.0976 |     0.1315 | fail      | fail              | fail                | fail                |
| V2 LMH mid=0.010             | validation  | fragmented-routing         | low                  | low               | match                    |       0.0952 |     0.1377 | fail      | fail              | fail                | fail                |

## Consistency Summary（一致性总结）

- available SFT（已有监督微调）样本数: 13; match（完全匹配）=7, partial（部分匹配）=5, mismatch（不匹配）=0, out-of-scope（超出适用域）=1.
- calibration（校准集）样本数: 5; match=4, partial=1, mismatch=0.

## Structure Distribution（码本结构分布）

| label                        |   active_l1 |   unique_l12 |   unique_sid |   collision_count |   top5_l1_cover |   l1_entropy_norm |   l1_gini |   avg_l2_per_l1 |   l12_singletons |   l12_ge5 |
|:-----------------------------|------------:|-------------:|-------------:|------------------:|----------------:|------------------:|----------:|----------------:|-----------------:|----------:|
| R690b L2=0.003 weak          |          56 |         2182 |         3673 |                13 |             683 |            0.9712 |    0.2668 |         38.9643 |             1443 |        87 |
| R690b L2=0.005 weak          |         131 |         2322 |         3671 |                15 |             488 |            0.9567 |    0.3404 |         17.7252 |             1611 |        68 |
| R690b L2=0.010 main          |          60 |         2330 |         3670 |                16 |             707 |            0.9630 |    0.3054 |         38.8333 |             1646 |        74 |
| R690b L2=0.015 fragmented    |         108 |         2619 |         3673 |                13 |             443 |            0.9616 |    0.3303 |         24.2500 |             2009 |        42 |
| R690b no L1 semantic         |         115 |         2649 |         3671 |                15 |             616 |            0.9514 |    0.3501 |         23.0348 |             2072 |        46 |
| R690b L3 ranking             |         105 |         2304 |         3675 |                11 |             598 |            0.9459 |    0.3873 |         21.9429 |             1612 |        76 |
| R690b L3=0.005 pending       |          83 |         2426 |         3675 |                11 |             532 |            0.9559 |    0.3476 |         29.2289 |             1735 |        62 |
| R690b L3=0.010 pending       |          60 |         2394 |         3673 |                13 |             738 |            0.9596 |    0.3120 |         39.9000 |             1724 |        65 |
| R690b L3=0.015 gate-failed   |          38 |          647 |         3579 |               107 |             844 |            0.9643 |    0.2820 |         17.0263 |              108 |       303 |
| Original L2 multihop ranking |          88 |         2449 |         3671 |                15 |             516 |            0.9569 |    0.3405 |         27.8295 |             1780 |        61 |
| Original semantic            |          48 |         2295 |         3670 |                16 |             833 |            0.9575 |    0.3108 |         47.8125 |             1604 |        73 |
| QCR L2 conflict ranking      |         117 |         2632 |         3675 |                11 |             496 |            0.9548 |    0.3419 |         22.4957 |             2057 |        45 |
| Stage3 prefix retained       |         256 |         2991 |         3675 |                11 |             203 |            0.9877 |    0.1916 |         11.6836 |             2543 |        23 |
| TAGCF attr mid               |         125 |         2570 |         3675 |                11 |             295 |            0.9889 |    0.1755 |         20.5600 |             1926 |        45 |
| V2 LMH mid=0.010             |         190 |         2722 |         3675 |                11 |             293 |            0.9803 |    0.2388 |         14.3263 |             2166 |        45 |
| V2 offline                   |         203 |         2680 |         3673 |                13 |             282 |            0.9760 |    0.2661 |         13.2020 |             2109 |        49 |

## Mid-Similarity Blind Spot（中等语义相似盲区）

S-mid（中等语义相似）使用语义相似度区间 `[0.80, 0.90)`，用于补充极端 S-near/S-far（语义近/远）之外的商品对。

| label                        |   smid_cfar_pair_count |   smid_cfar_same_l1 |   smid_cfar_same_l12 |   smid_cfar_split_after_l1 |   smid_cnear_pair_count |   smid_cnear_same_l1 |   smid_cnear_same_l12 |
|:-----------------------------|-----------------------:|--------------------:|---------------------:|---------------------------:|------------------------:|---------------------:|----------------------:|
| R690b L2=0.003 weak          |             10000.0000 |             61.6100 |               0.7300 |                    60.8800 |               1806.0000 |              53.0454 |                0.6091 |
| R690b L2=0.005 weak          |             10000.0000 |             35.1000 |               1.0100 |                    34.0900 |               1806.0000 |              31.7276 |                0.6645 |
| R690b L2=0.010 main          |             10000.0000 |             57.2700 |               0.4100 |                    56.8600 |               1806.0000 |              55.3710 |                0.3876 |
| R690b L2=0.015 fragmented    |             10000.0000 |             47.4600 |               0.2900 |                    47.1700 |               1806.0000 |              45.7364 |                0.2215 |
| R690b no L1 semantic         |             10000.0000 |             45.1000 |               0.3000 |                    44.8000 |               1806.0000 |              38.4275 |                0.2215 |
| R690b L3 ranking             |             10000.0000 |             46.4800 |               0.7700 |                    45.7100 |               1806.0000 |              42.3588 |                0.6091 |
| R690b L3=0.005 pending       |             10000.0000 |             56.7400 |               0.4500 |                    56.2900 |               1806.0000 |              52.7132 |                0.4430 |
| R690b L3=0.010 pending       |             10000.0000 |             61.0900 |               0.4700 |                    60.6200 |               1806.0000 |              58.0288 |                0.5537 |
| R690b L3=0.015 gate-failed   |             10000.0000 |             50.8900 |               8.2200 |                    42.6700 |               1806.0000 |              44.7398 |                7.6412 |
| Original L2 multihop ranking |             10000.0000 |             45.6700 |               0.2900 |                    45.3800 |               1806.0000 |              43.7431 |                0.4983 |
| Original semantic            |             10000.0000 |             59.2100 |               0.6700 |                    58.5400 |               1806.0000 |              54.5958 |                0.4430 |
| QCR L2 conflict ranking      |             10000.0000 |             42.2000 |               0.2200 |                    41.9800 |               1806.0000 |              41.9712 |                0.2215 |
| Stage3 prefix retained       |             10000.0000 |             20.9300 |               0.0800 |                    20.8500 |               1806.0000 |              25.9136 |                0.0000 |
| TAGCF attr mid               |             10000.0000 |             37.2200 |               0.1900 |                    37.0300 |               1806.0000 |              41.3068 |                0.3322 |
| V2 LMH mid=0.010             |             10000.0000 |             29.9600 |               0.1300 |                    29.8300 |               1806.0000 |              37.0986 |                0.2215 |
| V2 offline                   |             10000.0000 |             26.1000 |               0.2000 |                    25.9000 |               1806.0000 |              28.7375 |                0.3322 |

## Train-Only Learnability（仅训练集可学习性）

这些指标只使用 train interaction（训练交互），衡量 SID prefix（语义标识前缀）在 SFT（监督微调）训练数据中是否有足够曝光。

| label                        |   train_l1_entropy_norm |   train_l12_entropy_norm |   train_sid_entropy_norm |   catalog_l2_zero_train_pct |   catalog_l2_median_train_events |   catalog_l3_zero_train_pct |   catalog_l3_median_train_events |   catalog_item_zero_train_pct |
|:-----------------------------|------------------------:|-------------------------:|-------------------------:|----------------------------:|---------------------------------:|----------------------------:|---------------------------------:|------------------------------:|
| R690b L2=0.003 weak          |                  0.9486 |                   0.8994 |                   0.9248 |                      0.5426 |                          68.0000 |                      1.0309 |                          27.0000 |                        1.0581 |
| R690b L2=0.005 weak          |                  0.9348 |                   0.9011 |                   0.9248 |                      0.5426 |                          61.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| R690b L2=0.010 main          |                  0.9419 |                   0.8974 |                   0.9248 |                      0.5697 |                          61.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| R690b L2=0.015 fragmented    |                  0.9377 |                   0.9019 |                   0.9248 |                      0.6240 |                          48.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| R690b no L1 semantic         |                  0.9293 |                   0.9024 |                   0.9248 |                      0.5969 |                          47.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| R690b L3 ranking             |                  0.9219 |                   0.8969 |                   0.9248 |                      0.4341 |                          62.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| R690b L3=0.005 pending       |                  0.9334 |                   0.9013 |                   0.9248 |                      0.7325 |                          56.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| R690b L3=0.010 pending       |                  0.9362 |                   0.8962 |                   0.9248 |                      0.5969 |                          55.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| R690b L3=0.015 gate-failed   |                  0.9518 |                   0.9103 |                   0.9142 |                      0.0271 |                         331.0000 |                      0.9767 |                          28.0000 |                        1.0581 |
| Original L2 multihop ranking |                  0.9279 |                   0.8997 |                   0.9243 |                      0.6782 |                          52.5000 |                      1.0581 |                          27.0000 |                        1.0581 |
| Original semantic            |                  0.9403 |                   0.8926 |                   0.9248 |                      0.5697 |                          60.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| QCR L2 conflict ranking      |                  0.9274 |                   0.8982 |                   0.9248 |                      0.8139 |                          47.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| Stage3 prefix retained       |                  0.9578 |                   0.9085 |                   0.9248 |                      0.8681 |                          37.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| TAGCF attr mid               |                  0.9632 |                   0.9040 |                   0.9248 |                      0.7054 |                          50.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| V2 LMH mid=0.010             |                  0.9479 |                   0.9009 |                   0.9248 |                      0.7325 |                          43.0000 |                      1.0581 |                          27.0000 |                        1.0581 |
| V2 offline                   |                  0.9492 |                   0.9017 |                   0.9247 |                      0.7054 |                          45.0000 |                      1.0581 |                          27.0000 |                        1.0581 |

## Key Observations（关键观察）

1. R690b L2 sweep（第二层权重扫描）形成了清晰的结构趋势：`0.010` 是 balanced-positive（平衡正向），`0.003` 更像 under-separated（拆分不足），`0.015` 和 no-L1（无第一层语义）是过拆或路由不稳。
2. QCR 是核心反例：selective separation（选择性拆分）很好，但 preservation（协同保持）和 learnability（可学习性）有警告，因此不能只看 split-after-L1（同第一层后拆分）。
3. V2 offline（离线 v2）是跨方法族例外：它的 L1 semantic stability（第一层语义稳定性）不符合当前诊断假设，但下游仍强，说明本诊断最适合解释“语义层级协同注入”方法族，不应当当作跨所有 SID 的单一排名器。
4. L3=0.010 目前是 prospective（前瞻）样本：四轴均 pass（通过），所以结构上是 high-band candidate（高潜力候选），等待 SFT（监督微调）结果验证。
5. 新增 S-mid（中等语义相似）和 train-only learnability（仅训练集可学习性）是增强证据，不替代原四轴规则；它们主要用于发现“结构看着好但下游学不到”的情况。

## Output Files（输出文件）

- `diagnostic_metrics.csv`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-09_sid_structural_diagnostic_l3_ranking/diagnostic_metrics.csv`
- `diagnostic_rules.json`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-09_sid_structural_diagnostic_l3_ranking/diagnostic_rules.json`
- `diagnostic_case_studies.md`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-09_sid_structural_diagnostic_l3_ranking/diagnostic_case_studies.md`
- `diagnostic_pair_examples.md`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-09_sid_structural_diagnostic_l3_ranking/diagnostic_pair_examples.md`
- `metrics.json`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-09_sid_structural_diagnostic_l3_ranking/metrics.json`
