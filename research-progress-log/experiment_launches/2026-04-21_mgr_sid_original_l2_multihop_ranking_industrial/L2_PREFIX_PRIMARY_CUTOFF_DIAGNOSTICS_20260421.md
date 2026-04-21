# L2 Prefix Primary-Cutoff Diagnostics（第二层前缀主要截断诊断）

Status（状态）: `diagnostic_snapshot（诊断快照）`

This diagnostic uses existing evaluate outputs only. Paired bootstrap（配对自助法） measures test-sample uncertainty（测试样本不确定性）, not training-seed variance（训练随机性方差）.

## Metric Summary（指标摘要）

| run（运行） | NDCG@1 | HR@1 | NDCG@3 | HR@3 | NDCG@5 | HR@5 | NDCG@10 | HR@10 |
|---|---|---|---|---|---|---|---|---|
| recipe_original | 0.06860799 | 0.06860799 | 0.08416813 | 0.09574233 | 0.09054808 | 0.11118465 | 0.10182815 | 0.14626075 |
| v2_on_p05 | 0.07059343 | 0.07059343 | 0.08451223 | 0.09508052 | 0.09253300 | 0.11471432 | 0.10270767 | 0.14626075 |
| original_l2 | 0.06618134 | 0.06618134 | 0.08283132 | 0.09485992 | 0.09144148 | 0.11581734 | 0.10165136 | 0.14736378 |
| original_l3 | 0.06684315 | 0.06684315 | 0.08445174 | 0.09772777 | 0.09226315 | 0.11692036 | 0.10159264 | 0.14604015 |
| r720e | 0.06397529 | 0.06397529 | 0.08397369 | 0.09883080 | 0.09185278 | 0.11802338 | 0.10094471 | 0.14604015 |
| strongest_original_sft | 0.06706375 | 0.06706375 | 0.08500848 | 0.09838959 | 0.09315326 | 0.11824399 | 0.10372025 | 0.15089345 |

## Paired Bootstrap（配对自助法）

| comparison（对比） | metric（指标） | mean diff（均值差） | 95% CI（置信区间） | approx p |
|---|---|---|---|---|
| original_l2 - recipe_original | ndcg@1 | -0.00242665 | [-0.00706486, 0.00242665] | 0.3492 |
| original_l2 - recipe_original | hr@1 | -0.00242665 | [-0.00727995, 0.00242665] | 0.3516 |
| original_l2 - recipe_original | ndcg@3 | -0.00133681 | [-0.00617169, 0.00345122] | 0.5754 |
| original_l2 - recipe_original | hr@3 | -0.00088242 | [-0.00705934, 0.00529451] | 0.8042 |
| original_l2 - recipe_original | ndcg@5 | 0.00089340 | [-0.00379514, 0.00552093] | 0.7228 |
| original_l2 - recipe_original | hr@5 | 0.00463269 | [-0.00198544, 0.01147143] | 0.1896 |
| original_l2 - recipe_original | ndcg@10 | -0.00017679 | [-0.00463937, 0.00432165] | 0.9532 |
| original_l2 - recipe_original | hr@10 | 0.00110302 | [-0.00661813, 0.00882418] | 0.7924 |
| original_l3 - recipe_original | ndcg@1 | -0.00176484 | [-0.00661813, 0.00308846] | 0.5136 |
| original_l3 - recipe_original | hr@1 | -0.00176484 | [-0.00661813, 0.00308846] | 0.5024 |
| original_l3 - recipe_original | ndcg@3 | 0.00028360 | [-0.00471934, 0.00508960] | 0.9036 |
| original_l3 - recipe_original | hr@3 | 0.00198544 | [-0.00419148, 0.00838297] | 0.5614 |
| original_l3 - recipe_original | ndcg@5 | 0.00171507 | [-0.00311369, 0.00652930] | 0.4872 |
| original_l3 - recipe_original | hr@5 | 0.00573572 | [-0.00154423, 0.01279506] | 0.1292 |
| original_l3 - recipe_original | ndcg@10 | -0.00023551 | [-0.00484913, 0.00450863] | 0.9398 |
| original_l3 - recipe_original | hr@10 | -0.00022060 | [-0.00838297, 0.00750055] | 0.9790 |
| r720e - recipe_original | ndcg@1 | -0.00463269 | [-0.00993272, 0.00066181] | 0.1036 |
| r720e - recipe_original | hr@1 | -0.00463269 | [-0.01014780, 0.00088242] | 0.0962 |
| r720e - recipe_original | ndcg@3 | -0.00019444 | [-0.00560461, 0.00521301] | 0.9408 |
| r720e - recipe_original | hr@3 | 0.00308846 | [-0.00397088, 0.00970660] | 0.4044 |
| r720e - recipe_original | ndcg@5 | 0.00130470 | [-0.00399140, 0.00654564] | 0.6210 |
| r720e - recipe_original | hr@5 | 0.00683874 | [-0.00066181, 0.01433929] | 0.0784 |
| r720e - recipe_original | ndcg@10 | -0.00088343 | [-0.00615065, 0.00427086] | 0.7424 |
| r720e - recipe_original | hr@10 | -0.00022060 | [-0.00860357, 0.00816236] | 0.9752 |
| v2_on_p05 - recipe_original | ndcg@1 | 0.00198544 | [-0.00308846, 0.00705934] | 0.4638 |
| v2_on_p05 - recipe_original | hr@1 | 0.00198544 | [-0.00286786, 0.00705934] | 0.4618 |
| v2_on_p05 - recipe_original | ndcg@3 | 0.00034409 | [-0.00466939, 0.00526065] | 0.9108 |
| v2_on_p05 - recipe_original | hr@3 | -0.00066181 | [-0.00705934, 0.00551511] | 0.8624 |
| v2_on_p05 - recipe_original | ndcg@5 | 0.00198492 | [-0.00292997, 0.00706282] | 0.4332 |
| v2_on_p05 - recipe_original | hr@5 | 0.00352967 | [-0.00352967, 0.01036841] | 0.3478 |
| v2_on_p05 - recipe_original | ndcg@10 | 0.00087952 | [-0.00389175, 0.00566970] | 0.7232 |
| v2_on_p05 - recipe_original | hr@10 | 0.00000000 | [-0.00794728, 0.00816788] | 1.0000 |
| original_l2 - original_l3 | ndcg@1 | -0.00066181 | [-0.00529451, 0.00397088] | 0.8066 |
| original_l2 - original_l3 | hr@1 | -0.00066181 | [-0.00529451, 0.00397088] | 0.8038 |
| original_l2 - original_l3 | ndcg@3 | -0.00162042 | [-0.00610081, 0.00284957] | 0.4920 |
| original_l2 - original_l3 | hr@3 | -0.00286786 | [-0.00904478, 0.00308846] | 0.3630 |
| original_l2 - original_l3 | ndcg@5 | -0.00082167 | [-0.00523918, 0.00366758] | 0.7278 |
| original_l2 - original_l3 | hr@5 | -0.00110302 | [-0.00772116, 0.00573572] | 0.7874 |
| original_l2 - original_l3 | ndcg@10 | 0.00005872 | [-0.00431024, 0.00423496] | 0.9982 |
| original_l2 - original_l3 | hr@10 | 0.00132363 | [-0.00573572, 0.00860357] | 0.7436 |
| original_l2 - r720e | ndcg@1 | 0.00220604 | [-0.00308846, 0.00772116] | 0.4306 |
| original_l2 - r720e | hr@1 | 0.00220604 | [-0.00308846, 0.00772116] | 0.4394 |
| original_l2 - r720e | ndcg@3 | -0.00114237 | [-0.00639488, 0.00421527] | 0.6942 |
| original_l2 - r720e | hr@3 | -0.00397088 | [-0.01058901, 0.00286786] | 0.2592 |
| original_l2 - r720e | ndcg@5 | -0.00041130 | [-0.00535938, 0.00444858] | 0.8488 |
| original_l2 - r720e | hr@5 | -0.00220604 | [-0.00926539, 0.00463269] | 0.5616 |
| original_l2 - r720e | ndcg@10 | 0.00070664 | [-0.00409365, 0.00551056] | 0.7590 |
| original_l2 - r720e | hr@10 | 0.00132363 | [-0.00639753, 0.00904478] | 0.7748 |
| original_l2 - v2_on_p05 | ndcg@1 | -0.00441209 | [-0.00882418, 0.00000000] | 0.0588 |
| original_l2 - v2_on_p05 | hr@1 | -0.00441209 | [-0.00882418, -0.00022060] | 0.0494 |
| original_l2 - v2_on_p05 | ndcg@3 | -0.00168091 | [-0.00639760, 0.00293873] | 0.4702 |
| original_l2 - v2_on_p05 | hr@3 | -0.00022060 | [-0.00617692, 0.00595632] | 0.9758 |
| original_l2 - v2_on_p05 | ndcg@5 | -0.00109152 | [-0.00562683, 0.00334776] | 0.6214 |
| original_l2 - v2_on_p05 | hr@5 | 0.00110302 | [-0.00551511, 0.00750055] | 0.7692 |
| original_l2 - v2_on_p05 | ndcg@10 | -0.00105632 | [-0.00528884, 0.00335362] | 0.6308 |
| original_l2 - v2_on_p05 | hr@10 | 0.00110302 | [-0.00661813, 0.00882418] | 0.7876 |
| original_l2 - strongest_original_sft | ndcg@1 | -0.00088242 | [-0.00551511, 0.00397088] | 0.7646 |
| original_l2 - strongest_original_sft | hr@1 | -0.00088242 | [-0.00551511, 0.00397088] | 0.7602 |
| original_l2 - strongest_original_sft | ndcg@3 | -0.00217716 | [-0.00706484, 0.00266831] | 0.3754 |
| original_l2 - strongest_original_sft | hr@3 | -0.00352967 | [-0.00992720, 0.00286786] | 0.3066 |
| original_l2 - strongest_original_sft | ndcg@5 | -0.00171178 | [-0.00653228, 0.00307363] | 0.4938 |
| original_l2 - strongest_original_sft | hr@5 | -0.00242665 | [-0.00970660, 0.00485330] | 0.5484 |
| original_l2 - strongest_original_sft | ndcg@10 | -0.00206889 | [-0.00656072, 0.00238915] | 0.3718 |
| original_l2 - strongest_original_sft | hr@10 | -0.00352967 | [-0.01103022, 0.00419148] | 0.3972 |

## Final Top10 Proxy Diagnostics（最终前 10 代理诊断）

These are final-output proxies（最终输出代理）, not exact per-step beam survival（逐步束搜索存活）.

| run（运行） | GT L2 covered@1 | GT L2 covered@3 | GT L2 covered@5 | GT L2 covered@10 | same L2 frac@10 | graph-neighbor frac@10 | semantic-neighbor frac@10 | mean hit rank@10 |
|---|---|---|---|---|---|---|---|---|
| recipe_original | 0.13015663 | 0.16390911 | 0.18266049 | 0.20869182 | 0.07013016 | 0.08213104 | 0.12799471 | 3.26244344 |
| v2_on_p05 | 0.10103684 | 0.15552614 | 0.17471873 | 0.20295610 | 0.05442312 | 0.08700640 | 0.13373042 | 3.18250377 |
| original_l2 | 0.11052283 | 0.16015884 | 0.17935142 | 0.20670638 | 0.05605559 | 0.08682991 | 0.12993602 | 3.20059880 |
| original_l3 | 0.11956762 | 0.16082065 | 0.18111626 | 0.20758879 | 0.06196779 | 0.09055813 | 0.13406133 | 3.16163142 |
| r720e | 0.10500772 | 0.14251048 | 0.16545334 | 0.19964703 | 0.04531216 | 0.08563865 | 0.13275976 | 3.07854985 |

## Tokenizer-Level Neighbor Prefix Overlap（分词器级邻居前缀重叠）

| run（运行） | graph L1 overlap | graph L2 overlap | semantic L1 overlap | semantic L2 overlap |
|---|---|---|---|---|
| recipe_original | 0.11724132 | 0.01652553 | 0.53963086 | 0.03567338 |
| v2_on_p05 | 0.04564004 | 0.01178059 | 0.22303611 | 0.02215160 |
| original_l2 | 0.07870496 | 0.01445623 | 0.40942563 | 0.02648804 |
| original_l3 | 0.08634178 | 0.01355602 | 0.42352014 | 0.02311621 |
| r720e | 0.04864210 | 0.01050427 | 0.23989958 | 0.01748788 |

## Code Path Audit（代码路径审查）

- `original_l2_multihop_ranking` already sets `hierarchy_stopgrad_previous_levels=true`（前层停梯度为真）.
- In `train_v2._build_level_representations`, `level_representations[1] = detach(q1) + q2`, so the L2 ranking loss（第二层排序损失） does not backpropagate into `q1` through that auxiliary representation.
- Caveat（注意）: this protects the auxiliary L2 path（辅助第二层路径）, but base reconstruction/RQ losses（重建/量化损失） still train all levels.

Structured JSON（结构化结果）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_original_l2_multihop_ranking_industrial/l2_prefix_primary_cutoff_diagnostics_20260421.json`
