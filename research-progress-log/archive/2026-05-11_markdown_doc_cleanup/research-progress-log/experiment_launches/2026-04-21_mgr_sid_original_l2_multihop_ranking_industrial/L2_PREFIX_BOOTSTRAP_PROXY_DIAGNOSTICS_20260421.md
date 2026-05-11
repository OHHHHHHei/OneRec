# L2 Prefix Bootstrap / Proxy Diagnostics（第二层前缀自助法与代理诊断）

Status（状态）: `diagnostic_snapshot（诊断快照）`

This diagnostic uses existing evaluate outputs only. Paired bootstrap（配对自助法） measures test-sample uncertainty（测试样本不确定性）, not training-seed variance（训练随机性方差）.

## Metric Summary（指标摘要）

| run（运行） | NDCG@10 | HR@10 | HR@50 |
|---|---|---|---|
| recipe_original | 0.10182815 | 0.14626075 | 0.24244430 |
| v2_on_p05 | 0.10270767 | 0.14626075 | 0.24818001 |
| original_l2 | 0.10165136 | 0.14736378 | 0.25612177 |
| original_l3 | 0.10159264 | 0.14604015 | 0.24795941 |
| r720e | 0.10094471 | 0.14604015 | 0.25104787 |
| strongest_original_sft | 0.10372025 | 0.15089345 | 0.24531216 |

## Paired Bootstrap（配对自助法）

| comparison（对比） | metric（指标） | mean diff（均值差） | 95% CI（置信区间） | approx p |
|---|---|---|---|---|
| original_l2 - recipe_original | ndcg@10 | -0.00017679 | [-0.00471924, 0.00436304] | 0.9616 |
| original_l2 - recipe_original | hr@10 | 0.00110302 | [-0.00661813, 0.00860357] | 0.8100 |
| original_l2 - recipe_original | hr@50 | 0.01367748 | [0.00397088, 0.02338407] | 0.0048 |
| original_l3 - recipe_original | ndcg@10 | -0.00023551 | [-0.00488535, 0.00459894] | 0.9146 |
| original_l3 - recipe_original | hr@10 | -0.00022060 | [-0.00816236, 0.00772116] | 0.9530 |
| original_l3 - recipe_original | hr@50 | 0.00551511 | [-0.00419148, 0.01522171] | 0.2802 |
| r720e - recipe_original | ndcg@10 | -0.00088343 | [-0.00596559, 0.00428941] | 0.7420 |
| r720e - recipe_original | hr@10 | -0.00022060 | [-0.00860357, 0.00816236] | 0.9848 |
| r720e - recipe_original | hr@50 | 0.00860357 | [-0.00154423, 0.01897198] | 0.1000 |
| v2_on_p05 - recipe_original | ndcg@10 | 0.00087952 | [-0.00384376, 0.00565694] | 0.7140 |
| v2_on_p05 - recipe_original | hr@10 | 0.00000000 | [-0.00794176, 0.00816236] | 1.0000 |
| v2_on_p05 - recipe_original | hr@50 | 0.00573572 | [-0.00441209, 0.01610413] | 0.2906 |
| original_l2 - original_l3 | ndcg@10 | 0.00005872 | [-0.00419106, 0.00420631] | 0.9922 |
| original_l2 - original_l3 | hr@10 | 0.00132363 | [-0.00617692, 0.00860357] | 0.7412 |
| original_l2 - original_l3 | hr@50 | 0.00816236 | [-0.00154423, 0.01786896] | 0.1042 |
| original_l2 - r720e | ndcg@10 | 0.00070664 | [-0.00409481, 0.00547844] | 0.7768 |
| original_l2 - r720e | hr@10 | 0.00132363 | [-0.00639753, 0.00882418] | 0.7754 |
| original_l2 - r720e | hr@50 | 0.00507390 | [-0.00485330, 0.01522171] | 0.3198 |
| original_l2 - v2_on_p05 | ndcg@10 | -0.00105632 | [-0.00540947, 0.00337118] | 0.6424 |
| original_l2 - v2_on_p05 | hr@10 | 0.00110302 | [-0.00661813, 0.00926539] | 0.7926 |
| original_l2 - v2_on_p05 | hr@50 | 0.00794176 | [-0.00242665, 0.01831017] | 0.1376 |
| original_l2 - strongest_original_sft | ndcg@10 | -0.00206889 | [-0.00661445, 0.00260874] | 0.3832 |
| original_l2 - strongest_original_sft | hr@10 | -0.00352967 | [-0.01125083, 0.00397088] | 0.3786 |
| original_l2 - strongest_original_sft | hr@50 | 0.01080962 | [0.00088242, 0.02073682] | 0.0364 |

## Final Top50 Proxy Diagnostics（最终前 50 代理诊断）

These are final-output proxies（最终输出代理）, not exact per-step beam survival（逐步束搜索存活）.

| run（运行） | GT L2 covered@10 | GT L2 covered@50 | same L2 frac@10 | graph-neighbor frac@10 | semantic-neighbor frac@10 | mean hit rank@50 |
|---|---|---|---|---|---|---|
| recipe_original | 0.20869182 | 0.32120009 | 0.07013016 | 0.08213104 | 0.12799471 | 12.77343039 |
| v2_on_p05 | 0.20295610 | 0.31149349 | 0.05442312 | 0.08700640 | 0.13373042 | 13.07555556 |
| original_l2 | 0.20670638 | 0.32759762 | 0.05605559 | 0.08682991 | 0.12993602 | 13.73126615 |
| original_l3 | 0.20758879 | 0.31722921 | 0.06196779 | 0.09055813 | 0.13406133 | 12.70551601 |
| r720e | 0.19964703 | 0.31414075 | 0.04531216 | 0.08563865 | 0.13275976 | 12.64586995 |

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

Structured JSON（结构化结果）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_original_l2_multihop_ranking_industrial/l2_prefix_bootstrap_proxy_diagnostics_20260421.json`
