# MGR-SID Diagnostic Audit（MGR-SID 诊断脚本审计）

## Scope（范围）

- audit dataset（审计数据集）: `Industrial_and_Scientific`
- tokenizer entries with metrics（带结构指标的分词器数量）: `8`
- downstream comparable entries（可比下游条目数）: `18`
- comparable groups（可比组数）: `8`

## Main Verdicts（主要结论）

- `prefix_test_fraction_consistent_crowded` vs `mean_ndcg_at_10`: pairwise consistency（成对一致率）=`0.4783`, usable_pairs（有效成对数）=`23`
- `prefix_test_fraction_inconsistent_crowded` vs `mean_ndcg_at_10`: pairwise consistency（成对一致率）=`0.3478`, usable_pairs（有效成对数）=`23`
- `test_target_weighted_mean_l3_entropy_bits` vs `mean_ndcg_at_10`: pairwise consistency（成对一致率）=`0.3043`, usable_pairs（有效成对数）=`23`
- `test_target_weighted_mean_l2_leaf_count` vs `mean_ndcg_at_10`: pairwise consistency（成对一致率）=`0.2609`, usable_pairs（有效成对数）=`23`
- `tokenizer_generated_collision_rate` vs `mean_ndcg_at_10`: pairwise consistency（成对一致率）=`0.1111`, usable_pairs（有效成对数）=`18`

## Metrics Table（指标表）

| Metric | Score | Direction | Usable Pairs | Consistency | False Positive Rate |
|---|---|---|---:|---:|---:|
| `tokenizer_generated_collision_rate` | `mean_ndcg_at_10` | `lower_better` | 18 | 0.1111 | 0.8889 |
| `tokenizer_generated_collision_rate` | `mean_hr_at_10` | `lower_better` | 18 | 0.0556 | 0.9444 |
| `test_target_weighted_mean_l2_leaf_count` | `mean_ndcg_at_10` | `lower_better` | 23 | 0.2609 | 0.7391 |
| `test_target_weighted_mean_l2_leaf_count` | `mean_hr_at_10` | `lower_better` | 22 | 0.2273 | 0.7727 |
| `test_target_fraction_multileaf_l2` | `mean_ndcg_at_10` | `lower_better` | 23 | 0.4348 | 0.5652 |
| `test_target_fraction_multileaf_l2` | `mean_hr_at_10` | `lower_better` | 22 | 0.4091 | 0.5909 |
| `test_target_fraction_multileaf_l2_ge4` | `mean_ndcg_at_10` | `lower_better` | 23 | 0.3478 | 0.6522 |
| `test_target_fraction_multileaf_l2_ge4` | `mean_hr_at_10` | `lower_better` | 22 | 0.3182 | 0.6818 |
| `test_target_weighted_mean_l3_entropy_bits` | `mean_ndcg_at_10` | `lower_better` | 23 | 0.3043 | 0.6957 |
| `test_target_weighted_mean_l3_entropy_bits` | `mean_hr_at_10` | `lower_better` | 22 | 0.2727 | 0.7273 |
| `prefix_test_fraction_consistent_crowded` | `mean_ndcg_at_10` | `higher_better` | 23 | 0.4783 | 0.5217 |
| `prefix_test_fraction_consistent_crowded` | `mean_hr_at_10` | `higher_better` | 22 | 0.5455 | 0.4545 |
| `prefix_test_fraction_inconsistent_crowded` | `mean_ndcg_at_10` | `lower_better` | 23 | 0.3478 | 0.6522 |
| `prefix_test_fraction_inconsistent_crowded` | `mean_hr_at_10` | `lower_better` | 22 | 0.3182 | 0.6818 |
| `prefix_test_weighted_mean_graph_affinity` | `mean_ndcg_at_10` | `higher_better` | 23 | 0.5217 | 0.4783 |
| `prefix_test_weighted_mean_graph_affinity` | `mean_hr_at_10` | `higher_better` | 22 | 0.5455 | 0.4545 |

## Tokenizer Metrics（分词器结构指标）

| Tokenizer | Collision | Test Mean L2 Leaves | Test Multi-Leaf | Test Entropy | Consistent Crowded | Inconsistent Crowded |
|---|---:|---:|---:|---:|---:|---:|
| `tok_industrial_mgr_tokenizer_v2_offline` | 0.003527 | 4.342158 | 0.487315 | 1.100115 | 0.114714 | 0.371057 |
| `tok_industrial_mgr_upstream_baseline` | 0.003527 | 4.799912 | 0.689389 | 1.453294 | 0.114273 | 0.571145 |
| `tok_industrial_mgr_upstream_hierarchy` | 0.003256 | 4.449812 | 0.613060 | 1.293525 | 0.103684 | 0.467240 |
| `tok_industrial_original_semantic` | 0.004341 | 5.739687 | 0.606221 | 1.407254 | 0.117362 | 0.485330 |
| `tok_industrial_stage2_r202a_stopgrad` | 0.003527 | 3.614825 | 0.498787 | 1.030772 | 0.107876 | 0.362453 |
| `tok_industrial_stage3_r401b_g005` | 0.002984 | 2.696669 | 0.388043 | 0.737341 | 0.129715 | 0.237812 |
| `tok_industrial_stage3_r401d_g005_a005` | 0.002984 | 2.571145 | 0.381646 | 0.715644 | 0.134128 | 0.234503 |
| `tok_industrial_tagcf_r510_attr_mid` | 0.002984 | 3.684756 | 0.527686 | 1.089844 | 0.116259 | 0.408780 |

## Notes（备注）

- This audit only uses comparable downstream groups with the same stage and recipe（同阶段同配方）.
- Posterior explainers（后验解释器） such as evaluate error analysis are excluded from this table because they already consume evaluate outputs.
- Pairwise consistency（成对一致率） is the main decision criterion; low consistency means the diagnostic should not be used as a promotion gate（推进门槛）.
