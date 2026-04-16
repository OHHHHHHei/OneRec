# Coarse / Local Graph Diagnostics（粗图 / 局部图诊断）

- Dataset（数据集）: `Industrial_and_Scientific`
- Baseline tokenizer line（基线分词器线）: `v2_on_p05 tokenizer graph bank`

## D530: `G_local`（局部图）多跳扩散

| Variant（变体） | graph_nnz | outgoing_cov | connected_rate | overlap | topk_expansion | |
|---|---:|---:|---:|---:|---:|
| `local_multihop_a0.35_h2` | 570360 | 0.9715 | 0.9891 | 0.2450 | 2.8086 |
| `local_multihop_a0.35_h3` | 4747587 | 0.9726 | 0.9894 | 0.2370 | 3.0779 |
| `local_multihop_a0.50_h3` | 4747587 | 0.9726 | 0.9894 | 0.2347 | 3.0834 |

## D540: `G_coarse`（粗图）用户分群条件化

| Variant（变体） | graph_nnz | connected_rate | overlap | seg_mean | seg_ge2_rate | |
|---|---:|---:|---:|---:|---:|
| `coarse_user_segment_k4` | 70768 | 0.9721 | 0.9708 | 0.2519 | 0.0076 |
| `coarse_user_segment_k8` | 70768 | 0.9721 | 0.9675 | 0.1285 | 0.0272 |

## D541: `G_coarse`（粗图）`CIR`（边可靠性）重加权

- `graph_nnz`: `70768`
- `connected_item_rate`（连通物品比例）: `0.9721`
- `mean_neighbor_overlap_with_baseline`（与基线邻域重叠）: `0.9710`
- `cir_mean`（平均 CIR）: `0.0457`
- `cir_nonzero_rate`（非零 CIR 比例）: `1.0000`

## D542: `G_coarse`（粗图）`MGDCF`（全局同构物品图）重构

| Variant（变体） | graph_nnz | connected_rate | overlap | topk_expansion | |
|---|---:|---:|---:|---:|
| `coarse_mgdcf_r0.0500` | 10714 | 0.8850 | 0.0932 | 0.1166 |
| `coarse_mgdcf_r0.1000` | 21428 | 0.9723 | 0.1393 | 0.2417 |
| `coarse_mgdcf_r0.2000` | 42856 | 0.9886 | 0.1988 | 0.4962 |

## Quick Read（快速结论）

- `G_local`（局部图）最值得继续推进的候选：`local_multihop_a0.35_h3`
- `G_coarse`（粗图）同源重加权分支最值得继续推进的候选：`coarse_user_segment_k4`
- `G_coarse`（粗图）低风险对照是否值得推进：`promote`
- `G_coarse`（粗图）重构分支最值得继续推进的候选：`coarse_mgdcf_r0.2000`
