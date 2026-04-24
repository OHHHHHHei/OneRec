# Root Tmp Diagnostics（根目录临时诊断归档）

Status（状态）: `archived（归档）`
Snapshot date（快照日期）: `2026-04-24`

This folder preserves 68 tracked root-level `tmp_*` diagnostic artifacts（根目录临时诊断产物） that were moved out of the repository root during the MGR-SID negative research archive（负结果研究归档）.

They are retained for traceability（可追溯性） only.

## Groups（分组）

- `tmp_layerwise_prefix_hit_*`: layerwise prefix-hit diagnostics（逐层前缀命中诊断）.
- `tmp_local_multihop_*` and `tmp_mid_graph_*`: local-multihop / mid-graph comparison（局部多跳 / 中图对比）.
- `tmp_original_l2_sid_analysis_*`: original-L2 SID structure analysis（原版第二层 SID 结构分析）.
- `tmp_original_sft_error_*`: original SFT error analysis（原版监督微调错误分析）.
- `tmp_original_vs_semantic_l1_overlay_*`: original vs semantic-L1 overlay（原版与语义第一层覆盖对比）.
- `tmp_r720b_*`, `tmp_r720c_*`, `tmp_r720e_*`, `tmp_r720f_*`: R720-family diagnostic artifacts（R720 家族诊断产物）.

## Policy（使用规则）

- Do not treat these files as active result summaries（活跃结果总结）.
- If a future analysis（未来分析） needs them, cite this folder path（目录路径） directly.
- Do not move them back to the repository root（仓库根目录）.

