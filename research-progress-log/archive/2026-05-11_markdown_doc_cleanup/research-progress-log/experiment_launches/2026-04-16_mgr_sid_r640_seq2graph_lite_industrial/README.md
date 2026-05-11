# 2026-04-16 `R640` Seq2Graph-lite Tokenizer Screen（轻量 `Seq2Graph` 分词器筛选）

Status（状态）: `snapshot（快照）`
Snapshot date（快照日期）: `2026-04-17`

## 目的

这是 `D640`（图审计）之后的第一轮正式 tokenizer screen（分词器筛选）。

要回答的问题是：

> 当 `Seq2Graph-lite`（轻量 `Seq2Graph`） high-order rescue graph（高阶补盲图）真正接到 `v2` tokenizer（分词器）训练里时，`reliability`（可靠性）和 `direct-weak mask`（直接弱连接掩码）哪一种更值得保留。

这一步仍然不是下游最终裁决，但它是 `R645` 之前必须完成的最小工程筛选。

## 对应变体

- `R640b`
  - `L1 <- coarse_seq2g_rel`
  - `L2 <- fagsp_mid_seq2g_rel`
  - `L3 <- local_purified`
- `R640c`
  - `L1 <- coarse_seq2g_rel_masked`
  - `L2 <- fagsp_mid_seq2g_rel_masked`
  - `L3 <- local_purified`

## 启动方式

- 并行启动脚本：
  - [launch_mgr_sid_r640_seq2graph_tmux.sh](/home/leejt/OneRec/scripts/launch_mgr_sid_r640_seq2graph_tmux.sh)
- 单独训练加生成脚本：
  - [experiment_mgr_sid_r640b_seq2graph_rel_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r640b_seq2graph_rel_train_generate.sh)
  - [experiment_mgr_sid_r640c_seq2graph_rel_masked_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r640c_seq2graph_rel_masked_train_generate.sh)

## 对应配置

- [sid_train_industrial_mgr_sid_r640b_seq2graph_rel.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r640b_seq2graph_rel.yaml)
- [sid_train_industrial_mgr_sid_r640c_seq2graph_rel_masked.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r640c_seq2graph_rel_masked.yaml)

## 当前结果与判断

- `R640a` 不是“正常 `v2`（版本 2）”
  - 它仍然是 `Seq2Graph-lite`（轻量 `Seq2Graph`）分支，只是最朴素的 `context-only`（仅上下文）版本
- 真正的 current `v2`（当前 `v2`）仍然是：
  - `L1 <- coarse_purified`
  - `L2 <- fagsp_mid_base`
  - `L3 <- local_purified`
- `R640b`
  - first eval collision（首次评估冲突率）: `0.9997287032`
  - best train collision（训练最佳冲突率）: `0.9283776451`
  - generated collision（生成后冲突率）: `0.4120998372`
  - `num_collisions = 1519 / 3686`
  - `max_conflict = 310`
  - 结论：
    - 这是一次明确的 catastrophic failure（灾难性失败）
    - `rel`（可靠性版）相对 `rel_masked`（可靠性加掩码版）额外保留了 `17,960` 条边
    - 这批边全部满足 direct-strong（直接强连接）条件：`direct_support >= 0.5`
    - 它们的平均 direct support（直接支持度）为 `9.13`
    - 主要集中在 dense same-brand families（稠密同品牌家族），例如 `HATCHBOX`、`Small Parts`、`uxcell`、`3D Solutech`
- `R640c`
  - first eval collision（首次评估冲突率）: `0.9997287032`
  - best train collision（训练最佳冲突率）: `0.1242539338`
  - generated collision（生成后冲突率）: `0.0032555616`
  - `num_collisions = 12 / 3686`
  - `max_conflict = 3`
  - 结论：
    - 这是 `R640` 分支里唯一通过 catastrophic failure filter（灾难性失败过滤）的 tokenizer candidate（分词器候选）
    - 已正式推进到 `R645 = title_history2sid_on + desc_align_p05` downstream adjudication（下游裁决）
- 当前阶段判断：
  - `Seq2Graph-lite`（轻量 `Seq2Graph`）不是空分支，`rel_masked`（可靠性加掩码版）确实能留下可用候选
  - 但 reliability-only rescue（仅可靠性感知补盲）当前不能再作为主线，因为它会把 direct-strong family edges（直接强连接家族边）重新灌回 coarse graph（粗图）
