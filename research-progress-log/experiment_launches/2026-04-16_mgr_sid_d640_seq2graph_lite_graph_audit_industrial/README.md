# 2026-04-16 `D640` Seq2Graph-lite Graph Audit（轻量 `Seq2Graph` 图审计）

Status（状态）: `snapshot（快照）`
Snapshot date（快照日期）: `2026-04-16`

## 目的

这是 `Seq2Graph-lite`（轻量 `Seq2Graph`） high-order rescue carrier（高阶补盲载体）分支的第一个正式 offline audit（离线审计）。

它回答的问题是：

> 如果我们把 `Seq2Graph`（序列到图增广）里“跨序列前驱共享”的思想压缩成一张 pure collaborative（纯协同）的 item-item rescue graph（物品-物品补盲图），它能不能真正补到当前 dense semantic blind spots（稠密语义盲区），而不是只做一轮无差别加边。

这一步是 engineering filter（工程过滤），不是 scientific verdict（科学裁决）。
它只决定 `R640a / R640b / R640c` 是否值得进入 tokenizer screen（分词器筛选），不决定方法是否成立。

## 对应实现

- 构图实现：
  - [graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/graph_bank.py)
- 图库注册：
  - [transplanted_graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/transplanted_graph_bank.py)
- 训练接口对齐：
  - [train_v2.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py)
- 审计脚本：
  - [experiment_mgr_sid_seq2graph_lite_graph_audit.py](/home/leejt/OneRec/scripts/experiment_mgr_sid_seq2graph_lite_graph_audit.py)

## 审计设置

- 数据集：
  - `Industrial_and_Scientific`
- 共同设置：
  - `seq2g_mix_alpha = 0.35`
  - `seq2g_context_topk = 32`
  - `seq2g_candidate_topm = 32`
  - `seq2g_direct_tau = 0.5`
- 对比变体：
  - `coarse_seq2g_ctx_only`
  - `coarse_seq2g_rel`
  - `coarse_seq2g_rel_masked`

## 主要输出

- [SUMMARY.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_d640_seq2graph_lite_graph_audit_industrial/SUMMARY.md)
- [D640_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_d640_seq2graph_lite_graph_audit_industrial/D640_summary.json)
- [D640_hotspot_semantic_pairs.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_d640_seq2graph_lite_graph_audit_industrial/D640_hotspot_semantic_pairs.csv)

## 关键结果

- 三个 `Seq2Graph-lite`（轻量 `Seq2Graph`） coarse variants（粗图变体）都不是 near-baseline tweak（近基线微调）：
  - `overlap`（邻域重叠）只有 `0.3527 ~ 0.3731`
  - `rescue_edge_ratio`（补盲边比例）达到 `0.5325 ~ 0.6049`
- `coarse_seq2g_rel`（可靠性感知粗图）在 hotspot visibility（热点可见性）上最好：
  - `visible_fraction`: `0.1667 -> 0.3667`
  - `predecessor_sharing_direct_zero_visible_fraction`: `0.0 -> 0.8`
- `coarse_seq2g_rel_masked`（可靠性感知加掩码粗图）在 rescue purity（补盲纯度）上最好：
  - `rescue_edge_ratio = 0.6049`
  - `direct_zero_visible_fraction = 0.25`
  - `predecessor_sharing_direct_zero_visible_fraction = 0.8`

## 当前判断

`D640` 给出的不是下游胜负，而是一个很明确的 engineering answer（工程答案）：

- `Seq2Graph-lite`（轻量 `Seq2Graph`）确实补到了当前 graph carrier blind spot（图载体盲区）
- `reliability`（可靠性）不是多余项，它能带来更高的 hotspot visibility（热点可见性）
- `direct-weak mask`（直接弱连接掩码）也不是多余项，它能让补盲更聚焦于真正的 direct-zero / direct-weak pairs（直接零连接 / 直接弱连接物品对）

所以这一步的结论是：

- `R640a / R640b / R640c` 都可以进入 tokenizer screen（分词器筛选）
- 其中 `R640b` 和 `R640c` 最值得优先看
