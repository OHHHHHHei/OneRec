# 2026-04-16 `D542` MGDCF Coarse Diagnostics（`MGDCF` 粗图诊断）

Status（状态）: `archived（归档）`
Archived on（归档日期）: `2026-04-16`

This run is preserved as provenance（来源记录） only.
After `S000`, it is no longer an active diagnostics gate（活跃诊断门）.

## 目的

这是 coarse reconstruction（粗图重构）主线的第一个正式 diagnostics gate（诊断门）。

它回答的问题是：

> 如果不用当前基于局部序列共现的 `coarse_purified`（净化粗图），
> 而改用 `MGDCF` 风格的 homogeneous item-item graph（同构物品图），
> 我们能不能得到一张**真正重构**而不是“轻微重加权”的 `G_coarse`（粗粒度图）。

## 对应实现

- 构图实现：
  - [graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/graph_bank.py)
- 图库注册：
  - [transplanted_graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/transplanted_graph_bank.py)
- 诊断脚本：
  - [experiment_mgr_sid_coarse_local_graph_diagnostics.py](/home/leejt/OneRec/scripts/archive/retired_prior_diagnostics/experiment_mgr_sid_coarse_local_graph_diagnostics.py)

## 诊断设置

- 数据集：
  - `Industrial_and_Scientific`
- 候选：
  - `coarse_mgdcf_r0.0500`
  - `coarse_mgdcf_r0.1000`
  - `coarse_mgdcf_r0.2000`
- 共同设置：
  - `binarize_edges = True`
  - `graph_topk = 32`

## 主要输出

- [SUMMARY.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_mgdcf_coarse_diagnostics_industrial/SUMMARY.md)
- [summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_mgdcf_coarse_diagnostics_industrial/summary.json)
- [D542_mgdcf_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_mgdcf_coarse_diagnostics_industrial/D542_mgdcf_summary.json)

## 关键结果

- `r=0.05`
  - `connected_item_rate = 0.8850`
  - `overlap = 0.0932`
  - 太稀，连通性掉得明显
- `r=0.10`
  - `connected_item_rate = 0.9723`
  - `overlap = 0.1393`
  - 已经是合理 coarse（粗图）候选
- `r=0.20`
  - `connected_item_rate = 0.9886`
  - `overlap = 0.1988`
  - `topk_expansion_ratio = 0.4962`
  - 是当前最适合直接推进 tokenizer screen（分词器筛选）的版本

## 历史判断（仅供回溯）

`D542` 最重要的结论是：

- `MGDCF` 不是那种“几乎不改图”的弱 coarse（粗图）候选；
- 它和 baseline `coarse_purified`（基线粗图）差异很大，属于真正的 coarse reconstruction（粗图重构）；
- 在当前数据集上，`keep_ratio = 0.20` 是最合理的第一个 tokenizer（分词器）推进点。

所以这轮历史上给出的下一步是：

- `R542a`
  - `L1 <- coarse_mgdcf`
  - `L2 <- fagsp_mid_mgdcf`
  - `L3 <- local_purified`
  - `mgdcf_keep_ratio = 0.20`

## 当前状态

- 这类 coarse reconstruction diagnostics（粗图重构诊断）已经退役，不再直接驱动后续推进。
