# 2026-04-15 Coarse / Local Graph Diagnostics（粗图 / 局部图诊断）

Status（状态）: `archived（归档）`
Archived on（归档日期）: `2026-04-16`

This run is preserved as provenance（来源记录） only.
After `S000`, it is no longer an active diagnostics gate（活跃诊断门）.

## 目的

这是 coarse/local graph-carrier（粗图 / 局部图载体）计划的 `M0` diagnostics gate（诊断门）。

它不是 tokenizer（分词器）训练，也不是下游 `SFT -> evaluate`（监督微调到评测）；
它要先回答：

> 在当前 `v2` 图库里，被长期低估的 `G_coarse / G_local`（粗粒度图 / 局部图）到底有没有明显的结构改进空间，
> 以及哪些候选图真的改动了图，而不是只换了个名字。

## 对应计划

- [EXPERIMENT_PLAN_COARSE_LOCAL_GRAPH_CARRIERS.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_PLAN_COARSE_LOCAL_GRAPH_CARRIERS.md)
- [EXPERIMENT_TRACKER_COARSE_LOCAL_GRAPH_CARRIERS.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_TRACKER_COARSE_LOCAL_GRAPH_CARRIERS.md)

## 脚本

- [experiment_mgr_sid_coarse_local_graph_diagnostics.py](/home/leejt/OneRec/scripts/archive/retired_prior_diagnostics/experiment_mgr_sid_coarse_local_graph_diagnostics.py)

## 输入

- train:
  - `/home/leejt/OneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- test:
  - `/home/leejt/OneRec/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv`

## 诊断对象

- `D530`: `G_local`（局部图）multi-hop diffused transition（多跳扩散转移图）
- `D540`: `G_coarse`（粗图）user-segment-conditioned co-occurrence（用户分群条件化共现图）
- `D541`: `G_coarse`（粗图）`CIR`（边可靠性） reweighting（重加权）

## 主要输出

- [SUMMARY.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial/SUMMARY.md)
- [summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial/summary.json)
- [D530_local_multihop_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial/D530_local_multihop_summary.json)
- [D540_user_segment_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial/D540_user_segment_summary.json)
- [D541_cir_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial/D541_cir_summary.json)

## Quick Read（快速结论）

- `D530` 的结论比预期更复杂：
  - multi-hop `G_local`（多跳局部图）确实大幅扩展了邻域规模
  - 但当前 baseline `local_purified`（基线局部图）本身的 item coverage（物品覆盖）已经很高
  - 所以“`L3` 监督覆盖严重不足”这个最初假设，**没有被这轮诊断直接支持**
- `D540` 的当前实现过于接近 baseline `G_coarse`（基线粗图）：
  - user-segment（用户分群）图的拓扑几乎没变
  - 这说明当前这种“跨分群支持比例”写法，可能太弱，暂时不值得直接推进 tokenizer run（分词器实验）
- `D541` 也是类似情况：
  - `CIR`（边可靠性）信号存在
  - 但当前的低风险混合式重加权，离 baseline 粗图也很近
  - 它更像一个合理 control（对照），而不是高收益主线

## 历史判断（仅供回溯）

这轮 `M0` 最重要的结论不是“已经找到最优 coarse/local 候选”，而是：

> 我们已经知道哪些 coarse/local 候选在当前定义下**改动太小**，
> 以及“`G_local` coverage（局部图覆盖）不足”并不是一个强成立的简单解释。

因此，这个阶段历史上给出的下一步是：

1. 如果只推进一个 tokenizer-side（分词器侧）候选，优先考虑 **最小化的 shallow multi-hop `G_local`（浅层多跳局部图）**，因为它至少真的改变了图。
2. 对 `user-segment`（用户分群）和 `CIR` coarse（粗图）候选，先重新考虑更强的权重 / 过滤形式，而不是直接按当前定义推进。

## 当前状态

- 这类 coarse/local offline diagnostics（粗图 / 局部图离线诊断）已经退役。
- 后续不能再因为这类诊断“看起来更有结构变化”就推进新 tokenizer（分词器）分支。
