# TAGCF 支链

这个文件夹专门存放本轮 `TAGCF`（`Turning Semantics into Topology`）启发出来的支链材料。

这条支链的核心问题不是：

- 继续在当前图上做更复杂的 `loss`（损失）修补；

而是：

> 当前 `graph bank`（图库）会不会本身太粗，导致我们在一个不够好的图载体上反复微调 `SID`（语义 ID）？

## 一句话定位

这是一条 **探索性支链**：

- 目标：检查“语义转拓扑（semantics to topology，语义转拓扑）”能不能给当前 `MGR-SID` 带来更好的 `graph carrier`（图载体）
- 状态：已完成论文精读和首轮实验设计
- 与主线关系：并行探索，不替代当前 stage-3 主执行线

## 这个文件夹里有什么

1. [01_tagcf_paper_reading_20260414.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/side-branches/2026-04-15_tagcf_semantic_to_topology/01_tagcf_paper_reading_20260414.md)
   - `TAGCF` 论文精读
   - 重点整理“哪些思想适合借，哪些不该直接照搬”

## 配套实验计划

虽然这是支链 idea（想法）目录，但实验计划仍然统一放在 `refine-logs`（实验计划目录）里，便于和其他计划保持同一套索引习惯：

1. [EXPERIMENT_PLAN_TAGCF_SEMANTIC_TO_TOPOLOGY.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_PLAN_TAGCF_SEMANTIC_TO_TOPOLOGY.md)
2. [EXPERIMENT_TRACKER_TAGCF_SEMANTIC_TO_TOPOLOGY.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_TRACKER_TAGCF_SEMANTIC_TO_TOPOLOGY.md)

## 当前最重要的判断

- `TAGCF` 最值得借的是 **graph construction**（图构建）
- 第一轮不该照搬它的异构 `GNN`（图神经网络）主干
- 最值得先试的是：
  - `item-attribute-item`（物品-属性-物品）中介结构
  - 投影成新的 `item-item graph`（物品-物品图）
  - 检查它能不能成为更好的 `G_mid`（中尺度图）

## 建议阅读顺序

1. [01_tagcf_paper_reading_20260414.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/side-branches/2026-04-15_tagcf_semantic_to_topology/01_tagcf_paper_reading_20260414.md)
2. [EXPERIMENT_PLAN_TAGCF_SEMANTIC_TO_TOPOLOGY.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_PLAN_TAGCF_SEMANTIC_TO_TOPOLOGY.md)
3. [EXPERIMENT_TRACKER_TAGCF_SEMANTIC_TO_TOPOLOGY.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_TRACKER_TAGCF_SEMANTIC_TO_TOPOLOGY.md)

