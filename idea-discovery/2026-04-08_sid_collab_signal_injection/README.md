# 2026-04-08 SID Collaborative Signal Injection

这个目录现在只保留当前仍然有效的研究主线，并把已经阶段性放下的草案归档。

## Current Canonical Reading Order

1. `RESEARCH_DIRECTION.md`
2. `working_idea_graph_hierarchy_v1/00_ACTIVE_CONTEXT.md`
3. `working_idea_graph_hierarchy_v1/18_mgr_sid_v2_ambiguity_aware_method.md`
4. `working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`
5. `working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_PLAN_TOKENIZER_V2.md`
6. `working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_TRACKER_TOKENIZER_V2.md`

## 当前有效文档

- `RESEARCH_DIRECTION.md`
  - 当前主线方向说明
  - 内容定位：只保留 `motivation -> idea` 的顺向逻辑，不绑定具体实现

- `working_idea_graph_hierarchy_v1/`
  - 当前 active 主线
  - 主题已经收敛到：
    - graph structure as collaborative-information carrier
    - hierarchy-aware graph supervision
    - ambiguity-aware tokenizer refinement
    - semantic-structure retention

## 归档文档

- `archive/2026-04-08_initial_discovery_ambileaf/`
  - 这轮最初 discovery 过程中产生的阶段性材料
  - 主要围绕 `AmbiLeaf / local leaf retokenization` 展开
  - 这些文档仍然有参考价值，但不再代表当前最终主线

- `archive/2026-04-08_working_idea_hierarchy_aware_v1_superseded/`
  - 这一轮是从“multi-view collaborative fusion”过渡到“graph-native hierarchy-aware SID”的中间工作版本
  - 现在已经被 `working_idea_graph_hierarchy_v1/` 覆盖，不再作为当前主推进目录
  - 仍然保留完整 `idea-discovery` 产物，方便回看思路演化和负证据

归档内容包括：

- 文献图谱
- idea ranking
- novelty check
- critical review
- 初版 `IDEA_REPORT.md`
- `AmbiLeaf` 对应的 `refine-logs/`

- `working_idea_graph_hierarchy_v1/archive/2026-04-11_doc_cleanup_v1_superseded/`
  - 当前 graph-hierarchy 主线内部更早期的 `v1` discovery / proposal / plan 材料
  - 仍保留完整上下文，但不再作为 active 入口

- `working_idea_graph_hierarchy_v1/archive/2026-04-11_doc_cleanup_duplicates/`
  - 与当前 active 文档高度重叠的 related-work / design-review 文档
  - 以及已经被 tokenizer-first 执行路径替代的更宽版本 `v2` 计划 / tracker

## 当前建议

后续如果继续推进方法设计，优先基于：

- `RESEARCH_DIRECTION.md`
- `working_idea_graph_hierarchy_v1/00_ACTIVE_CONTEXT.md`
- `working_idea_graph_hierarchy_v1/18_mgr_sid_v2_ambiguity_aware_method.md`

如果需要回看从早期 local leaf 方向，到 graph-hierarchy，再到 `v2 tokenizer-first` 的演化证据，再去看 archive。
