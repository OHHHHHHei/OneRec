# 2026-04-08 SID Collaborative Signal Injection

这个目录现在只保留当前仍然有效的研究主线，并把已经阶段性放下的草案归档。

## 当前有效文档

- `RESEARCH_DIRECTION.md`
  - 当前主线方向说明
  - 主题：`Hierarchy-Aware Collaborative Signal Fusion for SID-based Generative Recommendation`
  - 内容定位：只保留 `motivation -> idea` 的顺向逻辑，不绑定具体公式和最终实现

- `working_idea_hierarchy_aware_v1/`
  - 基于当前主线方向做的一轮完整 `idea-discovery`
  - 包含：
    - literature survey
    - local pilot
    - idea ranking
    - novelty check
    - 两轮 reviewer-style review
    - refined proposal
    - detailed experiment plan
  - 当前如果要继续推进方法设计，优先阅读这里的：
    - `IDEA_REPORT.md`
    - `refine-logs/FINAL_PROPOSAL.md`
    - `refine-logs/EXPERIMENT_PLAN.md`

- `working_idea_graph_hierarchy_v1/`
  - 基于同一主线，进一步把问题推进到：
    - graph structure
    - view-specific denoising
    - hierarchy-aware graph supervision
  - 这一轮不再把重点放在简单 multi-view fusion，而是尝试把图结构直接写进 SID 学习本身
  - 当前如果要看更“论文级”的新候选方法，优先阅读这里的：
    - `IDEA_REPORT.md`
    - `refine-logs/FINAL_PROPOSAL.md`
    - `refine-logs/EXPERIMENT_PLAN.md`

## 归档文档

- `archive/2026-04-08_initial_discovery_ambileaf/`
  - 这轮最初 discovery 过程中产生的阶段性材料
  - 主要围绕 `AmbiLeaf / local leaf retokenization` 展开
  - 这些文档仍然有参考价值，但不再代表当前最终主线

归档内容包括：

- 文献图谱
- idea ranking
- novelty check
- critical review
- 初版 `IDEA_REPORT.md`
- `AmbiLeaf` 对应的 `refine-logs/`

## 当前建议

后续如果继续往下推进方法设计，优先基于：

- `RESEARCH_DIRECTION.md`
- `working_idea_hierarchy_aware_v1/`
- `working_idea_graph_hierarchy_v1/`

如果需要回看我们为什么一开始会想到局部 leaf 方向，再去看：

- `archive/2026-04-08_initial_discovery_ambileaf/`
