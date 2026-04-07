# SID / MiniOneRec Idea Discovery Workspace

Date: 2026-04-07

这个目录现在分成了 4 层，方便把“自动 idea discovery 过程”和“我们后续人工推进的报告/方案”分开看。

## 目录结构

### 1. 初始 idea discovery 过程文件

这些文件来自最早那轮自动化 idea discovery，保留原始顺序，方便回溯：

- `00_motivation_and_scope.md`
- `01_local_context.md`
- `02_arxiv_search_log.md`
- `03_latest_paper_notes.md`
- `04_literature_landscape.md`
- `05_idea_candidates.md`
- `06_novelty_check.md`
- `07_critical_review.md`
- `PROCESS_LOG.md`
- `IDEA_REPORT.md`
- `refine-logs/`

这部分更像“第一次找方向时的原始工作区”。

### 2. `10_project_reports/`

这部分放的是和当前仓库、实验复现、研究路线总结直接相关的报告：

- `01_project_understanding_report.md`
- `02_reproduction_progress.md`
- `03_research_story_summary.md`

如果你想快速回顾“我们现在到底做到了哪里”，先看这里。

### 3. `20_current_working_idea/`

这部分放的是我们后续真正围绕当前主线持续修改、讨论和收缩后的 idea 文档：

- `01_research_brief.md`
- `02_idea_draft.tex`
- `03_current_idea_report.md`
- `04_v0_5_review.md`
- `05_v0_5_experiment_plan.md`
- `06_refine_logs_current/`

这部分更像“当前有效工作区”。

### 4. `20_current_working_idea/06_refine_logs_current/`

这里保存的是我们后续人工 refine、评审、收缩主线时的日志与方案：

- `FINAL_PROPOSAL.md`
- `EXPERIMENT_PLAN.md`
- `EXPERIMENT_TRACKER.md`
- `PIPELINE_SUMMARY.md`
- `REFINEMENT_REPORT.md`
- `REVIEW_SUMMARY.md`
- `round-0-initial-proposal.md`
- `round-1-review.md`
- `round-1-refinement.md`
- `score-history.md`
- `REFINE_STATE.json`

这部分和根目录原先的 `refine-logs/` 对应，现在已经整体搬到这里。

## 当前建议阅读顺序

如果你要快速理解当前主线，建议按这个顺序看：

1. `10_project_reports/03_research_story_summary.md`
2. `20_current_working_idea/03_current_idea_report.md`
3. `20_current_working_idea/05_v0_5_experiment_plan.md`
4. `20_current_working_idea/06_refine_logs_current/FINAL_PROPOSAL.md`

## 当前最重要的结论

- 全局 collaborative tokenizer 重构赛道已经比较拥挤。
- 我们自己的诊断更支持 `local leaf ambiguity` 是当前真实瓶颈。
- 前端最小融合 `E1/C1` 已经 hard stop。
- 当前更值得继续推进的是 `ACLR / backend-local refinement` 主线。
