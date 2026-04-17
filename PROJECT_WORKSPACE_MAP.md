# Project Workspace Map（项目工作区地图）

Status（状态）: `reference（参考）`

This file is the shortest reliable entry point（最短可靠入口） for the current repository layout（当前仓库布局）.

If you need to sync to the project quickly, read in this order:

1. `DOCUMENTATION_MAINTENANCE_WORKFLOW.md`
2. `research-progress-log/CURRENT_STATE.md`
3. `research-progress-log/experiment_registry/README.md`
4. `research-progress-log/experiment_registry/downstream_scoreboard.csv`
5. `research-progress-log/experiment_launches/README.md`
6. `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/18_mgr_sid_v2_ambiguity_aware_method.md`
7. `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`

## Folder Roles（目录角色）

### `idea-discovery/`

Research-direction and method-design workspace（研究方向与方法设计工作区）.

- current mainline:
  `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/`
- archived earlier branches stay under sibling `archive/` folders

Use this directory for:

- direction notes（方向笔记）
- method design（方法设计）
- plan / tracker documents（计划 / 跟踪器文档）
- literature-to-method mapping（文献到方法映射）

### `research-progress-log/`

Canonical experiment-and-interpretation log（权威实验与解释日志）.

- primary current-state document（主要当前状态文档）:
  `CURRENT_STATE.md`
- experiment index:
  `experiment_launches/README.md`
- milestone narrative（里程碑叙事）:
  `research_progress_log.tex`

Use this directory for:

- stage summaries（阶段总结）
- per-run conclusions（单次运行结论）
- compact analysis reports（紧凑分析报告）

Do not treat this directory as raw artifact storage.

### `results/`

Lightweight result artifacts（轻量结果产物）.

Use this directory for:

- final merged `json` outputs（最终合并 `json` 输出）
- diagnostic summaries（诊断总结）
- recovered legacy outputs（恢复的历史输出）

### `logs/`

Training and evaluation logs（训练与评测日志）.

Keep logs here, not on the data disk.

### `output/`

Repository-side entry points only（仓库侧入口）.

Large checkpoints are now expected to live on the data disk under:

- `/data/leejt/OneRec/output_weights`

Local `output/` may contain only lightweight links or compatibility paths.

### `paper-mgr-sid-draft/` and `paper-mgr-sid-draft-zh/`

Current English and Chinese paper drafts（当前中英文论文草稿）.

Keep only:

- source `.tex`（源 `.tex` 文件）
- bibliography（参考文献）
- final `.pdf`（最终 `.pdf`）

Generated LaTeX auxiliaries should not be kept unless actively debugging compilation.

## Current Canonical Document Set（当前权威文档集合）

### Current status and registry（当前状态与总账）

- `DOCUMENTATION_MAINTENANCE_WORKFLOW.md`
- `research-progress-log/CURRENT_STATE.md`
- `research-progress-log/experiment_registry/README.md`
- `research-progress-log/experiment_registry/downstream_scoreboard.csv`
- `experiment_results.csv` as legacy wide registry（历史宽表总账）

### Method and direction（方法与方向）

- `idea-discovery/2026-04-08_sid_collab_signal_injection/README.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/RESEARCH_DIRECTION.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/CURRENT_TASK_ALIGNMENT.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/18_mgr_sid_v2_ambiguity_aware_method.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`

### Experiment status（实验状态）

- `research-progress-log/experiment_launches/README.md`
- `research-progress-log/research_progress_log.tex`

### Latest strongest run family（当前最强运行线）

- `research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/RESULTS.md`
- `research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/RESULTS.md`
- `research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/EVAL_ANALYSIS.md`

## Cleanup Policy（清理规则）

- keep a small number of canonical summaries（权威总结） in active folders（活跃目录）
- move superseded split notes（被替代的拆分笔记） into `archive/`
- keep raw `.json` / `.csv` analysis artifacts（分析产物） next to the run（运行） that generated them
- keep filenames stable（文件名稳定） for externally referenced core docs（外部引用的核心文档） when possible
