# CLAUDE.md

Status（状态）: `canonical-policy（权威规范）`

This file defines the repository-specific constraints for Claude Code（Claude Code） in this project.

It should stay aligned with:

- `/home/leejt/OneRec/AGENTS.md`
- `/home/leejt/OneRec/DOCUMENTATION_MAINTENANCE_WORKFLOW.md`

## Environment（环境）

Before running project commands in this repository, activate the Conda environment（Conda 环境）:

```bash
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec
```

Use `MiniOneRec` for:

- Python CLI checks（Python 命令行检查）
- unit tests（单元测试）
- config validation（配置校验）
- lightweight analysis scripts（轻量分析脚本）

## Output Policy（输出规则）

For training runs, store large checkpoints / weights（大型检查点 / 权重） on the data disk（数据盘）, not in the repository `./output` directory.

Preferred root（推荐根目录）:

```bash
/data/leejt/OneRec/output_weights
```

Keep lightweight artifacts（轻量产物） in the repository:

- `logs/`
- `results/`
- `research-progress-log/`
- `research-progress-log/experiment_registry/`
- `experiment_results.csv` as legacy wide registry（历史宽表总账）

## Writing Policy（写作规则）

When using English technical terms（英文技术术语） in user-facing writing（面向用户的写作）, add a Chinese translation（中文翻译） in parentheses.

Examples:

- `prefix stability`（前缀稳定性）
- `learnability`（可学习性）
- `routing`（路由）

## Documentation Maintenance Workflow（文档维护工作流）

When creating, updating, or cleaning documentation（文档）, follow:

- `/home/leejt/OneRec/DOCUMENTATION_MAINTENANCE_WORKFLOW.md`

Do not treat documentation maintenance（文档维护） as "update after every action".
Use an event-driven workflow（事件驱动工作流） instead.

You must update documentation after these events:

1. A new experiment（实验） is formally launched.
2. A new result（结果） is finalized and should enter the registry（总账）.
3. The implemented method（已实现方法） or code-aligned formula（代码对齐公式） changes.
4. The current strongest line（当前最强主线）, active question（当前活跃问题）, or next-step decision（下一步决策） changes.
5. A stage（阶段） is finished and old plans / trackers / summaries（计划 / 跟踪器 / 总结） should be downgraded, frozen, or archived（归档）.

Use these target documents by default:

- current state（当前状态）:
  - `/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md`
- experiment registry（实验总账）:
  - `/home/leejt/OneRec/research-progress-log/experiment_registry/README.md`
  - `/home/leejt/OneRec/research-progress-log/experiment_registry/tokenizer_registry.csv`
  - `/home/leejt/OneRec/research-progress-log/experiment_registry/sft_registry.csv`
  - `/home/leejt/OneRec/research-progress-log/experiment_registry/rl_registry.csv`
  - `/home/leejt/OneRec/research-progress-log/experiment_registry/downstream_scoreboard.csv`
- method narrative（方法叙事）:
  - `/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/18_mgr_sid_v2_ambiguity_aware_method.md`
- code-aligned method spec（代码对齐方法说明）:
  - `/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`
- stage index（阶段索引）:
  - `/home/leejt/OneRec/research-progress-log/experiment_launches/README.md`

Rules:

- Prefer updating an existing canonical document（权威文档） over creating a new summary document（总结文档）.
- Do not create another current-state summary（当前状态总结） if the change belongs in `CURRENT_STATE.md`.
- Do not record a finalized experiment result（已定稿实验结果） only in prose（文字）; update the corresponding split registry（分表总账） first or in the same task.
- Do not hand-write long rows（长行） into `/home/leejt/OneRec/experiment_results.csv`; it is now a legacy wide registry（历史宽表总账） for compatibility（兼容） and migration（迁移）.
- Follow the experiment recording pipeline（实验记录流水线） in `/home/leejt/OneRec/research-progress-log/experiment_registry/README.md`: launch/running status（启动/运行中状态） belongs in the stage README（阶段快照） and `CURRENT_STATE.md`; only finalized results（定稿结果） belong in split registries（分表总账）.
- Treat dated notes（带日期笔记） as `snapshot`（快照） or `discussion-only`（仅讨论） by default unless they are explicitly designated as canonical（权威）.

## Recommended Bootstrap Reading Order（推荐启动阅读顺序）

If Claude Code（Claude Code） needs to sync quickly, read in this order:

1. `/home/leejt/OneRec/DOCUMENTATION_MAINTENANCE_WORKFLOW.md`
2. `/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md`
3. `/home/leejt/OneRec/research-progress-log/experiment_registry/README.md`
4. `/home/leejt/OneRec/research-progress-log/experiment_registry/downstream_scoreboard.csv`
5. `/home/leejt/OneRec/PROJECT_WORKSPACE_MAP.md`
