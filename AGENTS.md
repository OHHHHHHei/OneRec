# AGENTS.md

## Environment

Before running project commands in this repository, activate the Conda environment:

```bash
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec
```

Use `MiniOneRec` for:

- Python CLI checks
- unit tests
- config validation
- lightweight analysis scripts

## Output Policy

For future training runs, store large weight/checkpoint outputs on the data disk instead of the repository `./output` directory.

Preferred root:

```bash
/data/leejt/OneRec/output_weights
```

Keep lightweight artifacts in the repository:

- `logs/`
- `results/`
- `research-progress-log/`
- `research-progress-log/experiment_registry/`
- `experiment_results.csv` as legacy wide registry（历史宽表总账）

## Writing Policy

When using English technical terms in user-facing writing, add a Chinese translation in parentheses.

Examples:

- `prefix stability` (`前缀稳定性`)
- `learnability` (`可学习性`)
- `routing` (`路由`)

## Documentation Maintenance Workflow

When creating, updating, or cleaning project documentation, follow:

- `/home/leejt/OneRec/DOCUMENTATION_MAINTENANCE_WORKFLOW.md`

Do not treat documentation maintenance as "update after every action".
Use a sparse checkpoint workflow instead.

Update documentation only at these checkpoints:

1. A new idea / method design is finalized enough to be reused or reviewed.
2. A new experiment validation is finalized and should enter the registry.
3. The implemented method or code-aligned formula changes.
4. The current strongest line, active question, or next-step decision changes.
5. A stage is finished and old plans / trackers / summaries should be downgraded, frozen, or archived.

Do not update long-term documentation merely because an experiment was launched, a tmux session started, a GPU assignment changed, or debugging / smoke testing happened.

Use these target documents by default:

- current state:
  - `/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md`
- experiment registry:
  - `/home/leejt/OneRec/research-progress-log/experiment_registry/README.md`
  - `/home/leejt/OneRec/research-progress-log/experiment_registry/tokenizer_registry.csv`
  - `/home/leejt/OneRec/research-progress-log/experiment_registry/sft_registry.csv`
  - `/home/leejt/OneRec/research-progress-log/experiment_registry/rl_registry.csv`
  - `/home/leejt/OneRec/research-progress-log/experiment_registry/downstream_scoreboard.csv`
- method narrative:
  - `/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/18_mgr_sid_v2_ambiguity_aware_method.md`
- code-aligned method spec:
  - `/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`
- stage index:
  - `/home/leejt/OneRec/research-progress-log/experiment_launches/README.md`

Rules:

- Prefer updating an existing canonical document over creating a new summary document.
- Do not create another current-state summary if the change belongs in `CURRENT_STATE.md`.
- Do not record a finalized experiment result only in prose; update the corresponding split registry（分表总账） first or in the same task.
- Do not hand-write long rows into `/home/leejt/OneRec/experiment_results.csv`; it is now a legacy wide registry（历史宽表总账） for compatibility and migration.
- Follow the experiment recording pipeline（实验记录流水线） in `/home/leejt/OneRec/research-progress-log/experiment_registry/README.md`: launch/running status（启动/运行中状态） should usually stay in the conversation/logs unless it is part of a finalized method design or changes the next-step decision; only finalized results（定稿结果） belong in split registries（分表总账）.
- Treat dated notes as `snapshot` (`快照`) or `discussion-only` (`仅讨论`) by default unless they are explicitly designated as canonical.
