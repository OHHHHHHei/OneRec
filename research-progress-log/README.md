# Research Progress Log（研究进展日志）

Status（状态）: `navigation（导航页）`

这个目录现在只保留两个核心入口：

1. 当前主线的唯一事实源
2. 实验总账与各实验阶段的可回溯记录入口

## Canonical Files（权威入口文件）

- `CURRENT_STATE.md`
- `experiment_registry/README.md`
- `experiment_registry/downstream_scoreboard.csv`
- `experiment_launches/README.md`

Legacy（历史遗留）:

- `../experiment_results.csv`

## Reference / Snapshot Files（参考 / 快照文件）

- `research_progress_log.tex`
- `research_progress_log.pdf`
- `archive/2026-04-14_post_stage2_review_materials/README.md`
  - 仅当你明确需要 archived brainstorm（归档头脑风暴） / external-review pack（外部评审包）时再读

## Recommended Reading Order（推荐阅读顺序）

1. `CURRENT_STATE.md`
2. `experiment_registry/README.md`
3. `experiment_registry/downstream_scoreboard.csv`
4. `experiment_launches/README.md`
5. 如需长篇里程碑叙事，再读 `research_progress_log.tex`
6. 如需查历史 brainstorm（头脑风暴） / review（评审），再读 `archive/.../README.md`

## Compile（编译）

```bash
cd /home/leejt/OneRec/research-progress-log
pdflatex -interaction=nonstopmode -halt-on-error research_progress_log.tex
```

## Usage Policy（使用规则）

- 当前状态只维护在 `CURRENT_STATE.md`
- 实验结果总账优先维护在 `experiment_registry/` 的 split registry（分表总账）中
- `../experiment_results.csv` 是 legacy wide registry（历史宽表总账），只用于旧脚本兼容和迁移追溯
- `research_progress_log.tex` 现在是 milestone narrative（里程碑叙事），不再承担 daily sync（日常同步）职责
- `experiment_launches/README.md` 只维护阶段索引，不再单独复述当前主线
- 细粒度 raw artifacts 保留在对应 run 目录下，不再在根目录平铺
- 顶层 brainstorm（头脑风暴） / external review（外部评审） / postmortem（复盘）文档如果不再属于当前活跃叙事，应移入 `archive/`
