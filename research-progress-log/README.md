# Research Progress Log（研究进展日志）

Status（状态）: `navigation（导航页）`

Last updated（更新日期）: `2026-05-11`

This directory should stay lightweight（保持轻量）. It is not a place for every launch note（启动记录） or temporary experiment thought（临时实验想法）.

## Live Documents（实时维护文档）

Use these as the default entry points（默认入口）:

1. [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md): current claim boundary（当前主张边界）, strongest evidence（最强证据）, and next checkpoint（下一检查点）.
2. [experiment_registry/README.md](/home/leejt/OneRec/research-progress-log/experiment_registry/README.md): registry workflow（总账流程）.
3. [experiment_registry/downstream_scoreboard.csv](/home/leejt/OneRec/research-progress-log/experiment_registry/downstream_scoreboard.csv): downstream result scoreboard（下游结果看板）.
4. [experiment_registry/tokenizer_registry.csv](/home/leejt/OneRec/research-progress-log/experiment_registry/tokenizer_registry.csv): tokenizer registry（分词器总账）.
5. [experiment_registry/sft_registry.csv](/home/leejt/OneRec/research-progress-log/experiment_registry/sft_registry.csv): SFT registry（监督微调总账）.
6. [experiment_registry/rl_registry.csv](/home/leejt/OneRec/research-progress-log/experiment_registry/rl_registry.csv): RL registry（强化学习总账）.

Current method branch（当前方法分支）:

- [MAINLINE.md](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/MAINLINE.md)

## Snapshot Documents（快照文档）

These are useful for history or advisor discussion（历史追溯或导师讨论）, but should not be treated as live state（实时状态）:

- [archive/](/home/leejt/OneRec/research-progress-log/archive)
- [2026-05-11_markdown_doc_cleanup/moved_files.txt](/home/leejt/OneRec/research-progress-log/archive/2026-05-11_markdown_doc_cleanup/moved_files.txt): list of archived Markdown files（已归档 Markdown 文件清单）.
- legacy LaTeX artifacts（历史 LaTeX 产物） were archived under [legacy_latex/](/home/leejt/OneRec/research-progress-log/archive/2026-05-11_markdown_doc_cleanup/research-progress-log/legacy_latex).

## Maintenance Policy（维护策略）

- Finalized results（定稿结果） go into split registries（分表总账） first.
- Current interpretation（当前解释） goes into `CURRENT_STATE.md`.
- Method-specific detail（方法细节） goes into the active branch `MAINLINE.md`.
- Advisor-facing text（导师汇报文字） goes under `advisor_reports/` as snapshot（快照）.
- Launch state（启动状态）, tmux sessions（会话）, GPU assignments（显卡分配）, and temporary debugging（临时调试） should stay in logs（日志） or conversation（对话）, not long-term docs（长期文档）.

Legacy（历史遗留）:

- [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv) remains a legacy wide registry（历史宽表总账） for compatibility（兼容） and migration（迁移） only.

## Reading Order（推荐阅读顺序）

1. [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
2. [MAINLINE.md](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/MAINLINE.md)
3. [experiment_registry/downstream_scoreboard.csv](/home/leejt/OneRec/research-progress-log/experiment_registry/downstream_scoreboard.csv)
4. Needed advisor report（需要的导师汇报） under [advisor_reports/](/home/leejt/OneRec/research-progress-log/advisor_reports)
5. Historical archive（历史归档） only when needed.
