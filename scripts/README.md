# Active Scripts（活跃脚本）

Status（状态）: `navigation（导航）`

Last updated（更新日期）: `2026-04-18`

## Mainline（主线）

当前主线入口是 collaborative-ranking SID（协同排序 SID）：

- `launch_mgr_sid_collab_ranking_tmux.sh`
- `experiment_mgr_sid_collab_ranking_train_generate.sh`
- `experiment_mgr_sid_collab_ranking_pair_source.py`
- `experiment_mgr_sid_collab_ranking_train.py`

通用依赖脚本：

- `experiment_mgr_sid_v2_train.py`
- `experiment_mgr_sid_v1_generate.py`
- `experiment_mgr_sid_prepare_data.py`

Registry utilities（总账工具）仍保留在明面：

- `validate_experiment_registry.py`
- `split_experiment_results_registry.py`
- `migrate_experiment_results_add_rl_columns.py`
- `migrate_experiment_results_add_tokenizer_registry.py`

## Temporary Legacy（临时历史保留）

`R690b` 的 SFT/evaluate（监督微调/评测）仍在 tmux（终端复用器）中运行，所以这些入口暂时不移动：

- `launch_mgr_sid_r690b_sft_eval_tmux.sh`
- `experiment_mgr_sid_r690b_sft_eval_chain.sh`
- `experiment_mgr_sid_r690b_hier_cost_guided_train_generate.sh`

等当前运行结束并完成结果记录后，应把这些脚本也移入 archive（归档）。

## Archive（归档）

旧分支脚本已经移动到：

- `/home/leejt/OneRec/scripts/archive/pre_r720_legacy_experiments_20260418/`

这些脚本只用于 provenance（追溯）或历史复现，不应作为新实验入口。
