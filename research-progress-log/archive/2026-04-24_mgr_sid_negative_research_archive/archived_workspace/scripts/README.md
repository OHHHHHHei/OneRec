# Archived Research Scripts（已归档研究脚本）

Status（状态）: `navigation（导航）`

Last updated（更新日期）: `2026-04-24`

## Archive Status（归档状态）

This directory contains archived research scripts（已归档研究脚本） for the MGR-SID / ACLR / QCR line（研究线）.

Do not launch them as active experiments（活跃实验） unless a future `CURRENT_STATE.md`（当前状态） explicitly reopens the line.

Archive entry（归档入口）:

- `/home/leejt/OneRec/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/README.md`
- `/home/leejt/OneRec/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/CLASSIFIED_STAGE_MANIFEST.md`

## Preserved Late-Stage Scripts（保留的后期脚本）

The following families（脚本族） remain at the root of this archived `scripts/` folder（已归档脚本目录） to preserve historical reproducibility（历史复现性） and avoid breaking artifact pointers（产物指针）:

- `experiment_mgr_sid_collab_ranking_*`
- `launch_mgr_sid_collab_ranking_*`
- `experiment_mgr_sid_original_*`
- `launch_mgr_sid_original_*`
- `experiment_mgr_sid_qcr_*`
- `launch_mgr_sid_qcr_*`
- late R690 / semantic-L1 / high-confidence L1 helpers（后期 R690 / 语义第一层 / 高置信第一层辅助脚本）

通用依赖脚本：

- `experiment_mgr_sid_v2_train.py`
- `experiment_mgr_sid_v1_generate.py`
- `experiment_mgr_sid_prepare_data.py`

Registry utilities（总账工具）也保留在这个归档脚本目录里：

- `validate_experiment_registry.py`
- `split_experiment_results_registry.py`
- `migrate_experiment_results_add_rl_columns.py`
- `migrate_experiment_results_add_tokenizer_registry.py`

## Archive（归档）

旧分支脚本已经移动到：

- `/home/leejt/OneRec/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/archive/pre_r720_legacy_experiments_20260418/`

这些脚本只用于 provenance（追溯）或 historical reproduction（历史复现），不应作为新实验入口。
