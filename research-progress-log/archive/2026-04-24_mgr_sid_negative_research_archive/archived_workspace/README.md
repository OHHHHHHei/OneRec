# Archived Research Workspace（已归档研究工作区）

Status（状态）: `archived（归档）`
Snapshot date（快照日期）: `2026-04-24`

This folder physically archives（物理归档） the research-only workspace（研究专用工作区） that was previously visible in the main repository layout（主仓库布局）.

The goal is to make the visible repository structure（显式仓库结构） look like the clean OneRec baseline（干净 OneRec 基线）, while preserving all research configs（研究配置）, scripts（脚本）, and method code（方法代码） for traceability（可追溯性）.

## Contents（内容）

- `config/experiments/`: late active MGR-SID / QCR experiment configs（后期 MGR-SID / QCR 实验配置）.
- `config/archive/`: pre-R720 legacy experiment configs（R720 前历史实验配置）.
- `config/legacy_top_level/`: ACLR-lite（协同重排） and old TDCF SID configs that used to live in top-level `config/`.
- `scripts/`: all research and registry-operation scripts（研究与总账运维脚本） previously under top-level `scripts/`.
- `src/onerec/experiments/`: MGR-SID method implementation（方法实现） previously under the installed package（包）.
- `src/onerec/evaluate/collaborative_rerank.py`: ACLR-lite / collaborative rerank（协同重排） implementation removed from the baseline evaluate module（基线评估模块）.

## Visible Baseline After Archive（归档后的显式基线结构）

After this archive:

- `config/` contains only standard OneRec yaml configs（标准 OneRec 配置）.
- top-level `scripts/` is removed from the visible baseline workspace（显式基线工作区）.
- `src/onerec/` contains only baseline packages（基线包）:
  - `preprocess`
  - `sid`
  - `convert`
  - `sft`
  - `rl`
  - `evaluate`
  - `utils`
- `src/onerec/evaluate/pipeline.py` no longer imports ACLR-lite（协同重排） code.

## Policy（使用规则）

- Do not launch new experiments（新实验） from this folder.
- Use it only for provenance（追溯）, audit（审计）, or historical reproduction（历史复现）.
- If a future research line starts, create a new dated workspace（带日期工作区） instead of reactivating this archived code.

