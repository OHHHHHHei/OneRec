# Retired Prior Diagnostics（已退役前验诊断）

Status（状态）: `archived（归档）`
Archived on（归档日期）: `2026-04-16`

This directory stores prior diagnostics（前验诊断） that were removed from the active
`scripts/` root after the `S000` diagnostic audit（诊断审计） concluded they were not
decision-usable（不可用于决策） for tokenizer promotion（分词器推进）.

These scripts are preserved only for provenance（来源追踪） and historical
reconstruction（历史复盘）.

They must not be used as:

- promotion gates（推进门槛）
- tokenizer ranking criteria（分词器排序准则）
- justification for skipping downstream `SFT -> evaluate`（监督微调到评测）

Retired scripts（已退役脚本）:

- `sid_diagnostics.py`
- `experiment_mgr_sid_v1_local_ambiguity_analysis.py`
- `experiment_mgr_sid_prefix_collaborative_consistency.py`
- `experiment_mgr_sid_stage2_interface_diagnostics.py`
- `experiment_mgr_sid_stage3_diagnostics.py`
- `experiment_mgr_sid_graph_bank_probe.py`
- `experiment_mgr_sid_coarse_local_graph_diagnostics.py`
- `experiment_mgr_sid_selective_separation_pair_diagnostics.py`

Explicit exception（明确例外）:

- `experiment_mgr_sid_v2_proxy_sanity.py` stays in the active `scripts/` root because it
  is still part of the code-aligned provenance（代码对齐来源） for the current `v2`
  ambiguity prior（歧义先验）, even though it must not be reused as a tokenizer
  promotion metric（分词器推进指标）.
