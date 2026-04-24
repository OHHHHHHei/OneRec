# Classified Stage Manifest（分类阶段清单）

Status（状态）: `archived（归档）`
Snapshot date（快照日期）: `2026-04-24`

This manifest（清单） classifies the archived research workspace（已归档研究工作区） after the 2026-04-24 physical archive（物理归档）.

## Asset Counts（资产计数）

- Experiment stage folders（实验阶段目录）: `61`
- Research configs（研究配置） in `archived_workspace/config/experiments/`: `38`
- Root-level MGR-SID scripts（根层 MGR-SID 脚本）: `52`
- Main method code（主方法代码）: `archived_workspace/src/onerec/experiments/mgr_sid/`

## Research Asset Boundaries（研究资产边界）

### Method Code（方法代码）

- `archived_workspace/src/onerec/experiments/mgr_sid/`
  - `train_v1.py`
  - `train_v2.py`
  - `train_collab_ranking_sid.py`
  - `graph_bank.py`
  - `transplanted_graph_bank.py`
  - `paper_transplants.py`
  - `probe.py`

### Configs（配置）

- Archived current-level research configs（已归档当前层研究配置）: `archived_workspace/config/experiments/`
- Archived legacy configs（已归档历史配置）: `archived_workspace/config/archive/2026-04-18_pre_r720_legacy_experiments/`
- Archived top-level ACLR/TDCF configs（已归档顶层 ACLR/TDCF 配置）: `archived_workspace/config/legacy_top_level/`
- Non-MGR OneRec baseline configs（非 MGR OneRec 基线配置） stay in top-level `config/*.yaml`.

### Scripts（脚本）

- Archived current-level research scripts（已归档当前层研究脚本）: `archived_workspace/scripts/experiment_mgr_sid_*`, `archived_workspace/scripts/launch_mgr_sid_*`
- Archived legacy scripts（已归档历史脚本）:
  - `archived_workspace/scripts/archive/pre_r720_legacy_experiments_20260418/`
  - `archived_workspace/scripts/archive/retired_prior_diagnostics/`
- Standard OneRec baseline entrypoints（标准 OneRec 基线入口） stay at repository root（仓库根目录）: `sid_train.sh`, `sid_generate.sh`, `sft.sh`, `rl.sh`, `evaluate.sh`.

### Data / Outputs（数据 / 输出）

- Research data variants（研究数据变体）: `data_experiment/`
- Lightweight result artifacts（轻量结果产物）: `results/experiments/`
- Archived root temp diagnostics（已归档根目录临时诊断）: `research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/root_tmp_diagnostics/`
- Large checkpoints（大模型权重）: `/data/leejt/OneRec/output_weights/experiments/`
- Logs（日志）: `logs/`, `wandb/`, `temp/`

## Stage Families（阶段族）

### A. Pipeline Reproduction and First Hierarchy Probe（流水线复现与首次层级探针）

- `2026-04-09_pipeline_alignment_and_reproduction`
- `2026-04-09_mgr_sid_v1_upstream`
- `2026-04-10_mgr_sid_data_experiment_convert`
- `2026-04-10_mgr_sid_sft_eval_industrial`

Verdict（裁决）:

- Useful for provenance（可追溯性）.
- Not an active method branch（活跃方法分支）.

### B. V2, Retention, and Codebook-Space Search（V2、保持项与码本空间搜索）

- `2026-04-11_mgr_sid_v2_proxy_sanity`
- `2026-04-11_mgr_sid_tokenizer_v2_r005`
- `2026-04-11_mgr_sid_v2_recipe_isolation_industrial`
- `2026-04-11_mgr_sid_v2_sft_desc_align_p05_industrial`
- `2026-04-11_mgr_sid_v2_sft_eval_industrial`
- `2026-04-12_mgr_sid_v2_rl_on_p05_industrial`
- `2026-04-13_mgr_sid_stage2_stopgrad_industrial`
- `2026-04-13_mgr_sid_stage2_semantic_retention_industrial`
- `2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial`
- `2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial`
- `2026-04-14_mgr_sid_stage3_prefix_retained_industrial`
- `2026-04-14_mgr_sid_stage3_sft_eval_industrial`
- `2026-04-14_mgr_sid_learnability_probe_baseline_check`

Verdict（裁决）:

- `v2_on_p05 -> RL` remains reference evidence（参考证据）.
- Stage-2 / Stage-3 variants did not establish a stronger downstream SID space（更强下游 SID 空间）.
- Retention-only refinements（仅保持项修复） are closed.

### C. Graph-Carrier Upgrade Search（图载体升级搜索）

- `2026-04-15_mgr_sid_tagcf_m0_attribute_graphs`
- `2026-04-15_mgr_sid_tagcf_r510_attr_mid`
- `2026-04-15_mgr_sid_tagcf_r510_sft_eval_industrial`
- `2026-04-15_mgr_sid_tagcf_r511_mix_mid`
- `2026-04-15_mgr_sid_fagsp_r520_mid_cascade`
- `2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial`
- `2026-04-16_mgr_sid_mgdcf_coarse_diagnostics_industrial`
- `2026-04-16_mgr_sid_mgdcf_coarse_isolation_industrial`
- `2026-04-16_mgr_sid_r542_mgdcf_coarse_industrial`

Verdict（裁决）:

- TAGCF / FaGSP / MGDCF graph carriers（图载体） did not become reliable promotion gates（可靠推进门槛）.
- Broad graph-carrier changes（宽图载体改动） often harmed routing（路由） or failed downstream transfer（下游迁移）.

### D. Selective Separation and Mid Pull-Push（选择性分离与中层推拉）

- `2026-04-16_mgr_sid_selective_separation_pair_diagnostics_industrial`
- `2026-04-16_mgr_sid_r610_selective_separation_industrial`
- `2026-04-16_mgr_sid_diagnostic_audit_industrial`
- `2026-04-16_mgr_sid_r630_mid_pull_push_industrial`
- `2026-04-16_mgr_sid_r630c_sft_eval_industrial`

Verdict（裁决）:

- Pair-source narrowing（样本对来源收窄） was conceptually useful.
- R630c downstream adjudication（下游裁决） was negative.
- Prior diagnostics（前验诊断） are not reliable promotion gates（可靠推进门槛）.

### E. Seq2Graph and Constraint Restoration（Seq2Graph 与约束恢复）

- `2026-04-16_mgr_sid_d640_seq2graph_lite_graph_audit_industrial`
- `2026-04-16_mgr_sid_r640_seq2graph_lite_industrial`
- `2026-04-17_mgr_sid_r640c_sft_eval_industrial`
- `2026-04-17_mgr_sid_r650_seq2graph_push_pull_industrial`
- `2026-04-17_mgr_sid_r650a_sft_eval_industrial`
- `2026-04-17_mgr_sid_r660_constraint_restoration_industrial`

Verdict（裁决）:

- Seq2Graph-lite（轻量 Seq2Graph） produced non-catastrophic tokenizer candidates（非灾难性分词器候选） in some settings.
- Downstream SFT（下游监督微调） did not validate the branch.
- Constraint restoration（约束恢复） did not rescue the core objective（核心目标）.

### F. Clean L2, CoST-Inspired, and Hierarchical Collaboration（干净 L2、CoST 启发与层级协同）

- `2026-04-18_mgr_sid_r670_clean_l1_semantic_l2_push_pull_industrial`
- `2026-04-18_mgr_sid_r680_l1_smooth_l2_contrastive_multihop_industrial`
- `2026-04-18_mgr_sid_r680a_sft_eval_industrial`
- `2026-04-18_mgr_sid_r690_cost_inspired_contrastive_quantization_industrial`
- `2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial`
- `2026-04-18_mgr_sid_r700_semantic_collab_intersection_industrial`
- `2026-04-18_mgr_sid_r710_v2_no_semantic_retention_industrial`
- `2026-04-18_mgr_sid_r720_l2_ranking_contrastive_industrial`

Verdict（裁决）:

- Clean L2 interfaces（干净第二层接口） were healthier than broad graph injection（宽图注入） but still not enough.
- CoST-inspired contrastive quantization（受 CoST 启发的对比量化） did not beat the reference baseline（参考基线）.
- R720 became the bridge to later collab-ranking（协同排序）, but R720a itself was negative.

### G. Collab-Ranking, Minimal Edit, and QCR（协同排序、最小编辑与 QCR）

- `2026-04-19_mgr_sid_collab_ranking_local_multihop_mid_industrial`
- `2026-04-19_mgr_sid_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_industrial`
- `2026-04-19_mgr_sid_collab_ranking_local_multihop_mid_l1_rescue_industrial`
- `2026-04-19_mgr_sid_collab_ranking_prism_coarse_local_multihop_mid_industrial`
- `2026-04-20_mgr_sid_collab_ranking_k1_128_l1_inverse_ambiguity_industrial`
- `2026-04-20_mgr_sid_collab_ranking_l1w075_l1_inverse_ambiguity_industrial`
- `2026-04-20_mgr_sid_highconf_l1_collab_ranking_industrial`
- `2026-04-20_mgr_sid_original_l3_collab_local_industrial`
- `2026-04-20_mgr_sid_semantic_l1_collab_deeper_industrial`
- `2026-04-21_mgr_sid_original_l2_ambiguity_aware_industrial`
- `2026-04-21_mgr_sid_original_l2_multihop_ranking_industrial`
- `2026-04-21_mgr_sid_original_l2_ranking_ambiguity_aware_industrial`
- `2026-04-21_mgr_sid_original_l3_ambiguity_aware_industrial`
- `2026-04-21_mgr_sid_qcr_l2_conflict_ranking_industrial`
- `2026-04-21_mgr_sid_v2_l1cap128_industrial`

Verdict（裁决）:

- Local multihop（局部多跳） and low-disturbance edits（低扰动编辑） were the most informative variants.
- None beat the strongest baseline（最强基线） on primary `NDCG@10`（归一化折损累计增益@10）.
- QCR（量化冲突感知排序） improved tokenizer health（分词器健康） but hurt downstream learnability（下游可学习性）.
- Hard L1 capacity reduction（硬性第一层容量缩减） is closed.

## Idea and Literature Workspaces（想法与文献工作区）

- `idea-discovery/2026-04-07-sid-collab/`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/`
- `papers/`

Verdict（裁决）:

- Preserved as archived background（归档背景） and literature support（文献支撑）.
- Not active current-state documents（活跃当前状态文档）.
