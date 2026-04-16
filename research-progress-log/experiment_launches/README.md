# Experiment Launch Index（实验启动索引）

Status（状态）: `stage-index（阶段索引）`

This directory keeps the experiment-by-experiment record for the current MGR-SID line.

The goal of this index is simple:

- one stage = one canonical entry
- detailed raw artifacts stay inside the stage folder
- old flat notes are archived after they are merged into a clearer stage summary

Every folder here is a historical stage record by default.
The latest stage is the current active execution path; earlier stages should be
read as provenance, not as the current optimization target.

This file is not the canonical current-state summary（权威当前状态摘要） anymore（不再承担该角色）.

If you want the live project status first, read:

1. `/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md`
2. `/home/leejt/OneRec/experiment_results.csv`

Diagnostic-only stages（仅诊断阶段） that failed the `S000` audit have been retired from
active reading and decision use（活跃阅读与决策用途）.

Archive pointer（归档指针）:

- `/home/leejt/OneRec/research-progress-log/archive/2026-04-16_retired_prior_diagnostics/README.md`

## Recommended Reading Order

1. `2026-04-09_pipeline_alignment_and_reproduction/README.md`
2. `2026-04-09_mgr_sid_v1_upstream/README.md`
3. `2026-04-10_mgr_sid_data_experiment_convert/README.md`
4. `2026-04-10_mgr_sid_sft_eval_industrial/RESULTS.md`
5. `2026-04-11_mgr_sid_tokenizer_v2_r005/RESULTS.md`
6. `2026-04-11_mgr_sid_v2_recipe_isolation_industrial/RESULTS.md`
7. `2026-04-12_mgr_sid_v2_rl_on_p05_industrial/RESULTS.md`
8. `2026-04-12_mgr_sid_v2_rl_on_p05_industrial/EVAL_ANALYSIS.md`
9. `2026-04-13_mgr_sid_stage2_stopgrad_industrial/README.md`
10. `2026-04-13_mgr_sid_stage2_semantic_retention_industrial/RESULTS.md`
11. `2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/RESULTS.md`
12. `2026-04-14_mgr_sid_stage3_prefix_retained_industrial/README.md`
13. `2026-04-14_mgr_sid_stage3_sft_eval_industrial/README.md`
14. `2026-04-15_mgr_sid_tagcf_m0_attribute_graphs/README.md`
15. `2026-04-15_mgr_sid_tagcf_r510_attr_mid/README.md`
16. `2026-04-15_mgr_sid_tagcf_r511_mix_mid/README.md`
17. `2026-04-15_mgr_sid_fagsp_r520_mid_cascade/README.md`
18. `2026-04-15_mgr_sid_r530_local_multihop_industrial/README.md`
19. `2026-04-16_mgr_sid_r542_mgdcf_coarse_industrial/README.md`
20. `2026-04-16_mgr_sid_mgdcf_coarse_isolation_industrial/README.md`
21. `2026-04-16_mgr_sid_r610_selective_separation_industrial/README.md`
22. `2026-04-16_mgr_sid_diagnostic_audit_industrial/SUMMARY.md`
23. `2026-04-16_mgr_sid_r630_mid_pull_push_industrial/README.md`

## Stage Map

### Stage A: Pipeline provenance and upstream-aligned reproduction

- canonical doc:
  `2026-04-09_pipeline_alignment_and_reproduction/README.md`
- covers:
  - default pipeline reproduction failure
  - upstream-style RQ-VAE launch/result
  - aligned training sanity
  - why final SID should be judged after `sid-generate`

### Stage B: Upstream-aligned `MGR-SID v1`

- canonical docs:
  - `2026-04-09_mgr_sid_v1_upstream/README.md`
  - `2026-04-09_mgr_sid_v1_upstream/RESULTS.md`
  - `2026-04-09_mgr_sid_v1_upstream/LOCAL_AMBIGUITY_BASELINE_VS_HIERARCHY.md`
- covers:
  - first positive tokenizer-side evidence
  - `hierarchy_reg` beating the semantic baseline on final SID structure

### Stage C: First downstream transfer and diagnostics

- canonical docs:
  - `2026-04-10_mgr_sid_data_experiment_convert/README.md`
  - `2026-04-10_mgr_sid_sft_eval_industrial/RESULTS.md`
  - `2026-04-10_mgr_sid_sft_eval_industrial/TOPK_STRUCTURAL_ANALYSIS.md`
- covers:
  - first `baseline vs hierarchy` SFT transfer
  - discovery that downstream gains are local and structure-sensitive

### Stage D: `v2` tokenizer construction

- canonical docs:
  - `2026-04-11_mgr_sid_v2_proxy_sanity/README.md`
  - `2026-04-11_mgr_sid_tokenizer_v2_r005/RESULTS.md`
  - `2026-04-11_mgr_sid_tokenizer_v2_r005/STRUCTURE_COMPARISON.md`
- covers:
  - ambiguity proxy sanity
  - first `v2` tokenizer run
  - tokenizer-side structural gains over both baseline and `v1`

### Stage E: `v2` downstream recipe isolation

- canonical docs:
  - `2026-04-11_mgr_sid_v2_recipe_isolation_industrial/RESULTS.md`
  - `2026-04-11_mgr_sid_v2_recipe_isolation_industrial/TOPK_ERROR_DISTRIBUTION_COMPARISON.md`
- covers:
  - four-cell recipe isolation
  - discovery that `title_history2sid_on + desc_align_p05` is the best current `v2` downstream recipe
  - strongest original recipe mismatch is mainly caused by `title_history2sid_off`

### Stage F: `v2_on_p05 -> RL`

- canonical docs:
  - `2026-04-12_mgr_sid_v2_rl_on_p05_industrial/RESULTS.md`
  - `2026-04-12_mgr_sid_v2_rl_on_p05_industrial/EVAL_ANALYSIS.md`
- covers:
  - end-to-end `v2` RL confirmation
  - current gap to strongest original MiniOneRec RL
  - the final remaining issue: mid-beam retention

### Stage G: Stage-2 retention-targeted tokenizer refinement

- canonical docs:
  - `2026-04-13_mgr_sid_stage2_stopgrad_industrial/README.md`
  - `2026-04-13_mgr_sid_stage2_semantic_retention_industrial/RESULTS.md`
  - `2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/RESULTS.md`
  - `../../idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/archive/2026-04-14_stage3_scope_cleanup/EXPERIMENT_PLAN_STAGE2_RETENTION.md`
- covers:
  - launch of the first retention-targeted tokenizer refinements
  - `R202a` as the best Block-2 structural branch
  - failure of the first semantic-retention KL implementation (`R205`)
  - first downstream screen of `R202a`, which shows that structural gains alone did not yet beat current `v2_on_p05`

### Stage H: Stage-2 interface diagnostics

- canonical docs:
  - `2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/README.md`
- covers:
  - explicit diagnosis that tokenizer-side structural cleanup did not
    automatically create a better downstream SID space
  - measurement of prefix rearrangement, code polysemy, and learnability
    probes
- current status:
  - retired from active decision workflow（活跃决策工作流） after `S000`

### Stage I: Stage-3 codebook-space search

- canonical docs:
  - `2026-04-14_mgr_sid_stage3_prefix_retained_industrial/README.md`
- covers:
  - current stage-3 tokenizer search
  - `R401b` and `R401d` as candidate SID codebook spaces
  - shift from “stay near baseline” to “find a better downstream SID space”
  - tokenizer-side diagnostics before downstream adjudication

### Stage J: Stage-3 downstream adjudication

- canonical docs:
  - `2026-04-14_mgr_sid_stage3_sft_eval_industrial/README.md`
- covers:
  - full downstream `SFT -> evaluate` for `R401b` and `R401d`
  - the negative verdict that structurally stronger stage-3 SID spaces did not beat current `v2_on_p05`
  - why stage-3 no longer remains the active execution stage

### Stage K: Learnability reinterpretation

- canonical docs:
  - `2026-04-14_mgr_sid_learnability_probe_baseline_check/README.md`
- covers:
  - baseline learnability probe for original semantic SID
  - evidence that original semantic may win on easier first-step routing while graph-informed SID spaces remain stronger on deeper conditional prediction
- current status:
  - retired from active decision workflow（活跃决策工作流） after `S000`

### Stage L: Graph-carrier upgrade exploration

- canonical docs:
  - `2026-04-15_mgr_sid_tagcf_m0_attribute_graphs/README.md`
  - `2026-04-15_mgr_sid_tagcf_r510_attr_mid/README.md`
  - `2026-04-15_mgr_sid_tagcf_r511_mix_mid/README.md`
  - `2026-04-15_mgr_sid_fagsp_r520_mid_cascade/README.md`
- covers:
  - the current shift from retention/codebook-space refinements to graph-carrier upgrades
  - `TAGCF`-inspired `semantics -> topology` attribute-mid exploration
  - deeper `FaGSP`-style `item-side cascade` `G_mid` exploration
  - this stage showed that repeated `G_mid` replacement still had no positive evidence

### Stage M: Coarse / Local diagnostics gate

- canonical docs:
  - `2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial/README.md`
- covers:
  - the first explicit offline diagnostics gate for under-tested `G_coarse / G_local`
  - evidence that current `user-segment` and `CIR` coarse candidates remain too close to baseline in their first formulations
  - evidence that multi-hop `G_local` really changes the graph, but does not by itself prove a baseline coverage failure
  - this stage decided that the first promoted tokenizer candidate should come from the local branch
- current status:
  - retired from active decision workflow（活跃决策工作流） after `S000`

### Stage N: First coarse/local tokenizer screen

- canonical docs:
  - `2026-04-15_mgr_sid_r530_local_multihop_industrial/README.md`
- covers:
  - the first tokenizer-side promotion after the coarse/local diagnostics gate
  - a shallow multi-hop `G_local`（浅层多跳局部图） screen that changes only `L3`
  - a clear negative verdict: changing only `L3` with the current local multi-hop formulation did not produce a viable tokenizer candidate
  - this stage re-centered the next exploration step on `G_coarse`（粗粒度图） reconstruction rather than deeper local expansion

### Stage O: `MGDCF` coarse reconstruction

- canonical docs:
  - `2026-04-16_mgr_sid_mgdcf_coarse_diagnostics_industrial/README.md`
  - `2026-04-16_mgr_sid_r542_mgdcf_coarse_industrial/README.md`
- covers:
  - the first true coarse-mother-graph reconstruction branch
  - diagnostics showing that `MGDCF` is not a near-baseline coarse tweak
  - the first tokenizer promotion that changes both `L1` and the mother graph of `G_mid`（中尺度图）
  - a mixed but ultimately negative tokenizer verdict: better than `R530a`, but still clearly weaker than current `v2` and stage-3 candidates
- current status:
  - the tokenizer run remains historical evidence（历史证据）, but its diagnostics-driven promotion path（诊断驱动推进路径） is retired after `S000`

### Stage P: `MGDCF` coarse-only isolation

- canonical docs:
  - `2026-04-16_mgr_sid_mgdcf_coarse_isolation_industrial/README.md`
- covers:
  - the direct follow-up to `R542a`
  - isolation of whether the `MGDCF` coarse branch itself is promising when `L2` is restored to `fagsp_mid_base`
  - parallel sensitivity check on `mgdcf_keep_ratio = 0.10 / 0.20`

### Stage Q: Selective-Separation pair diagnostics

- canonical docs:
  - `2026-04-16_mgr_sid_selective_separation_pair_diagnostics_industrial/README.md`
- covers:
  - the first explicit diagnostics gate for `semantic-close but collaboratively inconsistent`（语义接近但协同不一致） pair construction
  - evidence that `semantic-near + graph-non-neighbor`（语义接近 + 图上无邻接） is too broad as a first training rule
  - the first narrowed recommendation that phase-1 selective separation should start from `semantic-near + graph-weak`（语义接近 + 图弱连接）
- current status:
  - retired from active decision workflow（活跃决策工作流） after `S000`

### Stage R: Selective-Separation tokenizer screen

- canonical docs:
  - `2026-04-16_mgr_sid_r610_selective_separation_industrial/README.md`
- covers:
  - the first real training-time test of `reliability-aware selective separation`（可靠性感知选择性分离）
  - a minimal `L3`-only intervention on top of the current base `v2` tokenizer（分词器） backbone
  - promotion of `semantic-near + graph-weak`（语义接近 + 图弱连接） from diagnostics gate to formal tokenizer run
- current status:
  - preserved as a frozen tokenizer snapshot（冻结分词器快照）
  - not an active candidate（活跃候选）, because its supporting prior diagnostics（支撑前验诊断） were retired and no downstream verdict exists

### Stage S: Diagnostic audit

- canonical docs:
  - `2026-04-16_mgr_sid_diagnostic_audit_industrial/SUMMARY.md`
- covers:
  - the first retrospective audit（回顾性审计） of prior diagnostics（前验诊断） against historical downstream comparisons（历史下游对比）
  - a negative but important verdict: no current prior diagnostic（前验诊断） is strong enough to serve as a tokenizer promotion gate（分词器推进门槛）
  - retirement of generated collision（生成后冲突率）, local ambiguity（局部歧义）, and prefix collaborative consistency（前缀协同一致性） from the active decision workflow（活跃决策工作流）

### Stage T: Mid-only pull/push relaunch

- canonical docs:
  - `2026-04-16_mgr_sid_r630_mid_pull_push_industrial/README.md`
- covers:
  - the first selective-separation（选择性分离） relaunch after `S000`
  - compression of the objective（目标） into `pull-only / push-only / pull+push`（仅拉近 / 仅推远 / 拉近加推远）
  - restriction of auxiliary intervention（辅助干预） to `L2`（第 2 层） only
  - replacement of the old pair source（物品对来源） with `semantic-near + mid-graph-weak`（语义接近 + 中图弱连接）
- current status:
  - launched as the active tokenizer execution stage（活跃分词器执行阶段）
  - will be judged by downstream transfer（下游迁移）, not by retired prior diagnostics（已退役前验诊断）

## Artifact Policy Inside Run Folders

Inside each run folder:

- `README.md`:
  launch context and runtime metadata
- `RESULTS.md`:
  the canonical result summary
- extra `TOPK_*`, `EVAL_*`, or `*_diagnostics.*`:
  supporting analysis artifacts only; do not treat retired diagnostics（已退役诊断） as decision inputs
- raw `.json` / `.csv`:
  keep them for traceability; they do not need their own prose note unless they support a claim directly
