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
12. `2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/README.md`
13. `2026-04-14_mgr_sid_stage3_prefix_retained_industrial/README.md`
14. `2026-04-14_mgr_sid_stage3_sft_eval_industrial/README.md`
15. `2026-04-14_mgr_sid_learnability_probe_baseline_check/README.md`
16. `2026-04-15_mgr_sid_tagcf_m0_attribute_graphs/README.md`
17. `2026-04-15_mgr_sid_tagcf_r510_attr_mid/README.md`
18. `2026-04-15_mgr_sid_tagcf_r511_mix_mid/README.md`
19. `2026-04-15_mgr_sid_fagsp_r520_mid_cascade/README.md`

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
  - this is the current active execution stage

## Artifact Policy Inside Run Folders

Inside each run folder:

- `README.md`:
  launch context and runtime metadata
- `RESULTS.md`:
  the canonical result summary
- extra `TOPK_*`, `EVAL_*`, or `*_diagnostics.*`:
  supporting analysis artifacts
- raw `.json` / `.csv`:
  keep them for traceability; they do not need their own prose note unless they support a claim directly
