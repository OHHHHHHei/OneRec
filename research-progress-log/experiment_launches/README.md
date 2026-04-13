# Experiment Launch Index

This directory keeps the experiment-by-experiment record for the current MGR-SID line.

The goal of this index is simple:

- one stage = one canonical entry
- detailed raw artifacts stay inside the stage folder
- old flat notes are archived after they are merged into a clearer stage summary

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
  - `../../idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_PLAN_STAGE2_RETENTION.md`
- covers:
  - launch of the first retention-targeted tokenizer refinements
  - `R202a` as the best Block-2 structural branch
  - failure of the first semantic-retention KL implementation (`R205`)
  - first downstream screen of `R202a`, which shows that structural gains alone did not yet beat current `v2_on_p05`

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
