# Working Idea: Graph Hierarchy v1

This folder keeps the active materials for the current mainline:

- hierarchy-aware graph-structured collaborative integration into semantic SID
- graphs as collaborative-information carriers, not graph-encoder benchmarks
- tokenizer-first `MGR-SID v2` design and end-to-end validation

It is intentionally separated from the archived `../archive/2026-04-08_working_idea_hierarchy_aware_v1_superseded/` because this round shifts from simple multi-view fusion toward graph-native SID supervision and ambiguity-aware refinement.

## Canonical Reading Order

1. `00_ACTIVE_CONTEXT.md`
2. `CURRENT_TASK_ALIGNMENT.md`
3. `01_PROBE_AND_EARLY_EVIDENCE.md`
4. `02_RELATED_WORK_AND_MODULE_MAP.md`
5. `17_ambiguity_proxy_literature_scan.md`
6. `18_mgr_sid_v2_ambiguity_aware_method.md`
7. `19_mgr_sid_current_method_code_aligned_formulas.md`
8. `refine-logs/README.md`
9. `refine-logs/EXPERIMENT_PLAN_TOKENIZER_V2.md`
10. `refine-logs/EXPERIMENT_TRACKER_TOKENIZER_V2.md`
11. `refine-logs/EXPERIMENT_PLAN_STAGE2_RETENTION.md`
12. `refine-logs/EXPERIMENT_TRACKER_STAGE2_RETENTION.md`

## Archived Inside This Folder

- `archive/2026-04-11_doc_cleanup_v1_superseded/`
  - early discovery trail
  - original `v1` proposal and pre-`v2` experiment plan / tracker
- `archive/2026-04-11_doc_cleanup_duplicates/`
  - overlapping related-work and module-review notes
  - broader `v2` ambiguity-aware full-pipeline plan / tracker that is no longer the active execution path
- `archive/2026-04-12_doc_reorg_merged_sources/`
  - split notes that were merged into the new canonical summaries:
    - early probe notes
    - paper/module mapping notes
    - old process note

## Current status

- active mainline: `MGR-SID v2` tokenizer-first
- role of graphs: collaborative-information carriers for hierarchy-aware SID supervision
- strongest current `G_mid` candidate: `fagsp_mid_base`
- best current downstream recipe: `title_history2sid_on + desc_align_p05`
- current strongest end-to-end line: `v2_on_p05 -> RL`
- current bottleneck: mid-beam retention (`top5/top10`), not whether the tokenizer works at all
- current execution stage: stage-2 retention-targeted refinement with first-round results
  - `R202a`: best tokenizer-side branch so far
  - `R205`: negative semantic-retention result in current form
  - `R208`: downstream screen completed, but did not beat current `v2_on_p05`
