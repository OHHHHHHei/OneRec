# Working Idea: Graph Hierarchy v1

This folder now keeps only the active materials for the current mainline:

- hierarchy-aware graph-structured collaborative integration into semantic SID
- graphs as collaborative-information carriers, not graph-encoder benchmarks
- tokenizer-first `MGR-SID v2` design and validation

It is intentionally separated from the archived `../archive/2026-04-08_working_idea_hierarchy_aware_v1_superseded/` because this round shifts from simple multi-view fusion toward graph-native SID supervision.

## Active reading order

1. `CURRENT_TASK_ALIGNMENT.md`
2. `13_initial_probe_run_2026-04-09.md`
3. `14_paper_transplant_probe_run_2026-04-09.md`
4. `11_arxiv_related_work_by_question.md`
5. `12_modules_mapped_to_core_questions.md`
6. `17_ambiguity_proxy_literature_scan.md`
7. `18_mgr_sid_v2_ambiguity_aware_method.md`
8. `refine-logs/EXPERIMENT_PLAN_TOKENIZER_V2.md`
9. `refine-logs/EXPERIMENT_TRACKER_TOKENIZER_V2.md`

Optional running log:

- `PROCESS_LOG.md`

## Archived Inside This Folder

- `archive/2026-04-11_doc_cleanup_v1_superseded/`
  - early discovery trail
  - original `v1` proposal and pre-`v2` experiment plan / tracker
- `archive/2026-04-11_doc_cleanup_duplicates/`
  - overlapping related-work and module-review notes
  - broader `v2` ambiguity-aware full-pipeline plan / tracker that is no longer the active execution path

## Current status

- active mainline: `MGR-SID v2` tokenizer-first
- role of graphs: collaborative-information carriers for hierarchy-aware SID supervision
- strongest current `G_mid` candidate: `fagsp_mid_base`
- current bottleneck: tokenizer-to-SFT transfer and semantic-structure retention
- current evidence is still tokenizer / probe-stage support, not a final downstream paper result
