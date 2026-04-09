# Working Idea: Graph Hierarchy v1

This folder contains a fresh `idea-discovery` round focused on:

- graph-structured collaborative signals
- view-specific denoising
- hierarchy-aware collaborative allocation for 3-level SID tokenization

It is intentionally separated from `working_idea_hierarchy_aware_v1/` because this round shifts from simple multi-view fusion toward a more graph-native tokenizer design.

## Recommended reading order

1. `CURRENT_TASK_ALIGNMENT.md`
2. `IDEA_REPORT.md`
3. `refine-logs/FINAL_PROPOSAL.md`
4. `refine-logs/EXPERIMENT_PLAN.md`
5. `11_arxiv_related_work_by_question.md`
6. `12_modules_mapped_to_core_questions.md`

If you want the intermediate thinking trail:

1. `01_graph_pilot.md`
2. `02_literature_landscape.md`
3. `03_idea_candidates.md`
4. `04_novelty_check.md`
5. `05_review_round1.md`
6. `06_revision_after_round1.md`
7. `07_review_round2.md`

## Current status

- top recommendation: `MGR-SID`
- positioning: stronger than simple level-wise feature fusion, but still much more feasible than full graph-defined item tokenization
- key open design question: what is the best `mid-scale` graph view for Level 2 SID
