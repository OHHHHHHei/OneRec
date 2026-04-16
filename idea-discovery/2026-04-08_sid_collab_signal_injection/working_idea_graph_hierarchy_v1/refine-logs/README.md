# Refine Logs（细化日志）

Status（状态）: `plan-index（计划索引）`

This folder now keeps the most relevant planning entry（计划入口） for the current method line（当前方法主线）.

Older stage-specific plans（阶段计划） and trackers（跟踪器） were moved out of the root on
`2026-04-14` so they stop competing with the current narrative（当前叙事）.

This folder is now a planning index（计划索引）, not a current-state summary（当前状态摘要）.

Read this only after:

1. [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
2. [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)

## Recently Completed Mainline Decision Docs

- `EXPERIMENT_PLAN_STAGE3_PREFIX_RETAINED_HIERARCHY.md`
  - completed stage-3 execution plan
  - useful when you want to understand:
    - search for a better hierarchy-aware SID codebook space
    - why `R401b` / `R401d` were launched
    - why stage-3 was judged by full downstream `SFT -> evaluate`

- `../../../../research-progress-log/experiment_launches/2026-04-14_mgr_sid_stage3_sft_eval_industrial/README.md`
  - downstream verdict for stage-3
  - records that both `R401b` and `R401d` were negative relative to current `v2_on_p05`

## Current Mainline Reference Plans

- `EXPERIMENT_PLAN_SELECTIVE_SEPARATION.md`
  - current method-reference document（方法参考文档）
  - still useful for:
    - understanding the selective-separation（选择性分离） motivation and loss idea
    - reconstructing what was tried before `S000`
  - important update:
    - its diagnostics-first（诊断优先） execution blocks are no longer active after `S000`

- `EXPERIMENT_TRACKER_SELECTIVE_SEPARATION.md`
  - frozen tracker snapshot（冻结跟踪快照） for the last diagnostics-driven selective-separation branch
  - not an active execution tracker（活跃执行跟踪表） anymore

## Recently Active but Now Secondary Branch Plans

- `EXPERIMENT_PLAN_COARSE_LOCAL_GRAPH_CARRIERS.md`
  - previous active plan
  - still useful for reconstructing why `R530* / R542*` were launched
  - current role:
    - supporting branch history
    - graph-carrier evidence source for the new selective-separation phase

- `EXPERIMENT_TRACKER_COARSE_LOCAL_GRAPH_CARRIERS.md`
  - tracker for the coarse/local carrier branch
  - keep as execution history while the branch is being closed out

## Recently Completed Branch Plans

- `EXPERIMENT_PLAN_TAGCF_SEMANTIC_TO_TOPOLOGY.md`
  - `TAGCF`-inspired branch plan
  - useful for reconstructing why `R510 / R511` were launched
  - current status:
    - no positive downstream evidence so far

- `EXPERIMENT_TRACKER_TAGCF_SEMANTIC_TO_TOPOLOGY.md`
  - branch tracker for the `TAGCF` line
  - now mostly useful as a branch history document

- `EXPERIMENT_PLAN_FAGSP_CASCADE_GMID.md`
  - deeper `FaGSP` item-side cascade plan
  - useful for reconstructing why `R520` was launched
  - current status:
    - no positive tokenizer-side evidence so far

## Reading Policy（阅读规则）

Then read, in order:

1. `../18_mgr_sid_v2_ambiguity_aware_method.md`
2. `../19_mgr_sid_current_method_code_aligned_formulas.md`
3. `../20_sid_quality_beyond_structure.md`
4. `../21_graph_design_review_20260414.md`

## Historical Execution Chain

The following materials are preserved, but they are no longer active planning
documents:

- `archive/2026-04-14_stage3_scope_cleanup/README.md`
  - tokenizer `v2` plan / tracker
  - stage-2 retention plan / tracker
  - stage-2 interface diagnostics plan / tracker

If you need to reconstruct how the project got here, read that archive
directory intentionally.
Do not use those files as the default starting point for new graph-carrier decisions.

After `S000`, do not use any prior diagnostic（前验诊断） as the default starting point for
new tokenizer decisions（新分词器决策）.
