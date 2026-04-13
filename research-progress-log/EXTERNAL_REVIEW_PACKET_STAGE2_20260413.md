# External Review Packet (Stage-2 Snapshot, 2026-04-13)

This file is the recommended single-entry handoff for an external reviewer who
needs to understand the current project status in one pass.

It is designed to avoid a common failure mode:

- reading only the latest stage-2 tokenizer experiments,
- while missing the already-completed `v2_on_p05 -> RL` mainline result,
- and therefore misjudging the current project-wide state.

## What the reviewer should know first

Before reading any individual experiment note, the reviewer should anchor on
these project-level facts:

1. The strongest current `v2` end-to-end line is still:
   - `v2_on_p05 -> RL`
2. Stage-2 is **not** testing whether `v2` works at all.
   - That question has already been answered positively.
3. Stage-2 is testing a narrower question:
   - can retention-targeted tokenizer refinements close the remaining
     `top5/top10` mid-beam retention gap?
4. So far, no stage-2 branch has beaten the current `v2_on_p05` downstream
   line.

## Metric reading rule

For this project, reviewers should **not** judge runs from `@10` alone.

The minimum comparison set should always include:

- `NDCG@1 / @3 / @5 / @10 / @20`
- `HR@1 / @3 / @5 / @10 / @20`

And when useful, also inspect:

- `HR@50`

This matters because the current gaps are structured:

- some variants are better at the head (`@1/@3`),
- some are better in the mid beam (`@5/@10/@20`),
- some recover more long-tail mass at `@50`.

## Current scoreboard

### Strongest original MiniOneRec

- strongest original SFT
  - `NDCG@1/3/5/10/20 = 0.06706375 / 0.08500848 / 0.09315326 / 0.10372025 / 0.11358430`
  - `HR@1/3/5/10/20 = 0.06706375 / 0.09838959 / 0.11824399 / 0.15089345 / 0.18971983`
  - `HR@50 = 0.24531216`
- strongest original RL
  - `NDCG@1/3/5/10/20 = 0.07324068 / 0.08903190 / 0.09704467 / 0.10726345 / 0.11365951`
  - `HR@1/3/5/10/20 = 0.07324068 / 0.10037503 / 0.11978822 / 0.15133466 / 0.17670417`
  - `HR@50 = 0.21994264`

### Current strongest `v2` mainline

- `v2_on_p05 SFT`
  - `NDCG@1/3/5/10/20 = 0.07059343 / 0.08451223 / 0.09253300 / 0.10270767 / 0.11172619`
  - `HR@1/3/5/10/20 = 0.07059343 / 0.09508052 / 0.11471432 / 0.14626075 / 0.18243989`
  - `HR@50 = 0.24818001`
- `v2_on_p05 RL`
  - `NDCG@1/3/5/10/20 = 0.07434370 / 0.09053678 / 0.09629833 / 0.10431921 / 0.11269034`
  - `HR@1/3/5/10/20 = 0.07434370 / 0.10280168 / 0.11692036 / 0.14184867 / 0.17515994`
  - `HR@50 = 0.23737039`

### Stage-2 first-round results

- `R202a` (`stop-gradient`)
  - tokenizer-side structural winner
  - but downstream `R208` does **not** beat current `v2_on_p05`
  - `R208` SFT metrics:
    - `NDCG@1/3/5/10/20 = 0.06551952 / 0.08359820 / 0.09045320 / 0.09973721 / 0.10912282`
    - `HR@1/3/5/10/20 = 0.06551952 / 0.09728657 / 0.11383190 / 0.14251048 / 0.18001324`
    - `HR@50 = 0.23737039`
- `R202b-r075`
  - better final generated collision
  - worse retention-oriented structure
- `R205`
  - better final generated collision
  - worse retention-oriented structure

## Required reading order

The reviewer should read the following files in order.

### 1. Global status and project-wide interpretation

- `research-progress-log/research_progress_log.tex`

This is the main project-level summary.
It explains:

- how the project got from `v1` to `v2`,
- why `title_history2sid_on + desc_align_p05` became the current best `v2`
  downstream recipe,
- why stage-2 exists,
- and why the current strongest end-to-end line is still `v2_on_p05 -> RL`.

### 2. Unified experiment table

- `experiment_results.csv`

This is the canonical metric ledger.
If any narrative statement conflicts with the CSV, the CSV should be treated as
the source of truth for recorded metrics.

### 3. Why `v2_on_p05` is the current best downstream recipe

- `research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/RESULTS.md`
- `research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/TOPK_ERROR_DISTRIBUTION_COMPARISON.md`

These two files explain:

- why strongest original MiniOneRec recipe does **not** transfer directly to
  `v2`,
- why the conflict is mainly `title_history2sid_off`,
- and why `title_history2sid_on + desc_align_p05` became the correct stage-2
  downstream screen recipe.

### 4. Current strongest `v2` RL result

- `research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/RESULTS.md`
- `research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/EVAL_ANALYSIS.md`

These are essential.

They show that:

- `v2` already survives end-to-end RL,
- the remaining gap to strongest original RL is structured, not generic,
- and the gap should be read across `@1/@3/@5/@10/@20` rather than from
  `@10` alone:
  - stronger head ranking at `@1/@3`,
  - weaker mid-beam retention around `@5/@10/@20`,
  - different long-beam behavior at `@50`.

### 5. Stage-2 plan and tracker

- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_PLAN_STAGE2_RETENTION.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_TRACKER_STAGE2_RETENTION.md`

These show:

- what stage-2 was trying to test,
- what the success / failure gates were,
- and which branches are now completed, held, or discarded.

### 6. Stage-2 Block-2 results (`stop-gradient`)

- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/README.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/R204_v2_vs_r202a_local_ambiguity.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/R204b_v2_vs_r202b_retry075_local_ambiguity.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/R204c_r202a_vs_r202b_retry075_local_ambiguity.md`

These show:

- why `R202a` is the clean tokenizer-side structural winner,
- why `R202b-r075` is a collision-vs-structure tradeoff branch,
- and why stage-2 cannot be judged by final collision alone.

### 7. Stage-2 Block-3 results (`semantic retention KL`)

- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_semantic_retention_industrial/README.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_semantic_retention_industrial/RESULTS.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_semantic_retention_industrial/R207_v2_vs_r205_local_ambiguity.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_semantic_retention_industrial/R207b_r202a_vs_r205_local_ambiguity.md`

These show:

- why first-version batch-local KL is a negative result for the stage-2 goal,
- even though it improves final generated collision.

### 8. Stage-2 downstream screen

- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/RESULTS.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/EVAL_ANALYSIS.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/TOPK_V2_ON_P05_SFT_VS_R208.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/TOPK_STRONGEST_ORIG_SFT_VS_R208.md`

These files are the final downstream gate for `R202a`.

It shows:

- `R202a` is a meaningful tokenizer-side refinement,
- but it does **not** beat the current `v2_on_p05` downstream line when the
  comparison is made across the full cutoff set (`@1/@3/@5/@10/@20`), not just
  `@10`;
- the downstream change is highly structured rather than uniformly negative:
  - slight gain at `HR@3`
  - but weaker `NDCG@1`, weaker `HR@1`, and weaker beam retention at
    `@10/@20/@50`
- the gain is concentrated on hard crowded local cases, while a larger pool of
  already-stable examples regresses.

## Optional deep-dive documents

These are useful only if the reviewer wants more granularity.

- `research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_sft_eval_industrial/RESULTS.md`
- `research-progress-log/experiment_launches/2026-04-11_mgr_sid_tokenizer_v2_r005/STRUCTURE_COMPARISON.md`
- `research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/TOPK_STRONGEST_ORIG_RL_VS_V2_RL.md`
- `research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/TOPK_V2_SFT_VS_V2_RL.md`

## Questions the reviewer should answer

After reading the packet, the most useful external feedback would focus on:

1. Whether the current project-wide reading is correct:
   - strongest current line is still `v2_on_p05 -> RL`
   - stage-2 first round did not surpass it
2. Whether the current interpretation of the remaining gap is correct:
   - a structured ranking-profile gap across `@1/@3/@5/@10/@20`
   - especially a `top5/top10/@20` mid-beam retention gap
   - not a tokenizer collapse or a generic quality failure
3. Whether the next step should focus on:
   - RL-stage gap-closing,
   - a more conservative tokenizer refinement,
   - or a better tokenizer-to-downstream interface
4. Whether the new `R208` evaluate analysis changes the preferred next step:
   - because it now shows explicitly that `R202a` helps hard crowded local
     ambiguity,
   - but still fails to preserve enough already-stable examples in the beam.

## Short version

If the reviewer has very limited time, these six files are the minimum viable
packet:

- `research-progress-log/research_progress_log.tex`
- `experiment_results.csv`
- `research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/RESULTS.md`
- `research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/EVAL_ANALYSIS.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/README.md`
- `research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/EVAL_ANALYSIS.md`
