# Active Context

This is the shortest high-signal summary of the current mainline.

Use this file when you want to sync quickly without reading all active notes.

## One-Line Direction

Use graph structure as the carrier of collaborative information, and inject it into semantic SID construction through hierarchy-aware, ambiguity-aware structural supervision.

## Main Problem Statement

The current work is no longer trying to answer whether collaborative information matters at all.

That part is already supported.

The current problem is more specific:

> how do we inject graph-structured collaborative information into MiniOneRec’s semantic tokenizer so that hard local ambiguity is reduced without over-correcting already-stable semantic structure?

## Three Core Questions

### Q1. What graph should carry collaborative information?

Current stable answer:

- `G_coarse` for broad collaborative consistency
- `G_mid` as the most important view
- `G_local` for short-range transition structure

Current strongest `G_mid` candidate:

- `fagsp_mid_base`

### Q2. How should the method become hierarchy-aware?

Current stable answer:

- different SID levels should not receive the same graph signal in the same way
- hierarchy should be expressed through level-specific structural supervision, not generic feature fusion

### Q3. How should graph-structured collaboration enter MiniOneRec?

Current stable answer:

- semantic SID remains the backbone
- graph information acts as structural supervision
- the current `v2` direction is:
  - ambiguity-aware graph supervision
  - semantic-structure retention

## What Has Been Proven So Far

### Tokenizer side

- graph-aware tokenizer design can reduce local ambiguity
- `v2` produces cleaner SID structure than both the reproduced semantic baseline and `v1`
- the key improvement is on local same-`l2` ambiguity and conditional entropy

### Downstream SFT side

- `v2` is not only a tokenizer-side effect
- recipe isolation showed that the best current downstream recipe for `v2` is:
  - `title_history2sid_on + desc_align_p05`
- strongest original MiniOneRec recipe is not directly reusable for `v2`
- the main conflict came from `title_history2sid_off`, not from `desc_align_p05`

### Downstream RL side

- `v2_on_p05` survives end-to-end training into RL
- it already beats the strongest original MiniOneRec SFT on `NDCG@10`
- it still trails the strongest original MiniOneRec RL overall

Current remaining gap:

- not a generic quality gap
- mainly a `top5/top10` mid-beam retention gap

## Baseline Policy

Main baselines:

- strongest original MiniOneRec SFT
- strongest original MiniOneRec RL

Recipe-aligned original baseline:

- original MiniOneRec under the same task recipe as the `v2` comparison

Internal control:

- `mgr_upstream_baseline`
- `mgr_upstream_hierarchy`

Internal controls are useful for mechanism diagnosis, but they are not the main baselines for final claims.

## Current Mainline Recipe

If we continue from the strongest current `v2` line, the active path is:

- tokenizer:
  `v2`
- downstream recipe:
  `title_history2sid_on + desc_align_p05`
- current strongest run family:
  `v2_on_p05`

## Current Execution Stage

The active execution stage is now:

- retention-targeted stage-2 refinement, with first-round results now available
- current stage-2 outcomes:
  - `R202a`: best Block-2 tokenizer branch
  - `R205`: completed negative semantic-retention result
  - `R208`: completed downstream screen for `R202a`

This means the project is no longer asking whether `v2` works end-to-end.
That part is already supported, and the current strongest end-to-end line
remains `v2_on_p05 -> RL`.

The active question is:

> can a small tokenizer-side refinement close the remaining `top5/top10`
> retention gap without giving up the current ambiguity-cleanup gains?

Current answer so far:

- `R202a` improves retention-oriented tokenizer structure, but its downstream
  `SFT` screen does not beat current `v2_on_p05`.
- `R205` improves final generated collision (`13 -> 12`) but clearly worsens
  the local ambiguity structure, so it is not a valid replacement.

## Current Most Important Interpretation

The recipe preference inversion is now one of the clearest mechanism findings:

- original MiniOneRec prefers `title_history2sid_off`
- `v2` prefers `title_history2sid_on`

This suggests that `v2` changes the nature of the SID representation itself.
Its benefit is not recipe-invariant; it needs more explicit downstream SID-structure consumption.

## Canonical Follow-Up Docs

Read these next if you want more depth:

1. `CURRENT_TASK_ALIGNMENT.md`
2. `01_PROBE_AND_EARLY_EVIDENCE.md`
3. `02_RELATED_WORK_AND_MODULE_MAP.md`
4. `17_ambiguity_proxy_literature_scan.md`
5. `18_mgr_sid_v2_ambiguity_aware_method.md`
6. `19_mgr_sid_current_method_code_aligned_formulas.md`
7. `refine-logs/EXPERIMENT_PLAN_TOKENIZER_V2.md`
8. `refine-logs/EXPERIMENT_TRACKER_TOKENIZER_V2.md`
9. `refine-logs/EXPERIMENT_PLAN_STAGE2_RETENTION.md`
10. `refine-logs/EXPERIMENT_TRACKER_STAGE2_RETENTION.md`
