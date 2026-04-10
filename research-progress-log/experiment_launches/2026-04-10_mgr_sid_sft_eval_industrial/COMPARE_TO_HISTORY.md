# Compare Current MGR-SID SFT Results to Historical Industrial SFT Runs

## Current Runs

- `current_mgr_upstream_baseline_20260410`
- `current_mgr_upstream_hierarchy_20260410`

## Historical Reference From `experiment_results.csv`

### Closest historical setting

- `sft_industrial_noalign_20260323_235623`
- variant: `title_history2sid_on__desc_align_off`
- this is the fairest comparison to the current runs because it keeps:
  - `title_history2sid_enabled = true`
  - `alignment_enabled = false`

### Strongest historical SFT run

- `sft_industrial_title_history2sid_off__desc_align_p05_20260325_192249`
- variant: `title_history2sid_off__desc_align_p05`
- this is the best historical run on almost all Industrial SFT metrics in the CSV

## Raw Table

| run | variant | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical closest | `title_history2sid_on__desc_align_off` | 0.06243106 | 0.07981905 | 0.08688172 | 0.09872236 | 0.10803185 | 0.12073434 | 0.06243106 | 0.09287448 | 0.11008162 | 0.14714317 | 0.18442533 | 0.24840062 |
| historical best | `title_history2sid_off__desc_align_p05` | 0.06706375 | 0.08500848 | 0.09315326 | 0.10372025 | 0.11358430 | 0.12461898 | 0.06706375 | 0.09838959 | 0.11824399 | 0.15089345 | 0.18971983 | 0.24531216 |
| current | `mgr_upstream_baseline` | 0.06309287 | 0.07771807 | 0.08560683 | 0.09430022 | 0.10419997 | 0.11775708 | 0.06309287 | 0.08824178 | 0.10743437 | 0.13434811 | 0.17361571 | 0.24244430 |
| current | `mgr_upstream_hierarchy` | 0.06265167 | 0.08024707 | 0.08641766 | 0.09359572 | 0.10289594 | 0.11644559 | 0.06265167 | 0.09287448 | 0.10787558 | 0.13037723 | 0.16743878 | 0.23604677 |

## Delta vs Closest Historical Setting

### Current baseline minus closest historical

- `NDCG@1 = +0.00066181`
- `NDCG@3 = -0.00210098`
- `NDCG@5 = -0.00127489`
- `NDCG@10 = -0.00442214`
- `NDCG@20 = -0.00383188`
- `NDCG@50 = -0.00297726`
- `HR@1 = +0.00066181`
- `HR@3 = -0.00463270`
- `HR@5 = -0.00264725`
- `HR@10 = -0.01279506`
- `HR@20 = -0.01080962`
- `HR@50 = -0.00595632`

### Current hierarchy minus closest historical

- `NDCG@1 = +0.00022061`
- `NDCG@3 = +0.00042802`
- `NDCG@5 = -0.00046406`
- `NDCG@10 = -0.00512664`
- `NDCG@20 = -0.00513591`
- `NDCG@50 = -0.00428875`
- `HR@1 = +0.00022061`
- `HR@3 = +0.00000000`
- `HR@5 = -0.00220604`
- `HR@10 = -0.01676594`
- `HR@20 = -0.01698655`
- `HR@50 = -0.01235385`

## Delta vs Historical Best

### Current baseline minus historical best

- `NDCG@1 = -0.00397088`
- `NDCG@3 = -0.00729041`
- `NDCG@5 = -0.00754643`
- `NDCG@10 = -0.00942003`
- `NDCG@20 = -0.00938433`
- `NDCG@50 = -0.00686190`
- `HR@1 = -0.00397088`
- `HR@3 = -0.01014781`
- `HR@5 = -0.01080962`
- `HR@10 = -0.01654534`
- `HR@20 = -0.01610412`
- `HR@50 = -0.00286786`

### Current hierarchy minus historical best

- `NDCG@1 = -0.00441208`
- `NDCG@3 = -0.00476141`
- `NDCG@5 = -0.00673560`
- `NDCG@10 = -0.01012453`
- `NDCG@20 = -0.01068836`
- `NDCG@50 = -0.00817339`
- `HR@1 = -0.00441208`
- `HR@3 = -0.00551511`
- `HR@5 = -0.01036841`
- `HR@10 = -0.02051622`
- `HR@20 = -0.02228105`
- `HR@50 = -0.00926539`

## Reading

Current tokenizer changes do not yet beat the best historical Industrial SFT setting in the CSV.

More specifically:

- compared to the closest historical no-alignment baseline, the new `mgr_upstream_baseline` slightly improves only `@1`, but is worse on most `@3+` metrics
- `mgr_upstream_hierarchy` is slightly better than the closest historical run on `NDCG@3` and matches `HR@3`, but is still weaker on `@10+`
- compared to the best historical SFT run, both current runs are clearly behind across almost all metrics

The current evidence is therefore:

> hierarchy-aware SID shows tokenizer-level and local-ambiguity gains, but the present SFT setup has not yet converted those gains into a new end-to-end best Industrial SFT result.
