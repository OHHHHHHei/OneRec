# 2026-04-13 MGR-SID Stage-2 Semantic Retention Industrial

This stage implements Block 3 from `EXPERIMENT_PLAN_STAGE2_RETENTION.md`.

## Goal

Test whether a more structure-faithful semantic retention term can preserve
semantic neighborhood structure better than the current semantic smoothness
term, while keeping the retention-oriented tokenizer gains from stage-2.

## Variants

- `R205`: `R202a`-style stop-grad hierarchy isolation + batch-local semantic neighborhood KL
- `R205b`: fallback branch = current `v2` + batch-local semantic neighborhood KL (prepared, not launched by default)

## Retention Term

- semantic retention mode: `batch_local_kl`
- semantic retention temperature: `0.1`
- teacher space: original semantic embedding within batch
- student space: tokenizer representation within batch
- objective: `KL(p_sem || q_tok)`

## Configs

- primary:
  - `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage2_r205.yaml`
- fallback:
  - `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage2_r205b.yaml`

## Sanity Check

A short 2-epoch sanity run passed successfully:

- run dir:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/sanity_r205_stopgrad_kl/Apr-13-2026_02-13-34`
- best collision after sanity:
  - `0.0683667933`

## Final Status

- Date: `2026-04-13`
- `R205` finished normally
  - log:
    `/home/leejt/OneRec/logs/experiment_mgr_sid_stage2_r205_20260413.log`
  - summary:
    `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r205_stopgrad_kl/Apr-13-2026_02-15-01/summary.json`
  - best train collision:
    `0.1136733587`
  - best epoch:
    `1149`
  - final epoch collision:
    `0.1155724362`

## Interim Interpretation

`R205` did not collapse, but it also did not beat the simpler `R202a` stop-grad
branch:

- `R202a` best train collision:
  `0.1006511123`
- `R205` best train collision:
  `0.1136733587`

This means the first `batch_local_kl` semantic retention implementation is
currently a negative result at the tokenizer-training stage. The branch is kept
as evidence, but it should not replace `R202a` as the main stage-2 candidate
without additional retuning (for example lower semantic retention strength or a
higher temperature).

## R206: `sid-generate`

- generate summary:
  `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_semantic_retention_industrial/R206_r205_generate_summary.json`
- generated index:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/generated_indices/Industrial_and_Scientific.stage2_r205_stopgrad_kl.index.json`
- final generated collision:
  `0.0032555616`
  which is `12 / 3686`

So the semantic-retention KL branch does improve the final collision count after
generate, even though its tokenizer training collision was worse than `R202a`.

## R207: Structural Diagnosis After Generate

- `current v2` vs `R205`
  - markdown:
    `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_semantic_retention_industrial/R207_v2_vs_r205_local_ambiguity.md`
  - main readout:
    - mean target `l2` leaf count:
      `4.3422 -> 4.9572`
    - fraction targets in multi-leaf `same_l2`:
      `0.4873 -> 0.5449`
    - fraction targets in deep crowded `l2>=4`:
      `0.2228 -> 0.2621`
    - target-weighted entropy:
      `1.1001 -> 1.2623`

- `R202a` vs `R205`
  - markdown:
    `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_semantic_retention_industrial/R207b_r202a_vs_r205_local_ambiguity.md`
  - main readout:
    - mean target `l2` leaf count:
      `3.6148 -> 4.9572`
    - fraction targets in multi-leaf `same_l2`:
      `0.4988 -> 0.5449`
    - fraction targets in deep crowded `l2>=4`:
      `0.1994 -> 0.2621`
    - target-weighted entropy:
      `1.0308 -> 1.2623`

## Final Interpretation Update

`R205` is now a clearer negative result:

- it is stable numerically;
- it improves final generated collision from `13` to `12`;
- but the improvement comes with a large regression in the retention-oriented
  local structure that stage-2 is trying to improve.

So in the current form, `batch_local_kl` should **not** replace `R202a` as the
main stage-2 candidate, and it should **not** be pushed downstream.

## Output Roots

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r205_stopgrad_kl`
- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r205b_nosg_kl`
