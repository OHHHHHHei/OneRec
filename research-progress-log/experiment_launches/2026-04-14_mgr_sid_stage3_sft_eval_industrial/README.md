# 2026-04-14 MGR-SID Stage-3 SFT Evaluate Industrial

This launch pushes the two stage-3 tokenizer candidates to the current
strongest downstream `v2_on_p05` recipe.

## Goal

Evaluate whether the two stage-3 candidate SID codebook spaces:

- `R401b-g0.05`
- `R401d-g0.05-a0.05`

actually improve downstream recommendation quality under full:

- `title_history2sid_on + desc_align_p05`
- `SFT -> evaluate`

## Narrative Reminder

These runs are **not** testing whether a tokenizer candidate stays close to
`v2`.

They are testing whether the candidate codebook space is **better** for full
downstream learning.

Tokenizer-side structure, prefix drift, and codebook drift remain diagnostics.
The real selector here is final downstream `evaluate`.

## Execution Order

The chain is intentionally sequential on the same 4 GPUs:

1. `R401b SFT`
2. `R401b evaluate`
3. `R401d SFT`
4. `R401d evaluate`

## Configs

### `R401b`

- SFT:
  `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_stage3_r401b_title_on_desc_p05.yaml`
- Evaluate:
  `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_stage3_r401b_title_on_desc_p05.yaml`

### `R401d`

- SFT:
  `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_stage3_r401d_title_on_desc_p05.yaml`
- Evaluate:
  `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_stage3_r401d_title_on_desc_p05.yaml`

## Runtime

- launch date:
  `2026-04-14`
- tmux:
  `mgr_stage3_sft_eval_chain`
- GPUs:
  `2,3,4,5`
- chain script:
  `/home/leejt/OneRec/scripts/experiment_mgr_sid_stage3_r401b_r401d_sft_eval_chain.sh`
- status:
  `COMPLETED`

## Logs

### `R401b`

- SFT log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401b_sft_20260414.log`
- Evaluate log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401b_eval_20260414.log`

### `R401d`

- SFT log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401d_sft_20260414.log`
- Evaluate log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401d_eval_20260414.log`

## Output Targets

### `R401b`

- SFT output:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_sft_eval_20260414/r401b_title_on_desc_p05/sft`
- Evaluate result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_stage3_sft_eval_20260414/final_result_sft_mgr_stage3_r401b_title_on_desc_p05_Industrial_and_Scientific.json`

### `R401d`

- SFT output:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_sft_eval_20260414/r401d_title_on_desc_p05/sft`
- Evaluate result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_stage3_sft_eval_20260414/final_result_sft_mgr_stage3_r401d_title_on_desc_p05_Industrial_and_Scientific.json`

## Notes

- The recipe is intentionally matched to the strongest current `v2` downstream
  line.
- No tokenizer-side gate is used here; the point is to let downstream
  `evaluate` decide.
- After results land, this file should be updated with final metrics and a
  direct comparison against current `v2_on_p05`.

## Final Results

### `R401b`

- Evaluate result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_stage3_sft_eval_20260414/final_result_sft_mgr_stage3_r401b_title_on_desc_p05_Industrial_and_Scientific.json`
- `NDCG@1/3/5/10/20/50`
  - `0.06706375 / 0.08479311 / 0.09017463 / 0.09905007 / 0.10923107 / 0.12218252`
- `HR@1/3/5/10/20/50`
  - `0.06706375 / 0.09772777 / 0.11096404 / 0.13853960 / 0.17935142 / 0.24442974`
- `constraint_invalid_total`
  - `0`

### `R401d`

- Evaluate result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_stage3_sft_eval_20260414/final_result_sft_mgr_stage3_r401d_title_on_desc_p05_Industrial_and_Scientific.json`
- `NDCG@1/3/5/10/20/50`
  - `0.06353408 / 0.07871079 / 0.08548844 / 0.09353784 / 0.10334381 / 0.11669836`
- `HR@1/3/5/10/20/50`
  - `0.06353408 / 0.08978601 / 0.10633135 / 0.13148026 / 0.17030664 / 0.23803221`
- `constraint_invalid_total`
  - `0`

## Comparison

Relative to the current strongest `v2_on_p05` SFT line:

- `R401b` is negative:
  - `NDCG@10`: `0.09905` vs `0.10271`
  - `HR@10`: `0.13854` vs `0.14626`

- `R401d` is more negative:
  - `NDCG@10`: `0.09354` vs `0.10271`
  - `HR@10`: `0.13148` vs `0.14626`

Relative to `R401b`:

- `R401d` is also worse:
  - `NDCG@10`: `0.09354` vs `0.09905`
  - `HR@10`: `0.13148` vs `0.13854`

## Current Reading

Both stage-3 prefix-retained tokenizer candidates failed to beat the current
`v2_on_p05` downstream baseline under full `SFT -> evaluate`.

`R401b` is already a negative downstream result.
`R401d` is an even stronger negative result: adding codebook anchor on top of
the stage-3 branch did not recover downstream quality and instead degraded it
further.
