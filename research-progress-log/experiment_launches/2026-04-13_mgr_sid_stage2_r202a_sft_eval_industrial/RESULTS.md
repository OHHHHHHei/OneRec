# 2026-04-13 Stage-2 `R202a` -> SFT/Evaluate Results

## Run Identity

- record id:
  `sft_industrial_mgr_stage2_r202a_title_on_desc_p05_20260413_064109`
- result json:
  `/home/leejt/OneRec/results/experiments/mgr_sid_stage2_r202a_sft_eval_industrial_20260413/final_result_sft_mgr_stage2_r202a_title_on_desc_p05_Industrial_and_Scientific.json`
- output checkpoint:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_r202a_sft_eval_20260413/title_on_desc_p05/sft/final_checkpoint`

## Core Metrics

- `NDCG@1/3/5/10 = 0.06552 / 0.08360 / 0.09045 / 0.09974`
- `HR@1/3/5/10 = 0.06552 / 0.09729 / 0.11383 / 0.14251`

## Training Summary

- final eval loss:
  `1.6118812561`
- final train loss:
  `0.4657821171`
- early stop epoch:
  `5.5`

## Main Comparisons

### vs current best `v2_on_p05 SFT`

- `NDCG@10: 0.10271 -> 0.09974` (`-0.00297`)
- `HR@10: 0.14626 -> 0.14251` (`-0.00375`)

### vs strongest original MiniOneRec SFT

- strongest original:
  - `NDCG@10 = 0.10372`
  - `HR@10 = 0.15089`
- `R202a` remains below both metrics

## Interpretation

`R202a` is still a meaningful tokenizer-side result, because it improved the
retention-oriented local structure in Block 2. But once pushed into the fixed
`title_history2sid_on + desc_align_p05` downstream recipe, it does **not**
beat the current best `v2_on_p05` SFT line.

So the stage-2 conclusion for this branch is:

> `R202a` is a valid structural refinement, but it is **not** yet a downstream
> winner.

## Follow-up Analysis

Detailed evaluate-side analysis is recorded in:

- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/EVAL_ANALYSIS.md`
- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/TOPK_V2_ON_P05_SFT_VS_R208.md`
- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/TOPK_STRONGEST_ORIG_SFT_VS_R208.md`
