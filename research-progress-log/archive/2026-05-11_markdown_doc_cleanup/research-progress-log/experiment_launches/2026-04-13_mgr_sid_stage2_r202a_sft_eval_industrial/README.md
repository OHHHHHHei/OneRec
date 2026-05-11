# 2026-04-13 Stage-2 `R202a` SFT/Evaluate (Industrial)

## Goal

Push the current best Block-2 tokenizer candidate `R202a` into downstream validation:

- tokenizer: `R202a = stop-gradient hierarchy isolation`
- downstream recipe: `title_history2sid = on`
- downstream recipe: `SID-description alignment = on`
- `description_task_probability = 0.5`

This run corresponds to `R208` in the stage-2 tracker.

## Source Tokenizer Artifact

- generated index:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/generated_indices/Industrial_and_Scientific.stage2_r202a_stopgrad.index.json`

## Data Conversion

- variant root:
  `/home/leejt/OneRec/data_experiment/Amazon/stage2_r202a_stopgrad`
- conversion script:
  `/home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py`

## Config

- SFT:
  `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_stage2_r202a_title_on_desc_p05.yaml`
- evaluate:
  `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_stage2_r202a_title_on_desc_p05.yaml`
- chain script:
  `/home/leejt/OneRec/scripts/experiment_mgr_sid_stage2_r202a_sft_eval_chain.sh`

## Recipe Alignment

Aligned to the current best downstream recipe found for `v2`-style tokenizers:

- `batch_size = 1024`
- `micro_batch_size = 2`
- `world_size = 4`
- `num_epochs = 10`
- `learning_rate = 3e-4`
- `cutoff_len = 512`
- `eval_step = 0.05`
- `title_history2sid = on`
- `desc_align = on`
- `description_task_probability = 0.5`

## Runtime Plan

- physical GPUs: `2,3,4,5`
- launcher: `torchrun`
- tmux session: `mgr_stage2_r202a_sft`

## Logs

- SFT log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_stage2_r202a_sft_20260413.log`
- evaluate log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_stage2_r202a_eval_20260413.log`

## Outputs

- SFT root:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_r202a_sft_eval_20260413/title_on_desc_p05/sft`
- final result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_stage2_r202a_sft_eval_industrial_20260413/final_result_sft_mgr_stage2_r202a_title_on_desc_p05_Industrial_and_Scientific.json`

## Notes

- `R202a` is currently the strongest Block-2 retention-oriented tokenizer candidate.
- This run is launched in parallel with `R205` because `R208` remains informative regardless of whether `R205` later wins Block-3.
