# 2026-04-11 MGR-SID v2 SFT on Strongest Original MiniOneRec Recipe (Industrial)

## Goal

Run `v2` tokenizer under the strongest original MiniOneRec SFT recipe:

- `title_history2sid = off`
- `SID-description alignment = on`
- `description_task_probability = 0.5`

The purpose of this run is to answer the main next-step question:

> Can `v2` tokenizer improvements still hold when plugged into the strongest original MiniOneRec SFT recipe, rather than only the recipe-aligned noalign setting?

## Source Tokenizer Artifact

- `v2 offline` index:
  `/home/leejt/OneRec/output/experiments/mgr_sid_tokenizer_v2/generated_indices/Industrial_and_Scientific.mgr_tokenizer_v2_offline.index.json`

## Data Root

- `v2` data root:
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_tokenizer_v2_offline`

## Config

- SFT config:
  `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_tokenizer_v2_desc_align_p05.yaml`

## Recipe Alignment

This run is aligned to the strongest original MiniOneRec SFT recipe on:

- `batch_size = 1024`
- `micro_batch_size = 2`
- `world_size = 4`
- `num_epochs = 10`
- `learning_rate = 3e-4`
- `cutoff_len = 512`
- `eval_step = 0.05`
- `title_history2sid = off`
- `desc_align = on`
- `description_task_probability = 0.5`

Only the tokenizer source is changed from original MiniOneRec to `v2`.

## Runtime Plan

- physical GPUs: `2,3,4,5`
- launcher: `torchrun`
- tmux session: `mgr_v2_sft_desc_p05_ind`

## Logs

- SFT log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_sft_desc_align_p05_industrial_20260411.log`

## Outputs

- SFT root:
  `/home/leejt/OneRec/output/experiments/mgr_sid_v2_sft_desc_align_p05_industrial_20260411/mgr_tokenizer_v2_offline/sft`

## Notes

- `mgr_upstream_baseline` is not treated as the main baseline here.
- The main comparison target for this run is the strongest original MiniOneRec SFT:
  `sft_industrial_title_history2sid_off__desc_align_p05_20260325_192249`.
