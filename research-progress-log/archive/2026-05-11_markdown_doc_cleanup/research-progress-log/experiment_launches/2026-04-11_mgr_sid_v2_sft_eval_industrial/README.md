# 2026-04-11 MGR-SID v2 SFT + Evaluate (Industrial)

## Goal

Run the first downstream `SFT -> evaluate` chain for the tokenizer `v2` index:

- `mgr_tokenizer_v2_offline`

This run has three comparison layers:

- internal control:
  `/home/leejt/OneRec/results/experiments/mgr_sid_sft_eval_industrial_20260410/final_result_sft_mgr_upstream_baseline_Industrial_and_Scientific.json`
- recipe-aligned original MiniOneRec:
  `record_id = sft_industrial_noalign_20260323_235623`
- strongest original MiniOneRec anchors:
  - SFT: `sft_industrial_title_history2sid_off__desc_align_p05_20260325_192249`
  - RL: `rl_industrial_title_history2sid_off__desc_align_p05_batch256_20260329_152417`

## Source Tokenizer Artifacts

- MiniOneRec baseline index:
  `/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_baseline.index.json`
- `v2 offline` index:
  `/home/leejt/OneRec/output/experiments/mgr_sid_tokenizer_v2/generated_indices/Industrial_and_Scientific.mgr_tokenizer_v2_offline.index.json`

## Data Roots

- baseline data root already exists:
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_upstream_baseline`
- `v2` data root:
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_tokenizer_v2_offline`

## Configs

- SFT:
  `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_tokenizer_v2_offline.yaml`
- evaluate:
  `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_tokenizer_v2_offline.yaml`

## Runtime Plan

- convert writes the `v2` SID into `data_experiment/`
- `SFT` and `evaluate` run in one chained `tmux` session
- physical GPUs: `2,3,4,5`
- session name: `mgr_sft_eval_v2_ind`

## Logs

- convert:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_convert_industrial_20260411.log`
- SFT:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_sft_industrial_20260411.log`
- evaluate:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_eval_industrial_20260411.log`

## Outputs

- SFT root:
  `/home/leejt/OneRec/output/experiments/mgr_sid_v2_sft_eval_industrial_20260411/mgr_tokenizer_v2_offline/sft`
- evaluate result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_v2_sft_eval_industrial_20260411/final_result_sft_mgr_tokenizer_v2_offline_Industrial_and_Scientific.json`

## Notes

- This run does not re-run the original MiniOneRec baselines.
- `mgr_upstream_baseline` is treated as an internal control rather than the main baseline.
- Main conclusions should be based on:
  - recipe-aligned original MiniOneRec comparison
  - strongest original MiniOneRec SFT / RL comparison
- See `RESULTS.md` in the same folder for the current interpretation.
