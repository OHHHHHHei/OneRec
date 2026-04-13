# 2026-04-11 MGR-SID v2 Recipe Isolation (Industrial)

## Goal

Run the two missing cells of the `v2 tokenizer` recipe-isolation matrix:

1. `title_history2sid = on`, `desc_align = p05`
2. `title_history2sid = off`, `desc_align = off`

This isolates which factor is causing the strongest original MiniOneRec recipe mismatch:

- turning `title_history2sid` off
- turning `description alignment` on

## Fixed Elements

- tokenizer source:
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_tokenizer_v2_offline/index/Industrial_and_Scientific.index.json`
- data root:
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_tokenizer_v2_offline`
- training regime aligned to strongest original MiniOneRec SFT:
  - `batch_size = 1024`
  - `micro_batch_size = 2`
  - `world_size = 4`
  - `num_epochs = 10`
  - `learning_rate = 3e-4`
  - `cutoff_len = 512`
  - `eval_step = 0.05`

## Runs

### Run A

- recipe:
  - `title_history2sid = on`
  - `desc_align = p05`
- SFT config:
  `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_tokenizer_v2_title_on_desc_p05.yaml`
- evaluate config:
  `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_tokenizer_v2_title_on_desc_p05.yaml`
- tmux session:
  `mgr_v2_iso_ton_dp05_ind`

### Run B

- recipe:
  - `title_history2sid = off`
  - `desc_align = off`
- SFT config:
  `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_tokenizer_v2_title_off_desc_off.yaml`
- evaluate config:
  `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_tokenizer_v2_title_off_desc_off.yaml`
- tmux session:
  `mgr_v2_iso_toff_doff_ind`

## Scheduling

- both runs use physical GPUs `2,3,4,5`
- Run A starts immediately
- Run B waits in queue and starts only after the same GPU set becomes free

## Logs

- Run A SFT:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_title_on_desc_p05_sft_industrial_20260411.log`
- Run A evaluate:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_title_on_desc_p05_eval_industrial_20260411.log`
- Run B SFT:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_title_off_desc_off_sft_industrial_20260411.log`
- Run B evaluate:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_title_off_desc_off_eval_industrial_20260411.log`

## Outputs

- Run A SFT:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_v2_recipe_isolation_20260411/title_on_desc_p05/sft`
- Run A result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_v2_recipe_isolation_industrial_20260411/final_result_sft_mgr_tokenizer_v2_title_on_desc_p05_Industrial_and_Scientific.json`
- Run B SFT:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_v2_recipe_isolation_20260411/title_off_desc_off/sft`
- Run B result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_v2_recipe_isolation_industrial_20260411/final_result_sft_mgr_tokenizer_v2_title_off_desc_off_Industrial_and_Scientific.json`

## Results Summary

The full four-cell `v2` recipe matrix on Industrial is now:

| Run | title_history2sid | desc_align | NDCG@10 | HR@10 |
|---|---|---|---:|---:|
| `v2_on_off` | on | off | 0.10082 | 0.14251 |
| `v2_on_p05` | on | p05 | 0.10271 | 0.14626 |
| `v2_off_off` | off | off | 0.09125 | 0.13391 |
| `v2_off_p05` | off | p05 | 0.08993 | 0.13082 |

Main conclusions:

- the best current downstream recipe for `v2` is `title_history2sid_on + desc_align_p05`
- `desc_align_p05` is not the main cause of the earlier strongest-recipe failure
- the major negative factor is `title_history2sid_off`
- relative to strongest original MiniOneRec SFT, the remaining gap for `v2_on_p05` is already small:
  - `NDCG@10`: `0.10271` vs `0.10372`
  - `HR@10`: `0.14626` vs `0.15089`

Detailed structural analysis is recorded in:

- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/TOPK_ERROR_DISTRIBUTION_COMPARISON.md`
- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/TOPK_STRONGEST_ORIG_VS_V2_ON_P05.md`
- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/TOPK_V2_ON_OFF_VS_V2_ON_P05.md`
