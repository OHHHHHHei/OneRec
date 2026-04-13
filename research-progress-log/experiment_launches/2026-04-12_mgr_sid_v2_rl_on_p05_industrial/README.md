# 2026-04-12 MGR-SID V2 RL Launch on Industrial

## Goal

Continue from the current best downstream `v2` setting and test whether RL can close the remaining gap to the strongest original MiniOneRec RL result.

Current source checkpoint:

- `v2_on_p05` SFT final checkpoint:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_v2_recipe_isolation_20260411/title_on_desc_p05/sft/final_checkpoint`

Target comparison:

- strongest original MiniOneRec RL:
  `rl_industrial_title_history2sid_off__desc_align_p05_batch256_20260329_152417`

## Alignment Principle

This run aligns the RL training regime to the strongest recovered original MiniOneRec RL recipe as closely as the repository currently allows:

- `num_generations = 16`
- `gradient_accumulation_steps = 4`
- `world_size = 4`
- `effective_global_batch = 256`
- `per_device_train_batch_size = 16`
- `num_epochs = 2`
- `learning_rate = 1e-5`
- `reward_type = ranking`
- `beam_search = true`
- `max_completion_length = 128`

Implementation note:

- the current mainline RL pipeline always includes `SidDataset + RLTitle2SidDataset + RLSeqTitle2SidDataset`
- therefore this run is best understood as:
  - source SFT checkpoint = `v2_on_p05`
  - RL training regime aligned to the strongest recovered original MiniOneRec RL setup
  - with `per_device_train_batch_size = num_generations = 16` to satisfy the current mainline GRPO batch-shape requirement while keeping `effective_global_batch = 256`

## Configs

- RL config:
  `/home/leejt/OneRec/config/experiments/rl_industrial_mgr_tokenizer_v2_title_on_desc_p05.yaml`
- Evaluate config:
  `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_tokenizer_v2_title_on_desc_p05_rl.yaml`
- Chain script:
  `/home/leejt/OneRec/scripts/experiment_mgr_sid_v2_rl_eval_chain.sh`

## Runtime

- tmux session:
  `mgr_v2_rl_on_p05_ind`
- GPUs:
  `2,3,4,5`

## Outputs

- RL output:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_v2_rl_on_p05_industrial_20260412/rl`
- Final evaluate result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_v2_rl_on_p05_industrial_20260412/final_result_rl_mgr_tokenizer_v2_title_on_desc_p05_Industrial_and_Scientific.json`

## Logs

- RL log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_rl_on_p05_industrial_20260412.log`
- Evaluate log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_eval_on_p05_rl_industrial_20260412.log`
