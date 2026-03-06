# OneRec 参数一致性矩阵

这份文件只描述当前正式主配置，不再维护旧兼容入口映射。

## SFT

- 配置文件：`config/sft.yaml`
- 关键字段：
  - `training.batch_size = 1024`
  - `training.micro_batch_size = 4`
  - `training.warmup_steps = 20`
  - `training.group_by_length = false`
  - `training.load_best_model_at_end = true`
  - `training.freeze_llm = false`
  - `training.eval_step = 0.05`
  - `logging.wandb_project = OneRec`

## RL

- 配置文件：`config/rl.yaml`
- 关键字段：
  - `training.num_generations = 8`
  - `training.temperature = 1.0`
  - `training.eval_step = 0.0999`
  - `training.beam_search = true`
  - `training.test_during_training = false`
  - `training.reward_type = ranking`
  - `training.sync_ref_model = true`
  - `logging.wandb_project = OneRec`

## Evaluate

- 配置文件：`config/evaluate.yaml`
- 关键字段：
  - `batch_size = 8`
  - `num_beams = 50`
  - `max_new_tokens = 256`
  - `length_penalty = 0.0`
  - `temperature = 1.0`
  - `guidance_scale = null`

## 验收方式

执行：

```bash
python parity_check.py
```

如果输出 `Parity check passed.`，说明当前主配置仍处于既定一致性区间。
