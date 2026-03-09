# OneRec 参数一致性矩阵

当前 `config/sft.yaml`、`config/rl.yaml`、`config/evaluate.yaml` 已改为模板化配置。

也就是说，配置里会显式出现这些占位符：

- `%{category}`
- `%{split_stem}`
- `%{artifact_stem}`
- `%{eval_model_stage}`
- `%{eval_result_suffix}`

真正执行 `bash sft.sh industrial`、`bash rl.sh office`、`bash evaluate.sh rl industrial` 时，
shell 会先结合 [`config/datasets.yaml`](/e:/MiniOneRec-mainline/config/datasets.yaml) 渲染出临时配置，再启动对应阶段。

## 默认校验基线

`parity_check.py` 当前以这组默认渲染基线做检查：

- dataset key: `industrial`
- evaluate model stage: `sft`

## SFT

- 配置模板：`config/sft.yaml`
- 渲染后关键字段：
  - `training.batch_size = 1024`
  - `training.micro_batch_size = 4`
  - `training.warmup_steps = 20`
  - `training.group_by_length = false`
  - `training.load_best_model_at_end = true`
  - `training.freeze_llm = false`
  - `training.eval_step = 0.05`
  - `logging.wandb_project = OneRec`

## RL

- 配置模板：`config/rl.yaml`
- 渲染后关键字段：
  - `training.num_generations = 4`
  - `training.temperature = 1.0`
  - `training.eval_step = 0.05`
  - `training.beam_search = true`
  - `training.test_during_training = false`
  - `training.reward_type = ranking`
  - `training.sync_ref_model = true`
  - `logging.wandb_project = OneRec`

## Evaluate

- 配置模板：`config/evaluate.yaml`
- 渲染后关键字段：
  - `batch_size = 8`
  - `num_beams = 50`
  - `max_new_tokens = 256`
  - `length_penalty = 0.0`
  - `temperature = 1.0`
  - `guidance_scale = null`

## 校验方式

执行：

```bash
python parity_check.py
```

如果输出 `Parity check passed.`，说明默认模板渲染结果仍然落在当前既定基线内。
