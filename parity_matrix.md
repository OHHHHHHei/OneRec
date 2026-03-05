# OneRec 参数一致性矩阵（基线：`E:/MiniOneRec`）

本表用于锁定 `SFT / RL / Evaluate` 与原版脚本的关键参数语义，作为重构后的回归基线。

## SFT

| 新配置路径 | 旧参数名 | 期望默认值 | 必须一致 | 说明 |
|---|---|---:|---|---|
| `training.batch_size` | `batch_size` | `1024` | 是 | 用于梯度累积推导 |
| `training.micro_batch_size` | `micro_batch_size` | `4` | 是 | DDP 下每卡 batch |
| `training.warmup_steps` | `warmup_steps` | `20` | 是 | 原版固定 20 |
| `training.group_by_length` | `group_by_length` | `false` | 是 | 与原版默认一致 |
| `training.load_best_model_at_end` | `load_best_model_at_end` | `true` | 是 | 原版默认开启 |
| `training.freeze_llm` | `freeze_LLM` | `false` | 是 | `freeze_LLM` 兼容映射 |
| `training.eval_step` | `eval_step` | `0.05` | 是 | eval/save 步长 |

## RL

| 新配置路径 | 旧参数名 | 期望默认值 | 必须一致 | 说明 |
|---|---|---:|---|---|
| `training.num_generations` | `num_generations` | `8` | 是 | 训练生成数 |
| `training.temperature` | `temperature` | `1.0` | 是 | 采样温度 |
| `training.eval_step` | `eval_step` | `0.0999` | 是 | eval 频率 |
| `training.beam_search` | `beam_search` | `true` | 是 | 训练中约束 beam |
| `training.test_during_training` | `test_during_training` | `false` | 是 | 默认关闭在线测试 |
| `training.reward_type` | `reward_type` | `ranking` | 是 | 与当前主线一致 |
| `training.sync_ref_model` | `sync_ref_model` | `true` | 是 | GRPO 同步策略 |

## Evaluate

| 新配置路径 | 旧参数名 | 期望默认值 | 必须一致 | 说明 |
|---|---|---:|---|---|
| `batch_size` | `batch_size` | `8` | 是 | 分块推理大小 |
| `num_beams` | `num_beams` | `50` | 是 | 约束 beam 数 |
| `max_new_tokens` | `max_new_tokens` | `256` | 是 | 最大生成长度 |
| `length_penalty` | `length_penalty` | `0.0` | 是 | 长度惩罚 |
| `temperature` | `temperature` | `1.0` | 是 | 兼容字段，`do_sample=False` 下不参与 |
| `guidance_scale` | `guidance_scale` | `null` | 否 | 可选兼容字段 |

## Legacy 参数映射（必须存在）

- `sft.py` 兼容映射：`freeze_LLM -> training.freeze_llm`
- `rl.py` 兼容映射：`beam_search/test_during_training/eval_step/sync_ref_model`
- `evaluate.py` 兼容映射：`num_beams/max_new_tokens/length_penalty/temperature/guidance_scale`
