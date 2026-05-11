# 原始 SID Pipeline 复现检查（2026-04-09）

## 目的

这轮实验只做一件事：

> 用当前仓库里的原始主线 `sid-train -> sid-generate`，重新跑一遍纯语义 `qwen-td` baseline，检查能不能复现仓库里当前正式 `index.json` 的结果。

这次实验明确要求：

- 不和仓库中现有结果混在一起
- 使用全新的隔离输出目录
- 直接走当前主线代码，而不是实验分支

## 隔离输出位置

所有新产物都放在：

- `output/reproductions/2026-04-09_sid_pipeline_rerun/`

具体包括：

- `industrial_sid_train/`
- `office_sid_train/`
- `generated_indices/`

日志包括：

- `logs/repro_sid_train_industrial_qwen_td_20260409.log`
- `logs/repro_sid_train_office_qwen_td_20260409.log`
- `logs/repro_sid_generate_industrial_qwen_td_20260409.log`
- `logs/repro_sid_generate_office_qwen_td_20260409.log`

## 运行命令

### 1. sid-train

Industrial:

```bash
python -m onerec.main sid-train \
  --config config/sid_train.yaml \
  data_path=./data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
  device=cuda:2 \
  ckpt_dir=./output/reproductions/2026-04-09_sid_pipeline_rerun/industrial_sid_train
```

Office:

```bash
python -m onerec.main sid-train \
  --config config/sid_train.yaml \
  data_path=./data/Amazon/index/Office_Products.emb-qwen-td.npy \
  device=cuda:3 \
  ckpt_dir=./output/reproductions/2026-04-09_sid_pipeline_rerun/office_sid_train
```

### 2. sid-generate

Industrial:

```bash
python -m onerec.main sid-generate \
  --config config/sid_generate.yaml \
  ckpt_path=./output/reproductions/2026-04-09_sid_pipeline_rerun/industrial_sid_train/Apr-09-2026_21-40-30/best_collision_model.pth \
  output_file=./output/reproductions/2026-04-09_sid_pipeline_rerun/generated_indices/Industrial_and_Scientific.reproduced.index.json \
  device=cuda:2 \
  batch_size=64
```

Office:

```bash
python -m onerec.main sid-generate \
  --config config/sid_generate.yaml \
  ckpt_path=./output/reproductions/2026-04-09_sid_pipeline_rerun/office_sid_train/Apr-09-2026_21-40-30/best_collision_model.pth \
  output_file=./output/reproductions/2026-04-09_sid_pipeline_rerun/generated_indices/Office_Products.reproduced.index.json \
  device=cuda:3 \
  batch_size=64
```

## 结果

### A. sid-train 的 raw collision

Industrial:

- best collision: `0.9989148128052089`

Office:

- best collision: `0.9959525874530211`

这说明当前主线 `sid-train` 训练结束时，raw code collision 非常高。

### B. sid-generate 后的最终 index collision

重新生成的 index 文件：

- `output/reproductions/2026-04-09_sid_pipeline_rerun/generated_indices/Industrial_and_Scientific.reproduced.index.json`
- `output/reproductions/2026-04-09_sid_pipeline_rerun/generated_indices/Office_Products.reproduced.index.json`

其 collision 为：

- Industrial: `0.9940314704286489`
- Office: `0.9849667533969355`

最大冲突数：

- Industrial: `816`
- Office: `488`

也就是说，当前代码下的 `sid-generate` 并没有把 collision 修到很低。

### C. 与仓库当前正式 index 的对比

仓库中的正式 index：

- `data/Amazon/index/Industrial_and_Scientific.index.json`
- `data/Amazon/index/Office_Products.index.json`

它们的 collision 是：

- Industrial: `0.004340748779164406`
- Office: `0.004336513443191674`

最大冲突数：

- Industrial: `3`
- Office: `2`

### D. 新旧 index 的逐项一致性

将本轮复现出的 index 与当前正式 index 做 item-wise exact match：

- Industrial exact match: `0 / 3686`
- Office exact match: `0 / 3459`

也就是说：

> 当前主线 fresh rerun 生成的 index，与仓库当前正式 index 完全对不上。

## 最重要的结论

这轮复现给出的结论非常明确：

1. 当前主线 `sid-train` 的 raw collision 的确非常高。
2. 当前主线 `sid-generate` 也没有把 collision 修到历史记录中的低水平。
3. 用当前代码 fresh rerun，无法复现仓库中现有正式 `index.json` 的结果。

## 这说明什么

当前最合理的解释不是“我们这次手滑跑错了”，而是：

- 仓库里当前正式 `index.json` 很可能不是由当前这套默认 `sid-train -> sid-generate` 主线 fresh 生成出来的；
- 或者它来自历史代码版本、历史 checkpoint、额外修补步骤，或另一个未被当前默认入口覆盖的生成流程。

换句话说：

> 当前仓库存在一个明显的 provenance gap：  
> `data/Amazon/index/*.index.json` 的来源，和当前默认主线代码之间，至少有一个关键环节没有被完整保留或没有被当前入口复现出来。

## 当前建议

如果要继续追这个问题，下一步最值得做的不是再重复跑一遍，而是：

1. 回溯当前正式 `index.json` 的真实来源
2. 搜索是否存在旧版本 `sid-generate`、额外去重脚本或人工后处理
3. 检查历史 checkpoint / commit / 旧实验目录中是否保留了对应生成链
4. 单独审查当前 `rqvae_indices.py` 的 collision repair 逻辑，解释为什么它在 fresh rerun 中几乎没有修复冲突
