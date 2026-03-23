# OneRec 项目理解报告

## 项目整体结论

OneRec 是一个面向生成式推荐的工程化主链仓库，目标是把原始 MiniOneRec 的核心实验链路收敛为统一的 `bash + yaml + python stage` 体系，主链为：

```text
preprocess -> embed -> sid-train -> sid-generate -> convert -> sft -> rl -> evaluate
```

当前仓库真正的核心研究主线是 `convert -> sft -> rl -> evaluate`，其中：

- `SFT` 负责把语义 ID 预测、item 对齐、标题/描述对齐等多个任务拼成一个统一监督训练集。
- `RL` 在 SFT checkpoint 基础上，用约束生成 + 自定义 reward 做 GRPO 式优化。
- `Evaluate` 用 `info.txt` 构建合法 semantic ID 约束树，进行受约束 beam 解码，并打印 HR/NDCG。

从当前工作区状态看，这个仓库已经包含：

- 已处理好的 `Industrial_and_Scientific` / `Office_Products` 数据。
- 已训练好的部分 `SFT / RL` checkpoint。
- 已跑过的 evaluate 结果与分片临时目录。

但“从原始 preprocess 一路跑到最终 evaluate”的默认配置并不完全闭环，至少有几处明显的配置漂移或危险默认值：

- `config/sid_generate.yaml` 缺少默认必需参数，直接跑默认命令大概率失败。
- `config/convert.yaml` 指向 `./data/Amazon/index`，但当前工作区该目录下并没有 `.train.inter/.valid.inter/.test.inter`。
- 原仓曾包含 `parity_check.py`，但它的预期值已经和当前 `config/sft.yaml` / `config/rl.yaml` 漂移，现已删除。

另外，当前主线对 semantic ID 的“实际假设”是 3-level：

```text
<a_x><b_y><c_z>
```

虽然部分 SID 量化代码支持 4/5 层 token，但主线训练与评测多处只显式处理前三层或只 canonicalize 三层，因此后续若做 SID 深度创新，需要先补兼容性检查。（来源：`src/onerec/sft/datasets.py`、`src/onerec/evaluate/semantic_id.py`、`src/onerec/sid/generate/*.py`）

来源文件：

- `README.md`
- `src/onerec/main.py`
- `src/onerec/sft/*`
- `src/onerec/rl/*`
- `src/onerec/evaluate/*`

## 主链路概览

### 仓库结构

当前仓库主要分层如下：

- `config/`: 各 stage YAML、数据集模板映射、DeepSpeed 配置。
- `src/onerec/main.py`: 统一 stage 分发入口。
- `src/onerec/preprocess`: 原始 Amazon18/Amazon23 预处理。
- `src/onerec/sid`: embedding、量化训练、semantic ID 生成。
- `src/onerec/convert`: 把 `.inter + item.json + index.json` 转成 SFT/RL/Evaluate 消费的 CSV/info 契约。
- `src/onerec/sft`: 数据集构造、token 扩展、Trainer 封装。
- `src/onerec/rl`: RL 数据集、reward、GRPO trainer、DeepSpeed 兼容层。
- `src/onerec/evaluate`: 受约束解码、split/merge、metrics。
- `tests/unit`: 单测。
- `data/`: 当前仓库内的数据与索引。
- `output/`, `results/`, `temp/`: 已有实验产物。

来源文件：

- `README.md`
- `config/`
- `src/onerec/`
- `tests/`

### 从入口脚本到 Python stage 的运行图

常规 stage 的运行路径：

```text
bash <stage>.sh
  -> source _common.sh
  -> 解析 config / dataset key / eval stage / overrides
  -> (SFT/RL/Evaluate) 渲染 YAML 模板
  -> python -m onerec.main <stage> --config <path> [overrides]
  -> load_config(...)
  -> dispatch 到 src/onerec/<stage>/...
```

`SFT` 的外层运行图：

```text
sft.sh
  -> _common.sh.resolve_config_path
  -> _common.sh.use_first_positional_as_dataset_key
  -> _common.sh.render_stage_config
  -> 读取 runtime.launcher / gpus / nproc
  -> torchrun 或 python
  -> onerec.main:sft
  -> onerec.sft.pipeline:run_sft
```

`RL` 的外层运行图：

```text
rl.sh
  -> resolve_config_path
  -> use_first_positional_as_dataset_key
  -> render_stage_config
  -> 读取 runtime.launcher / accelerate_config / num_processes
  -> accelerate launch 或 torchrun 或 python
  -> onerec.main:rl
  -> onerec.rl.pipeline:run_rl
  -> onerec.rl.trainer:ReReTrainer
```

`Evaluate` 的外层运行图：

```text
evaluate.sh
  -> resolve_config_path
  -> resolve_evaluate_selection
  -> render_stage_config
  -> 若 parallel=false: 直接 onerec.main evaluate
  -> 若 parallel=true:
       split -> 多 worker evaluate -> merge -> metrics
```

来源文件：

- `_common.sh`
- `sft.sh`
- `rl.sh`
- `evaluate.sh`
- `src/onerec/main.py`

## 训练与评测主链路

### 各阶段总表

| 阶段 | Shell 入口 | Python 入口 | 主要输入 | 主要输出 | 与前后阶段关系 |
| --- | --- | --- | --- | --- | --- |
| preprocess | `preprocess_amazon18.sh` / `preprocess_amazon23.sh` | `onerec.preprocess.amazon18` / `amazon23` | 原始 review + metadata JSON/JSONL | `.train.inter/.valid.inter/.test.inter`、`.item.json`、`.review.json`、`.inter.json`、`.user2id`、`.item2id` | 为 `convert` 和 `embed` 提供原始结构化数据 |
| embed | `text2emb.sh` | `onerec.sid.embed` | `<dataset>.item.json` | `<dataset>.emb-<plm>-td.npy` | 为 `sid-train` 提供 item embedding |
| sid-train | `sid_train.sh` | `onerec.sid.quantizers.*` | `.emb-*.npy` | 量化模型 checkpoint 目录 | 为 `sid-generate` 提供量化器 |
| sid-generate | `sid_generate.sh` | `onerec.sid.generate.*` | `.emb-*.npy` + SID checkpoint | `<dataset>.index.json` | 为 `convert/SFT/RL` 提供 semantic ID 映射 |
| convert | `convert.sh` | `onerec.convert.pipeline:run_convert` | `.item.json` + `.index.json` + `.train/.valid/.test.inter` | `train/valid/test/*.csv` + `info/*.txt` | 把 preprocess/SID 产物转成后续训练评测契约 |
| sft | `sft.sh` | `onerec.sft.pipeline:run_sft` | `train.csv`、`valid.csv`、`item.json`、`index.json`、base model | `checkpoint-*`、`final_checkpoint/` | 为 RL 和 Evaluate 提供监督微调模型 |
| rl | `rl.sh` | `onerec.rl.pipeline:run_rl` | SFT checkpoint、`train.csv`、`valid.csv`、`item.json`、`index.json`、`info.txt` | `checkpoint-*`、`final_checkpoint/` | 在 SFT 基础上继续优化，并可被 Evaluate 使用 |
| evaluate | `evaluate.sh` | `onerec.evaluate.pipeline:run_evaluate` + `split/merge/metrics` | `test.csv`、`info.txt`、待评估 checkpoint | shard JSON、merged JSON、stdout metrics | 主链最终评测输出 |

### preprocess

- 入口脚本：`preprocess_amazon18.sh`、`preprocess_amazon23.sh`
- 主 Python 入口：`src/onerec/preprocess/amazon18.py`、`src/onerec/preprocess/amazon23.py`
- 输入：
  - Amazon18：原始 `meta_*.json` 与 `*_5.json`
  - Amazon23：原始 `*.jsonl` reviews 和 metadata
- 输出：
  - `<dataset>.train.inter`
  - `<dataset>.valid.inter`
  - `<dataset>.test.inter`
  - `<dataset>.item.json`
  - `<dataset>.review.json`
  - `<dataset>.inter.json`
  - `<dataset>.user2id`
  - `<dataset>.item2id`
- 链接关系：
  - `embed` 消费 `item.json`
  - `convert` 需要 `item.json + *.inter + index.json`

补充观察：

- Amazon18 版本有“若 item 数不足 3000，则向前扩时间窗口”的递归逻辑；Amazon23 没有这段递归扩展。（来源：`src/onerec/preprocess/amazon18.py`）
- 两个 preprocess 都是按全局时间排序后做 8:1:1 切分，并把历史长度截到最多 50。（来源：`src/onerec/preprocess/amazon18.py`、`src/onerec/preprocess/amazon23.py`）

### embed

- 入口脚本：`text2emb.sh`
- 主 Python 入口：`src/onerec/sid/embed.py`
- 输入：
  - `<root>/<dataset>.item.json`
- 输出：
  - `<root>/<dataset>.emb-<plm_name>-td.npy`
- 链接关系：
  - 输出 embedding 给 `sid-train`

关键实现：

- 读取 `item.json` 的 `title + description`
- 清洗文本后逐 item 编码
- 使用 Transformer hidden states 做 masked mean pooling
- 通过 `Accelerator` 做多进程切分与聚合

来源文件：

- `config/embed.yaml`
- `src/onerec/sid/embed.py`

### sid-train

- 入口脚本：`sid_train.sh`
- 主 Python 入口：`src/onerec/main.py` 根据 `kind` 分发到：
  - `src/onerec/sid/quantizers/rqvae.py`
  - `src/onerec/sid/quantizers/rqkmeans_faiss.py`
  - `src/onerec/sid/quantizers/rqkmeans_constrained.py`
  - `src/onerec/sid/quantizers/rqkmeans_plus.py`
- 输入：
  - `.emb-*.npy`
- 输出：
  - 量化模型 checkpoint
  - 某些量化实现还会直接输出 codebook / codes
- 链接关系：
  - checkpoint 供 `sid-generate` 使用

关键实现：

- 通用 trainer 在 `src/onerec/sid/trainer.py`
- “验证”指标是 collision rate，而且验证数据直接复用训练 data loader，不是单独 held-out 集。（来源：`src/onerec/sid/trainer.py`）

### sid-generate

- 入口脚本：`sid_generate.sh`
- 主 Python 入口：
  - `src/onerec/sid/generate/rqvae_indices.py`
  - `src/onerec/sid/generate/rqkmeans_plus_indices.py`
- 输入：
  - `.emb-*.npy`
  - SID checkpoint
- 输出：
  - `<dataset>.index.json`
- 链接关系：
  - `convert`、`sft`、`rl` 都依赖 `index.json`

关键实现：

- `rqvae_indices.py` 会先编码，再针对冲突样本做多轮 `use_sk=True` 的重编码，尽量降低 collision。
- `rqkmeans_plus_indices.py` 会在 raw codes 上做 `+1` offset，然后用 Polars 做去重修补。

待确认：

- `config/sid_generate.yaml` 当前只有 `kind: rqvae`，但 `rqvae_indices.py` 明确要求 `--ckpt_path` 和 `--output_file`。因此 README 中直接 `bash sid_generate.sh` 与当前默认 YAML 不一致。（来源：`config/sid_generate.yaml`、`src/onerec/sid/generate/rqvae_indices.py`）

### convert

- 入口脚本：`convert.sh`
- 主 Python 入口：`src/onerec/convert/pipeline.py`
- 输入：
  - `<dataset>.item.json`
  - `<dataset>.index.json`
  - `<dataset>.train.inter`
  - `<dataset>.valid.inter`
  - `<dataset>.test.inter`
- 输出：
  - `train/<category>_5_2016-10-2018-11.csv`
  - `valid/<category>_5_2016-10-2018-11.csv`
  - `test/<category>_5_2016-10-2018-11.csv`
  - `info/<category>_5_2016-10-2018-11.txt`
- 链接关系：
  - `SFT` 消费 `train.csv/valid.csv + item.json/index.json`
  - `RL` 消费 `train.csv/valid.csv + item.json/index.json + info.txt`
  - `Evaluate` 消费 `test.csv + info.txt`

关键实现：

- `index.json` 把 `item_id -> ["<a_...>", "<b_...>", "<c_...>"]`
- `convert` 直接把 token list 拼成 `item_sid`
- `info.txt` 每行写成：`semantic_id \t item_title \t item_id`
- CSV 里保留了 title / raw item id / semantic id 三套字段

危险点：

- `load_dataset()` 期望所有源文件都在同一个 `data_dir` 下；当前默认 `config/convert.yaml` 指向 `./data/Amazon/index`，但当前工作区这个目录下并没有 `.train.inter/.valid.inter/.test.inter`，因此默认 convert 配置与当前数据布局不匹配。（来源：`config/convert.yaml`、`src/onerec/convert/pipeline.py`、实际 `data/` 目录）

### sft

- 入口脚本：`sft.sh`
- 主 Python 入口：`src/onerec/sft/pipeline.py`
- 输入：
  - `data.train_file`
  - `data.eval_file`
  - `data.sid_index_path`
  - `data.item_meta_path`
  - `model.base_model`
- 输出：
  - `output.output_dir/checkpoint-*`
  - `output.output_dir/final_checkpoint`
- 链接关系：
  - RL 默认从 SFT `final_checkpoint` 启动
  - Evaluate 可直接评估 SFT checkpoint

### rl

- 入口脚本：`rl.sh`
- 主 Python 入口：`src/onerec/rl/pipeline.py`
- 输入：
  - `model.base_model`，默认是 SFT `final_checkpoint`
  - `train.csv`、`valid.csv`
  - `item.json`、`index.json`
  - `info.txt`
- 输出：
  - `output.output_dir/final_checkpoint`
  - 训练期间的 step-based checkpoint
- 链接关系：
  - Evaluate 可选择评估 RL checkpoint

### evaluate

- 入口脚本：`evaluate.sh`
- 主 Python 入口：
  - `src/onerec/evaluate/pipeline.py`
  - `src/onerec/evaluate/split_merge.py`
  - `src/onerec/evaluate/merge.py`
  - `src/onerec/evaluate/metrics.py`
- 输入：
  - `test.csv`
  - `info.txt`
  - 待评估模型目录
- 输出：
  - `temp/<stage>-<category>/<gpu>.csv`
  - `temp/<stage>-<category>/<gpu>.json`
  - `results/final_result_<stage>_<category>.json`
  - stdout 指标
- 链接关系：
  - 是主链的终点

来源文件：

- `sft.sh`
- `rl.sh`
- `evaluate.sh`
- `src/onerec/sft/pipeline.py`
- `src/onerec/rl/pipeline.py`
- `src/onerec/evaluate/pipeline.py`

## SFT 分析

### 数据集组成

SFT 训练集不是单一任务，而是 4 个 dataset 的拼接：

1. `SidSFTDataset`
2. `SidItemFeatDataset`
3. `FusionSeqRecDataset`
4. `TitleHistory2SidSFTDataset`

具体含义如下：

- `SidSFTDataset`: 输入历史 semantic ID 序列，输出下一个 `item_sid`。
- `SidItemFeatDataset`: 双向 item 对齐任务。
  - `sid -> title`
  - `title -> sid`
- `FusionSeqRecDataset`: 输入历史交互，输出目标 item 的 title 或 description。
- `TitleHistory2SidSFTDataset`: 输入历史 item title 序列，输出目标 `item_sid`。

当前工作区 `Industrial_and_Scientific` 的实际训练规模可按当前数据直接推得：

- `SidSFTDataset`: 36259
- `SidItemFeatDataset`: 7372
- `FusionSeqRecDataset`: 36259
- `TitleHistory2SidSFTDataset`: 36259
- 合计：116149

来源文件：

- `src/onerec/sft/pipeline.py`
- `src/onerec/sft/datasets.py`
- `data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- `data/Amazon/index/Industrial_and_Scientific.item.json`

### Prompt 与隐藏数据混合策略

几个重要但不在 YAML 里配置的“代码级默认值”：

- `FusionSeqRecDataset.enable_title_description_alignment = True`
- `FusionSeqRecDataset.description_task_probability = 0.5`
- `FusionSeqRecDataset` 会把 description 字段做自定义清洗，若是 list 或“字符串化 list”，优先取最长文本。

这意味着当前 SFT 实际并不是纯“history -> next SID”监督，而是混合了 title 对齐和 description 对齐任务；这一点对后续科研迭代很重要，因为它会直接影响对齐目标和训练梯度分布。（来源：`src/onerec/sft/datasets.py`）

### token 扩展

SFT 在加载 tokenizer 后，会读取 `sid_index_path` 里所有 unique semantic tokens，并执行：

- `tokenizer.add_tokens(new_tokens)`
- `model.resize_token_embeddings(len(tokenizer))`

当前 `Industrial_and_Scientific` 数据里，semantic token 的 unique 数量为：

- `a` 层：48
- `b` 层：256
- `c` 层：256
- 合计 SID token：560

当前已有 SFT checkpoint 的 `added_tokens.json` 中也确实包含这 560 个 SID token；另外还带有 26 个非 SID 的 Qwen 特殊 token。（来源：`src/onerec/sft/token_extension.py`、`data/Amazon/index/Industrial_and_Scientific.index.json`、`output/sft_Industrial_and_Scientific_refactor/final_checkpoint/added_tokens.json`）

### semantic ID 深度假设

虽然 SID 量化模块支持可变层数，但主线训练代码多处只显式使用前三层：

- `SidItemFeatDataset`: `sid = "".join(sids[:3])`
- `FusionSeqRecDataset`: `sid = "".join(sids[:3])`
- `TitleHistory2SidSFTDataset`: `id2sid[str(item_id)] = "".join(sids[:3])`

因此当前主线实际假设 semantic ID 是三层 `<a><b><c>`。如果后续要改到 4 层以上，不能只改 SID 生成端，还要同步检查 SFT/RL/Evaluate/metrics。（来源：`src/onerec/sft/datasets.py`、`src/onerec/evaluate/semantic_id.py`）

### 训练参数推导

SFT 的 batch 关系由 `_resolve_grad_accum_steps()` 决定：

```text
gradient_accumulation_steps
  = floor(batch_size / micro_batch_size)
  = 若 DDP, 再除以 world_size
```

当前 `config/sft.yaml` 是：

- `batch_size = 1024`
- `micro_batch_size = 2`
- `runtime.nproc_per_node = 4`

所以当前实际：

```text
grad_accum = 1024 / 2 / 4 = 128
effective_global_batch = micro_batch_size * grad_accum * world_size = 1024
```

这里要特别注意：

- 这段公式用的是整除/下界裁剪，不做一致性断言。
- 如果将来 `batch_size` 不能整除 `micro_batch_size * world_size`，会静默变成更小的有效 batch。

来源文件：

- `src/onerec/sft/pipeline.py`
- `config/sft.yaml`

### 其它关键训练行为

- CUDA 默认用 BF16，CPU 默认 FP32。（来源：`src/onerec/sft/pipeline.py`）
- `eval_step` 直接作为 float 传给 `TrainingArguments.eval_steps` / `save_steps`，当前值是 `0.05`。（来源：`src/onerec/sft/pipeline.py`、`config/sft.yaml`）
- 若 `freeze_llm=true`，代码会冻结所有参数，只保留 input embedding 可训练，并对旧词表部分做梯度清零；这本质上是在“只训练新 token embedding”。（来源：`src/onerec/sft/pipeline.py`）
- 早停只看 `eval_loss`，而 eval 集只使用 `SidSFTDataset(valid.csv)`，并不覆盖全部四种 SFT 子任务。（来源：`src/onerec/sft/pipeline.py`）
- 训练集 shuffle 用固定 `seed=42`，不是配置里的 `training.seed`；eval 被 shuffle 了两次，第二次也固定是 42。（来源：`src/onerec/sft/pipeline.py`）

## RL 分析

### RL 数据组织

RL 训练集由三部分拼接而成：

1. `SidDataset(train.csv)`
2. `RLTitle2SidDataset(item.json, index.json)`
3. `RLSeqTitle2SidDataset(train.csv, sample=10000)`

含义如下：

- `SidDataset`: history semantic ID -> target semantic ID
- `RLTitle2SidDataset`: title/description -> target semantic ID
- `RLSeqTitle2SidDataset`: history title sequence -> target semantic ID

两个重要的“代码硬编码”：

- `RLSeqTitle2SidDataset` 的 sample 固定写死为 `10000`
- `RLTitle2SidDataset` 对 description 的处理比 SFT 更粗糙，只要是非空字符串就直接拿来做 prompt，没有像 SFT 那样做 list/最长描述清洗

这意味着：

- RL 的数据混合比例当前不是 YAML 控制，而是代码写死。
- SFT 与 RL 对同一份 `item.json.description` 的清洗逻辑不一致，后续若做 prompt 或 reward 研究，这里值得优先统一。

来源文件：

- `src/onerec/rl/pipeline.py`
- `src/onerec/rl/datasets.py`
- `src/onerec/sft/datasets.py`

### num_generations 的作用

`num_generations` 是 RL 中最关键的控制参数之一，它同时影响：

- 每个 prompt 生成多少个 candidate
- sampler 如何重复样本
- reward 如何按组归一化
- ranking reward 如何按组计算
- beam_search 时的 `num_beams` / `num_return_sequences`

当前 RL 训练中，`RepeatRandomSampler` 会把每个样本重复 `num_generations` 次，然后训练器要求：

```text
global_train_batch = per_device_train_batch_size * num_processes
global_eval_batch = per_device_eval_batch_size * num_processes
```

都必须能被 `num_generations` 整除，否则直接报错。（来源：`src/onerec/rl/trainer.py`）

按当前 `config/rl.yaml`：

- `train_batch_size = 16`
- `eval_batch_size = 32`
- `num_processes = 4`
- `num_generations = 16`
- `gradient_accumulation_steps = 4`

则：

- 每个训练 step 全局共有 `16 * 4 = 64` 条“已重复样本”
- 对应 `64 / 16 = 4` 个 unique prompt
- 每个 optimizer update 再乘上 grad accumulation，约等于 `4 * 4 = 16` 个 unique prompt

这说明当前 RL 更新是“少量 prompt、多候选生成”的 regime，而不是“大量 prompt、少候选”的 regime。

来源文件：

- `config/rl.yaml`
- `src/onerec/rl/trainer.py`

### reward 组成

当前 `reward_type=ranking` 时，RL 实际使用两个 reward：

1. `rule_reward`
2. `ranking_reward`

其构成方式在 `run_rl()` 中是：

```python
reward_fun = [
    build_rule_reward(...),
    build_ranking_reward(..., num_generations)
]
```

各 reward 的语义：

- `rule_reward`: completion 与 target semantic ID 完全一致则得 1，否则 0。
- `ranking_reward`: 先按 `num_generations` 一组分块；若该组至少有一个 hit，则 miss 按负的 NDCG-like 权重给惩罚，hit 给 0；若整组没有 hit，则整组都给 0。
- `semantic_reward`: 使用外部 item embedding 的余弦相似度。
- `sasrec`: 预留分支，当前主线直接 `NotImplementedError`。

来源文件：

- `src/onerec/rl/pipeline.py`
- `src/onerec/rl/rewards.py`

### reward lookup 的一个核心风险

RL 并不是直接在 batch 内保存 target，而是先在 dataset 构建阶段把：

- `prompt -> history`
- `history -> target`

塞进字典，再在 reward 计算时查回来。

问题是：同一段 history / prompt 在训练数据里会重复出现，甚至会对应不同 target。当前工作区直接统计到：

- `Industrial_and_Scientific` 的 `SidDataset(train)`：
  - 总样本 36259
  - unique histories 30016
  - duplicate histories 1778
  - 对应多个 target 的 ambiguous histories 1550
- `Office_Products` 的 `SidDataset(train)`：
  - 总样本 38924
  - unique histories 31459
  - duplicate histories 1904
  - ambiguous histories 1617

这意味着当前 RL reward lookup 存在“后写覆盖前写”的语义风险：同一个 prompt 在 reward 时可能只保留了最后一次 target。（来源：`src/onerec/rl/datasets.py`、`src/onerec/rl/rewards.py`、当前 `train.csv` 抽样统计）

这条风险在做 reward 研究时尤其要优先处理，否则 reward 噪声会直接污染结论。

### 约束生成

RL trainer 内部会像 Evaluate 一样，从 `info.txt` 构建一棵 semantic ID 合法前缀树，然后把 `ConstrainedLogitsProcessor` 接到生成过程里。

这意味着 RL 训练并不是“自由文本采样后再比对 SID”，而是“生成时就被 semantic ID 合法空间约束住”。（来源：`src/onerec/rl/trainer.py`、`src/onerec/evaluate/constrained_decoding.py`）

### beam_search / dynamic_sampling / add_gt

几个重要开关：

- `beam_search`
- `dynamic_sampling`
- `add_gt`
- `test_during_training`
- `sync_ref_model`

当前默认 YAML：

- `beam_search: true`
- `dynamic_sampling: false`
- `add_gt: false`
- `test_during_training: false`
- `sync_ref_model: true`

待确认：

- 代码里当 `beam_search=true` 时，`GenerationConfig` 同时设置了 `num_beams=self.num_generations` 和 `do_sample=True`。这更像“带 beam 的采样”，并不是严格意义上的 deterministic beam search；命名与实际语义是否完全一致，建议后续实验前再确认一次。（来源：`src/onerec/rl/trainer.py`）

### checkpoint / DeepSpeed / wandb

- 默认 launcher 是 `accelerate`，配置来自 `config/zero2_opt.yaml`，采用 DeepSpeed ZeRO-2。（来源：`rl.sh`、`config/rl.yaml`、`config/zero2_opt.yaml`）
- 训练结束后，只有 main process 会把最终模型和 tokenizer 保存到 `output.output_dir/final_checkpoint`。（来源：`src/onerec/rl/pipeline.py`）
- 训练前会调用 `patch_deepspeed_cleanup()`，目的是绕过 DeepSpeed shutdown 阶段常见的 destroy/back-end assertion 问题。（来源：`src/onerec/rl/deepspeed_compat.py`）
- 启用 wandb 时，会设置 `WANDB_PROJECT`，并且在 logging step 上可把 completions 作为 `wandb.Table` 记录。（来源：`src/onerec/rl/pipeline.py`、`src/onerec/rl/trainer.py`）

## Evaluate 分析

### split / worker / merge / metrics 流程

`evaluate.sh` 的并行评测流程是：

1. 解析 `evaluate.sh [config] [sft|rl] [dataset_key] [overrides]`
2. 渲染 `config/evaluate.yaml`
3. 读取 `runtime.parallel` 和 `runtime.cuda_visible_devices`
4. `split`：把 `test.csv` 切成多个 shard
5. 每张 GPU 起一个 worker，分别跑 `onerec.main evaluate`
6. `merge`：合并 shard JSON
7. `metrics`：打印 NDCG / HR

来源文件：

- `evaluate.sh`
- `src/onerec/main.py`
- `src/onerec/evaluate/split_merge.py`
- `src/onerec/evaluate/merge.py`
- `src/onerec/evaluate/metrics.py`

### split

`split()` 的逻辑很直接：

- 读整份 `test.csv`
- 按 `cuda_list` 长度平均切片
- 输出到 `temp/<stage>-<category>/<gpu>.csv`

但它当前使用：

```python
df[start:end].to_csv(..., index=True)
```

所以 shard CSV 会带一个多余的首列索引。当前 `EvalSidDataset` 只读取命名字段，因此不会直接炸，但这会让 shard 文件 schema 与原始 test CSV 不一致，属于易误用点。（来源：`src/onerec/evaluate/split_merge.py`，实际 `temp/sft-Industrial_and_Scientific/2.csv` 抽样）

### worker evaluate

单 worker 的 `run_evaluate()` 逻辑是：

1. 读取模型 checkpoint
2. 读取 tokenizer
3. 从 `info.txt` 构建 semantic ID 合法前缀树
4. 读取 `test.csv`，构造 `EvalSidDataset`
5. batch generate，挂上 `ConstrainedLogitsProcessor`
6. decode 并 canonicalize semantic ID
7. 把每条样本整理成 `{input, output, predict}`
8. 输出 shard JSON

当前 `results/*.json` 的记录格式确实只有：

- `input`
- `output`
- `predict`

不会保留原始 `user_id/history_item_id/item_id` 等字段。换句话说，当前 evaluate 结果适合“整体指标统计”，不适合做细粒度 per-user 误差分析。（来源：`src/onerec/evaluate/pipeline.py`、实际 `results/final_result_sft_Industrial_and_Scientific.json` 抽样）

### 约束解码核心机制

合法 token 约束来自 `info.txt` 第一列 semantic ID：

1. 对每个 semantic ID，拼成 `### Response:\n<sid>\n`
2. tokenizer 编码
3. 基于 token prefix 构造 `hash_dict`
4. 生成时，`ConstrainedLogitsProcessor` 仅保留当前 prefix 下的合法下一个 token
5. 若合法集合为空，则记录 warning 并强制 EOS

这套机制的关键点：

- 约束是在 token 级而不是字符串级完成的。
- `num_beams` 如果大于第一步合法分支数，会触发 warning。
- decode 后还会用 `canonicalize_semantic_id()` 再抽取一次 `<a><b><c>` 模式。

来源文件：

- `src/onerec/evaluate/pipeline.py`
- `src/onerec/evaluate/constrained_decoding.py`
- `src/onerec/evaluate/semantic_id.py`

### split / merge / metrics 的结果文件生成方式

文件命名链路如下：

- shard CSV: `./temp/<stage>-<category>/<gpu>.csv`
- shard JSON: `./temp/<stage>-<category>/<gpu>.json`
- merged JSON: `./results/final_result_<stage>_<category>.json`

当前工作区现有结果中，`predict` 长度与 `num_beams=50` 一致。（来源：`config/evaluate.yaml`、`evaluate.sh`、现有 `results/*.json` 抽样）

### metrics

`metrics.gao()` 会：

- 读取 merged JSON
- 对 `predict` 和 `output` 做 `canonicalize_semantic_id()`
- 计算 `NDCG@1/3/5/10/20/50`
- 计算 `HR@1/3/5/10/20/50`
- 直接打印到 stdout

当前它不会把 metrics 持久化为单独 JSON/CSV 文件，因此实验结果可复现性依赖日志保存，而不是结果目录内的结构化 summary 文件。（来源：`src/onerec/evaluate/metrics.py`）

### Evaluate 里值得特别注意的“代码默认值”

- `temperature` 虽在 config 中保留，但当前 `GenerationConfig` 是 `do_sample=False`，所以它实际上不参与采样；代码注释也明确说是“legacy compatibility”。（来源：`src/onerec/evaluate/pipeline.py`）
- `K` 参数虽然从 YAML 传到了 `EvalSidDataset`，但 dataset 逻辑里没有用到它，当前是一个未生效参数。（来源：`config/evaluate.yaml`、`src/onerec/evaluate/datasets.py`）
- `guidance_scale` 在 dataclass 默认值里是 `1.0`，但当前 YAML 明确设为 `null`，这是一个典型的“代码默认值”和“实际运行值”不同的例子。（来源：`src/onerec/config.py`、`config/evaluate.yaml`）

## 配置系统分析

### YAML 配置

当前正式入口是 `config/*.yaml`：

- `config/datasets.yaml`: 数据集 key -> 模板变量
- `config/sft.yaml`
- `config/rl.yaml`
- `config/evaluate.yaml`
- 其它 stage 的 YAML

其中 `sft/rl/evaluate` 都使用模板变量：

```yaml
%{category}
%{split_stem}
%{artifact_stem}
%{eval_model_stage}
%{eval_result_suffix}
```

来源文件：

- `config/datasets.yaml`
- `src/onerec/utils/config_templates.py`

### shell wrapper 如何渲染配置

公共流程在 `_common.sh`：

1. `resolve_config_path`
2. `use_first_positional_as_dataset_key`
3. `resolve_evaluate_selection`
4. `render_stage_config`

`render_stage_config` 内部调用：

- `onerec.utils.config_templates.render_config_file`

该函数会：

- 读取原始 YAML
- 读取 `config/datasets.yaml`
- 把模板变量替换成实际数据集值
- 写到 `/tmp/onerec-rendered/...yaml`

来源文件：

- `_common.sh`
- `src/onerec/utils/config_templates.py`

### main.py 如何分发 stage

`src/onerec/main.py` 的 stage dispatch 分为两类：

- 对外 stage：
  - `preprocess`
  - `embed`
  - `sid-train`
  - `sid-generate`
  - `convert`
  - `sft`
  - `rl`
  - `evaluate`
- Evaluate 内部辅助 stage：
  - `split`
  - `merge`
  - `metrics`

其中：

- `preprocess/embed/sid-*` 多用 `_run_module(...)` 跳转到 legacy 风格脚本模块。
- `convert/sft/rl/evaluate` 直接在 mainline pipeline 里调用 `run_*()`。

来源文件：

- `src/onerec/main.py`

### override 参数如何生效

override 形式是：

```bash
key=value
```

生效过程有两层：

1. shell wrapper 的内嵌 Python 会先临时应用 override，用于算 launcher / batch / run summary。
2. `onerec.main -> load_config()` 会再次通过 `apply_overrides()` 把 override 深写回 payload。

override 支持：

- `true/false`
- `none/null`
- int
- float
- string

而且未知字段不会丢失，会落入 dataclass 的 `extras` 字段，这也是为何 `sid_train.yaml` / `sid_generate.yaml` 这种“几乎全靠 extras”的配置仍能工作。（来源：`_common.sh`、`src/onerec/config.py`）

### 代码默认值 vs shell/YAML 实际运行值

下面是几个最重要的差异：

| 参数 | 代码默认值 | 当前 YAML / wrapper 实际值 |
| --- | --- | --- |
| `SFT training.batch_size` | 32 | 1024 |
| `SFT training.micro_batch_size` | 4 | 2 |
| `SFT training.num_epochs` | 1 | 10 |
| `SFT training.learning_rate` | 1e-4 | 3e-4 |
| `RL training.train_batch_size` | 32 | 16 |
| `RL training.num_generations` | 8 | 16 |
| `RL training.reward_type` | `rule` | `ranking` |
| `RL training.test_during_training` | `True` | `False` |
| `RL training.sync_ref_model` | `False` | `True` |
| `Evaluate batch_size` | 4 | 8 |
| `Evaluate guidance_scale` | 1.0 | `null` |
| `SFT runtime.launcher` | wrapper fallback `torchrun` | `torchrun` |
| `RL runtime.launcher` | wrapper fallback `accelerate` | `accelerate` |
| `Evaluate runtime.launcher` | wrapper fallback `parallel` | `parallel` |

需要特别指出的漂移：

- 原仓的 `parity_check.py` 曾写着 `SFT micro_batch_size=4`、`RL num_generations=4` 的预期，但当时当前配置文件已经分别是 `2` 和 `16`；该脚本因配置漂移已删除。（来源：`config/sft.yaml`、`config/rl.yaml`）

### 额外的“隐藏代码默认值”

这些值不在 YAML 里，但会影响实验语义：

- `FusionSeqRecDataset.description_task_probability = 0.5`
- `RLSeqTitle2SidDataset(sample=10000)` 写死在 pipeline
- `evaluate.temperature` 当前不参与采样
- `EvalSidDataset(K=...)` 当前不使用 K
- `sample_train=true` 时，若 base model 路径含 `sft`，RL 会直接丢掉 train_dataset 前 20%

来源文件：

- `src/onerec/sft/datasets.py`
- `src/onerec/rl/pipeline.py`
- `src/onerec/evaluate/pipeline.py`
- `src/onerec/evaluate/datasets.py`

## 数据契约分析

### 当前工作区数据规模

当前仓库内两套主要数据：

| 数据集 | train | valid | test | item/index/info 行数 | unique semantic ID |
| --- | --- | --- | --- | --- | --- |
| `Industrial_and_Scientific` | 36259 | 4532 | 4533 | 3686 | 3670 |
| `Office_Products` | 38924 | 4866 | 4866 | 3459 | 3444 |

这里 `unique semantic ID < item 数`，说明当前两套数据都存在 semantic ID 冲突：

- `Industrial_and_Scientific`: 15 个重复 semantic ID
- `Office_Products`: 15 个重复 semantic ID

来源文件：

- `data/Amazon/train/*.csv`
- `data/Amazon/valid/*.csv`
- `data/Amazon/test/*.csv`
- `data/Amazon/index/*.index.json`
- `data/Amazon/index/*.item.json`
- `data/Amazon/info/*.txt`

### train / valid / test CSV

当前 CSV 字段为：

- `user_id`
- `history_item_title`
- `item_title`
- `history_item_id`
- `item_id`
- `history_item_sid`
- `item_sid`

字段语义：

- `user_id`: convert 后的人造字符串 ID，当前常见格式为 `A<number>`
- `history_item_title`: 历史 item title 列表，Python literal string
- `item_title`: 目标 item title
- `history_item_id`: 历史 raw item_id 列表，Python literal string
- `item_id`: 目标 raw item_id
- `history_item_sid`: 历史 semantic ID 列表，Python literal string
- `item_sid`: 目标 semantic ID

训练依赖：

- SFT 依赖全部字段中的一部分，尤其是 `history_item_sid`、`history_item_title`、`item_id`、`item_sid`
- RL 依赖 `history_item_sid`、`history_item_title`、`item_sid`
- Evaluate 只用 `history_item_sid` 与 `item_sid`

最容易出错的点：

- 多个 history 字段是“字符串化列表”，后续统一靠 `ast.literal_eval()` 解析，任何格式损坏都可能直接在 dataset 构造阶段报错。
- `history_item_title/history_item_id/history_item_sid` 必须保持对齐，否则训练 prompt 和 target 会错位。

来源文件：

- `src/onerec/convert/pipeline.py`
- `src/onerec/utils/parsing.py`
- `src/onerec/sft/datasets.py`
- `src/onerec/rl/datasets.py`
- `src/onerec/evaluate/datasets.py`

### info.txt

`info.txt` 每行格式为：

```text
semantic_id \t item_title \t item_id
```

它是 RL / Evaluate 的核心桥接文件：

- RL 用它构建 semantic ID 约束树
- Evaluate 用它构建 prefix trie
- metrics 用它建立合法 semantic ID 词表

注意：

- 当前 `build_ranking_reward` 不直接用 title，但 `build_semantic_reward` 会基于 `item2id` 建图；若同一 semantic ID 对应多个 item_id，这里会有歧义。

来源文件：

- `src/onerec/convert/pipeline.py`
- `src/onerec/rl/pipeline.py`
- `src/onerec/evaluate/pipeline.py`
- `src/onerec/evaluate/metrics.py`

### index.json

`index.json` 的格式是：

```json
{
  "0": ["<a_236>", "<b_231>", "<c_226>"],
  "1": ["<a_42>", "<b_80>", "<c_160>"]
}
```

语义：

- key 是 raw `item_id`（字符串）
- value 是 semantic token list

用途：

- SFT token 扩展
- SFT/RL title/sid 对齐任务
- convert 生成 `item_sid`

来源文件：

- `src/onerec/sft/token_extension.py`
- `src/onerec/sft/datasets.py`
- `src/onerec/rl/datasets.py`
- `src/onerec/convert/pipeline.py`

### item.json

`item.json` 的格式是：

```json
{
  "0": {"title": "...", "description": "...", ...}
}
```

语义：

- key 与 `index.json` / CSV 的 `item_id` 对齐
- value 是 item 元数据

用途：

- `embed`: 读 title + description 生成 embedding
- `SFT`: 构造 item 对齐与 title/description 对齐任务
- `RL`: 构造 title/description -> SID 任务

来源文件：

- `src/onerec/sid/embed.py`
- `src/onerec/sft/datasets.py`
- `src/onerec/rl/datasets.py`

### semantic ID 的格式和使用位置

当前主线里的 semantic ID 有三种层面的“格式假设”：

1. 数据层：
   - `index.json` 与 `info.txt` 当前都是 `<a_x><b_y><c_z>`
2. 训练层：
   - 多个 dataset 直接 `join(sids[:3])`
3. 评测层：
   - `canonicalize_semantic_id()` 只提取 `<a_x><b_y><c_z>`

这说明当前主线对 semantic ID 的使用虽然表面上是“字符串”，实际上是强依赖三层结构的。（来源：`src/onerec/sft/datasets.py`、`src/onerec/evaluate/semantic_id.py`）

### 文件关系总结

| 文件 | 生产者 | 消费者 | 关键关系 |
| --- | --- | --- | --- |
| `train/valid/test.csv` | `convert` | `sft/rl/evaluate` | `item_id` 必须能在 `item.json/index.json` 中找到；`item_sid` 应等于 `index.json[item_id]` 拼接结果 |
| `info.txt` | `convert` | `rl/evaluate/metrics` | 第一列 semantic ID 必须覆盖所有合法输出 |
| `index.json` | `sid-generate` | `convert/sft/rl` | key 必须与 `item.json` 和 CSV 的 `item_id` 对齐 |
| `item.json` | `preprocess` | `embed/convert/sft/rl` | key 必须与 `index.json` 和 CSV 对齐 |

## 潜在 bug、危险默认值与易误用点

下面这些我建议视为当前仓库最值得优先标注的风险项：

1. `config/sid_generate.yaml` 默认不包含 `ckpt_path/output_file`，与真实入口要求不一致。
2. `config/convert.yaml` 当前默认路径与工作区数据布局不匹配，默认 convert 很可能跑不起来。
3. 原仓的 `parity_check.py` 预期值曾与 `config/sft.yaml`、`config/rl.yaml` 漂移，现已删除。
4. `src/onerec/utils/io.py` 在模块 import 阶段就引入 `pandas`，导致即便只是 `--help`、config render 等轻量动作，也需要完整依赖环境。
5. `RL` 的 reward lookup 依赖 `prompt/history -> target` 字典，重复 history 会被覆盖；当前训练数据里这不是边缘情况，而是大量存在。
6. `BaseDataset` 的 cache key 没有显式包含 `sample` 和 `seed`，切 sample 或改 seed 时存在读到旧 cache 的风险。（来源：`src/onerec/utils/cache.py`、`src/onerec/utils/dataset_base.py`）
7. `split()` 会写出带 index 的 shard CSV，导致分片后的 CSV schema 漂移。
8. evaluate 输出 JSON 丢掉了原始 user/item 元信息，不利于 error analysis。
9. `Evaluate.K` 当前未生效，`Evaluate.temperature` 当前也基本未生效，容易给实验配置造成“以为改了，其实没用”的错觉。
10. 当前数据中 semantic ID 本身存在重复项，对严格 one-to-one item identification 不完全友好。
11. `beam_search` 在 RL 中的实现语义与字面名不完全一致，建议后续实验前复核。

## 测试与风险

### tests 当前覆盖了什么

当前 `tests/unit` 覆盖的主要是“工程骨架”而不是“训练语义”：

- CLI help
- config loader
- config template render
- constrained decoding
- convert contract smoke test
- deepspeed cleanup patch
- evaluate split/merge
- repo layout
- precision policy
- semantic ID canonicalization

来源文件：

- `tests/unit/test_cli_help.py`
- `tests/unit/test_config_loader.py`
- `tests/unit/test_config_templates.py`
- `tests/unit/test_constrained_decoding.py`
- `tests/unit/test_convert_contracts.py`
- `tests/unit/test_deepspeed_compat.py`
- `tests/unit/test_evaluate_split_merge.py`
- `tests/unit/test_flow_layout.py`
- `tests/unit/test_precision_policy.py`
- `tests/unit/test_semantic_id.py`

### 哪些关键训练语义还没有被覆盖

当前测试没有覆盖这些最关键的科研语义：

- preprocess 输出内容是否真的满足 convert 所需契约
- embed 生成 embedding 的 shape / 排序 / 稳定性
- sid-train / sid-generate 的集成正确性
- semantic ID 冲突修复逻辑是否可重复
- SFT 四任务混合比例与 prompt 模板是否正确
- token 扩展后 tokenizer/model vocab 是否完全对齐
- RL 的 sampler / num_generations / reward grouping 是否一致
- RL 重复 history 覆盖问题
- Evaluate 的真实约束解码质量与异常路径
- metrics 是否应该落盘而不仅是 stdout

### 当前仓库最容易出问题的工程环节

我认为最容易出问题的点有四类：

1. 配置与代码漂移。
2. 数据契约与路径漂移。
3. RL reward lookup 与重复 history 的语义冲突。
4. semantic ID 深度/唯一性假设与实际数据不完全一致。

另外，当前工作树里已经存在未提交修改：

- `config/sft.yaml`
- `config/evaluate.yaml`

本报告没有改动它们，只做了只读分析。（来源：`git status --short`）

## 后续科研迭代建议

### 最适合插入创新点的模块

如果后续目标是“在这个仓库上加生成式推荐创新点”，我建议优先从以下位置切入：

1. `src/onerec/sft/datasets.py`
   - 适合做数据混合策略、prompt 设计、title/description/history 对齐方式创新。
2. `src/onerec/rl/rewards.py`
   - 适合做 reward 设计、multi-objective reward、ranking/semantic/hybrid reward。
3. `src/onerec/rl/trainer.py`
   - 适合做 group sampling、candidate selection、约束生成策略、advantage 归一化方式创新。
4. `src/onerec/evaluate/pipeline.py`
   - 适合做 rerank、约束诊断、输出更细粒度分析结果。

### 改动风险最大的模块

以下模块一动就容易牵一发动全身：

1. semantic ID 结构本身
2. `src/onerec/evaluate/constrained_decoding.py`
3. `src/onerec/evaluate/semantic_id.py`
4. `src/onerec/rl/trainer.py`
5. shell wrapper + config render 层

原因是这些地方既影响训练，又影响评测，还会影响现有产物兼容性。

### 最值得先做的实验

如果以“低风险、快迭代、科研信号清楚”为原则，我建议第一批实验按这个顺序来：

1. SFT 数据混合消融。
   - 只保留 `SidSFTDataset`
   - `SidSFT + SidItemFeat`
   - 全量四任务
   - 开/关 `FusionSeqRecDataset` 的 description alignment
2. RL reward 消融。
   - `rule`
   - `ranking_only`
   - `rule + ranking`
   - 若补好语义映射，再测 `semantic`
3. Evaluate 诊断实验。
   - 统计 constraint mismatch
   - 比较不同 `num_beams`
   - 补 per-sample 诊断输出
4. history 表达创新。
   - 只用 SID history
   - 只用 title history
   - SID + title 混合 prompt

### 哪些地方需要先补 smoke test / unit test

在高频改科研代码前，我建议先补下面几类测试：

1. config/render smoke test
   - 至少保证 `sft.sh/rl.sh/evaluate.sh` 的 config render 与 override 还能跑通
2. data contract test
   - 检查 `CSV.item_id -> index.json/item.json`
   - 检查 `CSV.item_sid == ''.join(index.json[item_id])`
   - 检查 `info.txt` 与 `index.json` 一致性
3. SFT dataset smoke test
   - 检查 4 个 dataset 的样本数、tokenize 后字段、token 扩展数量
4. RL reward/grouping test
   - 检查 `num_generations` 分组、sampler、reward normalization
   - 特别检查重复 history 覆盖问题
5. Evaluate smoke test
   - 用最小 fake tokenizer / fake logits 验证 split -> worker -> merge -> metrics 整链
   - 检查结果 JSON 是否保留足够的分析字段

## 待确认

下面这些点我不建议假装确定，后续如果你要正式做实验，最好先再核一次：

1. preprocess 默认配置（`Toys_and_Games` / `All_Beauty`）与当前主线默认数据集（`industrial` / `office`）之间是否有一套未提交到仓库的中间转换流程。
2. RL 中 `beam_search=true` 是否就是你期望的“采样式 beam”而不是严格 deterministic beam。
3. 后续是否真的要支持 4-level 以上 semantic ID；如果要，SFT/RL/Evaluate/metrics 都要同步改。
4. 现有 `output/sft_Industrial_and_Scientific_refactor_align` 是否代表一条新的 SFT 数据配方分支；它与当前 `config/sft.yaml` 并不直接对应。

## 命令与轻量验证记录

本次分析实际运行过的命令包括：

- 仓库结构查看：
  - `ls -la`
  - `find . -maxdepth 2 -type d | sort`
  - `rg --files ...`
- 代码与配置阅读：
  - `sed -n ... README.md`
  - `sed -n ... _common.sh`
  - `sed -n ... src/onerec/**/*.py`
  - `sed -n ... config/*.yaml`
  - `sed -n ... tests/unit/*.py`
- 数据抽样：
  - 抽样查看 `train/valid/test.csv`
  - 抽样查看 `info.txt`
  - 抽样查看 `index.json`、`item.json`
  - 抽样查看 `results/*.json`、`temp/*`
- 轻量校验：
  - `PYTHONPATH=./src python -m onerec.main --help`
  - `python -m unittest discover -s tests/unit -v`
  - 若干只读 Python 统计脚本

轻量验证结论：

- 在 `conda activate MiniOneRec` 环境里，`pandas`、`transformers`、`torch` 等关键依赖都齐全。
- `PYTHONPATH=./src python -m onerec.main --help` 可以正常通过。
- 由于 `parity_check.py` 已删除，当前轻量校验以单测和 CLI 检查为主。
- `python -m unittest discover -s tests/unit -v` 在 `MiniOneRec` 环境里已不再依赖 parity check 这类额外脚本。
- 我最开始在默认 shell 环境里遇到的 `pandas` / `transformers` 缺失，更像是“没有进入项目指定环境”的问题，不应当被误判为仓库本身缺依赖。
- 现有 `results/*.json` 与 `temp/*` 已证实 evaluate 的 shard -> merge 产物链条确实跑过。
