# MiniOneRec 主链路迁移对照报告

## 1. 迁移结论摘要

### 总判断

- `SFT / RL / Evaluate` 三段的核心训练与推理逻辑整体上是从旧仓 `E:\MiniOneRec` 直接迁移到主仓 `E:\MiniOneRec-mainline`，不是重写算法。
- `RL trainer` 基本属于原样迁移。`E:\MiniOneRec\minionerec_trainer.py` 与 `src/onerec/rl/trainer.py` 做无索引 diff 时只看到导入路径和少量非语义文本差异，关键方法位置也一一对应：`ReReTrainer`、`prefix_allowed_tokens_fn`、`_prepare_inputs`、`compute_loss` 在旧仓分别位于 `121/588/665/1035` 行，新仓也位于 `121/588/665/1035` 行。  
  标签：`等价迁移`
- 真正显著的变化主要不在训练器主体，而在“工程入口”和“默认配置”：
  - 旧仓是脚本直调单文件。
  - 新仓是 `bash + YAML + stage runner + dataclass config + template render`。
  标签：`运行入口改变`
- 可能影响复现实验结果的点，不是“主算法变了”，而是默认参数和默认运行方式变了，尤其在 RL 和 Evaluate 上更明显。  
  标签：`默认值改变`、`可能影响训练/评估结果`

### 一句话结论

- `SFT`：训练数据拼接、semantic token 扩词、训练目标基本保持一致；最大的行为差异是精度从固定 `bf16` 改成运行时动态选择。  
  标签：`工程重构，行为不变` + `轻微行为变化`
- `RL`：核心 trainer 和 reward 组织基本保留；最大的差异是主仓默认 YAML 参数与旧仓 shell 实跑值差得较大，`reward_type=sasrec` 也不再直接可用。  
  标签：`等价迁移` + `默认值改变` + `明确行为变化`
- `Evaluate`：受约束解码仍然是同一套思路，但主仓新增了约束诊断、worker 上下文和模板化路径管理；`guidance_scale` 的默认配置从代码默认 `1.0` 变成 YAML 默认 `null`。  
  标签：`工程重构，行为不变` + `默认值改变`

### 术语说明：BF16 是什么

- `BF16` 是 `bfloat16`，一种 16 位浮点精度。
- 旧仓在 SFT / RL / Evaluate 中普遍直接写死 `torch_dtype=torch.bfloat16` 或 `bf16=True`，例如 `E:\MiniOneRec\sft.py:161,270`、`E:\MiniOneRec\rl.py:138,283`、`E:\MiniOneRec\evaluate.py:75`。
- 主仓把这件事改成“按当前 CUDA 能力动态选择 `bf16 / fp16 / fp32`”，例如 `src/onerec/sft/pipeline.py:20-27`、`src/onerec/rl/pipeline.py:23-27`、`src/onerec/evaluate/pipeline.py:56-57`。

## 2. 迁移链路概览

### 旧仓主链路

旧仓主链路是脚本直接调单文件：

| 阶段 | 旧仓入口 | 说明 |
| --- | --- | --- |
| preprocess | `data/amazon18_data_process.py`、`data/amazon23_data_process.py` | 数据清洗与 `.inter` 生成 |
| embed / SID | `rq/text2emb/*`、`rq/rqvae.py`、`rq/generate_indices.py` | embedding、量化训练、semantic ID 生成 |
| convert | `convert_dataset.py` | 生成当前训练使用的 CSV 和 `info.txt` |
| SFT | `sft.sh -> sft.py` | 单脚本训练 |
| RL | `rl.sh -> rl.py -> minionerec_trainer.py` | GRPO + 约束生成 |
| Evaluate | `evaluate.sh -> evaluate.py + split.py + merge.py + calc.py` | 多卡分片评估 |

### 新仓主链路

新仓把主链路拆成 stage：

- 统一入口：`src/onerec/main.py:22-175`
- 配置模型：`src/onerec/config.py:13-250`
- 模板渲染：`src/onerec/utils/config_templates.py:20-100`
- shell 统一解析：`_common.sh:6-95`

新的执行方式是：

```text
shell -> 渲染 YAML 模板 -> python -m onerec.main <stage> -> stage pipeline
```

### 入口映射结论

- 旧仓硬编码目录和脚本参数，主仓改成 `datasets.yaml` 的数据集键控模板：
  - `config/datasets.yaml:1-9`
  - `src/onerec/utils/config_templates.py:20-43`
  - `_common.sh:70-95`
- 这不会改变算法本身，但会改变默认路径、输出命名和默认数据集选择。  
  标签：`运行入口改变`

## 3. SFT 对照

### 3.1 核心逻辑是否保留

结论：

- 保留了相同的三路训练数据拼接：
  - `SidSFTDataset`
  - `SidItemFeatDataset`
  - `FusionSeqRecDataset`
- `TitleHistory2SidSFTDataset` 在旧仓和新仓都处于注释/未启用状态：
  - 旧仓 `E:\MiniOneRec\sft.py:229-231`
  - 新仓 `src/onerec/sft/pipeline.py:92`

证据：

- 旧仓 `E:\MiniOneRec\sft.py:223-231`
- 新仓 `src/onerec/sft/pipeline.py:87-93`
- 旧仓 dataset 定义在 `E:\MiniOneRec\data.py:465,746,1195,1401`
- 新仓 dataset 定义在 `src/onerec/sft/datasets.py:10,54,110,216`

结论标签：

- `等价迁移`

### 3.2 模型加载与精度策略

旧仓：

- `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)`  
  证据：`E:\MiniOneRec\sft.py:159-161`
- `TrainingArguments(..., bf16=True)`  
  证据：`E:\MiniOneRec\sft.py:267-270`

新仓：

- 先调用 `_resolve_precision()` 按环境选择 `bf16/fp16/fp32`  
  证据：`src/onerec/sft/pipeline.py:20-27`
- 再把 `torch_dtype=model_dtype` 传入模型  
  证据：`src/onerec/sft/pipeline.py:50-55`
- `TrainingArguments` 中写 `bf16=use_bf16, fp16=use_fp16`  
  证据：`src/onerec/sft/pipeline.py:104-115`

判断：

- 训练语义未改，但数值精度不再固定。若运行设备不支持 BF16，新仓会退到 FP16 或 FP32，而旧仓会继续尝试 BF16。
- 这属于环境适配增强，但也可能造成跨设备结果不完全一致。  
  标签：`轻微行为变化`、`可能影响训练/评估结果`

### 3.3 tokenizer 扩词与 semantic token 注入

旧仓：

- 内嵌 `TokenExtender`，从 `.index.json` 读取 semantic tokens 并 `tokenizer.add_tokens(...)`。  
  证据：`E:\MiniOneRec\sft.py:30-61,171-184`

新仓：

- 把逻辑拆成独立模块 `src/onerec/sft/token_extension.py`
- pipeline 读取 `sid_index_path` 后扩词并 resize embeddings。  
  证据：`src/onerec/sft/token_extension.py:5-21`、`src/onerec/sft/pipeline.py:61-69`

判断：

- 逻辑等价，只是职责拆分。  
  标签：`工程重构，行为不变`

### 3.4 `freeze_LLM` / `freeze_llm` 行为

旧仓：

- 参数名为 `freeze_LLM`，默认 `False`。  
  证据：`E:\MiniOneRec\sft.py:128`
- 逻辑是冻结全部参数，只给 embedding 保留梯度，并用 hook 将旧 vocab 梯度置零，仅训练新 token 行。  
  证据：`E:\MiniOneRec\sft.py:186-210`

新仓：

- 参数名改为 `freeze_llm`，默认 `false`。  
  证据：`config/sft.yaml:17`
- 逻辑仍是冻结全部参数，仅开放 embedding，并用 hook 屏蔽旧 vocab 梯度。  
  证据：`src/onerec/sft/pipeline.py:71-82`

判断：

- 命名风格从 CamelCase 变为 snake_case。
- 行为实现与旧仓一致，没有发现语义变化。  
  标签：`工程重构，行为不变`

### 3.5 数据解析实现差异

旧仓：

- dataset 大量用 `eval(...)` 解析 CSV 中的列表字符串。  
  证据：`E:\MiniOneRec\data.py:472,680,983,1276,1306,1443`

新仓：

- 统一改成 `parse_sequence()`，内部使用 `ast.literal_eval`。  
  证据：`src/onerec/utils/parsing.py:5-15`、`src/onerec/sft/datasets.py:16,151,167,235,242`

判断：

- 在常规数据格式下行为等价，但新仓更安全，异常面更小。
- 如果旧数据里混入了不规范的 Python 表达式字符串，新仓可能比旧仓更严格。  
  标签：`工程重构，行为不变`，必要时可标记为 `证据不足，需人工复核`

### 3.6 SFT 参数对照

| 参数 | 旧仓代码默认 | 旧仓 shell 实值 | 新仓 YAML 默认 | 新仓运行时推导 | 是否行为变化 | 影响说明 |
| --- | --- | --- | --- | --- | --- | --- |
| `batch_size` | `128`，`E:\MiniOneRec\sft.py:121` | `1024`，`E:\MiniOneRec\sft.sh:13` | `1024`，`config/sft.yaml:11` | 不变 | 否 | 新仓默认跟随旧仓 shell，而不是旧仓函数默认 |
| `micro_batch_size` | `4`，`E:\MiniOneRec\sft.py:122` | `4`，`E:\MiniOneRec\sft.sh:14` | `4`，`config/sft.yaml:12` | 不变 | 否 | 保持一致 |
| `gradient_accumulation_steps` | `batch_size // micro_batch_size`，`E:\MiniOneRec\sft.py:149-156` | `1024/4=256`，DDP 4 卡时为 `64` | 同旧逻辑 | `_resolve_grad_accum_steps()` 计算，`src/onerec/sft/pipeline.py:28-42` | 否 | 语义一致 |
| `num_epochs` | `10`，`E:\MiniOneRec\sft.py:123` | shell 未显式传，走默认 `10` | `10`，`config/sft.yaml:14` | 不变 | 否 | 保持一致 |
| `learning_rate` | `3e-4`，`E:\MiniOneRec\sft.py:124` | shell 未显式传，走默认 `3e-4` | `3e-4`，`config/sft.yaml:15` | 不变 | 否 | 保持一致 |
| `cutoff_len` | `512`，`E:\MiniOneRec\sft.py:125` | shell 未显式传 | `512`，`config/sft.yaml:13` | 不变 | 否 | 保持一致 |
| `warmup_steps` | 代码固定 `20`，`E:\MiniOneRec\sft.py:267` | `20` | `20`，`config/sft.yaml:19` | 不变 | 否 | 保持一致 |
| `eval_step` | 代码固定 `0.05`，`E:\MiniOneRec\sft.py:249` | `0.05` | `0.05`，`config/sft.yaml:22` | 不变 | 否 | 保持一致 |
| `group_by_length` | `False`，`E:\MiniOneRec\sft.py:127` | shell 未传 | `false`，`config/sft.yaml:18` | 不变 | 否 | 保持一致 |
| `load_best_model_at_end` | 固定 `True`，`E:\MiniOneRec\sft.py:279` | `True` | `true`，`config/sft.yaml:20` | 不变 | 否 | 保持一致 |
| `early_stopping_patience` | `3`，`E:\MiniOneRec\sft.py:139` | shell 未传 | `3`，`config/sft.yaml:21` | 不变 | 否 | 保持一致 |
| `freeze_LLM / freeze_llm` | `False`，`E:\MiniOneRec\sft.py:128` | `False`，`E:\MiniOneRec\sft.sh:25` | `false`，`config/sft.yaml:17` | 不变 | 否 | 只改命名，不改行为 |
| 精度 | 固定 BF16，`E:\MiniOneRec\sft.py:161,270` | BF16 | 动态精度 | 按设备选 `bf16/fp16/fp32` | 是 | 可能影响跨设备复现 |

### 3.7 SFT 入口与输出命名

旧仓：

- `sft.sh` 直接调用 `sft.py`
- 输出目录示例：`output/sft_${category}_1.7B_noalign_2`  
  证据：`E:\MiniOneRec\sft.sh:10-25`

新仓：

- `sft.sh` 先渲染配置，再调用 `python -m onerec.main sft`
- 输出目录默认：`./output/sft_%{category}_refactor`
- 多卡配置写进 YAML：`launcher=torchrun`、`cuda_visible_devices`、`nproc_per_node`  
  证据：`sft.sh:8-96`、`config/sft.yaml:28-33`

判断：

- 算法没变，但输出命名、路径来源、数据集选择方式都变了。  
  标签：`运行入口改变`

## 4. RL 对照

### 4.1 核心迁移关系

结论：

- `rl.py` 的大部分“拼装逻辑”迁到了 `src/onerec/rl/pipeline.py`
- `minionerec_trainer.py` 几乎原样迁到了 `src/onerec/rl/trainer.py`
- reward 函数从 `rl.py` 内联实现，拆到了 `src/onerec/rl/rewards.py`

证据：

- 旧仓 `E:\MiniOneRec\rl.py:30-312`
- 新仓 `src/onerec/rl/pipeline.py:70-188`
- trainer 定义位置在旧/新分别为 `121` 行  
  证据：`E:\MiniOneRec\minionerec_trainer.py:121`、`src/onerec/rl/trainer.py:121`

判断：

- 这是典型的“文件拆分+工程化整理”，不是算法重写。  
  标签：`工程重构，行为不变`

### 4.2 dataset 组成是否保持一致

旧仓：

- `SidDataset`
- `RLTitle2SidDataset`
- `RLSeqTitle2SidDataset(sample=10000)`  
  证据：`E:\MiniOneRec\rl.py:79-87`

新仓：

- 同样三路拼接，第三路仍是 `sample=10000`  
  证据：`src/onerec/rl/pipeline.py:86-90`

判断：

- 数据集组成保持一致。  
  标签：`等价迁移`

### 4.3 reward 逻辑

旧仓：

- 在 `rl.py` 中内联实现 `rule / ranking / ranking_only / semantic / sasrec`  
  证据：`E:\MiniOneRec\rl.py:160-260`

新仓：

- 规则奖励拆成 `build_rule_reward / build_ranking_reward / build_semantic_reward`
- `sasrec` 仍被识别，但主仓直接抛 `NotImplementedError`  
  证据：`src/onerec/rl/rewards.py:12-56`、`src/onerec/rl/pipeline.py:107-125`

判断：

- `rule / ranking / ranking_only / semantic` 仍保留。
- `sasrec` 从“旧仓可走 legacy CF 路径”变成“主仓保留接口，但不接线”。  
  标签：`明确行为变化`

### 4.4 `reward_type=sasrec` 差异

旧仓：

- 当 `reward_type == "sasrec"` 时，会构造 `SASRec` 并加载 `cf_path`。  
  证据：`E:\MiniOneRec\rl.py:146-149,259`

新仓：

- 当 `reward_type == "sasrec"` 时，会先校验 `cf_path`，然后直接抛 `NotImplementedError`。  
  证据：`src/onerec/rl/pipeline.py:121-125`

判断：

- 这是主仓当前最明确的语义退化之一。
- 如果旧实验依赖 `sasrec` reward，主仓无法直接复现。  
  标签：`明确行为变化`、`可能影响训练/评估结果`

### 4.5 RL 参数对照

| 参数 | 旧仓代码默认 | 旧仓 shell 实值 | 新仓 YAML 默认 | 新仓运行时推导 | 是否行为变化 | 影响说明 |
| --- | --- | --- | --- | --- | --- | --- |
| `train_batch_size` | `32`，`E:\MiniOneRec\rl.py:45` | `64`，`E:\MiniOneRec\rl.sh:18` | `16`，`config/rl.yaml:12` | 直接使用 | 是 | 主仓默认显著更小 |
| `eval_batch_size` | `32`，`E:\MiniOneRec\rl.py:46` | `128`，`E:\MiniOneRec\rl.sh:19` | `32`，`config/rl.yaml:13` | 直接使用 | 是 | 主仓默认回到代码默认，未跟随旧 shell |
| `gradient_accumulation_steps` | `1`，`E:\MiniOneRec\rl.py:47` | `2`，`E:\MiniOneRec\rl.sh:21` | `4`，`config/rl.yaml:14` | 直接使用 | 是 | 全局 batch 规模变化明显 |
| `num_train_epochs` | `1`，`E:\MiniOneRec\rl.py:52` | `2`，`E:\MiniOneRec\rl.sh:20` | `2`，`config/rl.yaml:15` | 直接使用 | 否 | 主仓默认跟随旧 shell |
| `num_generations` | `16`，`E:\MiniOneRec\rl.py:51` | `8`，`E:\MiniOneRec\rl.sh:29` | `4`，`config/rl.yaml:16` | 直接使用 | 是 | 采样组大小变小，影响 reward 归一化 |
| `learning_rate` | `1e-6`，`E:\MiniOneRec\rl.py:53` | `1e-5`，`E:\MiniOneRec\rl.sh:36` | `1e-5`，`config/rl.yaml:17` | 直接使用 | 否 | 主仓默认跟随旧 shell |
| `eval_step` | `0.199`，`E:\MiniOneRec\rl.py:50` | `0.0999`，`E:\MiniOneRec\rl.sh:27` | `0.05`，`config/rl.yaml:18` | 直接使用 | 是 | 更频繁评估与保存 |
| `reward_type` | `rule`，`E:\MiniOneRec\rl.py:61` | `ranking`，`E:\MiniOneRec\rl.sh:28` | `ranking`，`config/rl.yaml:19` | 直接使用 | 否 | 主仓默认跟随旧 shell |
| `temperature` | `1.0`，`E:\MiniOneRec\rl.py:48` | `1.0`，`E:\MiniOneRec\rl.sh:35` | `1.0`，`config/rl.yaml:20` | 直接使用 | 否 | 保持一致 |
| `beta` | `0.04`，`E:\MiniOneRec\rl.py:54` | `1e-3`，`E:\MiniOneRec\rl.sh:38` | `0.001`，`config/rl.yaml:21` | 直接使用 | 否 | 主仓默认跟随旧 shell |
| `beam_search` | `False`，`E:\MiniOneRec\rl.py:55` | `True`，`E:\MiniOneRec\rl.sh:33` | `true`，`config/rl.yaml:23` | 直接使用 | 否 | 主仓默认跟随旧 shell |
| `test_during_training` | `True`，`E:\MiniOneRec\rl.py:56` | `False`，`E:\MiniOneRec\rl.sh:34` | `false`，`config/rl.yaml:24` | 直接使用 | 否 | 主仓默认跟随旧 shell |
| `dynamic_sampling` | `False`，`E:\MiniOneRec\rl.py:57` | `False`，`E:\MiniOneRec\rl.sh:31` | `false`，`config/rl.yaml:25` | 直接使用 | 否 | 保持一致 |
| `sync_ref_model` | `False`，`E:\MiniOneRec\rl.py:59` | `True`，`E:\MiniOneRec\rl.sh:32` | `true`，`config/rl.yaml:26` | 直接使用 | 否 | 主仓默认跟随旧 shell |
| `test_beam` | `20`，`E:\MiniOneRec\rl.py:60` | shell 未传，默认 `20` | `20`，`config/rl.yaml:27` | 直接使用 | 否 | 保持一致 |
| `sample_train` | `False`，`E:\MiniOneRec\rl.py:62` | `False`，`E:\MiniOneRec\rl.sh:26` | `false`，`config/rl.yaml:28` | 直接使用 | 否 | 保持一致 |
| `dapo` | `False`，`E:\MiniOneRec\rl.py:67` | `False`，`E:\MiniOneRec\rl.sh:39` | `false`，`config/rl.yaml:29` | 直接使用 | 否 | 保持一致 |
| `gspo` | `False`，`E:\MiniOneRec\rl.py:68` | shell 未传，默认 `False` | `false`，`config/rl.yaml:30` | 直接使用 | 否 | 保持一致 |
| 精度 | 固定 BF16，`E:\MiniOneRec\rl.py:138,283` | BF16 | 动态精度，`model_init_kwargs={"torch_dtype": model_dtype}`，`src/onerec/rl/pipeline.py:127-146` | 按设备选择 | 是 | 与 SFT 一样会影响跨设备复现 |

### 4.6 RL 约束生成逻辑

旧仓：

- `rl.py` 通过 `info.txt` 生成前缀树，再交给 `minionerec_trainer.py` 的 `prefix_allowed_tokens_fn`
- trainer 内约束生成与 beam/sample 策略都写在 `ReReTrainer` 中  
  证据：`E:\MiniOneRec\rl.py:72-78`、`E:\MiniOneRec\minionerec_trainer.py:588,665`

新仓：

- `pipeline.py` 仍按 `info.txt` 读取 semantic IDs
- `trainer.py` 仍保留同名方法和整体流程  
  证据：`src/onerec/rl/pipeline.py:81-84`、`src/onerec/rl/trainer.py:588,665`

判断：

- 约束生成核心保留。  
  标签：`等价迁移`

### 4.7 RL 入口与工程差异

旧仓：

- `rl.sh` 直接 `accelerate launch rl.py`
- category、checkpoint、resume path 都硬编码在 shell 里。  
  证据：`E:\MiniOneRec\rl.sh:13-45`

新仓：

- `rl.sh` 从 YAML 提取 runtime，并可从模板和 override 推导参数
- launcher、GPU 列表、port、HF endpoint 都从配置读取  
  证据：`rl.sh:8-115`、`config/rl.yaml:41-47`

判断：

- 入口更强，但默认行为不再等于旧仓 shell。尤其当前主仓默认 `industrial/office` 数据集模板，而旧仓 `rl.sh` 示例硬编码 `Toys_and_Games`。  
  标签：`运行入口改变`

### 4.8 DeepSpeed / 清理兼容性说明

当前主仓新增了与旧仓无关的兼容性修复：

- `src/onerec/rl/deepspeed_compat.py:10-49`
- `src/onerec/rl/pipeline.py:73`

作用：

- 这是为当前主仓补的 BF16/DeepSpeed 退出阶段清理保护，不属于原训练语义的一部分。
- 在迁移对照中应归为“工程实现变化”，不应误判成 RL 训练逻辑变化。  
  标签：`工程重构，行为不变`

## 5. Evaluate 对照

### 5.1 核心迁移关系

旧仓：

- `evaluate.sh` 负责 split、多 GPU worker、merge、calc  
  证据：`E:\MiniOneRec\evaluate.sh:24-87`
- `evaluate.py` 做单 worker 推理
- `LogitProcessor.py` 做约束 logits processor

新仓：

- `evaluate.sh` 仍负责 split、多 worker、merge、metrics  
  证据：`evaluate.sh:120-173`
- `src/onerec/evaluate/pipeline.py` 做单 worker 推理
- `src/onerec/evaluate/constrained_decoding.py` 做约束 logits processor

判断：

- 评估分层关系保持一致。  
  标签：`等价迁移`

### 5.2 约束解码器是否改变语义

旧仓 `LogitProcessor.py`：

- 语义是：对每个 beam，根据当前 prefix 查合法 token；若无合法 token，则强制 EOS。  
  证据：`E:\MiniOneRec\LogitProcessor.py:24-66`

新仓 `constrained_decoding.py`：

- 核心约束语义未变，仍是“prefix -> allowed tokens，不合法时强制 EOS”
- 额外新增：
  - `warn_limit_per_step`
  - `enable_warning`
  - `invalid_total / invalid_by_step / top_invalid_hashes`
  - `get_diagnostics()`  
  证据：`src/onerec/evaluate/constrained_decoding.py:25-96`
- 无索引 diff 也显示新增部分全部是诊断计数和告警限流，没有改主判定分支。  
  证据：`git diff --no-index E:\MiniOneRec\LogitProcessor.py src/onerec/evaluate/constrained_decoding.py`

判断：

- 约束解码语义保持不变，新增的是可观测性。  
  标签：`工程重构，行为不变`

### 5.3 Evaluate 参数对照

| 参数 | 旧仓代码默认 | 旧仓 shell 实值 | 新仓 YAML 默认 | 新仓运行时推导 | 是否行为变化 | 影响说明 |
| --- | --- | --- | --- | --- | --- | --- |
| `batch_size` | `4`，`E:\MiniOneRec\evaluate.py:45` | `8`，`E:\MiniOneRec\evaluate.sh:48` | `8`，`config/evaluate.yaml:11` | 直接使用 | 否 | 主仓默认跟随旧 shell |
| `num_beams` | `50`，`E:\MiniOneRec\evaluate.py:50` | `50`，`E:\MiniOneRec\evaluate.sh:49` | `50`，`config/evaluate.yaml:13` | 直接使用 | 否 | 保持一致 |
| `max_new_tokens` | `256`，`E:\MiniOneRec\evaluate.py:49` | `256`，`E:\MiniOneRec\evaluate.sh:50` | `256`，`config/evaluate.yaml:14` | 直接使用 | 否 | 保持一致 |
| `length_penalty` | `0.0`，`E:\MiniOneRec\evaluate.py:48` | `0.0`，`E:\MiniOneRec\evaluate.sh:53` | `0.0`，`config/evaluate.yaml:15` | 直接使用 | 否 | 保持一致 |
| `temperature` | `1.0`，`E:\MiniOneRec\evaluate.py:51` | `1.0`，`E:\MiniOneRec\evaluate.sh:51` | `1.0`，`config/evaluate.yaml:16` | 直接使用 | 否 | 当前 beam 解码下主要是兼容字段 |
| `guidance_scale` | 代码默认 `1.0`，`E:\MiniOneRec\evaluate.py:52` | shell 传 `None`，`E:\MiniOneRec\evaluate.sh:52` | `null`，`config/evaluate.yaml:17` | 主仓只有非空时才传给 `generate()`，`src/onerec/evaluate/pipeline.py:67-69,165-166` | 是 | 代码默认与配置默认存在差异；旧仓 shell 的 `None` 是否被 Fire 解析为真正 `None` 需要人工确认 |

关于 `guidance_scale` 的结论：

- 代码默认层面：旧仓 `1.0`，新仓 YAML 默认 `null`，确实变化。
- shell 实跑层面：旧仓 `evaluate.sh` 已显式传 `None`，实际运行效果可能与新仓更接近。
- 由于旧仓使用 Fire，`None` 是否被解析成 Python `None` 仍建议人工复核。  
  标签：`默认值改变`、`证据不足，需人工复核`

### 5.4 多卡并行入口

旧仓：

- `evaluate.sh` 硬编码 `cuda_list="0,1,2,3"`，手工管理 `temp_dir`、worker 启动、merge 输出。  
  证据：`E:\MiniOneRec\evaluate.sh:24-77`

新仓：

- `evaluate.sh` 从 YAML 读取 `runtime.parallel`、`cuda_visible_devices`、`nproc_per_node`
- 先 split，再逐 GPU 启动 worker，并把 `worker_id / primary_worker` 写进环境变量
- 结束后自动 merge 和 metrics  
  证据：`evaluate.sh:51-173`

判断：

- 并行评估模式被保留，但主仓的 GPU 选择和输出路径不再硬编码在 shell 中。  
  标签：`运行入口改变`

### 5.5 结果路径与命名

旧仓：

- model 路径示例：`./output/sft_Industrial_and_Scientific_1.7B_noalign/final_checkpoint`
- 结果目录：`./results/${exp_name_clean}`
- 结果文件：`final_result_${category}.json`
- 临时目录：`./temp/${category}-${exp_name_clean}`  
  证据：`E:\MiniOneRec\evaluate.sh:6,24,67,76`

新仓：

- model 路径按模板展开：`./output/%{eval_model_stage}_%{category}_refactor/final_checkpoint`
- 结果文件直接命名为 `./results/final_result_{stage}_{category}.json`
- 临时目录：`./temp/{stage}-{category}`  
  证据：`config/evaluate.yaml:2,10`、`config_templates.py:42-43`、`evaluate.sh:121`

判断：

- 命名方案变化较大，会影响脚本复现路径和下游消费脚本。  
  标签：`运行入口改变`、`可能影响训练/评估结果`

## 6. 参数差异总表

下表只列会影响复现或最值得关注的项。

| 阶段 | 参数 | 旧仓代码默认 | 旧仓 shell 实值 | 新仓默认 | 结论 |
| --- | --- | --- | --- | --- | --- |
| SFT | 精度 | 固定 BF16 | 固定 BF16 | 动态 `bf16/fp16/fp32` | 主仓更稳，但跨设备复现不再严格等价 |
| SFT | `batch_size` | 128 | 1024 | 1024 | 主仓追随旧 shell，不追随旧代码默认 |
| SFT | `freeze_LLM / freeze_llm` | False | False | false | 仅命名改动，行为一致 |
| RL | `train_batch_size` | 32 | 64 | 16 | 主仓默认更小，训练动态会变 |
| RL | `eval_batch_size` | 32 | 128 | 32 | 主仓未追随旧 shell |
| RL | `gradient_accumulation_steps` | 1 | 2 | 4 | 主仓默认更大 |
| RL | `num_generations` | 16 | 8 | 4 | 主仓默认更小，直接影响 reward 归一化和搜索空间 |
| RL | `eval_step` | 0.199 | 0.0999 | 0.05 | 主仓评估更频繁 |
| RL | `reward_type` | rule | ranking | ranking | 主仓追随旧 shell |
| RL | `sasrec` reward | 可用 | 可用 | 直接 `NotImplementedError` | 明确语义退化 |
| Evaluate | `batch_size` | 4 | 8 | 8 | 主仓追随旧 shell |
| Evaluate | `guidance_scale` | 1.0 | shell 传 `None` | `null` | 代码默认与 shell 实跑存在分歧，需复核旧仓 Fire 解析 |
| Evaluate | 输出命名 | `results/<exp>/final_result_<category>.json` | 同左 | `results/final_result_<stage>_<category>.json` | 路径体系已变 |
| 全局 | 数据集路径 | shell 硬编码 | shell 硬编码 | `datasets.yaml` 模板渲染 | 更灵活，但默认键控路径成为新行为 |

## 7. 当前 tests / parity 覆盖了什么，没覆盖什么

### 已覆盖

- 配置模板渲染：
  - `tests/unit/test_config_templates.py:13-68`
- 配置文件存在和 `datasets.yaml` 结构：
  - `tests/unit/test_flow_layout.py:20-34`
- `parity_check.py` 对默认渲染值的硬校验：
  - SFT：`parity_check.py:43-56`
  - RL：`parity_check.py:57-70`
  - Evaluate：`parity_check.py:71-82`
- 评估 split/merge 接口：
  - `tests/unit/test_evaluate_split_merge.py:18-47`

### 未覆盖

- SFT 三路 dataset 拼接是否与旧仓完全一致
- RL reward 细节、`ReReTrainer` 训练步骤与旧仓行为等价性
- Evaluate 约束解码生成结果与旧仓的 beam 语义等价性
- `guidance_scale=None` 在旧仓 Fire 解析下的真实运行行为
- `sasrec` reward 在主仓不可用这一退化没有测试守护

结论：

- 当前 tests/parity 更偏“工程骨架”和“默认配置值”；
- 对真正的训练语义和评估语义覆盖不足。  
  标签：`工程重构，行为不变`，但存在 `证据不足，需人工复核` 区域

## 8. 风险与结论

### 高优先级风险

1. `reward_type=sasrec` 在主仓不可直接运行。  
   标签：`明确行为变化`  
   影响：旧实验若依赖 CF reward，主仓无法直接复现。

2. RL 默认参数与旧仓 shell 实跑值差异显著。  
   标签：`默认值改变`、`可能影响训练/评估结果`  
   重点项：`train_batch_size`、`eval_batch_size`、`gradient_accumulation_steps`、`num_generations`、`eval_step`。

3. SFT / RL / Evaluate 的精度从固定 BF16 改成动态精度。  
   标签：`轻微行为变化`、`可能影响训练/评估结果`  
   影响：不同 GPU 能力下可能出现数值行为差异。

### 中优先级风险

1. Evaluate 的 `guidance_scale` 默认策略变化。  
   标签：`默认值改变`、`证据不足，需人工复核`  
   影响：如果旧仓实际是 `None`，则主仓更接近旧 shell；如果旧仓实际仍按 `1.0` 运行，则主仓默认已变化。

2. 输出命名和目录结构变化。  
   标签：`运行入口改变`  
   影响：旧自动化脚本、结果收集脚本、checkpoint 引用路径会失效。

### 低优先级变化

1. dataset 解析从 `eval` 改为 `ast.literal_eval`
2. 约束解码器增加诊断与告警限流
3. RL 增加 DeepSpeed BF16 退出清理兼容层

这些都更偏工程性，不构成训练语义重写。  
标签：`工程重构，行为不变`

## 9. 最终结论

- 如果问题是“主仓有没有把旧仓 SFT / RL / Evaluate 的核心训练逻辑迁过来”，答案是：**有，尤其 RL trainer 和受约束解码基本是直接迁移。**
- 如果问题是“主仓默认运行出来的结果会不会天然等于旧仓以前的 shell 实验”，答案是：**不一定，主要因为默认参数、路径模板、输出命名和精度策略已经不再完全相同。**
- 若后续要做更严格的 parity 修复，建议优先按下面顺序排查：
  1. RL 默认 batch / generation / eval_step
  2. `reward_type=sasrec` 是否需要恢复
  3. Evaluate 的 `guidance_scale` 真实旧行为
  4. SFT / RL / Evaluate 在 BF16 固定与动态精度间的实验差异
