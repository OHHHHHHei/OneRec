# 重构版项目说明

## 1. 文档目的

这份文档说明当前重构后的项目结构、运行逻辑、阶段链路，以及推荐的操作方式。

当前主线流程保持为：

```text
preprocess -> embed -> SID -> convert -> SFT -> RL -> evaluate
```

这次重构的重点不是改变项目核心算法，而是在保留主流程和产物契约的前提下，重新组织代码结构，降低后续维护和扩展成本。

## 2. 当前项目结构

### 2.1 根目录职责

根目录现在主要承担以下职责：

- 保留旧入口兼容脚本
- 存放配置文件
- 存放数据脚本和本地产物
- 提供新的包化主线代码

关键目录如下：

```text
OneRec/
  archive/                 非主线代码和历史文档归档
  assets/                  文档图片资源
  config/                  旧配置目录，当前仍保留
  configs/                 新的 YAML 配置目录
  data/                    数据和预处理脚本目录
  minionerec/              顶层兼容导入包
  rq/                      旧 SID 子系统目录，保留兼容入口
  scripts/                 新的命令包装脚本
  src/minionerec/          新主线源码
  tests/                   本地契约测试和 smoke tests
  sft.py / rl.py / ...     旧入口兼容包装
```

### 2.2 新主线代码位置

真正的主线实现现在位于：

- `src/minionerec`

按职责划分如下：

- `cli/`
  - 统一命令入口
  - 各阶段子命令
- `common/`
  - IO、日志、路径解析、seed、tokenizer 辅助
- `config/`
  - 配置 schema
  - YAML 加载
  - CLI override 合并
- `preprocess/`
  - Amazon18 / Amazon23 数据预处理
- `sid/`
  - 文本 embedding
  - 量化训练
  - SID 生成
- `data/`
  - 数据契约
  - 数据缓存
  - SFT / RL / Eval 数据集
  - convert
- `training/sft/`
  - tokenizer 扩 SID
  - SFT pipeline
- `training/rl/`
  - reward
  - RL pipeline
  - trainer
- `training/cf/`
  - SASRec 相关实现
- `evaluation/`
  - 约束解码
  - split / merge / metric
- `compat/`
  - 旧参数到新配置的映射
  - 旧入口兼容逻辑

### 2.3 归档代码

以下内容已经从主链路剥离，但仍然保留作参考：

- `archive/gpr`
- `archive/old_docs`

这些内容不再作为当前主流程的一部分。

## 3. 主线运行逻辑

### 阶段 1：preprocess

目标：

- 将 Amazon 原始评论和元信息整理成训练可用的交互序列和 item 元信息

典型输出：

- `*.train.inter`
- `*.valid.inter`
- `*.test.inter`
- `*.item.json`

对应实现：

- `src/minionerec/preprocess/amazon18.py`
- `src/minionerec/preprocess/amazon23.py`

### 阶段 2：embed

目标：

- 将 item 的标题和描述编码为向量

典型输入：

- `*.item.json`

典型输出：

- `*.emb-*.npy`

对应实现：

- `src/minionerec/sid/text2emb.py`

### 阶段 3：SID

目标：

- 将 item embedding 量化成离散的 semantic ID

典型输入：

- embedding 文件

典型输出：

- `*.index.json`

对应实现：

- `src/minionerec/sid/quantizers/`
- `src/minionerec/sid/generate/`
- `src/minionerec/sid/models/`

### 阶段 4：convert

目标：

- 将交互数据、item 元信息和 SID 索引转换成 SFT、RL、评估所需的 CSV 和 `info.txt`

典型输入：

- `*.train.inter`
- `*.valid.inter`
- `*.test.inter`
- `*.item.json`
- `*.index.json`

典型输出：

- train / valid / test CSV
- `info/*.txt`

对应实现：

- `src/minionerec/data/convert.py`

### 阶段 5：SFT

目标：

- 为 tokenizer 扩展 SID token，并进行监督微调

保留的任务语义：

- history SID -> next SID
- SID <-> title
- history SID -> next item title

对应实现：

- `src/minionerec/training/sft/pipeline.py`
- `src/minionerec/training/sft/token_extension.py`
- `src/minionerec/data/datasets/sft.py`

典型产物：

- SFT checkpoint
- 扩展后的 tokenizer
- `final_checkpoint/`

### 阶段 6：RL

目标：

- 在 SFT 模型基础上继续做推荐导向的强化学习

保留的任务语义：

- SID history -> next SID
- title / description -> SID
- title sequence -> SID

对应实现：

- `src/minionerec/training/rl/pipeline.py`
- `src/minionerec/training/rl/rewards.py`
- `src/minionerec/training/rl/constrained_generation.py`
- `src/minionerec/data/datasets/rl.py`

关键逻辑：

- 生成阶段使用约束解码，只允许合法 SID
- reward 逻辑保持与原流程一致

### 阶段 7：evaluate

目标：

- 在合法 SID 空间上做约束解码评估，并计算指标

对应实现：

- `src/minionerec/evaluation/constrained_decoding.py`
- `src/minionerec/evaluation/pipeline.py`
- `src/minionerec/evaluation/split_merge.py`
- `src/minionerec/evaluation/metrics.py`

典型产物：

- 分片预测结果
- 合并后的结果
- HR / NDCG / 其他统计指标

## 4. 统一 CLI 用法

统一入口：

```bash
python -m minionerec.cli.main <stage> --config <yaml> [overrides...]
```

支持的阶段：

- `preprocess`
- `embed`
- `sid-train`
- `sid-generate`
- `convert`
- `sft`
- `rl`
- `evaluate`

示例：

```bash
python -m minionerec.cli.main preprocess --config configs/stages/preprocess/amazon18.yaml
python -m minionerec.cli.main embed --config configs/stages/embed/default.yaml
python -m minionerec.cli.main sid-train --config configs/stages/sid/rqvae_train.yaml
python -m minionerec.cli.main sid-generate --config configs/stages/sid/rqvae_generate.yaml
python -m minionerec.cli.main convert --config configs/stages/convert/default.yaml
python -m minionerec.cli.main sft --config configs/stages/sft/default.yaml
python -m minionerec.cli.main rl --config configs/stages/rl/default.yaml
python -m minionerec.cli.main evaluate --config configs/stages/evaluate/default.yaml
```

## 5. 配置方式

每个阶段都使用 YAML 配置，并允许命令行覆盖。

常见配置区块包括：

- `model`
- `data`
- `training`
- `output`
- `logging`

对应实现：

- `src/minionerec/config/loader.py`
- `src/minionerec/config/schema.py`

## 6. 兼容层

当前仓库仍保留旧脚本名作为兼容包装，例如：

- `sft.py`
- `rl.py`
- `evaluate.py`
- `convert_dataset.py`

这些入口的作用是保证旧的服务器命令不会立刻失效，但新开发应优先使用统一 CLI。

## 7. 推荐操作方式

### 本地

本地推荐做这些事情：

1. 修改 `src/minionerec` 下的主线代码
2. 修改 `configs/` 下的 YAML 配置
3. 运行本地单测和契约检查
4. 不把本地结果当成正式训练验收

本地常用检查：

```bash
python -m minionerec.cli.main --help
python -m unittest discover -s tests/unit -v
python -m compileall src minionerec
```

### 远程服务器

正式训练建议在远程服务器上按下面顺序执行：

1. preprocess
2. embed
3. sid-train
4. sid-generate
5. convert
6. sft
7. rl
8. evaluate

这样可以保持和原项目一致的逻辑链条。

## 8. 当前必须保持稳定的契约

这些内容在后续继续开发时应尽量保持不变：

- `*.item.json` 字段结构
- `*.index.json` SID 结构
- 转换后 CSV 的列结构
- `info/*.txt` 的格式
- tokenizer 扩 SID token 的逻辑
- SFT 输出供 RL / evaluate 继续加载的 checkpoint 结构

## 9. 当前验证边界

当前本地验证主要覆盖：

- 配置加载
- CLI 帮助信息
- 数据转换契约
- 编译检查

完整的 SFT、RL 和 evaluate，仍应在远程训练服务器上做最终验证。

