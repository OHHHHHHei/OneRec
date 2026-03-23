# OneRec

OneRec 是一个面向**生成式推荐**的开源训练与评估框架，保留了以下主流程：

```text
preprocess -> embed -> sid-train -> sid-generate -> convert -> sft -> rl -> evaluate
```

它的目标不是重新设计一套全新算法，而是把原始 MiniOneRec 项目的主链路收敛为一套更清晰、更稳定、可维护的工程实现：

- 保留 `SID -> SFT -> RL -> Evaluate` 的核心逻辑
- 保留原项目的主要产物契约与训练链条
- 用统一的 `bash + yaml` 入口替代分散脚本
- 用更清晰的目录结构组织代码、配置和产物

## 特性

- 统一的主入口：
  - `bash preprocess_amazon18.sh`
  - `bash preprocess_amazon23.sh`
  - `bash text2emb.sh`
  - `bash sid_train.sh`
  - `bash sid_generate.sh`
  - `bash convert.sh`
  - `bash sft.sh`
  - `bash rl.sh`
  - `bash evaluate.sh`
- 按阶段组织代码：
  - `src/onerec/preprocess`
  - `src/onerec/sid`
  - `src/onerec/convert`
  - `src/onerec/sft`
  - `src/onerec/rl`
  - `src/onerec/evaluate`
- YAML 配置模板化，支持用数据集 key 自动展开路径
- 支持 SFT / RL checkpoint 的阶段化评估
- 保留多卡训练与多卡并行评估能力

## 项目结构

```text
OneRec/
  config/
    datasets.yaml
    preprocess_amazon18.yaml
    preprocess_amazon23.yaml
    embed.yaml
    sid_train.yaml
    sid_generate.yaml
    convert.yaml
    sft.yaml
    rl.yaml
    evaluate.yaml
    zero2_opt.yaml
  src/onerec/
    main.py
    preprocess/
    sid/
    convert/
    sft/
    rl/
    evaluate/
    utils/
  data/
  tests/
  preprocess_amazon18.sh
  preprocess_amazon23.sh
  text2emb.sh
  sid_train.sh
  sid_generate.sh
  convert.sh
  sft.sh
  rl.sh
  evaluate.sh
```

## 安装

### 1. 创建环境

```bash
conda create -n OneRec python=3.11 -y
conda activate OneRec
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

## 配置方式

当前主仓使用 `config/*.yaml` 作为正式配置入口。

其中：

- `config/datasets.yaml` 定义数据集 key 到实际路径变量的映射
- `config/sft.yaml`、`config/rl.yaml`、`config/evaluate.yaml` 使用模板变量
- shell 脚本会在运行前自动渲染出最终配置

例如 `config/sft.yaml` 中会出现：

```yaml
data:
  train_file: ./data/Amazon/train/%{split_stem}.csv
  category: "%{category}"
output:
  output_dir: "./output/sft_%{category}_refactor"
```

这些占位符来自 `config/datasets.yaml`。

当前内置数据集 key：

- `industrial`
- `office`

## 快速开始

### 1. 直接跑默认数据集

默认数据集 key 是 `industrial`：

```bash
bash sft.sh
bash rl.sh
bash evaluate.sh
```

### 2. 显式指定数据集

```bash
bash sft.sh industrial
bash rl.sh industrial
bash evaluate.sh industrial
```

或者：

```bash
bash sft.sh office
bash rl.sh office
bash evaluate.sh office
```

### 3. 评估指定阶段的 checkpoint

默认 `evaluate` 评的是 `sft` checkpoint。你也可以显式指定：

```bash
bash evaluate.sh sft industrial
bash evaluate.sh rl industrial
```

## 常用命令

### 数据预处理

```bash
bash preprocess_amazon18.sh
bash preprocess_amazon23.sh
```

### 文本 embedding

```bash
bash text2emb.sh
```

### SID 训练与生成

```bash
bash sid_train.sh
bash sid_generate.sh
```

### 数据格式转换

```bash
bash convert.sh
```

### SFT

```bash
bash sft.sh industrial
```

### RL

```bash
bash rl.sh industrial
```

### Evaluate

```bash
bash evaluate.sh sft industrial
bash evaluate.sh rl industrial
```

## Override 用法

仍然支持在 shell 命令后追加 `key=value` 的 override：

```bash
bash sft.sh office training.num_epochs=3 training.batch_size=512
bash rl.sh industrial training.num_generations=4
bash evaluate.sh rl office num_beams=48
```

也支持显式指定配置文件：

```bash
bash sft.sh config/sft.yaml industrial
bash rl.sh config/rl.yaml office
bash evaluate.sh config/evaluate.yaml rl industrial
```

## 结果与产物命名

### SFT / RL 输出目录

默认命名规则：

- SFT：`./output/sft_<category>_refactor`
- RL：`./output/rl_<category>_refactor`

最终 checkpoint 目录：

- `final_checkpoint/`

训练中间 checkpoint：

- `checkpoint-*`

### Evaluate 输出目录

最终结果 JSON：

- SFT 评估：`./results/final_result_sft_<category>.json`
- RL 评估：`./results/final_result_rl_<category>.json`

临时分片目录：

- `./temp/sft-<category>`
- `./temp/rl-<category>`

分片结果文件：

- `4.json`
- `5.json`
- `6.json`
- `7.json`

## 当前代码主链

如果你只关心主链路代码，优先看这些目录：

- `src/onerec/sft`
- `src/onerec/rl`
- `src/onerec/evaluate`
- `src/onerec/utils`

其中：

- `src/onerec/sft`：SFT 数据集、token 扩展、训练流程
- `src/onerec/rl`：RL 数据集、reward、trainer、约束生成
- `src/onerec/evaluate`：约束解码、分片评估、merge、metrics
- `src/onerec/main.py`：统一 stage 入口

## 本地验证

本地不要求跑完整训练，当前推荐的检查方式是：

```bash
python -m unittest discover -s tests/unit -v
PYTHONPATH=./src python -m onerec.main --help
```

## 远程服务器验收建议

建议按下面顺序做：

1. SFT 小样本验证
2. RL 小样本验证
3. Evaluate 小样本验证

重点检查：

- 路径是否按数据集 key 正确展开
- 多卡参数是否正确生效
- `final_checkpoint/` 是否生成
- evaluate 的 split / worker / merge / metrics 是否完整跑通

## 已知说明

- 当前 `evaluate` 使用约束解码。如果 `num_beams` 大于约束树首层合法分支数，会出现 `Constraint mismatch summary` warning。
- 当前主仓以工程稳定和主链路清晰为优先，不主动改动原始算法语义。
- 这套实现以 `OneRec` 作为正式主仓名，但训练逻辑仍然继承 MiniOneRec 的核心流程。

## 许可证

本项目使用 [Apache-2.0](./LICENSE)。

## 致谢

本项目的设计和实现参考了原始 MiniOneRec 项目以及相关开源工作，尤其是：

- ReRe
- LC-Rec

如果你发现问题，建议在 issue 中附上：

- 运行命令
- 使用的数据集 key
- 关键 YAML 字段
- 报错日志或 warning 摘要
