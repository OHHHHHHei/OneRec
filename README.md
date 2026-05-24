# OneRec

OneRec 是一个面向生成式推荐的代码仓库。它沿用 MiniOneRec 风格系统中的
`SID -> SFT -> RL -> Evaluation` 主流程，并对核心训练与评估链路做了更清晰的工程整理。

本仓库定位为论文代码发布版本：主流程显式、配置集中，并且便于迁移到不同的
Amazon review 类目数据集。

## 概览

完整流程如下：

```text
preprocess -> embed -> sid-train -> sid-generate -> convert -> sft -> rl -> evaluate
```

主要模块包括：

- `SID`（`Semantic ID`，语义标识符）训练与生成
- `SFT`（`Supervised Fine-Tuning`，监督微调）
- `RL`（`Reinforcement Learning`，强化学习）
- 约束生成与排序评估（`Evaluation`，评估）
- 基于 YAML 的配置管理（`configuration`，配置）

## 仓库结构

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

创建 Python 环境：

```bash
conda create -n OneRec python=3.11 -y
conda activate OneRec
```

安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

## 数据准备

代码默认从 `data/` 目录读取处理后的数据。数据集路径统一在
`config/datasets.yaml` 中配置。

当前默认数据集 key 为 `industrial`，仓库中也提供了 `office` 的配置入口。

执行数据预处理：

```bash
bash preprocess_amazon18.sh
```

生成文本嵌入（`embedding`，嵌入）：

```bash
bash text2emb.sh
```

## 使用方式

### 训练并生成 SID

```bash
bash sid_train.sh industrial
bash sid_generate.sh industrial
```

### 转换下游训练数据

```bash
bash convert.sh industrial
```

### 运行 SFT

```bash
bash sft.sh industrial
```

### 运行 RL

```bash
bash rl.sh industrial
```

### 评估 checkpoint

评估 SFT checkpoint（检查点）：

```bash
bash evaluate.sh sft industrial
```

评估 RL checkpoint（检查点）：

```bash
bash evaluate.sh rl industrial
```

同样的命令也可以切换到其他数据集 key，例如：

```bash
bash sft.sh office
bash rl.sh office
bash evaluate.sh sft office
```

## 配置

所有主流程均由 `config/` 下的 YAML 文件控制。

主要配置文件包括：

- `config/datasets.yaml`：数据集 key 与路径模板
- `config/sid_train.yaml`：SID tokenizer（分词器）训练
- `config/sid_generate.yaml`：SID 生成
- `config/convert.yaml`：数据格式转换
- `config/sft.yaml`：监督微调
- `config/rl.yaml`：强化学习
- `config/evaluate.yaml`：评估

命令行支持 override（覆盖参数）：

```bash
bash sft.sh office training.num_epochs=3 training.batch_size=512
bash rl.sh industrial training.num_generations=4
bash evaluate.sh rl office num_beams=48
```

也可以显式指定配置文件：

```bash
bash sft.sh config/sft.yaml industrial
bash rl.sh config/rl.yaml office
bash evaluate.sh config/evaluate.yaml rl industrial
```

## 输出

默认 SFT 与 RL 输出目录：

```text
output/
  sft_<category>_refactor/
    final_checkpoint/
    checkpoint-*/
  rl_<category>_refactor/
    final_checkpoint/
    checkpoint-*/
```

默认评估结果：

```text
results/
  final_result_sft_<category>.json
  final_result_rl_<category>.json
```

临时评估分片会写入 `temp/` 目录。

## 许可证

本项目使用 [Apache-2.0](./LICENSE) 许可证。

## 致谢

本实现基于 MiniOneRec 风格的生成式推荐流程，并参考了 ReRec、LC-Rec 等相关开源推荐工作。
