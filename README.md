# OneRec

OneRec is a codebase for generative recommendation. It follows the
`SID -> SFT -> RL -> Evaluation` workflow used in MiniOneRec-style systems and
provides a cleaner implementation of the main training and evaluation stages.

This repository is intended to serve as a paper code release: the core pipeline
is kept explicit, configuration-driven, and easy to adapt to different Amazon
review categories.

## Overview

The full workflow is:

```text
preprocess -> embed -> sid-train -> sid-generate -> convert -> sft -> rl -> evaluate
```

Main components:

- `SID` (`Semantic ID`, 语义标识符) training and generation
- `SFT` (`Supervised Fine-Tuning`, 监督微调)
- `RL` (`Reinforcement Learning`, 强化学习)
- constrained generation and ranking evaluation (`Evaluation`, 评估)
- YAML-based configuration (`configuration`, 配置)

## Repository Structure

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

## Installation

Create a Python environment:

```bash
conda create -n OneRec python=3.11 -y
conda activate OneRec
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

The code expects processed data under `data/`. Dataset paths are configured in
`config/datasets.yaml`.

The default dataset key is `industrial`. The repository also includes
configuration entries for `office`.

To run preprocessing:

```bash
bash preprocess_amazon18.sh
```

To generate text embeddings (`embedding`, 嵌入):

```bash
bash text2emb.sh
```

## Usage

### Train and Generate SID

```bash
bash sid_train.sh industrial
bash sid_generate.sh industrial
```

### Convert Data for Downstream Training

```bash
bash convert.sh industrial
```

### Run SFT

```bash
bash sft.sh industrial
```

### Run RL

```bash
bash rl.sh industrial
```

### Evaluate Checkpoints

Evaluate an SFT checkpoint (`checkpoint`, 检查点):

```bash
bash evaluate.sh sft industrial
```

Evaluate an RL checkpoint:

```bash
bash evaluate.sh rl industrial
```

The same commands can be used with other dataset keys, for example:

```bash
bash sft.sh office
bash rl.sh office
bash evaluate.sh sft office
```

## Configuration

All main stages are controlled by YAML files in `config/`.

Important files:

- `config/datasets.yaml`: dataset keys and path templates
- `config/sid_train.yaml`: SID tokenizer training
- `config/sid_generate.yaml`: SID generation
- `config/convert.yaml`: data conversion
- `config/sft.yaml`: supervised fine-tuning
- `config/rl.yaml`: reinforcement learning
- `config/evaluate.yaml`: evaluation

Command-line overrides (`override`, 覆盖参数) are supported:

```bash
bash sft.sh office training.num_epochs=3 training.batch_size=512
bash rl.sh industrial training.num_generations=4
bash evaluate.sh rl office num_beams=48
```

You can also pass an explicit configuration file:

```bash
bash sft.sh config/sft.yaml industrial
bash rl.sh config/rl.yaml office
bash evaluate.sh config/evaluate.yaml rl industrial
```

## Outputs

Default SFT and RL outputs:

```text
output/
  sft_<category>_refactor/
    final_checkpoint/
    checkpoint-*/
  rl_<category>_refactor/
    final_checkpoint/
    checkpoint-*/
```

Default evaluation results:

```text
results/
  final_result_sft_<category>.json
  final_result_rl_<category>.json
```

Temporary evaluation shards are written under `temp/`.

## License

This project is released under the [Apache-2.0](./LICENSE) license.

## Acknowledgements

This implementation builds on the MiniOneRec-style generative recommendation
pipeline and related open-source recommendation work, including ReRec and
LC-Rec.
