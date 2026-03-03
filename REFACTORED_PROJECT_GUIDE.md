# Refactored Project Guide

## Purpose

This document explains the current project structure, execution logic, stage-to-stage data flow,
and recommended operating procedure for the refactored mainline repository.

The preserved mainline workflow is:

```text
preprocess -> embed -> SID -> convert -> SFT -> RL -> evaluate
```

The implementation style changed, but the core logic chain and artifact contracts remain aligned
with the original project.

## 1. Project structure

### Root-level directories

```text
archive/      Archived non-mainline code and historical notes
assets/       Images used by docs
config/       Legacy config files still kept for compatibility
configs/      New stage YAML configs
data/         Dataset scripts and local artifacts
minionerec/   Thin import shim for the packaged code
rq/           Legacy SID-related wrappers and old scripts
scripts/      Thin shell wrappers for common stage commands
src/          Main refactored source tree
tests/        Unit tests and small fixtures
```

### Main source tree

The real mainline code is under `src/minionerec`.

```text
src/minionerec/
  cli/           Unified command entrypoints
  common/        Shared utilities
  compat/        Legacy argument and path adapters
  config/        Typed config schema and YAML loader
  preprocess/    Amazon18 and Amazon23 preprocessing
  sid/           Embedding, quantization, SID generation
  data/          Contracts, cache, conversion, datasets
  training/
    sft/         SFT pipeline
    rl/          RL pipeline
    cf/          SASRec helpers
  evaluation/    Constrained decoding, split/merge, metrics
```

## 2. Logic chain

### Stage 1: preprocess

Goal:
- Convert raw Amazon review and metadata files into interaction splits and item metadata

Main outputs:
- `*.train.inter`
- `*.valid.inter`
- `*.test.inter`
- `*.item.json`

Main code:
- `src/minionerec/preprocess/amazon18.py`
- `src/minionerec/preprocess/amazon23.py`

### Stage 2: embed

Goal:
- Build text embeddings for items from title and description

Typical input:
- `*.item.json`

Typical output:
- item embedding arrays such as `*.emb-*.npy`

Main code:
- `src/minionerec/sid/text2emb.py`

### Stage 3: SID

Goal:
- Quantize item embeddings into discrete semantic IDs

Typical input:
- embedding arrays from the embed stage

Typical output:
- `*.index.json`

Main code:
- `src/minionerec/sid/quantizers/`
- `src/minionerec/sid/generate/`
- `src/minionerec/sid/models/`

### Stage 4: convert

Goal:
- Convert preprocessing artifacts and SID artifacts into the train, valid, test CSV files and
  info files used by SFT, RL, and evaluation

Typical inputs:
- `*.train.inter`
- `*.valid.inter`
- `*.test.inter`
- `*.item.json`
- `*.index.json`

Typical outputs:
- converted CSV files
- `info/*.txt`

Main code:
- `src/minionerec/data/convert.py`

### Stage 5: SFT

Goal:
- Extend tokenizer with SID tokens and train the supervised recommendation model

Main task semantics:
- history SID -> next SID
- SID <-> title
- history SID -> next item title

Main code:
- `src/minionerec/training/sft/pipeline.py`
- `src/minionerec/training/sft/token_extension.py`
- `src/minionerec/data/datasets/sft.py`

Typical output:
- model checkpoints
- tokenizer with SID extensions
- `final_checkpoint/`

### Stage 6: RL

Goal:
- Continue training from SFT with recommendation-oriented reinforcement learning

Main task semantics:
- SID history -> next SID
- title or description -> SID
- title sequence -> SID

Main code:
- `src/minionerec/training/rl/pipeline.py`
- `src/minionerec/training/rl/rewards.py`
- `src/minionerec/training/rl/constrained_generation.py`
- `src/minionerec/data/datasets/rl.py`

Important behavior:
- constrained generation only allows valid SID continuations
- reward computation stays aligned with the original workflow

### Stage 7: evaluate

Goal:
- Run constrained decoding on the trained model and compute ranking metrics

Main code:
- `src/minionerec/evaluation/constrained_decoding.py`
- `src/minionerec/evaluation/pipeline.py`
- `src/minionerec/evaluation/split_merge.py`
- `src/minionerec/evaluation/metrics.py`

Typical outputs:
- prediction shards
- merged results
- HR, NDCG, and related statistics

## 3. CLI usage

Unified entrypoint:

```bash
python -m minionerec.cli.main <stage> --config <yaml> [overrides...]
```

Examples:

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

## 4. Configuration model

Each stage uses YAML config plus optional CLI overrides.

Typical config areas:
- `model`
- `data`
- `training`
- `output`
- `logging`

The loader and schema live in:
- `src/minionerec/config/loader.py`
- `src/minionerec/config/schema.py`

## 5. Compatibility layer

The repository still ships the original script names as wrappers.

Examples:
- `sft.py`
- `rl.py`
- `evaluate.py`
- `convert_dataset.py`

These wrappers forward into the refactored mainline so older remote-server commands do not break
immediately.

## 6. Local operation procedure

Recommended local workflow:

1. Edit code and configs in the refactored layout.
2. Run unit tests and CLI help checks locally.
3. Validate data contracts with the sample fixtures in `tests/fixtures`.
4. Do not attempt full training locally unless the environment has the required GPU stack.

Useful checks:

```bash
python -m minionerec.cli.main --help
python -m unittest discover -s tests/unit -v
python -m compileall src minionerec
```

## 7. Remote server operation procedure

Recommended remote workflow:

1. Prepare the Python environment and deep learning dependencies.
2. Run preprocess only if raw data has not already been converted.
3. Run embed and SID stages to build semantic IDs.
4. Run convert to build SFT, RL, and evaluation inputs.
5. Run SFT.
6. Run RL starting from the SFT checkpoint.
7. Run evaluate on the final checkpoint.

This sequence preserves the original training logic and artifact chain.

## 8. Important contracts to keep stable

- item metadata format in `*.item.json`
- SID index format in `*.index.json`
- converted CSV column layout
- `info/*.txt` layout used by constrained decoding
- tokenizer SID extension logic
- checkpoint layout expected by RL and evaluation

## 9. Archived code

Code that is not part of the current mainline is kept under `archive/`.

Current archive areas:
- `archive/gpr/`
- `archive/old_docs/`

The mainline must not depend on archived code.

## 10. Current validation scope

This repository is validated locally through:
- config loading
- CLI help checks
- dataset conversion contract tests
- compile checks

Full SFT, RL, and evaluation runs should be validated on the remote training server.
