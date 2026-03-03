# OneRec Mainline

This repository is the independent mainline refactor of the original MiniOneRec codebase.
It keeps the same end-to-end recommendation workflow:

```text
preprocess -> embed -> SID -> convert -> SFT -> RL -> evaluate
```

The refactor keeps the main behavior contracts while moving the implementation into a
package-oriented layout under `src/minionerec`.

## What changed

- Unified CLI entrypoint
- YAML-based stage configuration
- Clear package boundaries for preprocess, SID, data, SFT, RL, and evaluation
- Compatibility wrappers for the original root scripts
- Archive area for non-mainline code

## Main entrypoint

```bash
python -m minionerec.cli.main <stage> --config <yaml> [overrides...]
```

Supported stages:

- `preprocess`
- `embed`
- `sid-train`
- `sid-generate`
- `convert`
- `sft`
- `rl`
- `evaluate`

Examples:

```bash
python -m minionerec.cli.main sft --config configs/stages/sft/default.yaml
python -m minionerec.cli.main rl --config configs/stages/rl/default.yaml
```

## Repository layout

```text
src/minionerec/
  cli/           Stage entrypoints
  common/        IO, logging, path resolution, validation, tokenizer helpers
  config/        YAML loading and typed config schema
  preprocess/    Amazon dataset preprocessing
  sid/           Text embedding, quantization, SID generation
  data/          Dataset contracts, cache, conversion, dataset builders
  training/sft/  SFT pipeline
  training/rl/   RL pipeline
  training/cf/   SASRec-related helpers
  evaluation/    Constrained decoding and metrics
  compat/        Legacy CLI and path compatibility
```

## Compatibility

The original script names are still available as thin wrappers:

- `sft.py`
- `rl.py`
- `evaluate.py`
- `convert_dataset.py`
- `split.py`
- `merge.py`
- `calc.py`

Use the new CLI for new work. Keep the wrappers only for gradual migration on remote servers.

## Data and outputs

Large datasets, generated embeddings, checkpoints, and experiment outputs are not meant to be
tracked in Git. The repository keeps only code, configuration, docs, and small test fixtures.

## Documentation

- Main operational guide: `REFACTORED_PROJECT_GUIDE.md`
- Architecture summary: `PROJECT_UNDERSTANDING.md`
- Archive notes: `archive/README.md`

## Local validation

Recommended local checks:

```bash
python -m minionerec.cli.main --help
python -m unittest discover -s tests/unit -v
python -m compileall src minionerec
```

This local environment is intended for contract checks and smoke tests. Full SFT, RL, and
evaluation runs should be executed on a remote training server.
