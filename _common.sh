#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

resolve_config_path() {
  local default_config="$1"
  shift
  CONFIG_PATH="${ONEREC_CONFIG:-$default_config}"
  DATASET_KEY="${ONEREC_DATASET:-}"
  POSITIONAL_ARGS=()
  PASSTHROUGH_ARGS=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      *.yaml|*.yml)
        CONFIG_PATH="$1"
        shift
        ;;
      --config)
        if [[ $# -lt 2 ]]; then
          echo "ERROR: --config requires a path" >&2
          exit 1
        fi
        CONFIG_PATH="$2"
        shift 2
        ;;
      *=*)
        PASSTHROUGH_ARGS+=("$1")
        shift
        ;;
      *)
        POSITIONAL_ARGS+=("$1")
        shift
        ;;
    esac
  done

  DATASET_OVERRIDES=()
}

use_first_positional_as_dataset_key() {
  if [[ -z "${DATASET_KEY:-}" && ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    DATASET_KEY="${POSITIONAL_ARGS[0]}"
    POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
  fi
  if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    echo "ERROR: unexpected positional arguments: ${POSITIONAL_ARGS[*]}" >&2
    exit 1
  fi
}

resolve_evaluate_selection() {
  EVAL_MODEL_STAGE="${ONEREC_EVAL_MODEL_STAGE:-}"
  if [[ -z "$EVAL_MODEL_STAGE" && ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    case "${POSITIONAL_ARGS[0],,}" in
      sft|rl)
        EVAL_MODEL_STAGE="${POSITIONAL_ARGS[0],,}"
        POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
        ;;
    esac
  fi
  if [[ -z "$EVAL_MODEL_STAGE" ]]; then
    EVAL_MODEL_STAGE="sft"
  fi
  use_first_positional_as_dataset_key
}

resolve_dataset_overrides() {
  local stage="$1"
  DATASET_OVERRIDES=()
  if [[ -z "${DATASET_KEY:-}" ]]; then
    return 0
  fi

  local datasets_path="$REPO_ROOT/config/datasets.yaml"
  if [[ ! -f "$datasets_path" ]]; then
    echo "ERROR: dataset mapping file not found: $datasets_path" >&2
    exit 1
  fi

  mapfile -t DATASET_OVERRIDES < <(python - "$stage" "$DATASET_KEY" "$datasets_path" "${EVAL_MODEL_STAGE:-}" <<'PY'
import sys
from pathlib import Path
import yaml

stage = sys.argv[1]
dataset_key = sys.argv[2].strip().lower()
datasets_path = Path(sys.argv[3])
eval_model_stage = sys.argv[4].strip().lower() or "sft"
with open(datasets_path, "r", encoding="utf-8") as handle:
    datasets = yaml.safe_load(handle) or {}
if dataset_key not in datasets:
    valid = ", ".join(sorted(datasets))
    raise SystemExit(f"ERROR: unknown dataset key {dataset_key!r}. Available: {valid}")
if eval_model_stage not in {"sft", "rl"}:
    raise SystemExit(f"ERROR: unsupported evaluate model stage {eval_model_stage!r}. Expected: sft or rl")

entry = datasets[dataset_key]
category = str(entry["category"]).strip()
split_stem = str(entry["split_stem"]).strip()
artifact_stem = str(entry.get("artifact_stem", category)).strip()
eval_result_suffix = "" if eval_model_stage == "sft" else f"_{eval_model_stage}"

overrides = {
    "sft": [
        f"data.train_file=./data/Amazon/train/{split_stem}.csv",
        f"data.eval_file=./data/Amazon/valid/{split_stem}.csv",
        f"data.sid_index_path=./data/Amazon/index/{artifact_stem}.index.json",
        f"data.item_meta_path=./data/Amazon/index/{artifact_stem}.item.json",
        f"data.category={category}",
        f"logging.wandb_run_name=sft_{category}",
        f"output.output_dir=./output/sft_{category}_refactor",
    ],
    "rl": [
        f"model.base_model=./output/sft_{category}_refactor/final_checkpoint",
        f"data.train_file=./data/Amazon/train/{split_stem}.csv",
        f"data.eval_file=./data/Amazon/valid/{split_stem}.csv",
        f"data.info_file=./data/Amazon/info/{split_stem}.txt",
        f"data.sid_index_path=./data/Amazon/index/{artifact_stem}.index.json",
        f"data.item_meta_path=./data/Amazon/index/{artifact_stem}.item.json",
        f"data.category={category}",
        f"logging.wandb_run_name=RL_{category}_refactor",
        f"output.output_dir=./output/rl_{category}_refactor",
    ],
    "evaluate": [
        f"model.base_model=./output/{eval_model_stage}_{category}_refactor/final_checkpoint",
        f"data.test_file=./data/Amazon/test/{split_stem}.csv",
        f"data.info_file=./data/Amazon/info/{split_stem}.txt",
        f"data.category={category}",
        f"output.output_dir=./results/final_result{eval_result_suffix}_{category}.json",
    ],
}

if stage not in overrides:
    raise SystemExit(f"ERROR: unsupported stage for dataset overrides: {stage}")

for item in overrides[stage]:
    print(item)
PY
  )
}

build_effective_override_args() {
  EFFECTIVE_OVERRIDES=("${DATASET_OVERRIDES[@]}" "${PASSTHROUGH_ARGS[@]}")
}

ensure_nproc_within_gpu_list() {
  local launcher="$1"
  local nproc="$2"
  local gpu_list="$3"
  local label="$4"

  if [[ -n "$gpu_list" ]]; then
    IFS=',' read -r -a _gpu_arr <<< "$gpu_list"
    local gpu_count="${#_gpu_arr[@]}"
    if [[ "$nproc" -gt "$gpu_count" ]]; then
      echo "ERROR: $label process count ($nproc) > gpu_count ($gpu_count) from runtime.cuda_visible_devices=$gpu_list" >&2
      exit 1
    fi
  fi
}
