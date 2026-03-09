#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

resolve_config_path() {
  local default_config="$1"
  shift
  CONFIG_PATH="${ONEREC_CONFIG:-$default_config}"
  DATASET_KEY="${ONEREC_DATASET:-}"
  EVAL_MODEL_STAGE="${ONEREC_EVAL_MODEL_STAGE:-}"
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
}

use_first_positional_as_dataset_key() {
  if [[ -z "${DATASET_KEY:-}" && ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    DATASET_KEY="${POSITIONAL_ARGS[0]}"
    POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
  fi
  if [[ -z "${DATASET_KEY:-}" ]]; then
    DATASET_KEY="industrial"
  fi
  if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    echo "ERROR: unexpected positional arguments: ${POSITIONAL_ARGS[*]}" >&2
    exit 1
  fi
}

resolve_evaluate_selection() {
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

render_stage_config() {
  local stage="$1"
  local datasets_path="$REPO_ROOT/config/datasets.yaml"
  if [[ ! -f "$datasets_path" ]]; then
    echo "ERROR: dataset mapping file not found: $datasets_path" >&2
    exit 1
  fi

  RENDERED_CONFIG_PATH="$(python - "$CONFIG_PATH" "$datasets_path" "$DATASET_KEY" "$EVAL_MODEL_STAGE" <<'PY'
import sys
from pathlib import Path

from onerec.utils.config_templates import render_config_file

config_path = sys.argv[1]
datasets_path = sys.argv[2]
dataset_key = sys.argv[3] or None
eval_model_stage = sys.argv[4] or "sft"
rendered = render_config_file(
    config_path=config_path,
    datasets_path=datasets_path,
    dataset_key=dataset_key,
    eval_model_stage=eval_model_stage,
)
print(Path(rendered).as_posix())
PY
)"
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
