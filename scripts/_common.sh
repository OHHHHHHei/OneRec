#!/usr/bin/env bash

resolve_config_path() {
  local default_config="$1"
  shift
  CONFIG_PATH="${MINIONEREC_CONFIG:-$default_config}"
  PASSTHROUGH_ARGS=()

  if [[ $# -gt 0 ]]; then
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
    esac
  fi

  PASSTHROUGH_ARGS=("$@")
}

ensure_nproc_within_gpu_list() {
  local launcher="$1"
  local nproc="$2"
  local gpu_list="$3"
  local label="$4"

  if [[ "$launcher" == "torchrun" && -n "$gpu_list" ]]; then
    IFS=',' read -r -a _gpu_arr <<< "$gpu_list"
    local gpu_count="${#_gpu_arr[@]}"
    if [[ "$nproc" -gt "$gpu_count" ]]; then
      echo "ERROR: $label nproc_per_node ($nproc) > gpu_count ($gpu_count) from runtime.cuda_visible_devices=$gpu_list" >&2
      exit 1
    fi
  fi
}

