#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_CONFIG="configs/stages/evaluate/default.yaml"
CONFIG_PATH="${MINIONEREC_CONFIG:-$DEFAULT_CONFIG}"

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

mapfile -t _launch < <(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
launcher = str(runtime.get("launcher", "parallel")).strip() or "parallel"
gpus = str(runtime.get("cuda_visible_devices", "")).strip()
nproc = runtime.get("nproc_per_node", None)
if nproc is None:
    if gpus:
        nproc = len([x for x in gpus.split(",") if x.strip()])
    else:
        nproc = 1
parallel = runtime.get("parallel", None)
if parallel is None:
    parallel = nproc > 1

data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
model = cfg.get("model", {}) if isinstance(cfg, dict) else {}
output = cfg.get("output", {}) if isinstance(cfg, dict) else {}
print(launcher)
print(gpus)
print(int(nproc))
print(str(bool(parallel)).lower())
print(str(model.get("base_model", "")).strip())
print(str(data.get("test_file", "")).strip())
print(str(data.get("info_file", "")).strip())
print(str(data.get("category", "")).strip())
print(str(output.get("output_dir", "")).strip())
PY
)

LAUNCHER="${_launch[0]:-parallel}"
GPU_LIST="${_launch[1]:-}"
NPROC="${_launch[2]:-1}"
PARALLEL="${_launch[3]:-false}"
MODEL_PATH="${_launch[4]:-}"
TEST_FILE="${_launch[5]:-}"
INFO_FILE="${_launch[6]:-}"
CATEGORY="${_launch[7]:-}"
RESULT_PATH="${_launch[8]:-./results/final_result.json}"

if [[ -n "$GPU_LIST" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi

echo "[EVAL] launcher=$LAUNCHER parallel=$PARALLEL gpus=${GPU_LIST:-<default>} config=$CONFIG_PATH"

if [[ "$LAUNCHER" == "python" || "$LAUNCHER" == "single" || "$PARALLEL" != "true" || "$NPROC" -le 1 ]]; then
  exec python -m minionerec.cli.main evaluate --config "$CONFIG_PATH" "$@"
fi

if [[ -z "$TEST_FILE" || -z "$INFO_FILE" ]]; then
  echo "ERROR: data.test_file and data.info_file are required in evaluate config for parallel mode." >&2
  exit 1
fi
if [[ -z "$GPU_LIST" ]]; then
  echo "ERROR: runtime.cuda_visible_devices is required in parallel evaluate mode." >&2
  exit 1
fi

mkdir -p "$(dirname "$RESULT_PATH")"
RESULT_STEM="$(basename "${RESULT_PATH%.*}")"
TEMP_DIR="./temp/${CATEGORY:-eval}-${RESULT_STEM}"
mkdir -p "$TEMP_DIR"

echo "[EVAL] splitting test file: $TEST_FILE -> $TEMP_DIR"
python ./split.py --input_path "$TEST_FILE" --output_path "$TEMP_DIR" --cuda_list "$GPU_LIST"

IFS=',' read -r -a _gpu_arr <<< "$GPU_LIST"
for gpu in "${_gpu_arr[@]}"; do
  gpu="$(echo "$gpu" | xargs)"
  if [[ -z "$gpu" ]]; then
    continue
  fi
  if [[ -f "$TEMP_DIR/${gpu}.csv" ]]; then
    echo "[EVAL] launch worker on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m minionerec.cli.main evaluate --config "$CONFIG_PATH" \
      "$@" \
      "data.test_file=$TEMP_DIR/${gpu}.csv" \
      "output.output_dir=$TEMP_DIR/${gpu}.json" &
  else
    echo "[EVAL] skip GPU $gpu: split file not found."
  fi
done

wait

actual_cuda_list="$(ls "$TEMP_DIR"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')"
if [[ -z "$actual_cuda_list" ]]; then
  echo "ERROR: no shard results generated under $TEMP_DIR" >&2
  exit 1
fi

echo "[EVAL] merging shard results: $actual_cuda_list -> $RESULT_PATH"
python ./merge.py --input_path "$TEMP_DIR" --output_path "$RESULT_PATH" --cuda_list "$actual_cuda_list"

echo "[EVAL] calculating metrics from merged result"
python ./calc.py --path "$RESULT_PATH" --item_path "$INFO_FILE"
