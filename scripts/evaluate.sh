#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/_common.sh"

DEFAULT_CONFIG="flows/evaluate/default.yaml"
resolve_config_path "$DEFAULT_CONFIG" "$@"

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
batch_size = cfg.get("batch_size", 8) if isinstance(cfg, dict) else 8
K = cfg.get("K", 0) if isinstance(cfg, dict) else 0
num_beams = cfg.get("num_beams", 50) if isinstance(cfg, dict) else 50
max_new_tokens = cfg.get("max_new_tokens", 256) if isinstance(cfg, dict) else 256
length_penalty = cfg.get("length_penalty", 0.0) if isinstance(cfg, dict) else 0.0
temperature = cfg.get("temperature", 1.0) if isinstance(cfg, dict) else 1.0
guidance_scale = cfg.get("guidance_scale", None) if isinstance(cfg, dict) else None
print(launcher)
print(gpus)
print(int(nproc))
print(str(bool(parallel)).lower())
print(str(model.get("base_model", "")).strip())
print(str(data.get("test_file", "")).strip())
print(str(data.get("info_file", "")).strip())
print(str(data.get("category", "")).strip())
print(str(output.get("output_dir", "")).strip())
print(batch_size)
print(K)
print(num_beams)
print(max_new_tokens)
print(length_penalty)
print(temperature)
print("None" if guidance_scale is None else guidance_scale)
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
BATCH_SIZE="${_launch[9]:-8}"
K_VALUE="${_launch[10]:-0}"
NUM_BEAMS="${_launch[11]:-50}"
MAX_NEW_TOKENS="${_launch[12]:-256}"
LENGTH_PENALTY="${_launch[13]:-0.0}"
TEMPERATURE="${_launch[14]:-1.0}"
GUIDANCE_SCALE="${_launch[15]:-None}"

if [[ -n "$GPU_LIST" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi

echo "[EVAL] launcher=$LAUNCHER parallel=$PARALLEL gpus=${GPU_LIST:-<default>} config=$CONFIG_PATH"
echo "[EVAL] summary batch_size=$BATCH_SIZE num_beams=$NUM_BEAMS max_new_tokens=$MAX_NEW_TOKENS length_penalty=$LENGTH_PENALTY temperature=$TEMPERATURE guidance_scale=$GUIDANCE_SCALE output=$RESULT_PATH"

if [[ "$LAUNCHER" == "python" || "$LAUNCHER" == "single" || "$PARALLEL" != "true" || "$NPROC" -le 1 ]]; then
  exec python -m minionerec.cli.main evaluate --config "$CONFIG_PATH" "${PASSTHROUGH_ARGS[@]}"
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
PRIMARY_WORKER=""
for gpu in "${_gpu_arr[@]}"; do
  gpu="$(echo "$gpu" | xargs)"
  if [[ -n "$gpu" ]]; then
    PRIMARY_WORKER="$gpu"
    break
  fi
done
if [[ -z "$PRIMARY_WORKER" ]]; then
  echo "ERROR: failed to resolve primary worker from runtime.cuda_visible_devices=$GPU_LIST" >&2
  exit 1
fi

for gpu in "${_gpu_arr[@]}"; do
  gpu="$(echo "$gpu" | xargs)"
  if [[ -z "$gpu" ]]; then
    continue
  fi
  if [[ -f "$TEMP_DIR/${gpu}.csv" ]]; then
    echo "[EVAL] launch worker on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" \
    MINIONEREC_EVAL_WORKER_ID="$gpu" \
    MINIONEREC_EVAL_PRIMARY_WORKER="$PRIMARY_WORKER" \
    python -u ./evaluate.py \
      --base_model "$MODEL_PATH" \
      --info_file "$INFO_FILE" \
      --category "$CATEGORY" \
      --test_data_path "$TEMP_DIR/${gpu}.csv" \
      --result_json_data "$TEMP_DIR/${gpu}.json" \
      --batch_size "$BATCH_SIZE" \
      --K "$K_VALUE" \
      --num_beams "$NUM_BEAMS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --length_penalty "$LENGTH_PENALTY" \
      --temperature "$TEMPERATURE" \
      --guidance_scale "$GUIDANCE_SCALE" &
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
