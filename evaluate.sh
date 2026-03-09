#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/_common.sh"

DEFAULT_CONFIG="config/evaluate.yaml"
resolve_config_path "$DEFAULT_CONFIG" "$@"
resolve_evaluate_selection
render_stage_config "evaluate"

mapfile -t _launch < <(python - "$RENDERED_CONFIG_PATH" "${PASSTHROUGH_ARGS[@]}" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def parse_value(raw):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw

for override in sys.argv[2:]:
    if "=" not in override:
        continue
    dotted, raw_value = override.split("=", 1)
    cur = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = parse_value(raw_value)

runtime = cfg.get("runtime", {})
launcher = str(runtime.get("launcher", "parallel")).strip() or "parallel"
gpus = str(runtime.get("cuda_visible_devices", "")).strip()
nproc = runtime.get("nproc_per_node")
if nproc is None:
    nproc = len([x for x in gpus.split(",") if x.strip()]) if gpus else 1
parallel = runtime.get("parallel")
if parallel is None:
    parallel = nproc > 1
data = cfg.get("data", {})
model = cfg.get("model", {})
output = cfg.get("output", {})
print(launcher)
print(gpus)
print(int(nproc))
print(str(bool(parallel)).lower())
print(str(model.get("base_model", "")).strip())
print(str(data.get("test_file", "")).strip())
print(str(data.get("info_file", "")).strip())
print(str(data.get("category", "")).strip())
print(str(output.get("output_dir", "")).strip())
print(cfg.get("batch_size", 8))
print(cfg.get("K", 0))
print(cfg.get("num_beams", 50))
print(cfg.get("max_new_tokens", 256))
print(cfg.get("length_penalty", 0.0))
print(cfg.get("temperature", 1.0))
guidance_scale = cfg.get("guidance_scale", None)
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

echo "[EVAL] launcher=$LAUNCHER parallel=$PARALLEL gpus=${GPU_LIST:-<default>} config=$RENDERED_CONFIG_PATH model_stage=${EVAL_MODEL_STAGE:-sft} dataset=${DATASET_KEY:-industrial}"
echo "[EVAL] summary batch_size=$BATCH_SIZE num_beams=$NUM_BEAMS max_new_tokens=$MAX_NEW_TOKENS length_penalty=$LENGTH_PENALTY temperature=$TEMPERATURE guidance_scale=$GUIDANCE_SCALE output=$RESULT_PATH"

if [[ "$LAUNCHER" == "python" || "$LAUNCHER" == "single" || "$PARALLEL" != "true" || "$NPROC" -le 1 ]]; then
  exec python -m onerec.main evaluate --config "$RENDERED_CONFIG_PATH" "${PASSTHROUGH_ARGS[@]}"
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
TEMP_DIR="./temp/${EVAL_MODEL_STAGE:-sft}-${CATEGORY:-eval}"
mkdir -p "$TEMP_DIR"

echo "[EVAL] splitting test file: $TEST_FILE -> $TEMP_DIR"
python -m onerec.main split --input_path "$TEST_FILE" --output_path "$TEMP_DIR" --cuda_list "$GPU_LIST"

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
    ONEREC_EVAL_WORKER_ID="$gpu" \
    ONEREC_EVAL_PRIMARY_WORKER="$PRIMARY_WORKER" \
    python -u -m onerec.main evaluate \
      --config "$RENDERED_CONFIG_PATH" \
      "${PASSTHROUGH_ARGS[@]}" \
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
python -m onerec.main merge --input_path "$TEMP_DIR" --output_path "$RESULT_PATH" --cuda_list "$actual_cuda_list"

echo "[EVAL] calculating metrics from merged result"
python -m onerec.main metrics --path "$RESULT_PATH" --item_path "$INFO_FILE"
