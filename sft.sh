#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/_common.sh"

DEFAULT_CONFIG="config/sft.yaml"
resolve_config_path "$DEFAULT_CONFIG" "$@"
use_first_positional_as_dataset_key
resolve_dataset_overrides "sft"
build_effective_override_args

mapfile -t _launch < <(python - "$CONFIG_PATH" "${EFFECTIVE_OVERRIDES[@]}" <<'PY'
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
launcher = str(runtime.get("launcher", "torchrun")).strip() or "torchrun"
gpus = str(runtime.get("cuda_visible_devices", "")).strip()
nproc = runtime.get("nproc_per_node")
if nproc is None:
    nproc = len([x for x in gpus.split(",") if x.strip()]) if gpus else 1
training = cfg.get("training", {})
logging = cfg.get("logging", {})
output = cfg.get("output", {})
print(launcher)
print(gpus)
print(int(nproc))
print(training.get("batch_size", ""))
print(training.get("micro_batch_size", ""))
print(training.get("num_epochs", ""))
print(training.get("learning_rate", ""))
print(logging.get("wandb_project", ""))
print(logging.get("wandb_run_name", ""))
print(output.get("output_dir", ""))
PY
)

LAUNCHER="${_launch[0]:-torchrun}"
GPU_LIST="${_launch[1]:-}"
NPROC="${_launch[2]:-1}"
TRAIN_BS="${_launch[3]:-}"
TRAIN_MBS="${_launch[4]:-}"
TRAIN_EPOCHS="${_launch[5]:-}"
TRAIN_LR="${_launch[6]:-}"
WANDB_PROJECT="${_launch[7]:-}"
WANDB_RUN_NAME="${_launch[8]:-}"
OUTPUT_DIR="${_launch[9]:-}"

if [[ -n "$GPU_LIST" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi

ensure_nproc_within_gpu_list "$LAUNCHER" "$NPROC" "$GPU_LIST" "SFT"

echo "[SFT] launcher=$LAUNCHER gpus=${GPU_LIST:-<default>} nproc_per_node=$NPROC config=$CONFIG_PATH dataset=${DATASET_KEY:-<config>}"
echo "[SFT] summary batch_size=$TRAIN_BS micro_batch_size=$TRAIN_MBS epochs=$TRAIN_EPOCHS lr=$TRAIN_LR wandb_project=$WANDB_PROJECT run_name=$WANDB_RUN_NAME output=$OUTPUT_DIR"

if [[ "$LAUNCHER" == "torchrun" && "$NPROC" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="$NPROC" -m onerec.main sft --config "$CONFIG_PATH" "${EFFECTIVE_OVERRIDES[@]}"
else
  exec python -m onerec.main sft --config "$CONFIG_PATH" "${EFFECTIVE_OVERRIDES[@]}"
fi
