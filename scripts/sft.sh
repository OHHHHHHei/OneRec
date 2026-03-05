#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/_common.sh"

DEFAULT_CONFIG="flows/sft/default.yaml"
resolve_config_path "$DEFAULT_CONFIG" "$@"

mapfile -t _launch < <(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
launcher = str(runtime.get("launcher", "torchrun")).strip() or "torchrun"
gpus = str(runtime.get("cuda_visible_devices", "")).strip()
nproc = runtime.get("nproc_per_node", None)
if nproc is None:
    if gpus:
        nproc = len([x for x in gpus.split(",") if x.strip()])
    else:
        nproc = 1
print(launcher)
print(gpus)
print(int(nproc))
training = cfg.get("training", {}) if isinstance(cfg, dict) else {}
logging = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
output = cfg.get("output", {}) if isinstance(cfg, dict) else {}
print(training.get("batch_size", ""))
print(training.get("micro_batch_size", ""))
print(training.get("num_epochs", ""))
print(training.get("learning_rate", ""))
print(logging.get("wandb_project", ""))
print(logging.get("wandb_run_name", ""))
print(output.get("output_dir", ""))
PY
)

LAUNCHER="${_launch[0]:-python}"
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

echo "[SFT] launcher=$LAUNCHER gpus=${GPU_LIST:-<default>} nproc_per_node=$NPROC config=$CONFIG_PATH"
echo "[SFT] summary batch_size=$TRAIN_BS micro_batch_size=$TRAIN_MBS epochs=$TRAIN_EPOCHS lr=$TRAIN_LR wandb_project=$WANDB_PROJECT run_name=$WANDB_RUN_NAME output=$OUTPUT_DIR"

if [[ "$LAUNCHER" == "torchrun" && "$NPROC" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="$NPROC" -m minionerec.cli.main sft --config "$CONFIG_PATH" "${PASSTHROUGH_ARGS[@]}"
else
  exec python -m minionerec.cli.main sft --config "$CONFIG_PATH" "${PASSTHROUGH_ARGS[@]}"
fi
