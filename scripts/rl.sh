#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/_common.sh"

DEFAULT_CONFIG="flows/rl/default.yaml"
resolve_config_path "$DEFAULT_CONFIG" "$@"

mapfile -t _launch < <(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
launcher = str(runtime.get("launcher", "accelerate")).strip() or "accelerate"
gpus = str(runtime.get("cuda_visible_devices", "")).strip()
nproc = runtime.get("num_processes", runtime.get("nproc_per_node", None))
if nproc is None:
    if gpus:
        nproc = len([x for x in gpus.split(",") if x.strip()])
    else:
        nproc = 1
acc_cfg = str(runtime.get("accelerate_config", "config/zero2_opt.yaml")).strip()
port = int(runtime.get("main_process_port", 29503))
hfe = str(runtime.get("hf_endpoint", "")).strip()
print(launcher)
print(gpus)
print(int(nproc))
print(acc_cfg)
print(port)
print(hfe)
training = cfg.get("training", {}) if isinstance(cfg, dict) else {}
logging = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
output = cfg.get("output", {}) if isinstance(cfg, dict) else {}
print(training.get("reward_type", ""))
print(training.get("num_generations", ""))
print(training.get("eval_step", ""))
print(training.get("beam_search", ""))
print(logging.get("wandb_project", ""))
print(logging.get("wandb_run_name", ""))
print(output.get("output_dir", ""))
PY
)

LAUNCHER="${_launch[0]:-accelerate}"
GPU_LIST="${_launch[1]:-}"
NPROC="${_launch[2]:-1}"
ACC_CONFIG="${_launch[3]:-config/zero2_opt.yaml}"
MAIN_PORT="${_launch[4]:-29503}"
HF_ENDPOINT_CFG="${_launch[5]:-}"
REWARD_TYPE="${_launch[6]:-}"
NUM_GENERATIONS="${_launch[7]:-}"
EVAL_STEP="${_launch[8]:-}"
BEAM_SEARCH="${_launch[9]:-}"
WANDB_PROJECT="${_launch[10]:-}"
WANDB_RUN_NAME="${_launch[11]:-}"
OUTPUT_DIR="${_launch[12]:-}"

if [[ -n "$GPU_LIST" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi

if [[ "$LAUNCHER" == "accelerate" && -n "$HF_ENDPOINT_CFG" ]]; then
  export HF_ENDPOINT="$HF_ENDPOINT_CFG"
fi

ensure_nproc_within_gpu_list "$LAUNCHER" "$NPROC" "$GPU_LIST" "RL"

echo "[RL] launcher=$LAUNCHER gpus=${GPU_LIST:-<default>} nproc_per_node=$NPROC config=$CONFIG_PATH"
echo "[RL] summary reward_type=$REWARD_TYPE num_generations=$NUM_GENERATIONS eval_step=$EVAL_STEP beam_search=$BEAM_SEARCH wandb_project=$WANDB_PROJECT run_name=$WANDB_RUN_NAME output=$OUTPUT_DIR"

if [[ "$LAUNCHER" == "accelerate" ]]; then
  exec accelerate launch \
    --config_file "$ACC_CONFIG" \
    --num_processes "$NPROC" \
    --main_process_port "$MAIN_PORT" \
    -m minionerec.cli.main rl --config "$CONFIG_PATH" "${PASSTHROUGH_ARGS[@]}"
elif [[ "$LAUNCHER" == "torchrun" && "$NPROC" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="$NPROC" -m minionerec.cli.main rl --config "$CONFIG_PATH" "${PASSTHROUGH_ARGS[@]}"
else
  exec python -m minionerec.cli.main rl --config "$CONFIG_PATH" "${PASSTHROUGH_ARGS[@]}"
fi
