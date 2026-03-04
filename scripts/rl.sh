#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_CONFIG="configs/stages/rl/default.yaml"
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
PY
)

LAUNCHER="${_launch[0]:-accelerate}"
GPU_LIST="${_launch[1]:-}"
NPROC="${_launch[2]:-1}"
ACC_CONFIG="${_launch[3]:-config/zero2_opt.yaml}"
MAIN_PORT="${_launch[4]:-29503}"
HF_ENDPOINT_CFG="${_launch[5]:-}"

if [[ -n "$GPU_LIST" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi

if [[ "$LAUNCHER" == "accelerate" && -n "$HF_ENDPOINT_CFG" ]]; then
  export HF_ENDPOINT="$HF_ENDPOINT_CFG"
fi

if [[ "$LAUNCHER" == "torchrun" && -n "$GPU_LIST" ]]; then
  IFS=',' read -r -a _gpu_arr <<< "$GPU_LIST"
  GPU_COUNT="${#_gpu_arr[@]}"
  if [[ "$NPROC" -gt "$GPU_COUNT" ]]; then
    echo "ERROR: nproc_per_node ($NPROC) > gpu_count ($GPU_COUNT) from runtime.cuda_visible_devices=$GPU_LIST" >&2
    exit 1
  fi
fi

echo "[RL] launcher=$LAUNCHER gpus=${GPU_LIST:-<default>} nproc_per_node=$NPROC config=$CONFIG_PATH"

if [[ "$LAUNCHER" == "accelerate" ]]; then
  exec accelerate launch \
    --config_file "$ACC_CONFIG" \
    --num_processes "$NPROC" \
    --main_process_port "$MAIN_PORT" \
    -m minionerec.cli.main rl --config "$CONFIG_PATH" "$@"
elif [[ "$LAUNCHER" == "torchrun" && "$NPROC" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="$NPROC" -m minionerec.cli.main rl --config "$CONFIG_PATH" "$@"
else
  exec python -m minionerec.cli.main rl --config "$CONFIG_PATH" "$@"
fi
