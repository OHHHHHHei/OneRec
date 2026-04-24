#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

CONFIG="$ROOT_DIR/config/experiments/sid_train_industrial_mgr_sid_r630c_mid_pull_push.yaml"
LOG="$ROOT_DIR/logs/experiment_mgr_sid_r630c_mid_pull_push_20260416.log"

mkdir -p "$(dirname "$LOG")"

python "$ROOT_DIR/scripts/experiment_mgr_sid_v2_train.py" \
  --config "$CONFIG" 2>&1 | tee "$LOG"
