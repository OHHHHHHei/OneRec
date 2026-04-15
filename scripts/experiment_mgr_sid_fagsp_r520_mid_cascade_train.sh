#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

python scripts/experiment_mgr_sid_v2_train.py \
  --config "$ROOT_DIR/config/experiments/sid_train_industrial_mgr_sid_fagsp_r520_mid_cascade.yaml"
