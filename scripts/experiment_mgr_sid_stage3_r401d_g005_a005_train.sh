#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

CONFIG="/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage3_r401d_g005_a005.yaml"
LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401d_g005_a005_20260414.log"

python /home/leejt/OneRec/scripts/experiment_mgr_sid_v2_train.py --config "${CONFIG}" 2>&1 | tee "${LOG}"
