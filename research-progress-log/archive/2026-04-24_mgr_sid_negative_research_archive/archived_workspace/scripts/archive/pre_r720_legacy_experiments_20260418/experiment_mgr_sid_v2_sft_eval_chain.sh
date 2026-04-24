#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

export CUDA_VISIBLE_DEVICES=2,3,4,5

SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_tokenizer_v2_offline.yaml"
EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_tokenizer_v2_offline.yaml"
SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_v2_sft_industrial_20260411.log"
EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_v2_eval_industrial_20260411.log"

bash /home/leejt/OneRec/sft.sh --config "${SFT_CONFIG}" 2>&1 | tee "${SFT_LOG}"
bash /home/leejt/OneRec/evaluate.sh --config "${EVAL_CONFIG}" 2>&1 | tee "${EVAL_LOG}"
