#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

export CUDA_VISIBLE_DEVICES=2,3,4,5

R510_SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_tagcf_r510_title_on_desc_p05.yaml"
R510_EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_tagcf_r510_title_on_desc_p05.yaml"

R510_SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_tagcf_r510_sft_20260415.log"
R510_EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_tagcf_r510_eval_20260415.log"

echo "[CHAIN] start R510 SFT"
bash /home/leejt/OneRec/sft.sh --config "${R510_SFT_CONFIG}" 2>&1 | tee "${R510_SFT_LOG}"

echo "[CHAIN] start R510 evaluate"
bash /home/leejt/OneRec/evaluate.sh --config "${R510_EVAL_CONFIG}" 2>&1 | tee "${R510_EVAL_LOG}"

echo "[CHAIN] R510 SFT/evaluate completed"
