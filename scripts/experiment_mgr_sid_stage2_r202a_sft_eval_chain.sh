#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

export CUDA_VISIBLE_DEVICES=2,3,4,5

SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_stage2_r202a_title_on_desc_p05.yaml"
EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_stage2_r202a_title_on_desc_p05.yaml"
SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_stage2_r202a_sft_20260413.log"
EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_stage2_r202a_eval_20260413.log"

bash /home/leejt/OneRec/sft.sh --config "${SFT_CONFIG}" 2>&1 | tee "${SFT_LOG}"
bash /home/leejt/OneRec/evaluate.sh --config "${EVAL_CONFIG}" 2>&1 | tee "${EVAL_LOG}"
