#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

export CUDA_VISIBLE_DEVICES=2,3,4,5

R401B_SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_stage3_r401b_title_on_desc_p05.yaml"
R401B_EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_stage3_r401b_title_on_desc_p05.yaml"
R401D_SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_stage3_r401d_title_on_desc_p05.yaml"
R401D_EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_stage3_r401d_title_on_desc_p05.yaml"

R401B_SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401b_sft_20260414.log"
R401B_EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401b_eval_20260414.log"
R401D_SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401d_sft_20260414.log"
R401D_EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401d_eval_20260414.log"

echo "[CHAIN] start R401b SFT"
bash /home/leejt/OneRec/sft.sh --config "${R401B_SFT_CONFIG}" 2>&1 | tee "${R401B_SFT_LOG}"

echo "[CHAIN] start R401b evaluate"
bash /home/leejt/OneRec/evaluate.sh --config "${R401B_EVAL_CONFIG}" 2>&1 | tee "${R401B_EVAL_LOG}"

echo "[CHAIN] start R401d SFT"
bash /home/leejt/OneRec/sft.sh --config "${R401D_SFT_CONFIG}" 2>&1 | tee "${R401D_SFT_LOG}"

echo "[CHAIN] start R401d evaluate"
bash /home/leejt/OneRec/evaluate.sh --config "${R401D_EVAL_CONFIG}" 2>&1 | tee "${R401D_EVAL_LOG}"

echo "[CHAIN] stage-3 SFT/eval chain completed"
