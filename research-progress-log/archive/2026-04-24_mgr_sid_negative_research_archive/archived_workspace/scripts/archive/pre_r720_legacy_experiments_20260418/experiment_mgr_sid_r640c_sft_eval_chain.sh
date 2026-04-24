#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

python /home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py \
  --variant "r640c_seq2graph_rel_masked=/data/leejt/OneRec/output_weights/experiments/mgr_sid_seq2graph_lite_20260416/generated_indices/Industrial_and_Scientific.r640c_seq2graph_rel_masked.index.json"

export CUDA_VISIBLE_DEVICES=2,3,4,5

SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r640c_title_on_desc_p05.yaml"
EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_r640c_title_on_desc_p05.yaml"

SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_r640c_sft_20260417.log"
EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_r640c_eval_20260417.log"

echo "[CHAIN] start R640c data prepare"
echo "[CHAIN] start R640c SFT"
bash /home/leejt/OneRec/sft.sh --config "${SFT_CONFIG}" 2>&1 | tee "${SFT_LOG}"

echo "[CHAIN] start R640c evaluate"
bash /home/leejt/OneRec/evaluate.sh --config "${EVAL_CONFIG}" 2>&1 | tee "${EVAL_LOG}"

echo "[CHAIN] R640c SFT/evaluate completed"
