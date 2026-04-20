#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

python /home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py \
  --variant "collab_ranking_mainline=/data/leejt/OneRec/output_weights/experiments/mgr_sid_r720_l2_ranking_contrastive_20260418/generated_indices/Industrial_and_Scientific.r720a_l2_ranking_contrastive.index.json"

export CUDA_VISIBLE_DEVICES=2,3,4,5

SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_collab_ranking_mainline_title_on_desc_p05.yaml"
EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_collab_ranking_mainline_title_on_desc_p05.yaml"

SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_collab_ranking_mainline_sft_20260419.log"
EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_collab_ranking_mainline_eval_20260419.log"

echo "[CHAIN] start collab-ranking mainline data prepare"
echo "[CHAIN] start collab-ranking mainline 4-GPU SFT"
echo "[CHAIN] recipe=title_history2sid_on + desc_align_p05"
bash /home/leejt/OneRec/sft.sh --config "${SFT_CONFIG}" 2>&1 | tee "${SFT_LOG}"

echo "[CHAIN] start collab-ranking mainline evaluate"
bash /home/leejt/OneRec/evaluate.sh --config "${EVAL_CONFIG}" 2>&1 | tee "${EVAL_LOG}"

echo "[CHAIN] collab-ranking mainline SFT/evaluate completed"
