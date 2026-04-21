#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

python /home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py \
  --variant "qcr_l2_conflict_ranking=/data/leejt/OneRec/output_weights/experiments/mgr_sid_qcr_l2_conflict_ranking_20260421/generated_indices/Industrial_and_Scientific.qcr_l2_conflict_ranking.index.json"

export CUDA_VISIBLE_DEVICES=2,3,4,5

SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_qcr_l2_conflict_ranking_title_on_desc_p05_4gpu.yaml"
EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_qcr_l2_conflict_ranking_title_on_desc_p05_4gpu.yaml"
SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_qcr_l2_conflict_ranking_title_on_desc_p05_4gpu_sft_20260421.log"
EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_qcr_l2_conflict_ranking_title_on_desc_p05_4gpu_eval_20260421.log"

echo "[CHAIN] prepared data_experiment variant: qcr_l2_conflict_ranking"
echo "[CHAIN] recipe=title_history2sid_on + desc_align_p05"
echo "[CHAIN] effective_batch=1024 micro_batch=2 world_size=4 grad_accum=128"

echo "[CHAIN] start qcr_l2_conflict_ranking SFT"
bash /home/leejt/OneRec/sft.sh --config "${SFT_CONFIG}" 2>&1 | tee "${SFT_LOG}"

echo "[CHAIN] start qcr_l2_conflict_ranking evaluate"
bash /home/leejt/OneRec/evaluate.sh --config "${EVAL_CONFIG}" 2>&1 | tee "${EVAL_LOG}"

echo "[CHAIN] qcr_l2_conflict_ranking SFT/evaluate completed"
