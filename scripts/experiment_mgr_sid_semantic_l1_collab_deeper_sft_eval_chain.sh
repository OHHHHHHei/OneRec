#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

python /home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py \
  --variant "semantic_l1_collab_deeper=/data/leejt/OneRec/output_weights/experiments/mgr_sid_semantic_l1_collab_deeper_20260420/generated_indices/Industrial_and_Scientific.semantic_l1_collab_deeper.index.json"

export CUDA_VISIBLE_DEVICES=2,3,4,5

SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_semantic_l1_collab_deeper_title_on_desc_p05_4gpu.yaml"
EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_semantic_l1_collab_deeper_title_on_desc_p05_4gpu.yaml"

SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_semantic_l1_collab_deeper_sft_20260420.log"
EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_semantic_l1_collab_deeper_eval_20260420.log"

echo "[CHAIN] start semantic-L1 collab-deeper data prepare"
echo "[CHAIN] start semantic-L1 collab-deeper 4-GPU SFT"
echo "[CHAIN] recipe=title_history2sid_on + desc_align_p05"
echo "[CHAIN] effective_batch=1024 micro_batch=2 world_size=4 grad_accum=128"
bash /home/leejt/OneRec/sft.sh --config "${SFT_CONFIG}" 2>&1 | tee "${SFT_LOG}"

echo "[CHAIN] start semantic-L1 collab-deeper evaluate"
bash /home/leejt/OneRec/evaluate.sh --config "${EVAL_CONFIG}" 2>&1 | tee "${EVAL_LOG}"

echo "[CHAIN] semantic-L1 collab-deeper SFT/evaluate completed"
