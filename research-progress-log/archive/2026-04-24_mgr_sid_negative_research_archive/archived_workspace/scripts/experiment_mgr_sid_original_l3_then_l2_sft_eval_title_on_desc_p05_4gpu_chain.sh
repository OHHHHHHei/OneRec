#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

python /home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py \
  --variant "original_l3_collab_local=/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l3_collab_local_20260420/generated_indices/Industrial_and_Scientific.original_l3_collab_local.index.json" \
  --variant "original_l2_multihop_ranking=/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l2_multihop_ranking_20260421/generated_indices/Industrial_and_Scientific.original_l2_multihop_ranking.index.json"

export CUDA_VISIBLE_DEVICES=2,3,4,5

L3_SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_original_l3_collab_local_title_on_desc_p05_4gpu.yaml"
L3_EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_original_l3_collab_local_title_on_desc_p05_4gpu.yaml"
L3_SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_original_l3_collab_local_title_on_desc_p05_4gpu_sft_20260421.log"
L3_EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_original_l3_collab_local_title_on_desc_p05_4gpu_eval_20260421.log"

L2_SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_original_l2_multihop_ranking_title_on_desc_p05_4gpu.yaml"
L2_EVAL_CONFIG="/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_original_l2_multihop_ranking_title_on_desc_p05_4gpu.yaml"
L2_SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_original_l2_multihop_ranking_title_on_desc_p05_4gpu_sft_20260421.log"
L2_EVAL_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_original_l2_multihop_ranking_title_on_desc_p05_4gpu_eval_20260421.log"

echo "[CHAIN] prepared data_experiment variants: original_l3_collab_local, original_l2_multihop_ranking"
echo "[CHAIN] recipe=title_history2sid_on + desc_align_p05"
echo "[CHAIN] effective_batch=1024 micro_batch=2 world_size=4 grad_accum=128"

echo "[CHAIN] start original_l3_collab_local SFT"
bash /home/leejt/OneRec/sft.sh --config "${L3_SFT_CONFIG}" 2>&1 | tee "${L3_SFT_LOG}"

echo "[CHAIN] start original_l3_collab_local evaluate"
bash /home/leejt/OneRec/evaluate.sh --config "${L3_EVAL_CONFIG}" 2>&1 | tee "${L3_EVAL_LOG}"

echo "[CHAIN] start original_l2_multihop_ranking SFT"
bash /home/leejt/OneRec/sft.sh --config "${L2_SFT_CONFIG}" 2>&1 | tee "${L2_SFT_LOG}"

echo "[CHAIN] start original_l2_multihop_ranking evaluate"
bash /home/leejt/OneRec/evaluate.sh --config "${L2_EVAL_CONFIG}" 2>&1 | tee "${L2_EVAL_LOG}"

echo "[CHAIN] original_l3 -> original_l2 SFT/evaluate completed"
