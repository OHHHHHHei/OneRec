#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

python /home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py \
  --variant "r680a_l1_smooth_l2_contrastive_multihop=/data/leejt/OneRec/output_weights/experiments/mgr_sid_r680_l1_smooth_l2_contrastive_multihop_20260418/generated_indices/Industrial_and_Scientific.r680a_l1_smooth_l2_contrastive_multihop.index.json"

export CUDA_VISIBLE_DEVICES=5,7

SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r680a_title_on_desc_p05_2gpu.yaml"
SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_r680a_sft_20260418.log"

echo "[CHAIN] start R680a data prepare"
echo "[CHAIN] start R680a 2-GPU SFT"
echo "[CHAIN] effective batch aligned to 4-GPU baseline via batch_size=1024, micro_batch_size=2, nproc_per_node=2"
bash /home/leejt/OneRec/sft.sh --config "${SFT_CONFIG}" 2>&1 | tee "${SFT_LOG}"

echo "[CHAIN] R680a SFT completed"
