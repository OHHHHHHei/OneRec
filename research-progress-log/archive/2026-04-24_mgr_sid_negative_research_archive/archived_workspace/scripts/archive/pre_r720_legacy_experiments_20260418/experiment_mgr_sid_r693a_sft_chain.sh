#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

python /home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py \
  --variant "r693a_hier_collab_only_multihop=/data/leejt/OneRec/output_weights/experiments/mgr_sid_r693_hier_collab_only_20260418/generated_indices/Industrial_and_Scientific.r693a_hier_collab_only_multihop.index.json"

export CUDA_VISIBLE_DEVICES=3,4,5,7

SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r693a_title_on_desc_p05.yaml"
SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_r693a_sft_20260418.log"

echo "[CHAIN] start R693a data prepare"
echo "[CHAIN] start R693a 4-GPU SFT"
echo "[CHAIN] recipe=title_history2sid_on + desc_align_p05"
bash /home/leejt/OneRec/sft.sh --config "${SFT_CONFIG}" 2>&1 | tee "${SFT_LOG}"

echo "[CHAIN] R693a SFT completed"
