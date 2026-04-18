#!/usr/bin/env bash
set -euo pipefail

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

cd /home/leejt/OneRec

python /home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py \
  --variant "r650a_seq2graph_mid_pull_push=/data/leejt/OneRec/output_weights/experiments/mgr_sid_seq2graph_push_pull_20260417/generated_indices/Industrial_and_Scientific.r650a_seq2graph_mid_pull_push.index.json"

export CUDA_VISIBLE_DEVICES=2,3,4,5

SFT_CONFIG="/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r650a_title_on_desc_p05.yaml"
SFT_LOG="/home/leejt/OneRec/logs/experiment_mgr_sid_r650a_sft_20260417.log"

echo "[CHAIN] start R650a data prepare"
echo "[CHAIN] start R650a SFT"
bash /home/leejt/OneRec/sft.sh --config "${SFT_CONFIG}" 2>&1 | tee "${SFT_LOG}"

echo "[CHAIN] R650a SFT completed"
