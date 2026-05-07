#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${#GPU_ARRAY[@]}"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/l2_lmh_sft results/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
INDEX_PATH="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507/generated_indices/Industrial_and_Scientific.v2_lmh_mid_weight001.index.json"
DATA_ROOT="data_experiment/Amazon/v2_lmh_mid_weight001"
PREPARE_SCRIPT="research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/experiment_mgr_sid_prepare_data.py"

SFT_CONFIG="$BRANCH_DIR/configs/sft_industrial_v2_lmh_mid_weight001_title_on_desc_p05_4gpu.yaml"
EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_industrial_v2_lmh_mid_weight001_title_on_desc_p05_4gpu.yaml"

if [[ ! -f "$INDEX_PATH" ]]; then
  echo "ERROR: missing generated index: $INDEX_PATH" >&2
  exit 1
fi
if [[ ! -f "$PREPARE_SCRIPT" ]]; then
  echo "ERROR: missing data prepare script: $PREPARE_SCRIPT" >&2
  exit 1
fi

echo "[CHAIN] prepare data_experiment variant: v2_lmh_mid_weight001"
python "$PREPARE_SCRIPT" \
  --variant "v2_lmh_mid_weight001=$INDEX_PATH"

for required_path in \
  "$DATA_ROOT/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/test/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/index/Industrial_and_Scientific.index.json" \
  "$DATA_ROOT/index/Industrial_and_Scientific.item.json" \
  "$DATA_ROOT/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"; do
  if [[ ! -f "$required_path" ]]; then
    echo "ERROR: prepared data missing: $required_path" >&2
    exit 1
  fi
done

echo "[CHAIN] recipe=title_history2sid_on + desc_align_p05"
echo "[CHAIN] effective_batch=1024 micro_batch=2 world_size=$NPROC grad_accum=$((1024 / (2 * NPROC)))"
echo "[CHAIN] start v2_lmh_mid_weight001 SFT on GPUs $GPU_LIST"
bash ./sft.sh industrial "$SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

echo "[CHAIN] start v2_lmh_mid_weight001 evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft industrial "$EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

echo "[CHAIN] v2_lmh_mid_weight001 SFT/evaluate completed"
