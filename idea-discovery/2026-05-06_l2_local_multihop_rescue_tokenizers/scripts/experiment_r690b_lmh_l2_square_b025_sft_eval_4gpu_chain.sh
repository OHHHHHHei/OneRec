#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${#GPU_ARRAY[@]}"

if [[ "$NPROC" -ne 4 ]]; then
  echo "ERROR: this SFT/eval chain is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/l2_square_sft results/experiments/mgr_sid_l2_square_sft_eval_20260509

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
VARIANT="r690b_lmh_l2_square_dominant_b025"
INDEX_PATH="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_square_20260509/generated_indices/Industrial_and_Scientific.${VARIANT}.index.json"
DATA_ROOT="data_experiment/Amazon/${VARIANT}"
PREPARE_SCRIPT="research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/experiment_mgr_sid_prepare_data.py"

SFT_CONFIG="$BRANCH_DIR/configs/sft_industrial_${VARIANT}_title_on_desc_p05_4gpu.yaml"
EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_industrial_${VARIANT}_title_on_desc_p05_4gpu.yaml"
SFT_MODEL="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_square_sft_eval_20260509/${VARIANT}_title_on_desc_p05_4gpu/sft/final_checkpoint"
RESULT_PATH="results/experiments/mgr_sid_l2_square_sft_eval_20260509/final_result_sft_mgr_${VARIANT}_title_on_desc_p05_4gpu_Industrial_and_Scientific.json"

for required_path in "$INDEX_PATH" "$PREPARE_SCRIPT" "$SFT_CONFIG" "$EVAL_CONFIG"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: required path missing: $required_path" >&2
    exit 1
  fi
done

echo "[L2-SQUARE-SFT] prepare data_experiment variant: $VARIANT"
python "$PREPARE_SCRIPT" \
  --variant "${VARIANT}=${INDEX_PATH}"

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

echo "[L2-SQUARE-SFT] recipe=title_history2sid_on + desc_align_p05"
echo "[L2-SQUARE-SFT] effective_batch=1024 micro_batch=2 world_size=$NPROC grad_accum=$((1024 / (2 * NPROC)))"
echo "[L2-SQUARE-SFT] start ${VARIANT} SFT on GPUs $GPU_LIST"
bash ./sft.sh industrial "$SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -d "$SFT_MODEL" ]]; then
  echo "ERROR: SFT final checkpoint missing: $SFT_MODEL" >&2
  exit 1
fi

echo "[L2-SQUARE-SFT] start ${VARIANT} evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft industrial "$EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -f "$RESULT_PATH" ]]; then
  echo "ERROR: eval result missing: $RESULT_PATH" >&2
  exit 1
fi

echo "[L2-SQUARE-SFT] ${VARIANT} SFT/evaluate completed: $RESULT_PATH"
