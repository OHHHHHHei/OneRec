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
  echo "ERROR: this baseline rerun chain is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/baseline_variance results/experiments/baseline_variance_sft_eval_20260509

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
VARIANT="original_semantic_title_off_desc_off_mb4_rerun_4gpu"
SFT_CONFIG="$BRANCH_DIR/configs/sft_industrial_${VARIANT}.yaml"
EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_industrial_${VARIANT}.yaml"
SFT_MODEL="/data/leejt/OneRec/output_weights/experiments/baseline_variance_sft_eval_20260509/${VARIANT}/sft/final_checkpoint"
RESULT_PATH="results/experiments/baseline_variance_sft_eval_20260509/final_result_sft_${VARIANT}_Industrial_and_Scientific.json"

for required_path in \
  "$SFT_CONFIG" \
  "$EVAL_CONFIG" \
  "./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "./data/Amazon/index/Industrial_and_Scientific.index.json" \
  "./data/Amazon/index/Industrial_and_Scientific.item.json" \
  "./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: required path missing: $required_path" >&2
    exit 1
  fi
done

echo "[BASELINE-RERUN] recipe=title_history2sid_off + desc_align_off"
echo "[BASELINE-RERUN] original SID=data/Amazon/index/Industrial_and_Scientific.index.json"
echo "[BASELINE-RERUN] effective_batch=1024 micro_batch=4 world_size=$NPROC grad_accum=$((1024 / (4 * NPROC))) seed=42"
echo "[BASELINE-RERUN] start SFT on GPUs $GPU_LIST"
bash ./sft.sh industrial "$SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -d "$SFT_MODEL" ]]; then
  echo "ERROR: SFT final checkpoint missing: $SFT_MODEL" >&2
  exit 1
fi

echo "[BASELINE-RERUN] start evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft industrial "$EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -f "$RESULT_PATH" ]]; then
  echo "ERROR: eval result missing: $RESULT_PATH" >&2
  exit 1
fi

echo "[BASELINE-RERUN] completed: $RESULT_PATH"
