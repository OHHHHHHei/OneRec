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
  echo "ERROR: this mainline rerun chain is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/mainline_variance results/experiments/mainline_variance_sft_eval_20260509

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
VARIANT="r690b_lmh_l2_contrastive_pull_weight001"
RERUN_VARIANT="${VARIANT}_rerun_4gpu"
INDEX_PATH="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507/generated_indices/Industrial_and_Scientific.${VARIANT}.index.json"
DATA_ROOT="data_experiment/Amazon/${VARIANT}"
PREPARE_SCRIPT="research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/experiment_mgr_sid_prepare_data.py"

SFT_CONFIG="$BRANCH_DIR/configs/sft_industrial_${RERUN_VARIANT}.yaml"
EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_industrial_${RERUN_VARIANT}.yaml"
SFT_MODEL="/data/leejt/OneRec/output_weights/experiments/mainline_variance_sft_eval_20260509/${RERUN_VARIANT}/sft/final_checkpoint"
RESULT_PATH="results/experiments/mainline_variance_sft_eval_20260509/final_result_sft_mgr_${RERUN_VARIANT}_Industrial_and_Scientific.json"

WAIT_RESULT="${WAIT_RESULT:-results/experiments/baseline_variance_sft_eval_20260509/final_result_sft_original_semantic_title_off_desc_p05_rerun_4gpu_Industrial_and_Scientific.json}"
WAIT_SESSION="${WAIT_SESSION:-strong_baseline_rerun_after_l2square_0509}"

echo "[MAINLINE-RERUN] waiting for upstream result: $WAIT_RESULT"
while [[ ! -f "$WAIT_RESULT" ]]; do
  if ! tmux has-session -t "$WAIT_SESSION" 2>/dev/null; then
    echo "ERROR: upstream tmux session ended before result appeared: $WAIT_SESSION" >&2
    exit 1
  fi
  sleep 300
done
echo "[MAINLINE-RERUN] upstream result found; starting mainline rerun"

for required_path in \
  "$INDEX_PATH" \
  "$PREPARE_SCRIPT" \
  "$SFT_CONFIG" \
  "$EVAL_CONFIG"; do
  if [[ ! -f "$required_path" ]]; then
    echo "ERROR: required path missing: $required_path" >&2
    exit 1
  fi
done

echo "[MAINLINE-RERUN] prepare data_experiment variant: $VARIANT"
python "$PREPARE_SCRIPT" \
  --variant "${VARIANT}=$INDEX_PATH"

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

echo "[MAINLINE-RERUN] recipe=title_history2sid_on + desc_align_p05"
echo "[MAINLINE-RERUN] tokenizer=$VARIANT"
echo "[MAINLINE-RERUN] effective_batch=1024 micro_batch=2 world_size=$NPROC grad_accum=$((1024 / (2 * NPROC))) seed=42"
echo "[MAINLINE-RERUN] start SFT on GPUs $GPU_LIST"
bash ./sft.sh industrial "$SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -d "$SFT_MODEL" ]]; then
  echo "ERROR: SFT final checkpoint missing: $SFT_MODEL" >&2
  exit 1
fi

echo "[MAINLINE-RERUN] start evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft industrial "$EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -f "$RESULT_PATH" ]]; then
  echo "ERROR: eval result missing: $RESULT_PATH" >&2
  exit 1
fi

echo "[MAINLINE-RERUN] completed: $RESULT_PATH"
