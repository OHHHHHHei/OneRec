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
  echo "ERROR: variance rerun2 chain is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/variance_rerun2 results/experiments/variance_rerun2_sft_eval_20260510

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
PREPARE_SCRIPT="research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/experiment_mgr_sid_prepare_data.py"
MAIN_VARIANT="r690b_lmh_l2_contrastive_pull_weight001"
MAIN_INDEX_PATH="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507/generated_indices/Industrial_and_Scientific.${MAIN_VARIANT}.index.json"
MAIN_DATA_ROOT="data_experiment/Amazon/${MAIN_VARIANT}"

WAIT_RESULT="${WAIT_RESULT:-results/experiments/mainline_variance_sft_eval_20260509/final_result_sft_mgr_r690b_lmh_l2_contrastive_pull_weight001_rerun_4gpu_Industrial_and_Scientific.json}"
WAIT_SESSION="${WAIT_SESSION:-mainline_rerun_after_strong_baseline_0509}"

BASELINE_SFT_CONFIG="$BRANCH_DIR/configs/sft_industrial_original_semantic_title_off_desc_p05_rerun2_4gpu.yaml"
BASELINE_EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_industrial_original_semantic_title_off_desc_p05_rerun2_4gpu.yaml"
BASELINE_MODEL="/data/leejt/OneRec/output_weights/experiments/variance_rerun2_sft_eval_20260510/original_semantic_title_off_desc_p05_rerun2_4gpu/sft/final_checkpoint"
BASELINE_RESULT="results/experiments/variance_rerun2_sft_eval_20260510/final_result_sft_original_semantic_title_off_desc_p05_rerun2_4gpu_Industrial_and_Scientific.json"

MAIN_SFT_CONFIG="$BRANCH_DIR/configs/sft_industrial_r690b_lmh_l2_contrastive_pull_weight001_rerun2_4gpu.yaml"
MAIN_EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_industrial_r690b_lmh_l2_contrastive_pull_weight001_rerun2_4gpu.yaml"
MAIN_MODEL="/data/leejt/OneRec/output_weights/experiments/variance_rerun2_sft_eval_20260510/r690b_lmh_l2_contrastive_pull_weight001_rerun2_4gpu/sft/final_checkpoint"
MAIN_RESULT="results/experiments/variance_rerun2_sft_eval_20260510/final_result_sft_mgr_r690b_lmh_l2_contrastive_pull_weight001_rerun2_4gpu_Industrial_and_Scientific.json"

echo "[VARIANCE-RERUN2] waiting for current mainline rerun result: $WAIT_RESULT"
while [[ ! -f "$WAIT_RESULT" ]]; do
  if ! tmux has-session -t "$WAIT_SESSION" 2>/dev/null; then
    echo "ERROR: upstream tmux session ended before result appeared: $WAIT_SESSION" >&2
    exit 1
  fi
  sleep 300
done
echo "[VARIANCE-RERUN2] upstream result found; starting paired rerun2"

for required_path in \
  "$BASELINE_SFT_CONFIG" \
  "$BASELINE_EVAL_CONFIG" \
  "$MAIN_SFT_CONFIG" \
  "$MAIN_EVAL_CONFIG" \
  "$MAIN_INDEX_PATH" \
  "$PREPARE_SCRIPT" \
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

echo "[VARIANCE-RERUN2] start strong baseline rerun2 on GPUs $GPU_LIST"
bash ./sft.sh industrial "$BASELINE_SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -d "$BASELINE_MODEL" ]]; then
  echo "ERROR: baseline final checkpoint missing: $BASELINE_MODEL" >&2
  exit 1
fi

echo "[VARIANCE-RERUN2] start strong baseline rerun2 evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft industrial "$BASELINE_EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -f "$BASELINE_RESULT" ]]; then
  echo "ERROR: baseline eval result missing: $BASELINE_RESULT" >&2
  exit 1
fi

echo "[VARIANCE-RERUN2] prepare mainline data_experiment variant: $MAIN_VARIANT"
python "$PREPARE_SCRIPT" \
  --variant "${MAIN_VARIANT}=$MAIN_INDEX_PATH"

for required_path in \
  "$MAIN_DATA_ROOT/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$MAIN_DATA_ROOT/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$MAIN_DATA_ROOT/test/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$MAIN_DATA_ROOT/index/Industrial_and_Scientific.index.json" \
  "$MAIN_DATA_ROOT/index/Industrial_and_Scientific.item.json" \
  "$MAIN_DATA_ROOT/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"; do
  if [[ ! -f "$required_path" ]]; then
    echo "ERROR: prepared mainline data missing: $required_path" >&2
    exit 1
  fi
done

echo "[VARIANCE-RERUN2] start mainline rerun2 on GPUs $GPU_LIST"
bash ./sft.sh industrial "$MAIN_SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -d "$MAIN_MODEL" ]]; then
  echo "ERROR: mainline final checkpoint missing: $MAIN_MODEL" >&2
  exit 1
fi

echo "[VARIANCE-RERUN2] start mainline rerun2 evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft industrial "$MAIN_EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -f "$MAIN_RESULT" ]]; then
  echo "ERROR: mainline eval result missing: $MAIN_RESULT" >&2
  exit 1
fi

echo "[VARIANCE-RERUN2] completed paired rerun2"
echo "[VARIANCE-RERUN2] baseline result: $BASELINE_RESULT"
echo "[VARIANCE-RERUN2] mainline result: $MAIN_RESULT"
