#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
WAIT_SESSION="${2:-toys_sft_baseline_then_mainline_0510}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${#GPU_ARRAY[@]}"

if [[ "$NPROC" -ne 4 ]]; then
  echo "ERROR: alpha050 SFT/eval is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/l2_alpha_sft results/experiments/mgr_sid_l2_alpha_sft_eval_20260511

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
VARIANT="r690b_lmh_l2_alpha050"
INDEX_PATH="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_alpha_20260510/generated_indices/Industrial_and_Scientific.${VARIANT}.index.json"
DATA_ROOT="data_experiment/Amazon/${VARIANT}"
PREPARE_SCRIPT="research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/experiment_mgr_sid_prepare_data.py"

SFT_CONFIG="$BRANCH_DIR/configs/sft_industrial_${VARIANT}_title_on_desc_p05_4gpu.yaml"
EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_industrial_${VARIANT}_title_on_desc_p05_4gpu.yaml"
SFT_MODEL="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_alpha_sft_eval_20260511/${VARIANT}_title_on_desc_p05_4gpu/sft/final_checkpoint"
RESULT_PATH="results/experiments/mgr_sid_l2_alpha_sft_eval_20260511/final_result_sft_mgr_${VARIANT}_title_on_desc_p05_4gpu_Industrial_and_Scientific.json"

wait_for_session_tail() {
  local session="$1"
  if [[ -z "$session" ]]; then
    return
  fi
  while tmux has-session -t "$session" 2>/dev/null; do
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$now] [ALPHA050-SFT] wait for queue session to finish: $session"
    sleep 300
  done
}

wait_gpus_free() {
  local threshold_mib=2500
  local stable_checks=0
  while true; do
    local busy=0
    for gpu in "${GPU_ARRAY[@]}"; do
      gpu="$(echo "$gpu" | xargs)"
      [[ -z "$gpu" ]] && continue
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | awk '{print $1}')"
      if [[ "$used" -gt "$threshold_mib" ]]; then
        busy=1
        break
      fi
    done

    if [[ "$busy" -eq 0 ]]; then
      stable_checks=$((stable_checks + 1))
      if [[ "$stable_checks" -ge 2 ]]; then
        return
      fi
      now="$(date '+%Y-%m-%d %H:%M:%S')"
      echo "[$now] [ALPHA050-SFT] GPUs $GPU_LIST look free; confirm once more before starting"
      sleep 60
      continue
    fi

    stable_checks=0
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$now] [ALPHA050-SFT] GPUs $GPU_LIST still busy; wait before starting"
    sleep 300
  done
}

for required_path in "$INDEX_PATH" "$PREPARE_SCRIPT" "$SFT_CONFIG" "$EVAL_CONFIG"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: required path missing: $required_path" >&2
    exit 1
  fi
done

echo "[ALPHA050-SFT] queued after session=$WAIT_SESSION on GPUs=$GPU_LIST"
wait_for_session_tail "$WAIT_SESSION"
wait_gpus_free

echo "[ALPHA050-SFT] prepare data_experiment variant: $VARIANT"
python "$PREPARE_SCRIPT" --variant "${VARIANT}=${INDEX_PATH}"

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

echo "[ALPHA050-SFT] recipe=title_history2sid_on + desc_align_p05"
echo "[ALPHA050-SFT] effective_batch=1024 micro_batch=2 world_size=$NPROC grad_accum=$((1024 / (2 * NPROC))) seed=42"

echo "[ALPHA050-SFT] start $VARIANT SFT on GPUs $GPU_LIST"
bash ./sft.sh industrial "$SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC" 2>&1 | tee logs/l2_alpha_sft/${VARIANT}_sft_0511.log

if [[ ! -e "$SFT_MODEL/config.json" ]]; then
  echo "ERROR: SFT final checkpoint missing: $SFT_MODEL" >&2
  exit 1
fi

echo "[ALPHA050-SFT] start $VARIANT evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft industrial "$EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC" 2>&1 | tee logs/l2_alpha_sft/${VARIANT}_eval_0511.log

if [[ ! -f "$RESULT_PATH" ]]; then
  echo "ERROR: eval result missing: $RESULT_PATH" >&2
  exit 1
fi

echo "[ALPHA050-SFT] completed $VARIANT SFT/evaluate"
echo "[ALPHA050-SFT] result=$RESULT_PATH"
