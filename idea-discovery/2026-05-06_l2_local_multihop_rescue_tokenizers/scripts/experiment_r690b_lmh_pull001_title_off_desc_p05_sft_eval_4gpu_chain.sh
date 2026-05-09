#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
WAIT_SESSION="${2:-mgr_office_r690b_lmh_sft_eval_0509}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${#GPU_ARRAY[@]}"

if [[ "$NPROC" -ne 4 ]]; then
  echo "ERROR: title-off SFT/eval is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/l2_lmh_sft results/experiments/mgr_sid_l2_lmh_recipe_ablation_20260509

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
PREPARE_SCRIPT="research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/experiment_mgr_sid_prepare_data.py"
VARIANT="r690b_lmh_l2_contrastive_pull_weight001"
RECIPE_TAG="${VARIANT}_title_off_desc_p05_4gpu"
INDEX_PATH="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507/generated_indices/Industrial_and_Scientific.${VARIANT}.index.json"
DATA_ROOT="data_experiment/Amazon/${VARIANT}"
SFT_CONFIG="$BRANCH_DIR/configs/sft_industrial_${RECIPE_TAG}.yaml"
EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_industrial_${RECIPE_TAG}.yaml"
SFT_MODEL="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_recipe_ablation_20260509/${RECIPE_TAG}/sft/final_checkpoint"
EVAL_RESULT="results/experiments/mgr_sid_l2_lmh_recipe_ablation_20260509/final_result_sft_mgr_${RECIPE_TAG}_Industrial_and_Scientific.json"

wait_tmux_session() {
  local session="$1"
  if [[ -z "$session" || "$session" == "-" ]]; then
    return
  fi
  echo "[TITLE-OFF-SFT] waiting for tmux session '$session'"
  while tmux has-session -t "$session" 2>/dev/null; do
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$now] [TITLE-OFF-SFT] still waiting for $session"
    sleep 300
  done
}

wait_gpus_free() {
  local threshold_mib=2500
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
      return
    fi
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$now] [TITLE-OFF-SFT] GPUs $GPU_LIST still busy; wait before starting title-off SFT"
    sleep 300
  done
}

wait_tmux_session "$WAIT_SESSION"
wait_gpus_free

for required_path in "$INDEX_PATH" "$PREPARE_SCRIPT" "$SFT_CONFIG" "$EVAL_CONFIG"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: required path missing: $required_path" >&2
    exit 1
  fi
done

if [[ ! -f "$DATA_ROOT/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" ]]; then
  echo "[TITLE-OFF-SFT] prepare data_experiment variant: $VARIANT"
  python "$PREPARE_SCRIPT" \
    --variant "${VARIANT}=${INDEX_PATH}"
else
  echo "[TITLE-OFF-SFT] data_experiment variant already exists: $DATA_ROOT"
fi

for required_path in \
  "$DATA_ROOT/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/test/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/index/Industrial_and_Scientific.index.json" \
  "$DATA_ROOT/index/Industrial_and_Scientific.item.json" \
  "$DATA_ROOT/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"; do
  if [[ ! -f "$required_path" ]]; then
    echo "ERROR: prepared Industrial data missing: $required_path" >&2
    exit 1
  fi
done

echo "[TITLE-OFF-SFT] recipe=title_history2sid_off + desc_align_p05"
echo "[TITLE-OFF-SFT] effective_batch=1024 micro_batch=2 world_size=$NPROC grad_accum=$((1024 / (2 * NPROC)))"
echo "[TITLE-OFF-SFT] start SFT on GPUs $GPU_LIST"
bash ./sft.sh industrial "$SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -e "$SFT_MODEL/config.json" ]]; then
  echo "ERROR: SFT final checkpoint missing: $SFT_MODEL" >&2
  exit 1
fi

echo "[TITLE-OFF-SFT] start evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft industrial "$EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -f "$EVAL_RESULT" ]]; then
  echo "ERROR: SFT eval result missing: $EVAL_RESULT" >&2
  exit 1
fi

echo "[TITLE-OFF-SFT] completed SFT/evaluate: $EVAL_RESULT"
