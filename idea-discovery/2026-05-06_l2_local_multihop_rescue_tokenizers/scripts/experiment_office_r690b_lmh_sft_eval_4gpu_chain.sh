#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
WAIT_SESSION="${2:-mgr_r690b_lmh_l1_w040_sft_eval_0508}"
TOKENIZER_SESSION="${3:-mgr_office_r690b_lmh_tok_0509}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${#GPU_ARRAY[@]}"

if [[ "$NPROC" -ne 4 ]]; then
  echo "ERROR: Office SFT/eval is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/office_lmh_sft results/experiments/mgr_sid_office_lmh_sft_eval_20260509

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
PREPARE_SCRIPT="research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/experiment_mgr_sid_prepare_data.py"
VARIANT="office_r690b_lmh_l1w030_l2w010_l3w020"
INDEX_PATH="/data/leejt/OneRec/output_weights/experiments/mgr_sid_office_lmh_main_20260509/generated_indices/Office_Products.${VARIANT}.index.json"
DATA_ROOT="data_experiment/Amazon/${VARIANT}"
SFT_CONFIG="$BRANCH_DIR/configs/sft_office_${VARIANT}_title_on_desc_p05_4gpu.yaml"
EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_office_${VARIANT}_title_on_desc_p05_4gpu.yaml"
SFT_MODEL="/data/leejt/OneRec/output_weights/experiments/mgr_sid_office_lmh_sft_eval_20260509/${VARIANT}_title_on_desc_p05_4gpu/sft/final_checkpoint"
EVAL_RESULT="results/experiments/mgr_sid_office_lmh_sft_eval_20260509/final_result_sft_mgr_${VARIANT}_title_on_desc_p05_4gpu_Office_Products.json"

wait_tmux_session() {
  local session="$1"
  local label="$2"
  if [[ -z "$session" || "$session" == "-" ]]; then
    return
  fi
  echo "[OFFICE-SFT] waiting for $label tmux session '$session'"
  while tmux has-session -t "$session" 2>/dev/null; do
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$now] [OFFICE-SFT] still waiting for $session"
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
    echo "[$now] [OFFICE-SFT] GPUs $GPU_LIST still busy; wait before starting Office SFT"
    sleep 300
  done
}

wait_tmux_session "$TOKENIZER_SESSION" "Office tokenizer"

if [[ ! -f "$INDEX_PATH" ]]; then
  echo "ERROR: Office tokenizer finished/went away but index is missing: $INDEX_PATH" >&2
  exit 1
fi

wait_tmux_session "$WAIT_SESSION" "current 4-GPU"
wait_gpus_free

for required_path in "$INDEX_PATH" "$PREPARE_SCRIPT" "$SFT_CONFIG" "$EVAL_CONFIG"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: required path missing: $required_path" >&2
    exit 1
  fi
done

echo "[OFFICE-SFT] prepare Office data_experiment variant: $VARIANT"
python "$PREPARE_SCRIPT" \
  --dataset Office_Products \
  --variant "${VARIANT}=${INDEX_PATH}"

for required_path in \
  "$DATA_ROOT/train/Office_Products_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/valid/Office_Products_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/test/Office_Products_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/index/Office_Products.index.json" \
  "$DATA_ROOT/index/Office_Products.item.json" \
  "$DATA_ROOT/info/Office_Products_5_2016-10-2018-11.txt"; do
  if [[ ! -f "$required_path" ]]; then
    echo "ERROR: prepared Office data missing: $required_path" >&2
    exit 1
  fi
done

echo "[OFFICE-SFT] recipe=title_history2sid_on + desc_align_p05"
echo "[OFFICE-SFT] effective_batch=1024 micro_batch=2 world_size=$NPROC grad_accum=$((1024 / (2 * NPROC)))"
echo "[OFFICE-SFT] start Office SFT on GPUs $GPU_LIST"
bash ./sft.sh office "$SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -e "$SFT_MODEL/config.json" ]]; then
  echo "ERROR: SFT final checkpoint missing: $SFT_MODEL" >&2
  exit 1
fi

echo "[OFFICE-SFT] start Office evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft office "$EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -f "$EVAL_RESULT" ]]; then
  echo "ERROR: Office SFT eval result missing: $EVAL_RESULT" >&2
  exit 1
fi

echo "[OFFICE-SFT] completed Office SFT/evaluate: $EVAL_RESULT"
