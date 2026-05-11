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
  echo "ERROR: Toys SFT/eval is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/toys_sft_eval_20260510 results/experiments/toys_sft_eval_20260510

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
PREPARE_SCRIPT="scripts/prepare_amazon18_sft_data_from_inter.py"
SOURCE_ROOT="data/Amazon18/Toys_and_Games"
DATASET="Toys_and_Games"
SPLIT_STEM="Toys_and_Games_5_2016-10-2018-11"

BASELINE_VARIANT="toys_baseline_rqvae_onerec_aligned"
MAINLINE_VARIANT="toys_r690b_lmh_l1w030_l2w010_l3w020"

BASELINE_INDEX="/data/leejt/OneRec/output_weights/experiments/toys_tokenizer_20260510/generated_indices/Toys_and_Games.baseline_rqvae_onerec_aligned.index.json"
MAINLINE_INDEX="/data/leejt/OneRec/output_weights/experiments/toys_tokenizer_20260510/generated_indices/Toys_and_Games.r690b_lmh_l1w030_l2w010_l3w020.index.json"

BASELINE_SFT_CONFIG="$BRANCH_DIR/configs/toys/sft_toys_baseline_rqvae_onerec_aligned_title_off_desc_p05_4gpu.yaml"
BASELINE_EVAL_CONFIG="$BRANCH_DIR/configs/toys/evaluate_toys_baseline_rqvae_onerec_aligned_title_off_desc_p05_4gpu.yaml"
MAINLINE_SFT_CONFIG="$BRANCH_DIR/configs/toys/sft_toys_r690b_lmh_l1w030_l2w010_l3w020_title_on_desc_p05_4gpu.yaml"
MAINLINE_EVAL_CONFIG="$BRANCH_DIR/configs/toys/evaluate_toys_r690b_lmh_l1w030_l2w010_l3w020_title_on_desc_p05_4gpu.yaml"

BASELINE_SFT_MODEL="/data/leejt/OneRec/output_weights/experiments/toys_sft_eval_20260510/baseline_rqvae_onerec_aligned_title_off_desc_p05_4gpu/sft/final_checkpoint"
MAINLINE_SFT_MODEL="/data/leejt/OneRec/output_weights/experiments/toys_sft_eval_20260510/r690b_lmh_l1w030_l2w010_l3w020_title_on_desc_p05_4gpu/sft/final_checkpoint"
BASELINE_RESULT="results/experiments/toys_sft_eval_20260510/final_result_sft_toys_baseline_rqvae_onerec_aligned_title_off_desc_p05_4gpu_Toys_and_Games.json"
MAINLINE_RESULT="results/experiments/toys_sft_eval_20260510/final_result_sft_toys_r690b_lmh_l1w030_l2w010_l3w020_title_on_desc_p05_4gpu_Toys_and_Games.json"

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
    echo "[$now] [TOYS-SFT] GPUs $GPU_LIST still busy; wait before starting"
    sleep 300
  done
}

prepare_variant() {
  local variant="$1"
  local index_path="$2"
  local data_root="data_experiment/Amazon/$variant"
  echo "[TOYS-SFT] prepare data_experiment variant=$variant"
  python "$PREPARE_SCRIPT" \
    --source-root "$SOURCE_ROOT" \
    --dataset "$DATASET" \
    --split-stem "$SPLIT_STEM" \
    --index-json "$index_path" \
    --output-root "$data_root"

  for required_path in \
    "$data_root/train/${SPLIT_STEM}.csv" \
    "$data_root/valid/${SPLIT_STEM}.csv" \
    "$data_root/test/${SPLIT_STEM}.csv" \
    "$data_root/index/${DATASET}.index.json" \
    "$data_root/index/${DATASET}.item.json" \
    "$data_root/info/${SPLIT_STEM}.txt"; do
    if [[ ! -f "$required_path" ]]; then
      echo "ERROR: prepared Toys data missing: $required_path" >&2
      exit 1
    fi
  done
}

for required_path in \
  "$PREPARE_SCRIPT" "$BASELINE_INDEX" "$MAINLINE_INDEX" \
  "$BASELINE_SFT_CONFIG" "$BASELINE_EVAL_CONFIG" \
  "$MAINLINE_SFT_CONFIG" "$MAINLINE_EVAL_CONFIG"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: required path missing: $required_path" >&2
    exit 1
  fi
done

prepare_variant "$BASELINE_VARIANT" "$BASELINE_INDEX"
prepare_variant "$MAINLINE_VARIANT" "$MAINLINE_INDEX"
wait_gpus_free

echo "[TOYS-SFT] baseline_recipe=title_history2sid_off + desc_align_p05"
echo "[TOYS-SFT] mainline_recipe=title_history2sid_on + desc_align_p05"
echo "[TOYS-SFT] effective_batch=1024 micro_batch=2 world_size=$NPROC grad_accum=$((1024 / (2 * NPROC))) seed=42"

echo "[TOYS-SFT] start baseline SFT on GPUs $GPU_LIST"
bash ./sft.sh toys "$BASELINE_SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC" 2>&1 | tee logs/toys_sft_eval_20260510/baseline_sft.log

if [[ ! -e "$BASELINE_SFT_MODEL/config.json" ]]; then
  echo "ERROR: baseline SFT final checkpoint missing: $BASELINE_SFT_MODEL" >&2
  exit 1
fi

echo "[TOYS-SFT] start baseline evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft toys "$BASELINE_EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC" 2>&1 | tee logs/toys_sft_eval_20260510/baseline_eval.log

if [[ ! -f "$BASELINE_RESULT" ]]; then
  echo "ERROR: baseline eval result missing: $BASELINE_RESULT" >&2
  exit 1
fi

echo "[TOYS-SFT] start mainline SFT on GPUs $GPU_LIST"
bash ./sft.sh toys "$MAINLINE_SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC" 2>&1 | tee logs/toys_sft_eval_20260510/mainline_sft.log

if [[ ! -e "$MAINLINE_SFT_MODEL/config.json" ]]; then
  echo "ERROR: mainline SFT final checkpoint missing: $MAINLINE_SFT_MODEL" >&2
  exit 1
fi

echo "[TOYS-SFT] start mainline evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft toys "$MAINLINE_EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC" 2>&1 | tee logs/toys_sft_eval_20260510/mainline_eval.log

if [[ ! -f "$MAINLINE_RESULT" ]]; then
  echo "ERROR: mainline eval result missing: $MAINLINE_RESULT" >&2
  exit 1
fi

echo "[TOYS-SFT] completed baseline and mainline SFT/evaluate"
echo "[TOYS-SFT] baseline_result=$BASELINE_RESULT"
echo "[TOYS-SFT] mainline_result=$MAINLINE_RESULT"
