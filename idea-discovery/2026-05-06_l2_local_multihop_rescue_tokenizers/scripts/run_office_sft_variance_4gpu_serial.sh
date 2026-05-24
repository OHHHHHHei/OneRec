#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
WAIT_SESSION="${2:--}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${#GPU_ARRAY[@]}"

if [[ "$NPROC" -ne 4 ]]; then
  echo "ERROR: Office variance SFT/eval is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
PREPARE_SCRIPT="research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/experiment_mgr_sid_prepare_data.py"
OUT_ROOT="/data/leejt/OneRec/output_weights/experiments/office_variance_sft_eval_20260511"
RESULT_ROOT="results/experiments/office_variance_sft_eval_20260511"
LOG_ROOT="logs/office_variance_sft"
TMP_CONFIG_ROOT="/tmp/onerec-office-variance-20260511"

MAINLINE_VARIANT="office_r690b_lmh_l1w030_l2w010_l3w020"
MAINLINE_INDEX="/data/leejt/OneRec/output_weights/experiments/mgr_sid_office_lmh_main_20260509/generated_indices/Office_Products.${MAINLINE_VARIANT}.index.json"
MAINLINE_DATA_ROOT="data_experiment/Amazon/${MAINLINE_VARIANT}"

mkdir -p "$OUT_ROOT" "$RESULT_ROOT" "$LOG_ROOT" "$TMP_CONFIG_ROOT"

wait_tmux_session() {
  local session="$1"
  if [[ -z "$session" || "$session" == "-" ]]; then
    return
  fi
  echo "[OFFICE-VARIANCE] waiting for tmux session '$session'"
  while tmux has-session -t "$session" 2>/dev/null; do
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$now] [OFFICE-VARIANCE] still waiting for $session"
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
    echo "[$now] [OFFICE-VARIANCE] GPUs $GPU_LIST still busy; wait before next SFT"
    sleep 300
  done
}

prepare_mainline_data() {
  if [[ ! -f "$MAINLINE_INDEX" ]]; then
    echo "ERROR: missing Office mainline tokenizer index: $MAINLINE_INDEX" >&2
    exit 1
  fi
  if [[ ! -f "$MAINLINE_DATA_ROOT/train/Office_Products_5_2016-10-2018-11.csv" ]]; then
    echo "[OFFICE-VARIANCE] prepare data_experiment variant: $MAINLINE_VARIANT"
    python "$PREPARE_SCRIPT" \
      --dataset Office_Products \
      --variant "${MAINLINE_VARIANT}=${MAINLINE_INDEX}"
  fi
}

write_sft_config() {
  local path="$1"
  local train_file="$2"
  local eval_file="$3"
  local sid_index_path="$4"
  local item_meta_path="$5"
  local title_history="$6"
  local desc_align="$7"
  local micro_batch="$8"
  local wandb_name="$9"
  local output_dir="${10}"

  python - "$path" "$train_file" "$eval_file" "$sid_index_path" "$item_meta_path" "$title_history" "$desc_align" "$micro_batch" "$wandb_name" "$output_dir" <<'PY'
import sys
import yaml

(
    path,
    train_file,
    eval_file,
    sid_index_path,
    item_meta_path,
    title_history,
    desc_align,
    micro_batch,
    wandb_name,
    output_dir,
) = sys.argv[1:]

cfg = {
    "model": {"base_model": "../Qwen3-1.7B", "train_from_scratch": False},
    "data": {
        "train_file": train_file,
        "eval_file": eval_file,
        "sid_index_path": sid_index_path,
        "item_meta_path": item_meta_path,
        "category": "Office_Products",
    },
    "training": {
        "batch_size": 1024,
        "micro_batch_size": int(micro_batch),
        "cutoff_len": 512,
        "enable_title_history2sid_dataset": title_history.lower() == "true",
        "enable_title_description_alignment": desc_align.lower() == "true",
        "description_task_probability": 0.5,
        "num_epochs": 10,
        "learning_rate": 0.0003,
        "seed": 42,
        "freeze_llm": False,
        "group_by_length": False,
        "warmup_steps": 20,
        "load_best_model_at_end": True,
        "early_stopping_patience": 3,
        "eval_step": 0.05,
    },
    "logging": {
        "wandb_project": "OneRec",
        "wandb_run_name": wandb_name,
        "report_to": "wandb",
    },
    "output": {"output_dir": output_dir, "save_total_limit": 2},
    "runtime": {
        "launcher": "torchrun",
        "cuda_visible_devices": "2,3,4,5",
        "nproc_per_node": 4,
    },
}

with open(path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
PY
}

write_eval_config() {
  local path="$1"
  local model_path="$2"
  local test_file="$3"
  local info_file="$4"
  local result_path="$5"

  python - "$path" "$model_path" "$test_file" "$info_file" "$result_path" <<'PY'
import sys
import yaml

path, model_path, test_file, info_file, result_path = sys.argv[1:]
cfg = {
    "model": {"base_model": model_path},
    "data": {
        "test_file": test_file,
        "info_file": info_file,
        "category": "Office_Products",
    },
    "training": {"seed": 42},
    "output": {"output_dir": result_path},
    "batch_size": 8,
    "K": 0,
    "num_beams": 50,
    "max_new_tokens": 256,
    "length_penalty": 0.0,
    "temperature": 1.0,
    "guidance_scale": None,
    "runtime": {
        "launcher": "parallel",
        "parallel": True,
        "cuda_visible_devices": "2,3,4,5",
        "nproc_per_node": 4,
    },
}
with open(path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
PY
}

run_one() {
  local label="$1"
  local train_file="$2"
  local valid_file="$3"
  local test_file="$4"
  local sid_index_path="$5"
  local item_meta_path="$6"
  local info_file="$7"
  local title_history="$8"
  local desc_align="$9"
  local micro_batch="${10}"

  local output_dir="$OUT_ROOT/$label/sft"
  local model_path="$output_dir/final_checkpoint"
  local result_path="$RESULT_ROOT/final_result_sft_${label}_Office_Products.json"
  local sft_config="$TMP_CONFIG_ROOT/${label}.sft.yaml"
  local eval_config="$TMP_CONFIG_ROOT/${label}.eval.yaml"
  local wandb_name="sft_${label}_office"

  for required_path in "$train_file" "$valid_file" "$test_file" "$sid_index_path" "$item_meta_path" "$info_file"; do
    if [[ ! -f "$required_path" ]]; then
      echo "ERROR: required Office path missing for $label: $required_path" >&2
      exit 1
    fi
  done

  if [[ -f "$result_path" ]]; then
    echo "[OFFICE-VARIANCE] skip existing result: $result_path"
    return
  fi

  write_sft_config "$sft_config" "$train_file" "$valid_file" "$sid_index_path" "$item_meta_path" "$title_history" "$desc_align" "$micro_batch" "$wandb_name" "$output_dir"
  write_eval_config "$eval_config" "$model_path" "$test_file" "$info_file" "$result_path"

  wait_gpus_free
  echo "[OFFICE-VARIANCE] start SFT: $label"
  echo "[OFFICE-VARIANCE] title_history2sid=$title_history desc_align=$desc_align batch=1024 micro_batch=$micro_batch gpus=$GPU_LIST"
  bash ./sft.sh office "$sft_config" \
    "runtime.cuda_visible_devices=$GPU_LIST" \
    "runtime.nproc_per_node=$NPROC"

  if [[ ! -f "$model_path/config.json" ]]; then
    echo "ERROR: SFT final checkpoint missing for $label: $model_path" >&2
    exit 1
  fi

  wait_gpus_free
  echo "[OFFICE-VARIANCE] start eval: $label"
  bash ./evaluate.sh sft office "$eval_config" \
    "runtime.cuda_visible_devices=$GPU_LIST" \
    "runtime.nproc_per_node=$NPROC"

  if [[ ! -f "$result_path" ]]; then
    echo "ERROR: eval result missing for $label: $result_path" >&2
    exit 1
  fi
  echo "[OFFICE-VARIANCE] completed: $label -> $result_path"
}

wait_tmux_session "$WAIT_SESSION"
prepare_mainline_data

run_one \
  "office_original_semantic_title_off_desc_off_rerun1_4gpu" \
  "./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv" \
  "./data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv" \
  "./data/Amazon/test/Office_Products_5_2016-10-2018-11.csv" \
  "./data/Amazon/index/Office_Products.index.json" \
  "./data/Amazon/index/Office_Products.item.json" \
  "./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt" \
  "false" \
  "false" \
  "4"

run_one \
  "office_r690b_lmh_l1w030_l2w010_l3w020_title_on_desc_p05_rerun1_4gpu" \
  "./${MAINLINE_DATA_ROOT}/train/Office_Products_5_2016-10-2018-11.csv" \
  "./${MAINLINE_DATA_ROOT}/valid/Office_Products_5_2016-10-2018-11.csv" \
  "./${MAINLINE_DATA_ROOT}/test/Office_Products_5_2016-10-2018-11.csv" \
  "./${MAINLINE_DATA_ROOT}/index/Office_Products.index.json" \
  "./${MAINLINE_DATA_ROOT}/index/Office_Products.item.json" \
  "./${MAINLINE_DATA_ROOT}/info/Office_Products_5_2016-10-2018-11.txt" \
  "true" \
  "true" \
  "2"

run_one \
  "office_original_semantic_title_off_desc_off_rerun2_4gpu" \
  "./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv" \
  "./data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv" \
  "./data/Amazon/test/Office_Products_5_2016-10-2018-11.csv" \
  "./data/Amazon/index/Office_Products.index.json" \
  "./data/Amazon/index/Office_Products.item.json" \
  "./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt" \
  "false" \
  "false" \
  "4"

run_one \
  "office_r690b_lmh_l1w030_l2w010_l3w020_title_on_desc_p05_rerun2_4gpu" \
  "./${MAINLINE_DATA_ROOT}/train/Office_Products_5_2016-10-2018-11.csv" \
  "./${MAINLINE_DATA_ROOT}/valid/Office_Products_5_2016-10-2018-11.csv" \
  "./${MAINLINE_DATA_ROOT}/test/Office_Products_5_2016-10-2018-11.csv" \
  "./${MAINLINE_DATA_ROOT}/index/Office_Products.index.json" \
  "./${MAINLINE_DATA_ROOT}/index/Office_Products.item.json" \
  "./${MAINLINE_DATA_ROOT}/info/Office_Products_5_2016-10-2018-11.txt" \
  "true" \
  "true" \
  "2"

echo "[OFFICE-VARIANCE] all Office SFT/eval reruns completed"
