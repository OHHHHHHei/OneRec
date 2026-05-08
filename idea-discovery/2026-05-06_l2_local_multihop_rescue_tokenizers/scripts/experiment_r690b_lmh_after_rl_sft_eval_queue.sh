#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
WAIT_SESSION="${2:-mgr_r690b_lmh_pull001_rl_eval_0507}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${#GPU_ARRAY[@]}"

if [[ "$NPROC" -ne 4 ]]; then
  echo "ERROR: this SFT/eval queue is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/l2_lmh_sft results/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
PREPARE_SCRIPT="research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/experiment_mgr_sid_prepare_data.py"
RL_EVAL_RESULT="results/experiments/mgr_sid_l2_lmh_sweep_rl_eval_20260507/final_result_rl_mgr_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu_Industrial_and_Scientific.json"

echo "[AFTER-RL-SFT] waiting for tmux session '$WAIT_SESSION' to finish RL/eval before using GPUs $GPU_LIST"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$now] [AFTER-RL-SFT] still waiting for $WAIT_SESSION"
  sleep 300
done

if [[ ! -f "$RL_EVAL_RESULT" ]]; then
  echo "ERROR: waited for RL session '$WAIT_SESSION', but RL eval result is missing: $RL_EVAL_RESULT" >&2
  echo "ERROR: refusing to start SFT queue so RL failure is not silently skipped." >&2
  exit 1
fi

echo "[AFTER-RL-SFT] detected RL/eval completion: $RL_EVAL_RESULT"

run_one_variant() {
  local tag="$1"
  local index_path="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507/generated_indices/Industrial_and_Scientific.${tag}.index.json"
  local data_root="data_experiment/Amazon/${tag}"
  local sft_config="$BRANCH_DIR/configs/sft_industrial_${tag}_title_on_desc_p05_4gpu.yaml"
  local eval_config="$BRANCH_DIR/configs/evaluate_industrial_${tag}_title_on_desc_p05_4gpu.yaml"
  local sft_model="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507/${tag}_title_on_desc_p05_4gpu/sft/final_checkpoint"
  local eval_result="results/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507/final_result_sft_mgr_${tag}_title_on_desc_p05_4gpu_Industrial_and_Scientific.json"

  echo "============================================"
  echo "[AFTER-RL-SFT] variant=$tag"
  echo "[AFTER-RL-SFT] index=$index_path"
  echo "[AFTER-RL-SFT] config=$sft_config"
  echo "============================================"

  for required_path in "$index_path" "$PREPARE_SCRIPT" "$sft_config" "$eval_config"; do
    if [[ ! -e "$required_path" ]]; then
      echo "ERROR: required path missing for $tag: $required_path" >&2
      exit 1
    fi
  done

  echo "[AFTER-RL-SFT] prepare data_experiment variant: $tag"
  python "$PREPARE_SCRIPT" \
    --variant "${tag}=${index_path}"

  for required_path in \
    "$data_root/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
    "$data_root/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
    "$data_root/test/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
    "$data_root/index/Industrial_and_Scientific.index.json" \
    "$data_root/index/Industrial_and_Scientific.item.json" \
    "$data_root/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"; do
    if [[ ! -f "$required_path" ]]; then
      echo "ERROR: prepared data missing for $tag: $required_path" >&2
      exit 1
    fi
  done

  echo "[AFTER-RL-SFT] recipe=title_history2sid_on + desc_align_p05"
  echo "[AFTER-RL-SFT] effective_batch=1024 micro_batch=2 world_size=$NPROC grad_accum=$((1024 / (2 * NPROC)))"
  echo "[AFTER-RL-SFT] start $tag SFT on GPUs $GPU_LIST"
  bash ./sft.sh industrial "$sft_config" \
    "runtime.cuda_visible_devices=$GPU_LIST" \
    "runtime.nproc_per_node=$NPROC"

  if [[ ! -e "$sft_model/config.json" ]]; then
    echo "ERROR: SFT final checkpoint missing for $tag: $sft_model" >&2
    exit 1
  fi

  echo "[AFTER-RL-SFT] start $tag evaluate on GPUs $GPU_LIST"
  bash ./evaluate.sh sft industrial "$eval_config" \
    "runtime.cuda_visible_devices=$GPU_LIST" \
    "runtime.nproc_per_node=$NPROC"

  if [[ ! -f "$eval_result" ]]; then
    echo "ERROR: SFT eval result missing for $tag: $eval_result" >&2
    exit 1
  fi
  echo "[AFTER-RL-SFT] completed $tag SFT/evaluate: $eval_result"
}

run_one_variant "r690b_lmh_l2_contrastive_pull_weight0003"
run_one_variant "r690b_lmh_l2_contrastive_pull_weight0015"

echo "[AFTER-RL-SFT] all queued SFT/evaluate jobs completed"
