#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${#GPU_ARRAY[@]}"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/attnres results/attnres

BRANCH_DIR="idea-discovery/2026-05-06_hierarchy_aware_attnres_sid_readout"

run_sft_eval() {
  local run_id="$1"
  local sft_config="$2"
  local eval_config="$3"
  local start_ts
  start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$start_ts] START $run_id SFT on GPUs $GPU_LIST"
  bash ./sft.sh industrial "$sft_config" \
    "runtime.cuda_visible_devices=$GPU_LIST" \
    "runtime.nproc_per_node=$NPROC"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $run_id EVAL on GPUs $GPU_LIST"
  bash ./evaluate.sh sft industrial "$eval_config" \
    "runtime.cuda_visible_devices=$GPU_LIST" \
    "runtime.nproc_per_node=$NPROC"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $run_id"
}

run_sft_eval \
  "attnres_h1_global" \
  "$BRANCH_DIR/configs/sft_attnres_h1_global_title_off_desc_p05_4gpu.yaml" \
  "$BRANCH_DIR/configs/eval_attnres_h1_global_title_off_desc_p05_4gpu.yaml"

run_sft_eval \
  "attnres_h2_sid_only" \
  "$BRANCH_DIR/configs/sft_attnres_h2_sid_only_title_off_desc_p05_4gpu.yaml" \
  "$BRANCH_DIR/configs/eval_attnres_h2_sid_only_title_off_desc_p05_4gpu.yaml"

run_sft_eval \
  "attnres_h3_level_aware" \
  "$BRANCH_DIR/configs/sft_attnres_h3_level_aware_title_off_desc_p05_4gpu.yaml" \
  "$BRANCH_DIR/configs/eval_attnres_h3_level_aware_title_off_desc_p05_4gpu.yaml"
