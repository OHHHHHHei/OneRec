#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/leejt/OneRec"
cd "$ROOT_DIR"

CONFIG_ROOT="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/configs/stage1_l3_infonce"
SCRIPT="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/scripts/train_generate_tokenizer_lmh.sh"
OUT_ROOT="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l3_infonce_20260509"

run_one() {
  local tag="$1"
  local config="$CONFIG_ROOT/sid_train_industrial_r690b_lmh_l2_weight001_${tag}.yaml"
  local ckpt_root="$OUT_ROOT/r690b_lmh_l2_weight001_${tag}"
  local generated_index="$OUT_ROOT/generated_indices/Industrial_and_Scientific.r690b_lmh_l2_weight001_${tag}.index.json"
  local generate_summary="$OUT_ROOT/generated_indices/Industrial_and_Scientific.r690b_lmh_l2_weight001_${tag}.summary.json"
  local log="logs/l3_infonce/mgr_${tag}_tokenizer_0509.log"

  echo "[L3-INFONCE-SWEEP] start ${tag}"
  bash "$SCRIPT" "$config" "$ckpt_root" "$generated_index" "$generate_summary" "$log"
  echo "[L3-INFONCE-SWEEP] done ${tag}"
}

run_one "l3_infonce010"
run_one "l3_infonce020"
