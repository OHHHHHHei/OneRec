#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

STAGE_DIR="$ROOT_DIR/research-progress-log/experiment_launches/2026-04-19_mgr_sid_collab_ranking_local_multihop_mid_l1_rescue_industrial"
CONFIG="$ROOT_DIR/config/experiments/sid_train_industrial_mgr_sid_collab_ranking_local_multihop_mid_l1_rescue.yaml"
CKPT_ROOT="/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_l1_rescue_20260419/industrial_r720c_local_multihop_mid_l1_rescue"
GENERATED_INDEX="/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_l1_rescue_20260419/generated_indices/Industrial_and_Scientific.r720c_local_multihop_mid_l1_rescue.index.json"
GENERATE_SUMMARY="$STAGE_DIR/R720c_generate_summary.json"
LOG="$ROOT_DIR/logs/experiment_mgr_sid_collab_ranking_local_multihop_mid_l1_rescue_20260419.log"

mkdir -p "$(dirname "$LOG")"
mkdir -p "$(dirname "$GENERATED_INDEX")"
mkdir -p "$(dirname "$GENERATE_SUMMARY")"
mkdir -p "$STAGE_DIR"

python "$ROOT_DIR/scripts/experiment_mgr_sid_collab_ranking_train.py" \
  --config "$CONFIG" 2>&1 | tee "$LOG"

RUN_DIR="$(
python - <<'PY'
from pathlib import Path
root = Path("/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_l1_rescue_20260419/industrial_r720c_local_multihop_mid_l1_rescue")
dirs = [p for p in root.iterdir() if p.is_dir()]
if not dirs:
    raise SystemExit("No run directory found after training.")
print(max(dirs, key=lambda p: p.stat().st_mtime))
PY
)"

python "$ROOT_DIR/scripts/experiment_mgr_sid_v1_generate.py" \
  --ckpt_path "$RUN_DIR/best_collision_model.pth" \
  --output_file "$GENERATED_INDEX" \
  --summary_file "$GENERATE_SUMMARY" \
  --device "cuda:0" \
  --batch_size 64 \
  --max_collision_rounds 20 2>&1 | tee -a "$LOG"
