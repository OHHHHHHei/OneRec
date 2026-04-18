#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

CONFIG="$ROOT_DIR/config/experiments/sid_train_industrial_mgr_sid_r640b_seq2graph_rel.yaml"
CKPT_ROOT="/data/leejt/OneRec/output_weights/experiments/mgr_sid_seq2graph_lite_20260416/industrial_r640b_seq2graph_rel"
GENERATED_INDEX="/data/leejt/OneRec/output_weights/experiments/mgr_sid_seq2graph_lite_20260416/generated_indices/Industrial_and_Scientific.r640b_seq2graph_rel.index.json"
GENERATE_SUMMARY="$ROOT_DIR/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r640_seq2graph_lite_industrial/R640b_generate_summary.json"

python "$ROOT_DIR/scripts/experiment_mgr_sid_v2_train.py" \
  --config "$CONFIG"

RUN_DIR="$(
python - <<'PY'
from pathlib import Path
root = Path('/data/leejt/OneRec/output_weights/experiments/mgr_sid_seq2graph_lite_20260416/industrial_r640b_seq2graph_rel')
dirs = [p for p in root.iterdir() if p.is_dir()]
if not dirs:
    raise SystemExit('No run directory found after training.')
print(max(dirs, key=lambda p: p.stat().st_mtime))
PY
)"

python "$ROOT_DIR/scripts/experiment_mgr_sid_v1_generate.py" \
  --ckpt_path "$RUN_DIR/best_collision_model.pth" \
  --output_file "$GENERATED_INDEX" \
  --summary_file "$GENERATE_SUMMARY" \
  --device "cuda:0" \
  --batch_size 64 \
  --max_collision_rounds 20
