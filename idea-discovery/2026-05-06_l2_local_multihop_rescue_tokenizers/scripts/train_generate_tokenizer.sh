#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 <config> <ckpt_root> <generated_index> <generate_summary> <log>" >&2
  exit 2
fi

CONFIG="$1"
CKPT_ROOT="$2"
GENERATED_INDEX="$3"
GENERATE_SUMMARY="$4"
LOG="$5"

ROOT_DIR="/home/leejt/OneRec"
ARCHIVE_DIR="$ROOT_DIR/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace"
OVERLAY_DIR="$ROOT_DIR/temp/mgr_sid_runtime_overlay"

cd "$ROOT_DIR"
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

mkdir -p "$OVERLAY_DIR/onerec"
ln -sfn "$ROOT_DIR/src/onerec/__init__.py" "$OVERLAY_DIR/onerec/__init__.py"
ln -sfn "$ROOT_DIR/src/onerec/config.py" "$OVERLAY_DIR/onerec/config.py"
ln -sfn "$ARCHIVE_DIR/src/onerec/experiments" "$OVERLAY_DIR/onerec/experiments"
ln -sfn "$ROOT_DIR/src/onerec/sid" "$OVERLAY_DIR/onerec/sid"
ln -sfn "$ROOT_DIR/src/onerec/utils" "$OVERLAY_DIR/onerec/utils"

export PYTHONPATH="$OVERLAY_DIR:$ARCHIVE_DIR/src:$ROOT_DIR/src:${PYTHONPATH:-}"

mkdir -p "$(dirname "$LOG")"
mkdir -p "$(dirname "$GENERATED_INDEX")"
mkdir -p "$(dirname "$GENERATE_SUMMARY")"
mkdir -p "$CKPT_ROOT"

python "$ARCHIVE_DIR/scripts/experiment_mgr_sid_v2_train.py" \
  --config "$CONFIG" 2>&1 | tee "$LOG"

RUN_DIR="$(
python - "$CKPT_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
dirs = [p for p in root.iterdir() if p.is_dir()]
if not dirs:
    raise SystemExit(f"No run directory found after training: {root}")
print(max(dirs, key=lambda p: p.stat().st_mtime))
PY
)"

python "$ARCHIVE_DIR/scripts/experiment_mgr_sid_v1_generate.py" \
  --ckpt_path "$RUN_DIR/best_collision_model.pth" \
  --output_file "$GENERATED_INDEX" \
  --summary_file "$GENERATE_SUMMARY" \
  --device "cuda:0" \
  --batch_size 64 \
  --max_collision_rounds 20 2>&1 | tee -a "$LOG"

