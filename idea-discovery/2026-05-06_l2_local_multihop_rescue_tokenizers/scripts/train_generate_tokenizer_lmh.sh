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
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

mkdir -p "$(dirname "$LOG")"
mkdir -p "$(dirname "$GENERATED_INDEX")"
mkdir -p "$(dirname "$GENERATE_SUMMARY")"
mkdir -p "$CKPT_ROOT"

python -m onerec.experiments.hcsid.train_entry \
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

python -m onerec.experiments.hcsid.generate_entry \
  --ckpt_path "$RUN_DIR/best_collision_model.pth" \
  --output_file "$GENERATED_INDEX" \
  --summary_file "$GENERATE_SUMMARY" \
  --device "cuda:0" \
  --batch_size 64 \
  --max_collision_rounds 20 2>&1 | tee -a "$LOG"
