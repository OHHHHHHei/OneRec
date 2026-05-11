#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <gpu_id>" >&2
  exit 2
fi

GPU_ID="$1"
ROOT_DIR="/home/leejt/OneRec"
BRANCH_DIR="$ROOT_DIR/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
CONFIG="$BRANCH_DIR/configs/toys/sid_train_toys_baseline_rqvae_onerec_aligned.yaml"
CKPT_ROOT="/data/leejt/OneRec/output_weights/experiments/toys_tokenizer_20260510/baseline_rqvae_onerec_aligned"
INDEX_DIR="/data/leejt/OneRec/output_weights/experiments/toys_tokenizer_20260510/generated_indices"
INDEX_FILE="$INDEX_DIR/Toys_and_Games.baseline_rqvae_onerec_aligned.index.json"
STRUCTURE_SUMMARY="$BRANCH_DIR/toys_baseline_rqvae_onerec_aligned_structure_summary.json"
LOG="$ROOT_DIR/logs/toys_tokenizer_20260510/toys_baseline_rqvae_onerec_aligned.log"

cd "$ROOT_DIR"
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

mkdir -p "$CKPT_ROOT" "$INDEX_DIR" "$(dirname "$LOG")"

echo "[TOYS-BASELINE] config=$CONFIG"
echo "[TOYS-BASELINE] ckpt=$CKPT_ROOT"
echo "[TOYS-BASELINE] index=$INDEX_FILE"
echo "[TOYS-BASELINE] log=$LOG"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m onerec.main sid-train --config "$CONFIG" 2>&1 | tee "$LOG"

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

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m onerec.sid.generate.rqvae_indices \
  --ckpt_path "$RUN_DIR/best_collision_model.pth" \
  --output_file "$INDEX_FILE" \
  --device "cuda:0" \
  --batch_size 64 \
  --max_collision_rounds 20 2>&1 | tee -a "$LOG"

python - "$INDEX_FILE" "$STRUCTURE_SUMMARY" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

index_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
obj = json.loads(index_path.read_text())
vals = list(obj.values()) if isinstance(obj, dict) else obj
sids = [tuple(v[:3]) for v in vals]
c1 = Counter(s[0] for s in sids)
c12 = Counter(s[:2] for s in sids)
c123 = Counter(sids)
summary = {
    "index_path": str(index_path),
    "num_items": len(sids),
    "active_l1": len(c1),
    "unique_l12": len(c12),
    "unique_sid": len(c123),
    "collision_count": len(sids) - len(c123),
    "collision_rate": (len(sids) - len(c123)) / max(len(sids), 1),
    "max_conflict": max(Counter(sids).values()) if sids else 0,
    "max_l1_bucket": max(c1.values()) if c1 else 0,
    "top5_l1_cover": sum(v for _, v in c1.most_common(5)),
    "top5_l1": c1.most_common(5),
}
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False))
PY

echo "[TOYS-BASELINE] completed."
