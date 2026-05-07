#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
GPU="${2:-}"

if [[ "$MODE" != "v2" && "$MODE" != "r690b" ]]; then
  echo "Usage: $0 <v2|r690b> <gpu_id>" >&2
  exit 2
fi

ROOT_DIR="/home/leejt/OneRec"
ARCHIVE_DIR="$ROOT_DIR/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace"
BRANCH_DIR="$ROOT_DIR/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
OVERLAY_DIR="$ROOT_DIR/temp/mgr_sid_runtime_overlay"
PAIR_DIR="$BRANCH_DIR/pairs"
TRAIN_SCRIPT="$BRANCH_DIR/scripts/train_generate_tokenizer.sh"

cd "$ROOT_DIR"
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

# Setup runtime overlay
mkdir -p "$OVERLAY_DIR/onerec"
ln -sfn "$ROOT_DIR/src/onerec/__init__.py" "$OVERLAY_DIR/onerec/__init__.py"
ln -sfn "$ROOT_DIR/src/onerec/config.py" "$OVERLAY_DIR/onerec/config.py"
ln -sfn "$ARCHIVE_DIR/src/onerec/experiments" "$OVERLAY_DIR/onerec/experiments"
ln -sfn "$ROOT_DIR/src/onerec/sid" "$OVERLAY_DIR/onerec/sid"
ln -sfn "$ROOT_DIR/src/onerec/utils" "$OVERLAY_DIR/onerec/utils"
export PYTHONPATH="$OVERLAY_DIR:$ARCHIVE_DIR/src:$ROOT_DIR/src:${PYTHONPATH:-}"

if [[ "$MODE" == "v2" ]]; then
  BASE_CONFIG="$BRANCH_DIR/configs/sid_train_industrial_mgr_sid_v2_l2_local_multihop.yaml"
  PARAM="mid_weight"
  SWEEPS=(0.03 0.01 0.005)
  TAG_PREFIX="v2_lmh"
  CKPT_BASE="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507"
else
  BASE_CONFIG="$BRANCH_DIR/configs/sid_train_industrial_mgr_sid_r690b_l2_infonce_local_multihop.yaml"
  PARAM="l2_contrastive_pull_weight"
  SWEEPS=(0.03 0.01 0.005)
  TAG_PREFIX="r690b_lmh"
  CKPT_BASE="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507"

  # Regenerate negative pairs with local_multihop
  echo "[R690b] Regenerating negative pairs from local_multihop..."
  mkdir -p "$PAIR_DIR"
  python "$ARCHIVE_DIR/scripts/archive/pre_r720_legacy_experiments_20260418/experiment_mgr_sid_mid_pull_push_pair_source.py" \
    --tag R690bLMH \
    --mid-view-name local_multihop \
    --output-dir "$PAIR_DIR"
  echo "[R690b] Pairs ready."
fi

for w in "${SWEEPS[@]}"; do
  WSTR=$(echo "$w" | sed 's/\.//g')
  TAG="${TAG_PREFIX}_${PARAM}${WSTR}"
  SWEEP_CONFIG="/tmp/${TAG}.yaml"
  CKPT_DIR="$CKPT_BASE/$TAG"
  INDEX_DIR="$CKPT_BASE/generated_indices"
  INDEX_FILE="$INDEX_DIR/Industrial_and_Scientific.${TAG}.index.json"
  SUMMARY="$BRANCH_DIR/${TAG}_generate_summary.json"
  LOG="$ROOT_DIR/logs/experiment_mgr_sid_${TAG}_$(date +%Y%m%d).log"

  # Generate swept config by modifying the single parameter + ckpt_dir
  python3 -c "
import yaml
with open('$BASE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['$PARAM'] = float('$w')
cfg['ckpt_dir'] = '$CKPT_DIR'
with open('$SWEEP_CONFIG', 'w') as f:
    yaml.dump(cfg, f)
"

  echo "============================================"
  echo "  $TAG_PREFIX: $PARAM = $w  (gpu=$GPU)"
  echo "  ckpt: $CKPT_DIR"
  echo "============================================"

  mkdir -p "$CKPT_DIR" "$INDEX_DIR" "$(dirname "$LOG")"

  CUDA_VISIBLE_DEVICES="$GPU" bash "$TRAIN_SCRIPT" \
    "$SWEEP_CONFIG" "$CKPT_DIR" "$INDEX_FILE" "$SUMMARY" "$LOG"

  # Check collision
  if [[ -s "$SUMMARY" ]]; then
    COLL=$(python3 -c "
import json
with open('$SUMMARY') as f:
    d = json.load(f)
r = d.get('collision_rate', d.get('generated_collision_rate', 1.0))
print(f'{r:.4f}')
" 2>/dev/null || echo "1.0000")
    echo "  => generated collision_rate = $COLL"
  else
    echo "  => WARNING: no summary JSON, collision unknown"
  fi
  echo ""
done

echo "[DONE] $TAG_PREFIX sweep finished."
