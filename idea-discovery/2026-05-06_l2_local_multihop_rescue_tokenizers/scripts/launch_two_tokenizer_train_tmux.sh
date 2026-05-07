#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/leejt/OneRec"
ARCHIVE_DIR="$ROOT_DIR/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace"
BRANCH_DIR="$ROOT_DIR/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
OVERLAY_DIR="$ROOT_DIR/temp/mgr_sid_runtime_overlay"
PAIR_DIR="$BRANCH_DIR/pairs"

GPU_V2="${GPU_V2:-6}"
GPU_R690B="${GPU_R690B:-7}"
SESSION_V2="${SESSION_V2:-mgr_v2_l2_lmh_tok_0506}"
SESSION_R690B="${SESSION_R690B:-mgr_r690b_l2_lmh_tok_0506}"

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

mkdir -p "$PAIR_DIR"
python "$ARCHIVE_DIR/scripts/archive/pre_r720_legacy_experiments_20260418/experiment_mgr_sid_mid_pull_push_pair_source.py" \
  --tag R690bLMH \
  --mid-view-name local_multihop \
  --output-dir "$PAIR_DIR"

PAIR_CSV="$PAIR_DIR/R690bLMH_all_mid_graph_weak_pairs.csv"
if [[ ! -s "$PAIR_CSV" ]]; then
  echo "Pair source CSV missing or empty: $PAIR_CSV" >&2
  exit 1
fi

if tmux has-session -t "$SESSION_V2" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_V2" >&2
  exit 1
fi
if tmux has-session -t "$SESSION_R690B" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_R690B" >&2
  exit 1
fi

TRAIN_SCRIPT="$BRANCH_DIR/scripts/train_generate_tokenizer.sh"

CONFIG_V2="$BRANCH_DIR/configs/sid_train_industrial_mgr_sid_v2_l2_local_multihop.yaml"
CKPT_V2="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_local_multihop_rescue_20260506/industrial_v2_l2_local_multihop"
INDEX_V2="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_local_multihop_rescue_20260506/generated_indices/Industrial_and_Scientific.v2_l2_local_multihop.index.json"
SUMMARY_V2="$BRANCH_DIR/v2_l2_local_multihop_generate_summary.json"
LOG_V2="$ROOT_DIR/logs/experiment_mgr_sid_v2_l2_local_multihop_20260506.log"

CONFIG_R690B="$BRANCH_DIR/configs/sid_train_industrial_mgr_sid_r690b_l2_infonce_local_multihop.yaml"
CKPT_R690B="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_local_multihop_rescue_20260506/industrial_r690b_l2_infonce_local_multihop"
INDEX_R690B="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_local_multihop_rescue_20260506/generated_indices/Industrial_and_Scientific.r690b_l2_infonce_local_multihop.index.json"
SUMMARY_R690B="$BRANCH_DIR/r690b_l2_infonce_local_multihop_generate_summary.json"
LOG_R690B="$ROOT_DIR/logs/experiment_mgr_sid_r690b_l2_infonce_local_multihop_20260506.log"

tmux new-session -d -s "$SESSION_V2" \
  "CUDA_VISIBLE_DEVICES=$GPU_V2 bash '$TRAIN_SCRIPT' '$CONFIG_V2' '$CKPT_V2' '$INDEX_V2' '$SUMMARY_V2' '$LOG_V2'"

tmux new-session -d -s "$SESSION_R690B" \
  "CUDA_VISIBLE_DEVICES=$GPU_R690B bash '$TRAIN_SCRIPT' '$CONFIG_R690B' '$CKPT_R690B' '$INDEX_R690B' '$SUMMARY_R690B' '$LOG_R690B'"

echo "Launched tokenizer training tmux sessions:"
echo "  - $SESSION_V2 on GPU $GPU_V2"
echo "  - $SESSION_R690B on GPU $GPU_R690B"
echo "Pair source: $PAIR_CSV"
tmux ls

