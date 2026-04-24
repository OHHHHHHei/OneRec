#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

OUTPUT_DIR="$ROOT_DIR/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r670_clean_l1_semantic_l2_push_pull_industrial"
SEMANTIC_GRAPH="$OUTPUT_DIR/R670a_l1_high_conf_semantic_graph.npz"
PAIR_CSV="$OUTPUT_DIR/R670a_all_mid_graph_weak_pairs.csv"

python "$ROOT_DIR/scripts/experiment_mgr_sid_r670a_l1_semantic_graph.py" \
  --tag R670a \
  --output-dir "$OUTPUT_DIR"

if [[ ! -f "$SEMANTIC_GRAPH" ]]; then
  echo "L1 semantic graph not found: $SEMANTIC_GRAPH" >&2
  exit 1
fi

python "$ROOT_DIR/scripts/experiment_mgr_sid_mid_pull_push_pair_source.py" \
  --tag R670a \
  --mid-view-name fagsp_mid_base \
  --output-dir "$OUTPUT_DIR"

if [[ ! -f "$PAIR_CSV" ]]; then
  echo "Pair source CSV not found: $PAIR_CSV" >&2
  exit 1
fi

SESSION="mgr_r670a_clean_l1_semantic_l2_push_pull"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

GPU="${CUDA_VISIBLE_DEVICES:-7}"
tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=\"$GPU\" && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r670a_clean_l1_semantic_l2_push_pull_train_generate.sh\"'"

echo "Launched tmux session:"
echo "  - $SESSION on GPU $GPU"
echo
tmux ls
