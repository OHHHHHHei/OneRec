#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

OUTPUT_DIR="$ROOT_DIR/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r680_l1_smooth_l2_contrastive_multihop_industrial"
PAIR_CSV="$OUTPUT_DIR/R680a_all_mid_graph_weak_pairs.csv"

python "$ROOT_DIR/scripts/experiment_mgr_sid_mid_pull_push_pair_source.py" \
  --tag R680a \
  --mid-view-name local_multihop \
  --output-dir "$OUTPUT_DIR"

if [[ ! -f "$PAIR_CSV" ]]; then
  echo "Pair source CSV not found: $PAIR_CSV" >&2
  exit 1
fi

SESSION="mgr_r680a_l1_smooth_l2_contrastive_multihop"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

GPU="${CUDA_VISIBLE_DEVICES:-7}"
tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=\"$GPU\" && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r680a_l1_smooth_l2_contrastive_multihop_train_generate.sh\"'"

echo "Launched tmux session:"
echo "  - $SESSION on GPU $GPU"
echo
tmux ls
