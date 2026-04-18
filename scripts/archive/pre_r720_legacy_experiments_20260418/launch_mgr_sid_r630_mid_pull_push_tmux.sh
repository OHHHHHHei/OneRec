#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

PAIR_OUTPUT_DIR="$ROOT_DIR/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630_mid_pull_push_industrial"
PAIR_CSV="$PAIR_OUTPUT_DIR/R630_all_mid_graph_weak_pairs.csv"

python "$ROOT_DIR/scripts/experiment_mgr_sid_mid_pull_push_pair_source.py" \
  --output-dir "$PAIR_OUTPUT_DIR"

if [[ ! -f "$PAIR_CSV" ]]; then
  echo "Pair source CSV not found: $PAIR_CSV" >&2
  exit 1
fi

SESSION_A="mgr_r630a_mid_pull_only"
SESSION_B="mgr_r630b_mid_push_only"
SESSION_C="mgr_r630c_mid_pull_push"

for session in "$SESSION_A" "$SESSION_B" "$SESSION_C"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

tmux new-session -d -s "$SESSION_A" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=2 && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r630a_mid_pull_only_train.sh\"'"
tmux new-session -d -s "$SESSION_B" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=3 && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r630b_mid_push_only_train.sh\"'"
tmux new-session -d -s "$SESSION_C" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=4 && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r630c_mid_pull_push_train.sh\"'"

echo "Launched tmux sessions:"
echo "  - $SESSION_A on GPU 2"
echo "  - $SESSION_B on GPU 3"
echo "  - $SESSION_C on GPU 4"
echo
tmux ls
