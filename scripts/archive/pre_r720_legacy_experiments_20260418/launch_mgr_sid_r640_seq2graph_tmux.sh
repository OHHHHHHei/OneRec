#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

SESSION_B="mgr_r640b_seq2graph_rel"
SESSION_C="mgr_r640c_seq2graph_rel_masked"

for session in "$SESSION_B" "$SESSION_C"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

tmux new-session -d -s "$SESSION_B" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=2 && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r640b_seq2graph_rel_train_generate.sh\"'"
tmux new-session -d -s "$SESSION_C" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=3 && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r640c_seq2graph_rel_masked_train_generate.sh\"'"

echo "Launched tmux sessions:"
echo "  - $SESSION_B on GPU 2"
echo "  - $SESSION_C on GPU 3"
echo
tmux ls
