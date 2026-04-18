#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

SESSION="mgr_r710a_v2_no_semantic_retention"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

GPU="${R710A_CUDA_VISIBLE_DEVICES:-7}"

tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=\"$GPU\" && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r710a_v2_no_semantic_retention_train_generate.sh\"'"

echo "Launched tmux session:"
echo "  - $SESSION on GPU $GPU"
echo
tmux ls
