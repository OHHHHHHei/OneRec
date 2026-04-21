#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_LIST="${1:-2,3,4,5}"
SESSION="mgr_sid_qcr_l2_conflict_ranking_sft_eval"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=\"$GPU_LIST\" && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_qcr_l2_conflict_ranking_sft_eval_title_on_desc_p05_4gpu_chain.sh\"'"

echo "Started tmux session: $SESSION on GPUs $GPU_LIST"
echo "Attach with: tmux attach -t $SESSION"
