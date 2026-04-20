#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

SESSION="mgr_semantic_l1_collab_deeper_sft_eval_4gpu"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

GPUS="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=\"$GPUS\" && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_semantic_l1_collab_deeper_sft_eval_chain.sh\"'"

echo "Launched tmux session:"
echo "  - $SESSION on GPUs $GPUS"
echo
tmux ls
