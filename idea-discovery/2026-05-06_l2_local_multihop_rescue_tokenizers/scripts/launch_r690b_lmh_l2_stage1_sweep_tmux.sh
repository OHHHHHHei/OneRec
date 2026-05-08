#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/leejt/OneRec"
BRANCH_DIR="$ROOT_DIR/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
QUEUE_SCRIPT="$BRANCH_DIR/scripts/run_r690b_lmh_l2_stage1_queue.sh"
LOG_DIR="$ROOT_DIR/logs/l2_lmh_stage1"

GPU_A="${GPU_A:-6}"
GPU_B="${GPU_B:-7}"
SESSION_A="${SESSION_A:-mgr_r690b_l2_stage1_A_0507}"
SESSION_B="${SESSION_B:-mgr_r690b_l2_stage1_B_0507}"

cd "$ROOT_DIR"
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_A" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_A" >&2
  exit 1
fi
if tmux has-session -t "$SESSION_B" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_B" >&2
  exit 1
fi

chmod +x "$QUEUE_SCRIPT"

LOG_A="$LOG_DIR/${SESSION_A}.log"
LOG_B="$LOG_DIR/${SESSION_B}.log"

# Queue A tests the upper side around the successful 0.01 anchor.
# The script gates 0.02 behind a non-collapsed 0.015 result.
tmux new-session -d -s "$SESSION_A" \
  "bash '$QUEUE_SCRIPT' '$GPU_A' A 0.015 0.02 2>&1 | tee '$LOG_A'"

# Queue B tests the lower side around the successful 0.01 anchor.
tmux new-session -d -s "$SESSION_B" \
  "bash '$QUEUE_SCRIPT' '$GPU_B' B 0.003 2>&1 | tee '$LOG_B'"

echo "Launched R690b LMH L2 stage-1 sweep:"
echo "  - $SESSION_A on GPU $GPU_A: 0.015 -> gated 0.02"
echo "  - $SESSION_B on GPU $GPU_B: 0.003"
echo "Logs:"
echo "  - $LOG_A"
echo "  - $LOG_B"
tmux ls | grep -E "$SESSION_A|$SESSION_B" || true
