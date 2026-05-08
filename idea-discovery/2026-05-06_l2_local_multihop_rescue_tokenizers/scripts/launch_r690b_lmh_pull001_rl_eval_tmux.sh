#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPU_LIST="${1:-2,3,4,5}"
SESSION_NAME="${2:-mgr_r690b_lmh_pull001_rl_eval_0507}"
LOG_PATH="logs/l2_lmh_rl/${SESSION_NAME}.log"
CHAIN_SCRIPT="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/scripts/experiment_r690b_lmh_pull001_rl_eval_chain.sh"

mkdir -p "$(dirname "$LOG_PATH")"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "bash '$CHAIN_SCRIPT' '$GPU_LIST' 2>&1 | tee '$LOG_PATH'"

echo "launched tmux session: $SESSION_NAME"
echo "gpu_list: $GPU_LIST"
echo "log: $LOG_PATH"
echo "attach: tmux attach -t $SESSION_NAME"
