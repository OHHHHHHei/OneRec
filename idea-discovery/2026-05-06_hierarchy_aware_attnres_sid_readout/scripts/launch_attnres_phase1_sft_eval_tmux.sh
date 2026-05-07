#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPU_LIST="${1:-2,3,4,5}"
SESSION_NAME="${2:-attnres_phase1_sft_eval_0506}"
LOG_PATH="logs/attnres/${SESSION_NAME}.log"

mkdir -p logs/attnres

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "bash idea-discovery/2026-05-06_hierarchy_aware_attnres_sid_readout/scripts/experiment_attnres_phase1_sft_eval_chain.sh '$GPU_LIST' 2>&1 | tee '$LOG_PATH'"

echo "Launched tmux session: $SESSION_NAME"
echo "GPU list: $GPU_LIST"
echo "Log: $LOG_PATH"
