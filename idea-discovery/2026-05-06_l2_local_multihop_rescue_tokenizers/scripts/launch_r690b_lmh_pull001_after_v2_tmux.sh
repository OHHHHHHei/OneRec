#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPU_LIST="${1:-2,3,4,5}"
WAIT_SESSION="${2:-mgr_v2_lmh_mid001_sft_eval_0507}"
SESSION_NAME="${3:-mgr_r690b_lmh_pull001_after_v2_0507}"
LOG_PATH="logs/l2_lmh_sft/${SESSION_NAME}.log"
WAIT_RESULT="results/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507/final_result_sft_mgr_v2_lmh_mid_weight001_title_on_desc_p05_4gpu_Industrial_and_Scientific.json"

mkdir -p logs/l2_lmh_sft

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "bash -lc 'cd \"$REPO_ROOT\"; \
    echo \"[QUEUE] waiting for tmux session: $WAIT_SESSION\"; \
    while tmux has-session -t \"$WAIT_SESSION\" 2>/dev/null; do sleep 60; done; \
    if [[ ! -f \"$WAIT_RESULT\" ]]; then \
      echo \"ERROR: upstream v2 eval result not found after $WAIT_SESSION ended: $WAIT_RESULT\" >&2; \
      exit 1; \
    fi; \
    echo \"[QUEUE] upstream v2 eval result found: $WAIT_RESULT\"; \
    echo \"[QUEUE] launching r690b pull=0.01 SFT/eval on GPUs $GPU_LIST\"; \
    bash idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/scripts/experiment_r690b_lmh_pull001_sft_eval_chain.sh \"$GPU_LIST\"' \
    2>&1 | tee '$LOG_PATH'"

echo "Launched queued tmux session: $SESSION_NAME"
echo "Waiting for session: $WAIT_SESSION"
echo "GPU list for queued run: $GPU_LIST"
echo "Log: $LOG_PATH"
