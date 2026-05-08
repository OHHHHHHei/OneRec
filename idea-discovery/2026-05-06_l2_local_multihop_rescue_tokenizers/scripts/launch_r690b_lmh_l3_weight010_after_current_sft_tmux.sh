#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
WAIT_SESSION="${2:-mgr_r690b_lmh_after_rl_sft003_0015_0507}"
SESSION_NAME="${3:-mgr_r690b_lmh_l3_w010_after_current_sft_0508}"
LOG_PATH="${4:-logs/l2_lmh_sft/mgr_r690b_lmh_l3_w010_after_current_sft_0508.log}"
CHAIN_SCRIPT="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/scripts/experiment_r690b_lmh_l3_weight010_sft_eval_4gpu_chain.sh"
WAIT_RESULT="results/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507/final_result_sft_mgr_r690b_lmh_l2_contrastive_pull_weight0015_title_on_desc_p05_4gpu_Industrial_and_Scientific.json"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")"

tmux new-session -d -s "$SESSION_NAME" \
  "source /home/leejt/miniconda3/etc/profile.d/conda.sh && conda activate MiniOneRec && cd '$REPO_ROOT' && \
   echo '[L3-AFTER] waiting for tmux session $WAIT_SESSION before launching L3=0.010 SFT/eval on GPUs $GPU_LIST' && \
   while tmux has-session -t '$WAIT_SESSION' 2>/dev/null; do date '+[%Y-%m-%d %H:%M:%S] [L3-AFTER] still waiting for $WAIT_SESSION'; sleep 300; done && \
   if [ ! -f '$WAIT_RESULT' ]; then echo 'ERROR: waited for $WAIT_SESSION but expected eval result is missing: $WAIT_RESULT' >&2; exit 1; fi && \
   echo '[L3-AFTER] prerequisite eval result detected: $WAIT_RESULT' && \
   bash '$CHAIN_SCRIPT' '$GPU_LIST' 2>&1 | tee '$LOG_PATH'"

echo "[L3-AFTER] launched watcher session: $SESSION_NAME"
echo "[L3-AFTER] log: $LOG_PATH"
