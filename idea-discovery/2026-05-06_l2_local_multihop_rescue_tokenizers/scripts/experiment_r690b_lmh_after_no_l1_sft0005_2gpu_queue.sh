#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-6,7}"
WAIT_SESSION="${2:-mgr_r690b_lmh_no_l1_sft_eval_2gpu_0508}"
NO_L1_RESULT="results/experiments/mgr_sid_l1_ablation_sft_eval_20260508/final_result_sft_mgr_r690b_lmh_l2_contrastive_pull_weight001_no_l1_semantic_title_on_desc_p05_2gpu_Industrial_and_Scientific.json"
CHAIN_SCRIPT="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/scripts/experiment_r690b_lmh_pull0005_sft_eval_2gpu_chain.sh"

echo "[AFTER-NO-L1-SFT] waiting for tmux session '$WAIT_SESSION' before using GPUs $GPU_LIST"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$now] [AFTER-NO-L1-SFT] still waiting for $WAIT_SESSION"
  sleep 300
done

if [[ ! -f "$NO_L1_RESULT" ]]; then
  echo "ERROR: waited for '$WAIT_SESSION', but no-L1 eval result is missing: $NO_L1_RESULT" >&2
  echo "ERROR: refusing to start 0.005 SFT so a failed upstream run is not silently skipped." >&2
  exit 1
fi

echo "[AFTER-NO-L1-SFT] detected no-L1 SFT/eval completion: $NO_L1_RESULT"
echo "[AFTER-NO-L1-SFT] starting 0.005 SFT/eval on GPUs $GPU_LIST"
bash "$CHAIN_SCRIPT" "$GPU_LIST"
