#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

OUTPUT_DIR="$ROOT_DIR/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r690_cost_inspired_contrastive_quantization_industrial"
PAIR_CSV="$OUTPUT_DIR/R690_all_mid_graph_weak_pairs.csv"

python "$ROOT_DIR/scripts/experiment_mgr_sid_mid_pull_push_pair_source.py" \
  --tag R690 \
  --mid-view-name fagsp_mid_base \
  --output-dir "$OUTPUT_DIR"

if [[ ! -f "$PAIR_CSV" ]]; then
  echo "Pair source CSV not found: $PAIR_CSV" >&2
  exit 1
fi

SESSION_A="mgr_r690a_l2_graph_infonce"
SESSION_B="mgr_r690b_hier_cost_guided"
if tmux has-session -t "$SESSION_A" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_A" >&2
  exit 1
fi
if tmux has-session -t "$SESSION_B" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_B" >&2
  exit 1
fi

GPU_A="${R690A_CUDA_VISIBLE_DEVICES:-3}"
GPU_B="${R690B_CUDA_VISIBLE_DEVICES:-4}"

tmux new-session -d -s "$SESSION_A" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=\"$GPU_A\" && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r690a_l2_graph_infonce_train_generate.sh\"'"

tmux new-session -d -s "$SESSION_B" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=\"$GPU_B\" && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r690b_hier_cost_guided_train_generate.sh\"'"

echo "Launched tmux sessions:"
echo "  - $SESSION_A on GPU $GPU_A"
echo "  - $SESSION_B on GPU $GPU_B"
echo
tmux ls
