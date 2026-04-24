#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

OUTPUT_DIR="$ROOT_DIR/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial"
L1_GRAPH="$OUTPUT_DIR/R693a_l1_coarse_highconf_graph.npz"
PAIR_CSV="$OUTPUT_DIR/R693a_all_mid_graph_weak_pairs.csv"

python "$ROOT_DIR/scripts/experiment_mgr_sid_hier_collab_graph_sources.py" \
  --tag R693a \
  --mid-view-name local_multihop \
  --coarse-view-name coarse_purified \
  --output-dir "$OUTPUT_DIR"

if [[ ! -f "$L1_GRAPH" ]]; then
  echo "L1 graph file not found: $L1_GRAPH" >&2
  exit 1
fi
if [[ ! -f "$PAIR_CSV" ]]; then
  echo "Negative pair CSV not found: $PAIR_CSV" >&2
  exit 1
fi

SESSION="mgr_r693a_hier_collab_only_multihop"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

GPU="${R693A_CUDA_VISIBLE_DEVICES:-3}"

tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=\"$GPU\" && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r693a_hier_collab_only_multihop_train_generate.sh\"'"

echo "Launched tmux session:"
echo "  - $SESSION on GPU $GPU"
echo
tmux ls
