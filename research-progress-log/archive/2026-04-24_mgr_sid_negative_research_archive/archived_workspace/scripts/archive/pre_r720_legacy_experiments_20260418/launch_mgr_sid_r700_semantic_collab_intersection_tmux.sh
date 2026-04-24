#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

OUTPUT_DIR="$ROOT_DIR/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r700_semantic_collab_intersection_industrial"
L1_GRAPH="$OUTPUT_DIR/R700a_l1_semantic_collab_intersection_graph.npz"
L2_POS_GRAPH="$OUTPUT_DIR/R700a_l2_semantic_multihop_positive_graph.npz"
L2_NEG_CSV="$OUTPUT_DIR/R700a_l2_semantic_near_multihop_weak_pairs.csv"

python "$ROOT_DIR/scripts/experiment_mgr_sid_r700a_graph_sources.py" \
  --tag R700a \
  --coarse-view-name coarse_purified \
  --mid-view-name local_multihop \
  --output-dir "$OUTPUT_DIR"

for file in "$L1_GRAPH" "$L2_POS_GRAPH" "$L2_NEG_CSV"; do
  if [[ ! -f "$file" ]]; then
    echo "Required R700a graph source not found: $file" >&2
    exit 1
  fi
done

SESSION="mgr_r700a_semantic_collab_intersection"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

GPU="${R700A_CUDA_VISIBLE_DEVICES:-2}"

tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=\"$GPU\" && bash \"$ROOT_DIR/scripts/experiment_mgr_sid_r700a_semantic_collab_intersection_train_generate.sh\"'"

echo "Launched tmux session:"
echo "  - $SESSION on GPU $GPU"
echo
tmux ls
