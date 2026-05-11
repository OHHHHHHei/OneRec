#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <gpu_id>" >&2
  exit 2
fi

GPU_ID="$1"
ROOT_DIR="/home/leejt/OneRec"
ARCHIVE_DIR="$ROOT_DIR/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace"
BRANCH_DIR="$ROOT_DIR/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
TRAIN_SCRIPT="$BRANCH_DIR/scripts/train_generate_tokenizer_lmh.sh"
PAIR_SOURCE_SCRIPT="$ARCHIVE_DIR/scripts/archive/pre_r720_legacy_experiments_20260418/experiment_mgr_sid_mid_pull_push_pair_source.py"
PROXY_SCRIPT="$BRANCH_DIR/scripts/recompute_proxy_local_multihop.py"

CONFIG="$BRANCH_DIR/configs/toys/sid_train_toys_r690b_lmh_l1w030_l2w010_l3w020.yaml"
PAIR_DIR="$BRANCH_DIR/pairs/toys"
LOG_DIR="$ROOT_DIR/logs/toys_tokenizer_20260510"
CKPT_DIR="/data/leejt/OneRec/output_weights/experiments/toys_tokenizer_20260510/r690b_lmh_l1w030_l2w010_l3w020"
INDEX_DIR="/data/leejt/OneRec/output_weights/experiments/toys_tokenizer_20260510/generated_indices"

CATEGORY="Toys_and_Games"
TRAIN_CSV="./data/Amazon18/Toys_and_Games/Toys_and_Games.tokenizer_train.csv"
EMB_PATH="./data/Amazon18/Toys_and_Games/Toys_and_Games.emb-qwen-td.npy"
PAIR_TAG="ToysR690bLMH"
PAIR_CSV="$PAIR_DIR/${PAIR_TAG}_all_mid_graph_weak_pairs.csv"
PROXY_CSV="$PAIR_DIR/proxy_item_scores_toys_local_multihop.csv"
INDEX_FILE="$INDEX_DIR/${CATEGORY}.r690b_lmh_l1w030_l2w010_l3w020.index.json"
GENERATE_SUMMARY="$BRANCH_DIR/toys_r690b_lmh_l1w030_l2w010_l3w020_generate_summary.json"
STRUCTURE_SUMMARY="$BRANCH_DIR/toys_r690b_lmh_l1w030_l2w010_l3w020_structure_summary.json"
RUN_LOG="$LOG_DIR/toys_r690b_lmh_l1w030_l2w010_l3w020.log"

cd "$ROOT_DIR"
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec
export PYTHONPATH="$ARCHIVE_DIR/src:$ROOT_DIR/src:${PYTHONPATH:-}"

mkdir -p "$PAIR_DIR" "$LOG_DIR" "$CKPT_DIR" "$INDEX_DIR"

if [[ ! -s "$PAIR_CSV" ]]; then
  echo "[TOYS-LMH] building Toys local-multihop weak pairs: $PAIR_CSV"
  python "$PAIR_SOURCE_SCRIPT" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TRAIN_CSV" \
    --semantic-embedding-path "$EMB_PATH" \
    --output-dir "$PAIR_DIR" \
    --tag "$PAIR_TAG" \
    --mid-view-name local_multihop \
    --semantic-topk 32 \
    --graph-topk 32 \
    --graph-weak-quantile 0.25 \
    --local-multihop-alpha 0.35 \
    --local-multihop-max-hop 2 \
    --mgdcf-binarize-edges 2>&1 | tee "$RUN_LOG"
else
  echo "[TOYS-LMH] existing pair CSV found: $PAIR_CSV" | tee "$RUN_LOG"
fi

if [[ ! -s "$PROXY_CSV" ]]; then
  echo "[TOYS-LMH] building Toys local-multihop ambiguity prior: $PROXY_CSV" | tee -a "$RUN_LOG"
  python "$PROXY_SCRIPT" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TRAIN_CSV" \
    --semantic-embedding "$EMB_PATH" \
    --output-csv "$PROXY_CSV" \
    --semantic-topk 32 \
    --graph-topk 32 \
    --history-k 10 2>&1 | tee -a "$RUN_LOG"
else
  echo "[TOYS-LMH] existing proxy CSV found: $PROXY_CSV" | tee -a "$RUN_LOG"
fi

echo "[TOYS-LMH] config=$CONFIG" | tee -a "$RUN_LOG"
echo "[TOYS-LMH] ckpt=$CKPT_DIR" | tee -a "$RUN_LOG"
echo "[TOYS-LMH] index=$INDEX_FILE" | tee -a "$RUN_LOG"
echo "[TOYS-LMH] log=$RUN_LOG" | tee -a "$RUN_LOG"

CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$TRAIN_SCRIPT" \
  "$CONFIG" "$CKPT_DIR" "$INDEX_FILE" "$GENERATE_SUMMARY" "$RUN_LOG"

python - "$INDEX_FILE" "$STRUCTURE_SUMMARY" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

index_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
obj = json.loads(index_path.read_text())
vals = list(obj.values()) if isinstance(obj, dict) else obj
sids = [tuple(v[:3]) for v in vals]
c1 = Counter(s[0] for s in sids)
c12 = Counter(s[:2] for s in sids)
c123 = Counter(sids)
summary = {
    "index_path": str(index_path),
    "num_items": len(sids),
    "active_l1": len(c1),
    "unique_l12": len(c12),
    "unique_sid": len(c123),
    "collision_count": len(sids) - len(c123),
    "collision_rate": (len(sids) - len(c123)) / max(len(sids), 1),
    "max_conflict": max(Counter(sids).values()) if sids else 0,
    "max_l1_bucket": max(c1.values()) if c1 else 0,
    "top5_l1_cover": sum(v for _, v in c1.most_common(5)),
    "top5_l1": c1.most_common(5),
}
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False))
PY

echo "[TOYS-LMH] completed."
