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
BASE_CONFIG="$BRANCH_DIR/configs/sid_train_industrial_mgr_sid_r690b_l2_infonce_local_multihop.yaml"
TRAIN_SCRIPT="$BRANCH_DIR/scripts/train_generate_tokenizer.sh"
PAIR_SOURCE_SCRIPT="$ARCHIVE_DIR/scripts/archive/pre_r720_legacy_experiments_20260418/experiment_mgr_sid_mid_pull_push_pair_source.py"
PROXY_SCRIPT="$BRANCH_DIR/scripts/recompute_proxy_local_multihop.py"

CONFIG_DIR="$BRANCH_DIR/configs/office_stage1"
PAIR_DIR="$BRANCH_DIR/pairs/office"
LOG_DIR="$ROOT_DIR/logs/office_lmh_stage1"
CKPT_BASE="/data/leejt/OneRec/output_weights/experiments/mgr_sid_office_lmh_main_20260509"
INDEX_DIR="$CKPT_BASE/generated_indices"

CATEGORY="Office_Products"
SPLIT_STEM="Office_Products_5_2016-10-2018-11"
TRAIN_CSV="./data/Amazon/train/${SPLIT_STEM}.csv"
EMB_PATH="./data/Amazon/index/${CATEGORY}.emb-qwen-td.npy"

TAG="office_r690b_lmh_l1w030_l2w010_l3w020"
PAIR_TAG="OfficeR690bLMH"
PAIR_CSV="$PAIR_DIR/${PAIR_TAG}_all_mid_graph_weak_pairs.csv"
PROXY_CSV="$PAIR_DIR/proxy_item_scores_office_local_multihop.csv"
CONFIG_PATH="$CONFIG_DIR/sid_train_${TAG}.yaml"
CKPT_DIR="$CKPT_BASE/$TAG"
INDEX_FILE="$INDEX_DIR/${CATEGORY}.${TAG}.index.json"
GENERATE_SUMMARY="$BRANCH_DIR/${TAG}_generate_summary.json"
STRUCTURE_SUMMARY="$BRANCH_DIR/${TAG}_structure_summary.json"
RUN_LOG="$LOG_DIR/experiment_mgr_sid_${TAG}_$(date +%Y%m%d).log"

cd "$ROOT_DIR"
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec
export PYTHONPATH="$ARCHIVE_DIR/src:$ROOT_DIR/src:${PYTHONPATH:-}"

mkdir -p "$CONFIG_DIR" "$PAIR_DIR" "$LOG_DIR" "$CKPT_DIR" "$INDEX_DIR"

if [[ ! -s "$PAIR_CSV" ]]; then
  echo "[OFFICE-STAGE1] building Office local-multihop weak pairs: $PAIR_CSV"
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
    --mgdcf-binarize-edges
else
  echo "[OFFICE-STAGE1] existing pair CSV found: $PAIR_CSV"
fi

if [[ ! -s "$PROXY_CSV" ]]; then
  echo "[OFFICE-STAGE1] building Office local-multihop ambiguity prior: $PROXY_CSV"
  python "$PROXY_SCRIPT" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TRAIN_CSV" \
    --semantic-embedding "$EMB_PATH" \
    --output-csv "$PROXY_CSV" \
    --semantic-topk 32 \
    --graph-topk 32 \
    --history-k 10
else
  echo "[OFFICE-STAGE1] existing proxy CSV found: $PROXY_CSV"
fi

python - "$BASE_CONFIG" "$CONFIG_PATH" "$CKPT_DIR" "$PAIR_CSV" "$PROXY_CSV" "$TRAIN_CSV" "$EMB_PATH" <<'PY'
import sys
import yaml

base_config, out_path, ckpt_dir, pair_csv, proxy_csv, train_csv, emb_path = sys.argv[1:8]
with open(base_config) as f:
    cfg = yaml.safe_load(f)

cfg["data_path"] = emb_path
cfg["train_csv"] = train_csv
cfg["semantic_embedding_path"] = emb_path
cfg["ambiguity_csv"] = proxy_csv
cfg["ambiguity_column"] = "offline_combined"
cfg["ckpt_dir"] = ckpt_dir

# Office port of the current strongest Industrial R690b-LMH tokenizer setting.
cfg["coarse_weight"] = 0.0
cfg["mid_weight"] = 0.0
cfg["local_weight"] = 0.0
cfg["l1_contrastive_pull_weight"] = 0.03
cfg["l2_contrastive_pull_weight"] = 0.01
cfg["l2_contrastive_mode"] = "graph_infonce"
cfg["l2_infonce_temperature"] = 0.1
cfg["l2_infonce_negative_pair_csv"] = pair_csv
cfg["l2_infonce_negative_pair_rule"] = "semantic_near_mid_graph_weak"
cfg["l2_infonce_use_pair_reliability"] = True
cfg["l3_contrastive_pull_weight"] = 0.02
cfg["semantic_coarse_weight"] = 0.0
cfg["semantic_mid_weight"] = 0.0
cfg["coarse_view_name"] = "coarse_purified"
cfg["mid_view_name"] = "local_multihop"
cfg["local_view_name"] = "local_purified"
cfg["local_multihop_alpha"] = 0.35
cfg["local_multihop_max_hop"] = 2
cfg["hierarchy_stopgrad_previous_levels"] = True
cfg["epochs"] = 10000
cfg["batch_size"] = 20480
cfg["lr"] = 0.001
cfg["num_emb_list"] = [256, 256, 256]

with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "[OFFICE-STAGE1] config=$CONFIG_PATH"
echo "[OFFICE-STAGE1] ckpt=$CKPT_DIR"
echo "[OFFICE-STAGE1] index=$INDEX_FILE"
echo "[OFFICE-STAGE1] log=$RUN_LOG"

if [[ -s "$GENERATE_SUMMARY" && -s "$INDEX_FILE" ]]; then
  echo "[OFFICE-STAGE1] generated artifacts already exist, skip tokenizer training."
else
  CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$TRAIN_SCRIPT" \
    "$CONFIG_PATH" "$CKPT_DIR" "$INDEX_FILE" "$GENERATE_SUMMARY" "$RUN_LOG"
fi

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
    "collision_rate": (len(sids) - len(c123)) / len(sids),
    "max_conflict": max(Counter(sids).values()) if sids else 0,
    "max_l1_bucket": max(c1.values()) if c1 else 0,
    "top5_l1_cover": sum(v for _, v in c1.most_common(5)),
    "top5_l1": c1.most_common(5),
}
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False))
PY

echo "[OFFICE-STAGE1] completed."
