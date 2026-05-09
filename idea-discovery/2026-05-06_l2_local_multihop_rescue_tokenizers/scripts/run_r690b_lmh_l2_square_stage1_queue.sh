#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 <gpu_id> <queue_name> <variant> [<variant> ...]" >&2
  echo "Variants: square_dominant_b025, square_only" >&2
  exit 2
fi

GPU_ID="$1"
QUEUE_NAME="$2"
shift 2

ROOT_DIR="/home/leejt/OneRec"
BRANCH_DIR="$ROOT_DIR/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
BASE_CONFIG="$BRANCH_DIR/configs/sid_train_industrial_mgr_sid_r690b_l2_infonce_local_multihop.yaml"
TRAIN_SCRIPT="$BRANCH_DIR/scripts/train_generate_tokenizer_lmh.sh"
CONFIG_DIR="$BRANCH_DIR/configs/stage1_l2_square"
CKPT_BASE="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_square_20260509"
INDEX_DIR="$CKPT_BASE/generated_indices"
LOG_DIR="$ROOT_DIR/logs/l2_square_stage1"
PAIR_CSV="$BRANCH_DIR/pairs/R690bLMH_all_mid_graph_weak_pairs.csv"

cd "$ROOT_DIR"
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

if [[ ! -s "$PAIR_CSV" ]]; then
  echo "[ERROR] missing negative-pair CSV: $PAIR_CSV" >&2
  exit 1
fi

mkdir -p "$CONFIG_DIR" "$INDEX_DIR" "$LOG_DIR"

variant_params() {
  case "$1" in
    square_dominant_b025)
      echo "r690b_lmh_l2_square_dominant_b025 0.25 1.0"
      ;;
    square_only)
      echo "r690b_lmh_l2_square_only 0.0 1.0"
      ;;
    *)
      echo "[ERROR] unknown variant: $1" >&2
      exit 2
      ;;
  esac
}

write_config() {
  local config_path="$1"
  local ckpt_dir="$2"
  local base_weight="$3"
  local alpha="$4"

  python - "$BASE_CONFIG" "$config_path" "$ckpt_dir" "$base_weight" "$alpha" <<'PY'
import sys
import yaml

base_config, out_path, ckpt_dir, base_weight, alpha = sys.argv[1:6]
with open(base_config) as f:
    cfg = yaml.safe_load(f)

cfg["ckpt_dir"] = ckpt_dir

# Keep hyperparameters aligned with the current strongest R690b-LMH mainline.
cfg["coarse_weight"] = 0.0
cfg["mid_weight"] = 0.0
cfg["local_weight"] = 0.0
cfg["l1_contrastive_pull_weight"] = 0.03
cfg["l2_contrastive_pull_weight"] = 0.01
cfg["l2_contrastive_mode"] = "graph_infonce"
cfg["l2_infonce_temperature"] = 0.1
cfg["l2_infonce_negative_pair_csv"] = "./idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/pairs/R690bLMH_all_mid_graph_weak_pairs.csv"
cfg["l2_infonce_negative_pair_rule"] = "semantic_near_mid_graph_weak"
cfg["l3_contrastive_pull_weight"] = 0.02
cfg["l3_contrastive_mode"] = "pairwise_pull"
cfg["semantic_coarse_weight"] = 0.0
cfg["semantic_mid_weight"] = 0.0
cfg["coarse_view_name"] = "coarse_purified"
cfg["mid_view_name"] = "local_multihop"
cfg["local_view_name"] = "local_purified"
cfg["local_multihop_base_weight"] = float(base_weight)
cfg["local_multihop_alpha"] = float(alpha)
cfg["local_multihop_max_hop"] = 2
cfg["hierarchy_stopgrad_previous_levels"] = True
cfg["epochs"] = 10000
cfg["batch_size"] = 20480
cfg["lr"] = 0.001
cfg["num_emb_list"] = [256, 256, 256]

with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

diagnose_index() {
  local index_file="$1"
  local out_file="$2"
  python - "$index_file" "$out_file" <<'PY'
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
}

passes_gate() {
  local structure_summary="$1"
  python - "$structure_summary" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
ok = (
    d["collision_rate"] <= 0.02
    and d["active_l1"] >= 30
    and d["unique_l12"] >= 1500
    and d["max_conflict"] <= 5
)
raise SystemExit(0 if ok else 1)
PY
}

echo "[QUEUE:$QUEUE_NAME] GPU=$GPU_ID variants=$*"

for variant in "$@"; do
  read -r tag base_weight alpha <<<"$(variant_params "$variant")"
  ckpt_dir="$CKPT_BASE/$tag"
  config_path="$CONFIG_DIR/sid_train_industrial_${tag}.yaml"
  index_file="$INDEX_DIR/Industrial_and_Scientific.${tag}.index.json"
  generate_summary="$BRANCH_DIR/${tag}_generate_summary.json"
  structure_summary="$BRANCH_DIR/${tag}_structure_summary.json"
  run_log="$LOG_DIR/experiment_mgr_sid_${tag}_$(date +%Y%m%d).log"

  echo "============================================"
  echo "[QUEUE:$QUEUE_NAME] variant=$variant tag=$tag"
  echo "[QUEUE:$QUEUE_NAME] L2 graph = RowNorm(${base_weight} * A_local + ${alpha} * A_local^2)"
  echo "[QUEUE:$QUEUE_NAME] config=$config_path"
  echo "[QUEUE:$QUEUE_NAME] ckpt=$ckpt_dir"
  echo "============================================"

  if [[ -s "$generate_summary" && -s "$index_file" ]]; then
    echo "[QUEUE:$QUEUE_NAME] existing generated artifacts found, skip training."
  else
    write_config "$config_path" "$ckpt_dir" "$base_weight" "$alpha"
    mkdir -p "$ckpt_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$TRAIN_SCRIPT" \
      "$config_path" "$ckpt_dir" "$index_file" "$generate_summary" "$run_log"
  fi

  if [[ ! -s "$index_file" ]]; then
    echo "[ERROR] missing generated index after run: $index_file" >&2
    exit 1
  fi

  echo "[QUEUE:$QUEUE_NAME] structure diagnostics:"
  diagnose_index "$index_file" "$structure_summary"

  if ! passes_gate "$structure_summary"; then
    echo "[QUEUE:$QUEUE_NAME] gate failed after $tag; stop this queue to avoid wasting GPU."
    exit 0
  fi
done

echo "[QUEUE:$QUEUE_NAME] completed."
