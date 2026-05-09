#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${#GPU_ARRAY[@]}"
TOKENIZER_GPU="${GPU_ARRAY[0]}"
RUN_SFT="${RUN_SFT:-1}"

if [[ "$RUN_SFT" == "1" && "$NPROC" -ne 4 ]]; then
  echo "ERROR: this full chain is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
BASE_TOKENIZER_CONFIG="$BRANCH_DIR/configs/sid_train_industrial_mgr_sid_r690b_l2_infonce_local_multihop.yaml"
TRAIN_SCRIPT="$BRANCH_DIR/scripts/train_generate_tokenizer.sh"
PREPARE_SCRIPT="research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/scripts/experiment_mgr_sid_prepare_data.py"

VARIANT="r690b_lmh_l2_weight001_l3_ranking002"
TOKENIZER_EXP_ROOT="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l3_ranking_20260509"
CKPT_ROOT="$TOKENIZER_EXP_ROOT/$VARIANT"
INDEX_DIR="$TOKENIZER_EXP_ROOT/generated_indices"
INDEX_PATH="$INDEX_DIR/Industrial_and_Scientific.${VARIANT}.index.json"
GENERATE_SUMMARY="$BRANCH_DIR/${VARIANT}_generate_summary.json"
STRUCTURE_SUMMARY="$BRANCH_DIR/${VARIANT}_structure_summary.json"
TOKENIZER_CONFIG="$BRANCH_DIR/configs/stage1_l3_ranking/sid_train_industrial_${VARIANT}.yaml"
TOKENIZER_LOG="logs/l3_ranking/experiment_mgr_sid_${VARIANT}_$(date +%Y%m%d).log"

SFT_EXP_ROOT="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l3_ranking_sft_eval_20260509"
SFT_CONFIG="$BRANCH_DIR/configs/sft_industrial_${VARIANT}_title_on_desc_p05_4gpu.yaml"
EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_industrial_${VARIANT}_title_on_desc_p05_4gpu.yaml"
SFT_MODEL="$SFT_EXP_ROOT/${VARIANT}_title_on_desc_p05_4gpu/sft/final_checkpoint"
RESULT_DIR="results/experiments/mgr_sid_l3_ranking_sft_eval_20260509"
RESULT_PATH="$RESULT_DIR/final_result_sft_mgr_${VARIANT}_title_on_desc_p05_4gpu_Industrial_and_Scientific.json"
DATA_ROOT="data_experiment/Amazon/$VARIANT"

PAIR_CSV="$BRANCH_DIR/pairs/R690bLMH_all_mid_graph_weak_pairs.csv"
if [[ ! -s "$PAIR_CSV" ]]; then
  echo "ERROR: missing negative-pair CSV: $PAIR_CSV" >&2
  exit 1
fi

mkdir -p "$(dirname "$TOKENIZER_CONFIG")" "$INDEX_DIR" "$(dirname "$TOKENIZER_LOG")" "$RESULT_DIR"

python - "$BASE_TOKENIZER_CONFIG" "$TOKENIZER_CONFIG" "$CKPT_ROOT" <<'PY'
import sys
import yaml

base_config, out_path, ckpt_root = sys.argv[1:4]
with open(base_config) as f:
    cfg = yaml.safe_load(f)

cfg["ckpt_dir"] = ckpt_root
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
cfg["l3_contrastive_mode"] = "ranking"
cfg["l3_ranking_margin"] = 0.1
cfg["l3_ranking_positive_topk"] = 8
cfg["l3_ranking_negative_topk"] = 16
cfg["l3_ranking_negative_pair_csv"] = "./idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/pairs/R690bLMH_all_mid_graph_weak_pairs.csv"
cfg["l3_ranking_negative_pair_rule"] = "semantic_near_mid_graph_weak"
cfg["l3_ranking_use_pair_reliability"] = True
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

echo "[L3-RANK] tokenizer config: $TOKENIZER_CONFIG"
if [[ -s "$GENERATE_SUMMARY" && -s "$INDEX_PATH" ]]; then
  echo "[L3-RANK] existing generated tokenizer artifacts found; skip tokenizer training."
else
  CUDA_VISIBLE_DEVICES="$TOKENIZER_GPU" bash "$TRAIN_SCRIPT" \
    "$TOKENIZER_CONFIG" "$CKPT_ROOT" "$INDEX_PATH" "$GENERATE_SUMMARY" "$TOKENIZER_LOG"
fi

if [[ ! -s "$INDEX_PATH" ]]; then
  echo "ERROR: generated index missing: $INDEX_PATH" >&2
  exit 1
fi

python - "$INDEX_PATH" "$STRUCTURE_SUMMARY" <<'PY'
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
    "max_conflict": max(c123.values()) if c123 else 0,
    "max_l1_bucket": max(c1.values()) if c1 else 0,
    "top5_l1_cover": sum(v for _, v in c1.most_common(5)),
    "top5_l1": c1.most_common(5),
}
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False))
if not (
    summary["collision_rate"] <= 0.02
    and summary["active_l1"] >= 30
    and summary["unique_l12"] >= 1500
    and summary["max_conflict"] <= 5
):
    raise SystemExit("Tokenizer structure gate failed; stop before SFT.")
PY

if [[ "$RUN_SFT" != "1" ]]; then
  echo "[L3-RANK] RUN_SFT=$RUN_SFT; stop after tokenizer generation and structure gate."
  echo "[L3-RANK] resume later with: bash $0 2,3,4,5"
  exit 0
fi

echo "[L3-RANK] prepare data_experiment variant: $VARIANT"
python "$PREPARE_SCRIPT" --variant "${VARIANT}=${INDEX_PATH}"

python - "$SFT_CONFIG" "$EVAL_CONFIG" "$SFT_EXP_ROOT" "$VARIANT" "$RESULT_PATH" <<'PY'
import sys
import yaml

sft_out, eval_out, sft_root, variant, result_path = sys.argv[1:6]

sft = {
    "model": {"base_model": "../Qwen3-1.7B", "train_from_scratch": False},
    "data": {
        "train_file": f"./data_experiment/Amazon/{variant}/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "eval_file": f"./data_experiment/Amazon/{variant}/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "sid_index_path": f"./data_experiment/Amazon/{variant}/index/Industrial_and_Scientific.index.json",
        "item_meta_path": f"./data_experiment/Amazon/{variant}/index/Industrial_and_Scientific.item.json",
        "category": "Industrial_and_Scientific",
    },
    "training": {
        "batch_size": 1024,
        "micro_batch_size": 2,
        "cutoff_len": 512,
        "enable_title_history2sid_dataset": True,
        "enable_title_description_alignment": True,
        "description_task_probability": 0.5,
        "num_epochs": 10,
        "learning_rate": 0.0003,
        "seed": 42,
        "freeze_llm": False,
        "group_by_length": False,
        "warmup_steps": 20,
        "load_best_model_at_end": True,
        "early_stopping_patience": 3,
        "eval_step": 0.05,
    },
    "logging": {
        "wandb_project": "OneRec",
        "wandb_run_name": f"sft_mgr_{variant}_title_on_desc_p05_4gpu_industrial",
        "report_to": "wandb",
    },
    "output": {
        "output_dir": f"{sft_root}/{variant}_title_on_desc_p05_4gpu/sft",
        "save_total_limit": 2,
    },
    "runtime": {
        "launcher": "torchrun",
        "cuda_visible_devices": "2,3,4,5",
        "nproc_per_node": 4,
    },
}

eval_cfg = {
    "model": {"base_model": f"{sft_root}/{variant}_title_on_desc_p05_4gpu/sft/final_checkpoint"},
    "data": {
        "test_file": f"./data_experiment/Amazon/{variant}/test/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "info_file": f"./data_experiment/Amazon/{variant}/info/Industrial_and_Scientific_5_2016-10-2018-11.txt",
        "category": "Industrial_and_Scientific",
    },
    "training": {"seed": 42},
    "output": {"output_dir": result_path},
    "batch_size": 8,
    "K": 0,
    "num_beams": 50,
    "max_new_tokens": 256,
    "length_penalty": 0.0,
    "temperature": 1.0,
    "guidance_scale": None,
    "runtime": {
        "launcher": "parallel",
        "parallel": True,
        "cuda_visible_devices": "2,3,4,5",
        "nproc_per_node": 4,
    },
}

with open(sft_out, "w") as f:
    yaml.safe_dump(sft, f, sort_keys=False)
with open(eval_out, "w") as f:
    yaml.safe_dump(eval_cfg, f, sort_keys=False)
PY

for required_path in \
  "$DATA_ROOT/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/test/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "$DATA_ROOT/index/Industrial_and_Scientific.index.json" \
  "$DATA_ROOT/index/Industrial_and_Scientific.item.json" \
  "$DATA_ROOT/info/Industrial_and_Scientific_5_2016-10-2018-11.txt" \
  "$SFT_CONFIG" \
  "$EVAL_CONFIG"; do
  if [[ ! -f "$required_path" ]]; then
    echo "ERROR: required path missing: $required_path" >&2
    exit 1
  fi
done

echo "[L3-RANK] start SFT on GPUs $GPU_LIST"
bash ./sft.sh industrial "$SFT_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -d "$SFT_MODEL" ]]; then
  echo "ERROR: SFT final checkpoint missing: $SFT_MODEL" >&2
  exit 1
fi

echo "[L3-RANK] start evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh sft industrial "$EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

if [[ ! -f "$RESULT_PATH" ]]; then
  echo "ERROR: eval result missing: $RESULT_PATH" >&2
  exit 1
fi

echo "[L3-RANK] completed: $RESULT_PATH"
