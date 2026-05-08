#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

GPU_LIST="${1:-2,3,4,5}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${#GPU_ARRAY[@]}"

if [[ "$NPROC" -ne 4 ]]; then
  echo "ERROR: this RL config is calibrated for 4 GPUs; got GPU_LIST=$GPU_LIST (nproc=$NPROC)" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/l2_lmh_rl results/experiments/mgr_sid_l2_lmh_sweep_rl_eval_20260507

BRANCH_DIR="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers"
SFT_MODEL="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507/r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu/sft/final_checkpoint"
RL_MODEL="/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_rl_eval_20260507/r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu/rl/final_checkpoint"
RL_CONFIG="$BRANCH_DIR/configs/rl_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu.yaml"
EVAL_CONFIG="$BRANCH_DIR/configs/evaluate_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_rl_4gpu.yaml"

for required_path in \
  "$SFT_MODEL/config.json" \
  "$SFT_MODEL/model.safetensors" \
  "$SFT_MODEL/tokenizer.json" \
  "$RL_CONFIG" \
  "$EVAL_CONFIG" \
  "data_experiment/Amazon/r690b_lmh_l2_contrastive_pull_weight001/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "data_experiment/Amazon/r690b_lmh_l2_contrastive_pull_weight001/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "data_experiment/Amazon/r690b_lmh_l2_contrastive_pull_weight001/test/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  "data_experiment/Amazon/r690b_lmh_l2_contrastive_pull_weight001/index/Industrial_and_Scientific.index.json" \
  "data_experiment/Amazon/r690b_lmh_l2_contrastive_pull_weight001/index/Industrial_and_Scientific.item.json" \
  "data_experiment/Amazon/r690b_lmh_l2_contrastive_pull_weight001/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: required path missing: $required_path" >&2
    exit 1
  fi
done

echo "[CHAIN] start r690b_lmh_l2_contrastive_pull_weight001 RL on GPUs $GPU_LIST"
echo "[CHAIN] source SFT model: $SFT_MODEL"
echo "[CHAIN] recipe=title_history2sid_on + desc_align_p05 + ranking RL"
echo "[CHAIN] RL hyperparams: train_batch_size=16 grad_accum=4 num_generations=16 epochs=2 lr=1e-5"
bash ./rl.sh industrial "$RL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.num_processes=$NPROC"

if [[ ! -e "$RL_MODEL/config.json" ]]; then
  echo "ERROR: RL final checkpoint missing after training: $RL_MODEL" >&2
  exit 1
fi

echo "[CHAIN] start r690b_lmh_l2_contrastive_pull_weight001 RL evaluate on GPUs $GPU_LIST"
bash ./evaluate.sh rl industrial "$EVAL_CONFIG" \
  "runtime.cuda_visible_devices=$GPU_LIST" \
  "runtime.nproc_per_node=$NPROC"

echo "[CHAIN] r690b_lmh_l2_contrastive_pull_weight001 RL/evaluate completed"
