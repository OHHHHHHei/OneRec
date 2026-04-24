#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec

RL_CONFIG="config/experiments/rl_industrial_mgr_tokenizer_v2_title_on_desc_p05.yaml"
EVAL_CONFIG="config/experiments/evaluate_industrial_mgr_tokenizer_v2_title_on_desc_p05_rl.yaml"

./rl.sh industrial --config "$RL_CONFIG"
./evaluate.sh industrial --config "$EVAL_CONFIG"
