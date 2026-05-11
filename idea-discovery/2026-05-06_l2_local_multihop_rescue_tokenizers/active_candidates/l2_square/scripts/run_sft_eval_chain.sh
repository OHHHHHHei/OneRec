#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/leejt/OneRec"
cd "$REPO_ROOT"

exec bash idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/scripts/experiment_r690b_lmh_l2_square_b025_sft_eval_4gpu_chain.sh "$@"
