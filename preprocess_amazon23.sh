#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/_common.sh"

DEFAULT_CONFIG="config/preprocess_amazon23.yaml"
resolve_config_path "$DEFAULT_CONFIG" "$@"

exec python -m onerec.main preprocess --config "$CONFIG_PATH" "${PASSTHROUGH_ARGS[@]}"
