from __future__ import annotations

import runpy
import sys

from minionerec.cli._shared import build_config
from minionerec.config.schema import EmbedConfig


def run_embed_cli(config_path: str | None, overrides: list[str] | None):
    config = build_config(EmbedConfig, config_path, overrides)
    argv = [
        "minionerec.sid.text2emb",
        "--dataset", config.data.dataset_name,
        "--root", config.data.data_dir,
        "--plm_name", config.extras.get("plm_name", "qwen"),
        "--plm_checkpoint", config.model.base_model,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_module("minionerec.sid.text2emb", run_name="__main__")
    finally:
        sys.argv = old_argv
