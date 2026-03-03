from __future__ import annotations

import runpy
import sys
from pathlib import Path

from minionerec.cli._shared import build_config
from minionerec.config.schema import PreprocessConfig


def run_preprocess_cli(config_path: str | None, overrides: list[str] | None):
    config = build_config(PreprocessConfig, config_path, overrides)
    target = "minionerec.preprocess.amazon18" if config.extras.get("source", "amazon18") == "amazon18" else "minionerec.preprocess.amazon23"
    argv = [target]
    for key, value in config.extras.items():
        if key == "source":
            continue
        argv.extend([f"--{key}", str(value)])
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_module(target, run_name="__main__")
    finally:
        sys.argv = old_argv
