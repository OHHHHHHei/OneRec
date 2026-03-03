from __future__ import annotations

import runpy
import sys

from minionerec.cli._shared import build_config
from minionerec.config.schema import SidGenerateConfig, SidTrainConfig


def run_sid_train_cli(config_path: str | None, overrides: list[str] | None):
    config = build_config(SidTrainConfig, config_path, overrides)
    kind = config.extras.get("kind", "rqvae")
    module_name = {
        "rqvae": "minionerec.sid.quantizers.rqvae",
        "rqkmeans_faiss": "minionerec.sid.quantizers.rqkmeans_faiss",
        "rqkmeans_constrained": "minionerec.sid.quantizers.rqkmeans_constrained",
        "rqkmeans_plus": "minionerec.sid.quantizers.rqkmeans_plus",
    }[kind]
    argv = [module_name]
    for key, value in config.extras.items():
        if key == "kind":
            continue
        argv.extend([f"--{key}", str(value)])
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = old_argv


def run_sid_generate_cli(config_path: str | None, overrides: list[str] | None):
    config = build_config(SidGenerateConfig, config_path, overrides)
    kind = config.extras.get("kind", "rqvae")
    module_name = "minionerec.sid.generate.rqvae_indices" if kind == "rqvae" else "minionerec.sid.generate.rqkmeans_plus_indices"
    argv = [module_name]
    for key, value in config.extras.items():
        if key == "kind":
            continue
        argv.extend([f"--{key}", str(value)])
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = old_argv
