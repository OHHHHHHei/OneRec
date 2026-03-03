from __future__ import annotations

import argparse
import importlib

from minionerec.common.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniOneRec unified CLI")
    parser.add_argument("stage", choices=["preprocess", "embed", "sid-train", "sid-generate", "convert", "sft", "rl", "evaluate"])
    parser.add_argument("--config", dest="config_path", default=None)
    parser.add_argument("overrides", nargs="*")
    return parser


def main():
    configure_logging()
    args = build_parser().parse_args()
    dispatch = {
        "preprocess": ("minionerec.cli.preprocess", "run_preprocess_cli"),
        "embed": ("minionerec.cli.embed", "run_embed_cli"),
        "sid-train": ("minionerec.cli.sid", "run_sid_train_cli"),
        "sid-generate": ("minionerec.cli.sid", "run_sid_generate_cli"),
        "convert": ("minionerec.cli.convert", "run_convert_cli"),
        "sft": ("minionerec.cli.sft", "run_sft_cli"),
        "rl": ("minionerec.cli.rl", "run_rl_cli"),
        "evaluate": ("minionerec.cli.evaluate", "run_evaluate_cli"),
    }
    module_name, fn_name = dispatch[args.stage]
    module = importlib.import_module(module_name)
    result = getattr(module, fn_name)(args.config_path, args.overrides)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
