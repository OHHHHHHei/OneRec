#!/usr/bin/env python
"""Canonical trainer entrypoint for the current LMH SID tokenizer line."""

from __future__ import annotations

import argparse
import json

from .bridge import load_train_config, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ckpt_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "device": args.device,
        "epochs": args.epochs,
        "ckpt_dir": args.ckpt_dir,
    }
    config = load_train_config(args.config, overrides=overrides)
    summary = run_training(config)
    print(json.dumps({"run_dir": summary["run_dir"], "best": summary["best"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

