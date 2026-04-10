#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onerec.experiments.mgr_sid.train_v2 import load_train_config, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental MGR-SID tokenizer v2 training runner.")
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
