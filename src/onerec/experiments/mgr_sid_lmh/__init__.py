"""Backward-compatible import path for the LMH-HCSID tokenizer mainline.

New code should import :mod:`onerec.experiments.hcsid` directly. This module
remains only so existing launch scripts keep working.
"""

from onerec.experiments.hcsid import load_train_config, run_training

__all__ = ["load_train_config", "run_training"]
