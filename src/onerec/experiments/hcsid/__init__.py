"""HCSID tokenizer training package.

HCSID stands for Hierarchical Collaborative Semantic ID.  The current
implementation is LMH-HCSID, which injects local-multihop collaborative
structure into the SID tokenizer while keeping the OneRec downstream stack
unchanged.
"""

from .config import HcsidTrainConfig, load_train_config
from .trainer import run_training

__all__ = ["HcsidTrainConfig", "load_train_config", "run_training"]
