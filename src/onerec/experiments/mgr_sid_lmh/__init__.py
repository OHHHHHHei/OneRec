"""Current MGR-SID local-multihop tokenizer mainline.

This package is the canonical entrypoint for the active SID tokenizer research
line. It intentionally wraps the historically validated trainer while giving
new experiments a clean import path.
"""

from .bridge import load_train_config, run_training

__all__ = ["load_train_config", "run_training"]

