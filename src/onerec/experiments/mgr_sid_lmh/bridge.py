"""Bridge from the clean LMH mainline package to the validated legacy trainer.

The active LMH tokenizer line was developed in an archived experiment package.
This bridge isolates that dependency behind a stable current-path API so future
launch scripts no longer import archived modules directly.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
LEGACY_WORKSPACE = (
    REPO_ROOT
    / "research-progress-log"
    / "archive"
    / "2026-04-24_mgr_sid_negative_research_archive"
    / "archived_workspace"
)
LEGACY_EXPERIMENTS_DIR = LEGACY_WORKSPACE / "src" / "onerec" / "experiments"


def _attach_legacy_experiments_namespace() -> None:
    """Allow importing legacy experiment modules under ``onerec.experiments``."""
    if not LEGACY_EXPERIMENTS_DIR.exists():
        raise FileNotFoundError(f"Missing legacy experiments directory: {LEGACY_EXPERIMENTS_DIR}")

    import onerec.experiments as experiments_pkg

    legacy_path = str(LEGACY_EXPERIMENTS_DIR)
    if legacy_path not in experiments_pkg.__path__:
        experiments_pkg.__path__.append(legacy_path)


def _legacy_train_module():
    _attach_legacy_experiments_namespace()
    return importlib.import_module("onerec.experiments.mgr_sid.train_v2")


def load_train_config(config_path: str, overrides: dict[str, Any] | None = None):
    """Load an LMH tokenizer config using the current mainline API."""
    return _legacy_train_module().load_train_config(config_path, overrides=overrides)


def run_training(config) -> dict[str, Any]:
    """Run current LMH tokenizer training via the validated backend."""
    return _legacy_train_module().run_training(config)

