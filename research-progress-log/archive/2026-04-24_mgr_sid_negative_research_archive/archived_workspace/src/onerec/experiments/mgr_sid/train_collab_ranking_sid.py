from __future__ import annotations

from .train_v2 import MgrSidV2TrainConfig as CollabRankingSidTrainConfig
from .train_v2 import load_train_config as _load_train_config
from .train_v2 import run_training as _run_training


def _validate_mainline_config(cfg: CollabRankingSidTrainConfig) -> CollabRankingSidTrainConfig:
    if cfg.l2_ranking_contrastive_weight <= 0.0:
        raise ValueError(
            "Mainline collaborative-ranking SID requires l2_ranking_contrastive_weight > 0."
        )
    if cfg.mid_weight != 0.0:
        raise ValueError("Mainline collaborative-ranking SID expects mid_weight = 0.0.")
    if cfg.selective_separation_weight != 0.0:
        raise ValueError(
            "Mainline collaborative-ranking SID expects selective_separation_weight = 0.0."
        )
    return cfg


def load_train_config(
    config_path: str,
    overrides: dict[str, object] | None = None,
) -> CollabRankingSidTrainConfig:
    cfg = _load_train_config(config_path, overrides=overrides)
    return _validate_mainline_config(cfg)


def run_training(cfg: CollabRankingSidTrainConfig):
    _validate_mainline_config(cfg)
    return _run_training(cfg)
