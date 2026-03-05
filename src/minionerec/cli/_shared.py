from __future__ import annotations

import warnings
from pathlib import Path

from minionerec.config.loader import _construct, apply_overrides
from minionerec.common.io import read_yaml


_STAGE_DEFAULTS = {
    "sft": "flows/sft/default.yaml",
    "rl": "flows/rl/default.yaml",
    "evaluate": "flows/evaluate/default.yaml",
    "preprocess": "configs/stages/preprocess/amazon18.yaml",
    "embed": "configs/stages/embed/default.yaml",
    "sid-train": "configs/stages/sid/rqvae_train.yaml",
    "sid-generate": "configs/stages/sid/rqvae_generate.yaml",
    "convert": "configs/stages/convert/default.yaml",
}

_FLOW_STAGES = {"sft", "rl", "evaluate"}

_LEGACY_PATHS = {
    "sft": "configs/stages/sft/default.yaml",
    "rl": "configs/stages/rl/default.yaml",
    "evaluate": "configs/stages/evaluate/default.yaml",
}


def _resolve_config_path(stage: str, config_path: str | None) -> str:
    preferred = _STAGE_DEFAULTS.get(stage, "")
    legacy = _LEGACY_PATHS.get(stage)

    if config_path:
        normalized = config_path.replace("\\", "/")
        if stage in _FLOW_STAGES and legacy and normalized == legacy:
            raise ValueError(
                f"Config path `{config_path}` is deprecated for stage `{stage}`. "
                f"Use `{preferred}`."
            )
        return config_path

    if preferred and Path(preferred).exists():
        return preferred

    if stage in _FLOW_STAGES:
        raise FileNotFoundError(
            f"Default flow config not found for stage `{stage}`: `{preferred}`. "
            "Please restore the file or pass --config <path> explicitly."
        )

    if preferred:
        warnings.warn(
            f"Default config path for stage `{stage}` does not exist: `{preferred}`.",
            RuntimeWarning,
            stacklevel=3,
        )
    return preferred


def build_config(config_cls, config_path: str | None, overrides: list[str] | None, stage: str):
    resolved_path = _resolve_config_path(stage, config_path)
    payload = read_yaml(resolved_path) if resolved_path else {}
    apply_overrides(payload, overrides or [])
    return _construct(config_cls, payload)
