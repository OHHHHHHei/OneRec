from __future__ import annotations

import warnings
from pathlib import Path

from minionerec.config.loader import _construct, apply_overrides
from minionerec.common.io import read_yaml


_STAGE_DEFAULTS = {
    "sft": ("flows/sft/default.yaml", "configs/stages/sft/default.yaml"),
    "rl": ("flows/rl/default.yaml", "configs/stages/rl/default.yaml"),
    "evaluate": ("flows/evaluate/default.yaml", "configs/stages/evaluate/default.yaml"),
    "preprocess": ("configs/stages/preprocess/amazon18.yaml", "configs/stages/preprocess/amazon18.yaml"),
    "embed": ("configs/stages/embed/default.yaml", "configs/stages/embed/default.yaml"),
    "sid-train": ("configs/stages/sid/rqvae_train.yaml", "configs/stages/sid/rqvae_train.yaml"),
    "sid-generate": ("configs/stages/sid/rqvae_generate.yaml", "configs/stages/sid/rqvae_generate.yaml"),
    "convert": ("configs/stages/convert/default.yaml", "configs/stages/convert/default.yaml"),
}


def _resolve_config_path(stage: str, config_path: str | None) -> str:
    if config_path:
        normalized = config_path.replace("\\", "/")
        if normalized.startswith("configs/stages/"):
            warnings.warn(
                f"Config path `{config_path}` is deprecated for stage `{stage}`. "
                f"Prefer flow config path `{_STAGE_DEFAULTS.get(stage, ('', ''))[0]}`.",
                DeprecationWarning,
                stacklevel=3,
            )
        return config_path

    preferred, fallback = _STAGE_DEFAULTS.get(stage, ("", ""))
    if preferred and Path(preferred).exists():
        return preferred
    if fallback and Path(fallback).exists():
        if preferred != fallback:
            warnings.warn(
                f"Flow config `{preferred}` not found for stage `{stage}`. Falling back to `{fallback}`.",
                RuntimeWarning,
                stacklevel=3,
            )
        return fallback
    return ""


def build_config(config_cls, config_path: str | None, overrides: list[str] | None, stage: str):
    resolved_path = _resolve_config_path(stage, config_path)
    payload = read_yaml(resolved_path) if resolved_path else {}
    apply_overrides(payload, overrides or [])
    return _construct(config_cls, payload)
