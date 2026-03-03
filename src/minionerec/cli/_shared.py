from __future__ import annotations

from minionerec.config.loader import _construct, apply_overrides
from minionerec.common.io import read_yaml


def build_config(config_cls, config_path: str | None, overrides: list[str] | None):
    payload = read_yaml(config_path) if config_path else {}
    apply_overrides(payload, overrides or [])
    return _construct(config_cls, payload)
