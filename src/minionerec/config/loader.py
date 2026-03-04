from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

from minionerec.common.io import read_yaml

T = TypeVar("T")


def _deep_set(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def apply_overrides(payload: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override: {override}")
        key, raw_value = override.split("=", 1)
        _deep_set(payload, key, _coerce_value(raw_value))
    return payload


def _construct(cls: type[T], payload: Any) -> T:
    if not is_dataclass(cls):
        return payload
    kwargs = {}
    consumed = set()
    for field in fields(cls):
        if not isinstance(payload, dict) or field.name not in payload:
            continue
        value = payload[field.name]
        if value is None:
            kwargs[field.name] = None
            consumed.add(field.name)
            continue
        field_type = field.type
        try:
            kwargs[field.name] = _construct(field_type, value)
        except TypeError:
            kwargs[field.name] = value
        consumed.add(field.name)
    if "extras" in {field.name for field in fields(cls)} and isinstance(payload, dict):
        kwargs["extras"] = {key: value for key, value in payload.items() if key not in consumed}
    return cls(**kwargs)


def load_config(config_cls: type[T], config_path: str, overrides: list[str] | None = None) -> T:
    payload = read_yaml(config_path)
    payload = apply_overrides(payload, overrides or [])
    return _construct(config_cls, payload)
