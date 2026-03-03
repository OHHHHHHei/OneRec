import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    target = ensure_parent(path)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    target = ensure_parent(path)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    target = ensure_parent(path)
    frame.to_csv(target, index=False)


def read_npy(path: str | Path) -> np.ndarray:
    return np.load(path)


def write_npy(path: str | Path, array: np.ndarray) -> None:
    target = ensure_parent(path)
    np.save(target, array)


def read_pickle(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def write_pickle(path: str | Path, payload: Any) -> None:
    target = ensure_parent(path)
    with open(target, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
