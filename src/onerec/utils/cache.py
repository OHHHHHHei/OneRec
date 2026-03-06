import hashlib
import os
from pathlib import Path
from typing import Any

from onerec.utils.io import read_pickle, write_pickle


def build_cache_path(class_name: str, sources: list[str], extra: list[str]) -> Path | None:
    if not sources:
        return None
    key_parts = [class_name]
    for source in sources:
        source_path = Path(source).resolve()
        key_parts.append(str(source_path))
        key_parts.append(str(source_path.stat().st_mtime if source_path.exists() else 0))
    key_parts.extend(extra)
    digest = hashlib.md5("|".join(key_parts).encode("utf-8")).hexdigest()
    cache_dir = Path(sources[0]).resolve().parent / ".cache"
    return cache_dir / f"{class_name}_{digest}.pkl"


def load_cache(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return read_pickle(path)


def save_cache(path: Path | None, payload: Any) -> None:
    if path is None:
        return
    os.makedirs(path.parent, exist_ok=True)
    write_pickle(path, payload)

