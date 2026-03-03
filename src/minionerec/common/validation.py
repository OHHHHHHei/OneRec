from pathlib import Path


def require_exists(path: str | Path, label: str) -> Path:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"{label} not found: {target}")
    return target


def require_non_empty(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value
