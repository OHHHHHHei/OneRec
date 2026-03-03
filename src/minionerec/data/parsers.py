import ast
from typing import Any


def parse_sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, list):
            return parsed
        raise ValueError(f"Expected list literal, got: {value}")
    raise TypeError(f"Unsupported sequence value: {type(value)!r}")
