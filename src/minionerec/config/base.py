from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass
class StageConfig:
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError("StageConfig must be a dataclass")
        return asdict(self)
