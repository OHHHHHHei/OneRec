from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactResolver:
    root: Path

    @classmethod
    def from_cwd(cls, cwd: str | Path) -> "ArtifactResolver":
        return cls(Path(cwd).resolve())

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.root / candidate).resolve()

    def data_dir(self) -> Path:
        return self.resolve("data")

    def output_dir(self) -> Path:
        return self.resolve("output")

    def results_dir(self) -> Path:
        return self.resolve("results")
