from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT.parent / "src" / "minionerec"

__path__ = [str(_ROOT)]
if _SRC.exists():
    __path__.append(str(_SRC))
