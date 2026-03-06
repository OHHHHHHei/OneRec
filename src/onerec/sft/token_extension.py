import json
from pathlib import Path


class TokenExtender:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self._tokens = None

    def get_new_tokens(self) -> list[str]:
        if self._tokens is not None:
            return self._tokens
        with open(self.index_path, "r", encoding="utf-8") as handle:
            indices = json.load(handle)
        tokens = set()
        for item_tokens in indices.values():
            for token in item_tokens:
                tokens.add(token)
        self._tokens = sorted(tokens)
        return self._tokens
