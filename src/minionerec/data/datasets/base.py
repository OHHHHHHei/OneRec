from __future__ import annotations

import random
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from minionerec.common.tokenizer import WrappedTokenizer
from minionerec.data.cache import build_cache_path, load_cache, save_cache


class BaseDataset(Dataset):
    def __init__(self, tokenizer=None, max_len: int = 2048, test: bool = False, category: str = "", dedup: bool = False, seed: int | None = None):
        super().__init__()
        self.data = None
        self.inputs: list[Any] = []
        self._cache_sources: list[str] = []
        self.max_len = max_len
        self.test = test
        self.category = category
        self.dedup = dedup
        if tokenizer is not None:
            self.tokenizer = WrappedTokenizer(tokenizer)
        if seed is not None:
            random.seed(seed)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index]

    def generate_prompt(self, data_point: dict[str, Any]) -> str:
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def pre(self, idx: int):
        raise NotImplementedError

    def get_history(self, row):
        raise NotImplementedError

    def get_all(self) -> list[Any]:
        if self.data is None:
            return []
        if isinstance(self.data, pd.DataFrame):
            return [self.get_history(self.data.iloc[i]) for i in range(len(self.data))]
        return list(self.data)

    def get_inputs(self) -> None:
        cache_path = build_cache_path(
            self.__class__.__name__,
            self._cache_sources,
            [str(self.max_len), str(self.category), str(self.dedup), str(self.test)],
        )
        cached = load_cache(cache_path)
        if isinstance(cached, dict) and "inputs" in cached:
            self.inputs = cached["inputs"]
            if "prompt2history" in cached and hasattr(self, "prompt2history"):
                self.prompt2history = cached["prompt2history"]
            if "history2target" in cached and hasattr(self, "history2target"):
                self.history2target = cached["history2target"]
            return

        built_inputs = []
        for i in range(len(self.data)):
            sample = self.pre(i)
            if sample is not None:
                built_inputs.append(sample)
        self.inputs = built_inputs

        payload = {"inputs": self.inputs}
        if hasattr(self, "prompt2history"):
            payload["prompt2history"] = self.prompt2history
        if hasattr(self, "history2target"):
            payload["history2target"] = self.history2target
        save_cache(cache_path, payload)


class CSVBaseDataset(BaseDataset):
    def __init__(self, train_file: str, sample: int = -1, seed: int = 0, max_len: int = 2048, category: str = "", dedup: bool = False, tokenizer=None, test: bool = False):
        super().__init__(tokenizer=tokenizer, max_len=max_len, test=test, category=category, dedup=dedup, seed=seed)
        self.data = pd.read_csv(train_file)
        self._cache_sources = [train_file]
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)


class JSONBaseDataset(BaseDataset):
    def __init__(self, item_file: str, index_file: str, tokenizer=None, max_len: int = 2048, test: bool = False, category: str = "", dedup: bool = False, seed: int | None = None):
        super().__init__(tokenizer=tokenizer, max_len=max_len, test=test, category=category, dedup=dedup, seed=seed)
        from minionerec.common.io import read_json

        self.item_feat = read_json(item_file)
        self.indices = read_json(index_file)
        self._cache_sources = [item_file, index_file]
