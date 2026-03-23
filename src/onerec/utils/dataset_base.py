from __future__ import annotations

import random
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from onerec.utils.tokenizer import WrappedTokenizer


class BaseDataset(Dataset):
    def __init__(self, tokenizer=None, max_len: int = 2048, test: bool = False, category: str = "", dedup: bool = False, seed: int | None = None):
        super().__init__()
        self.data = None
        self.inputs: list[Any] = []
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
        built_inputs = []
        for i in range(len(self.data)):
            sample = self.pre(i)
            if sample is not None:
                built_inputs.append(sample)
        self.inputs = built_inputs


class CSVBaseDataset(BaseDataset):
    def __init__(self, train_file: str, sample: int = -1, seed: int = 0, max_len: int = 2048, category: str = "", dedup: bool = False, tokenizer=None, test: bool = False):
        super().__init__(tokenizer=tokenizer, max_len=max_len, test=test, category=category, dedup=dedup, seed=seed)
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)


class JSONBaseDataset(BaseDataset):
    def __init__(self, item_file: str, index_file: str, tokenizer=None, max_len: int = 2048, test: bool = False, category: str = "", dedup: bool = False, seed: int | None = None):
        super().__init__(tokenizer=tokenizer, max_len=max_len, test=test, category=category, dedup=dedup, seed=seed)
        from onerec.utils.io import read_json

        self.item_feat = read_json(item_file)
        self.indices = read_json(index_file)
