from __future__ import annotations

import ast
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_id_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    result: list[int] = []
    for item in parsed:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


class IndexedEmbDataset(Dataset):
    def __init__(self, data_path: str):
        embeddings = np.load(data_path).astype(np.float32)
        embeddings[np.isnan(embeddings)] = 0.0
        embeddings[np.isinf(embeddings)] = 0.0
        self.embeddings = embeddings
        self.dim = embeddings.shape[-1]

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        return index, torch.tensor(self.embeddings[index], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.embeddings)
