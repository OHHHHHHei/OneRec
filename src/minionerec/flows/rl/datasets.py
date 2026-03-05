from __future__ import annotations

import random

from minionerec.data.datasets.base import CSVBaseDataset, JSONBaseDataset
from minionerec.data.parsers import parse_sequence


class SidDataset(CSVBaseDataset):
    def __init__(self, train_file, max_len=2048, sample=-1, seed=0, category="", dedup=False):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer=None, test=False)
        self.prompt2history = {}
        self.history2target = {}
        self.get_inputs()

    def get_history(self, row):
        history_sids = parse_sequence(row["history_item_sid"])
        history = ", ".join(history_sids)
        history_str = "::".join(history_sids)
        target_sid = str(row["item_sid"])
        last_history = history_sids[-1] if history_sids else None
        return {
            "input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
            "output": target_sid + "\n",
            "history_str": history_str,
            "dedup": target_sid == last_history,
        }

    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx])
        target = history["output"]
        history["output"] = ""
        prompt = self.generate_prompt(history)
        self.prompt2history[prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target
        return {"prompt": prompt, "completion": target}


class RLTitle2SidDataset(JSONBaseDataset):
    def __init__(self, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        super().__init__(item_file, index_file, tokenizer=None, max_len=1024, test=False, category=category, dedup=dedup, seed=seed)
        self.prompt2history = {}
        self.history2target = {}
        self.data = []
        title2sid = {}
        description2sid = {}
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 3:
                sid = "".join(sids[:3])
                title2sid[self.item_feat[item_id]["title"]] = sid
                description = self.item_feat[item_id].get("description", "")
                if isinstance(description, str) and description.strip():
                    description2sid[description] = sid
        for title, sid in title2sid.items():
            self.data.append({"task": "title2sid", "input": title, "output": sid})
        for description, sid in description2sid.items():
            self.data.append({"task": "description2sid", "input": description, "output": sid})
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        self.get_inputs()

    def generate_prompt(self, data_point):
        if data_point["task"] == "title2sid":
            prompt = f"Which item has the title: {data_point['input']}?"
        else:
            prompt = f'An item can be described as follows: "{data_point["input"]}". Which item is it describing?'
        return f"""### User Input: 
{prompt}

### Response:\n"""

    def pre(self, idx):
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        target = data_point["output"] + "\n"
        self.prompt2history[prompt] = data_point["input"]
        self.history2target[data_point["input"]] = target
        return {"prompt": prompt, "completion": target}


class RLSeqTitle2SidDataset(CSVBaseDataset):
    def __init__(self, train_file, sample=-1, seed=0, category="", dedup=False):
        super().__init__(train_file, sample, seed, max_len=1024, category=category, dedup=dedup, tokenizer=None, test=False)
        self.prompt2history = {}
        self.history2target = {}
        self.get_inputs()

    def get_history(self, row):
        titles = parse_sequence(row["history_item_title"])
        target_sid = row["item_sid"]
        return {
            "inter_titles": ", ".join([f'"{title}"' for title in titles]),
            "target_sid": target_sid,
            "history_str": "::".join(titles),
        }

    def pre(self, idx):
        history_data = self.get_history(self.data.iloc[idx])
        prompt = f"""### User Input: 
Given the title sequence of user historical interactive items: {history_data["inter_titles"]}, can you recommend a suitable next item for the user?

### Response:\n"""
        target = history_data["target_sid"] + "\n"
        self.prompt2history[prompt] = history_data["history_str"]
        self.history2target[history_data["history_str"]] = target
        return {"prompt": prompt, "completion": target}
