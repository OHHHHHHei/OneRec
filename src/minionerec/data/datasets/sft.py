from __future__ import annotations

import copy
import random

from minionerec.data.datasets.base import BaseDataset, CSVBaseDataset, JSONBaseDataset
from minionerec.data.parsers import parse_sequence


class SidSFTDataset(CSVBaseDataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)
        self.get_inputs()

    def get_history(self, row):
        history_sids = parse_sequence(row["history_item_sid"])
        history = ", ".join(history_sids)
        target_sid = str(row["item_sid"])
        last_history = history_sids[-1] if history_sids else None
        return {
            "input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
            "output": target_sid + "\n",
            "history_str": history,
            "dedup": target_sid == last_history,
        }

    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history = self.get_history(self.data.iloc[idx])
        target = history["output"]
        history["output"] = ""
        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        if self.test:
            return {"input_ids": tokens, "attention_mask": attention_mask}
        golden = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": ([1] * len(tokens))[-self.max_len :],
            "labels": labels[-self.max_len :],
        }


class SidItemFeatDataset(JSONBaseDataset):
    def __init__(self, item_file, index_file, tokenizer=None, max_len=2048, sample=-1, test=False, seed=0, category=""):
        super().__init__(item_file=item_file, index_file=index_file, tokenizer=tokenizer, max_len=max_len, test=test, category=category, dedup=False, seed=seed)
        self.sid2title = {}
        self.title2sid = {}
        self.data = []
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 3:
                title = self.item_feat[item_id]["title"]
                sid = "".join(sids[:3])
                self.sid2title[sid] = title
                self.title2sid[title] = sid
        for sid, title in self.sid2title.items():
            self.data.append({"task": "sid2title", "input": sid, "output": title})
        for title, sid in self.title2sid.items():
            self.data.append({"task": "title2sid", "input": title, "output": sid})
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        self.get_inputs()

    def generate_prompt(self, data_point):
        if data_point["task"] == "title2sid":
            prompt = f"Which item has the title: {data_point['input']}?"
        else:
            prompt = f'What is the title of item "{data_point["input"]}"?'
        return f"""### User Input: 
{prompt}

### Response:\n"""

    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        if self.test:
            return {"input_ids": tokens, "attention_mask": attention_mask}
        target = data_point["output"] + "\n"
        golden = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": ([1] * len(tokens))[-self.max_len :],
            "labels": labels[-self.max_len :],
        }


class FusionSeqRecDataset(BaseDataset):
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        import pandas as pd
        from minionerec.common.io import read_json

        BaseDataset.__init__(self, tokenizer, max_len, test, category, dedup, seed)
        self.data = pd.read_csv(train_file)
        self._cache_sources = [train_file, item_file, index_file]
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.item_feat = read_json(item_file)
        self.indices = read_json(index_file)
        self.sid2title = {}
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 3:
                self.sid2title["".join(sids[:3])] = self.item_feat[item_id]["title"]
        self.get_inputs()

    def get_history(self, row):
        history_item_sid = parse_sequence(row["history_item_sid"])
        history_str = ", ".join(history_item_sid)
        target_sid = row["item_sid"]
        target_title = self.sid2title.get(target_sid, target_sid)
        last_history = history_item_sid[-1] if history_item_sid else None
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_sid": target_sid,
            "dedup": target_sid == last_history,
        }

    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history_data = self.get_history(self.data.iloc[idx])
        if self.dedup and history_data["dedup"]:
            return None
        prompt = f"""### User Input: 
The user has sequentially interacted with items {history_data["history_str"]}. Can you recommend the next item for him? Tell me the title of the item

### Response:\n"""
        target = history_data["target_title"] + "\n"
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        if self.test:
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
        golden = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": ([1] * len(tokens))[-self.max_len :],
            "labels": labels[-self.max_len :],
        }
