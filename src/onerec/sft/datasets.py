from __future__ import annotations

import ast
import random

from onerec.utils.dataset_base import BaseDataset, CSVBaseDataset, JSONBaseDataset
from onerec.utils.parsing import parse_sequence


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
    def __init__(
        self,
        train_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
        enable_title_description_alignment=True,
        description_task_probability=0.5,
    ):
        import pandas as pd
        from onerec.utils.io import read_json

        BaseDataset.__init__(self, tokenizer, max_len, test, category, dedup, seed)
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.item_feat = read_json(item_file)
        self.indices = read_json(index_file)
        self.sid2title = {}
        self.sid2description = {}
        self.enable_title_description_alignment = bool(enable_title_description_alignment)
        self.description_task_probability = float(description_task_probability)
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 3:
                sid = "".join(sids[:3])
                title = self.item_feat[item_id]["title"]
                description = self.item_feat[item_id].get("description", "")
                self.sid2title[sid] = title
                self.sid2description[sid] = self._process_description(description, title)
        self.get_inputs()

    def _process_description(self, description, title: str) -> str:
        if description is None:
            return title
        if isinstance(description, list):
            candidates = [str(item).strip() for item in description if str(item).strip()]
            return max(candidates, key=len) if candidates else title
        if isinstance(description, str):
            text = description.strip()
            if not text:
                return title
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None
                if isinstance(parsed, list):
                    candidates = [str(item).strip() for item in parsed if str(item).strip()]
                    return max(candidates, key=len) if candidates else title
            return text
        return str(description).strip() or title

    def generate_prompt_title(self, history: str) -> str:
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Tell me the title of the item"

    def generate_prompt_description(self, history: str) -> str:
        return f"Please review the user's historical interactions: {history}, and describe what kind of item he still needs."

    def get_history(self, row):
        history_item_sid = parse_sequence(row["history_item_sid"])
        history_str = ", ".join(history_item_sid)
        target_sid = str(row["item_sid"])
        target_title = self.sid2title.get(target_sid, target_sid)
        target_description = self.sid2description.get(target_sid, target_title)
        last_history = history_item_sid[-1] if history_item_sid else None
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_description": target_description,
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
        if self.enable_title_description_alignment and random.random() < self.description_task_probability:
            prompt_text = self.generate_prompt_description(history_data["history_str"])
            target = history_data["target_description"] + "\n"
        else:
            prompt_text = self.generate_prompt_title(history_data["history_str"])
            target = history_data["target_title"] + "\n"
        prompt = f"""### User Input: 
{prompt_text}

### Response:\n"""
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


class TitleHistory2SidSFTDataset(BaseDataset):
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        import pandas as pd
        from onerec.utils.io import read_json

        BaseDataset.__init__(self, tokenizer, max_len, test, category, dedup, seed)
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.item_feat = read_json(item_file)
        self.indices = read_json(index_file)
        self.id2sid = {}
        for item_id, sids in self.indices.items():
            if len(sids) >= 3:
                self.id2sid[str(item_id)] = "".join(sids[:3])
        self.get_inputs()

    def get_history(self, row):
        history_item_title = parse_sequence(row["history_item_title"])
        history_titles = ", ".join([f'"{title}"' for title in history_item_title])
        target_item_id = str(row["item_id"])
        target_sid = self.id2sid.get(target_item_id, target_item_id)
        is_duplicate = False
        if self.dedup and "history_item_id" in row:
            try:
                history_item_id = parse_sequence(row["history_item_id"])
                last_history_item_id = str(history_item_id[-1]) if history_item_id else None
                is_duplicate = target_item_id == last_history_item_id
            except (ValueError, TypeError):
                is_duplicate = False
        return {
            "input": f"The user has interacted with the following {self.category} items in chronological order: {history_titles}. Can you predict the next item the user may expect?",
            "output": target_sid + "\n",
            "history_titles": history_titles,
            "target_sid": target_sid,
            "dedup": is_duplicate,
        }

    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Based on the user's historical interaction with item titles, predict the semantic ID of the next item they may expect.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history_data = self.get_history(self.data.iloc[idx])
        if self.dedup and history_data["dedup"]:
            return None
        target_output = history_data["output"]
        history_data["output"] = ""
        prompt = self.generate_prompt(history_data)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }
        golden_tokens = self.tokenizer.encode(target_output, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }
