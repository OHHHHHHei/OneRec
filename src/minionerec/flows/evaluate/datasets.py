from __future__ import annotations

from minionerec.data.datasets.base import CSVBaseDataset
from minionerec.data.parsers import parse_sequence


class EvalSidDataset(CSVBaseDataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)
        self.get_inputs()

    def get_history(self, row):
        history_sids = parse_sequence(row["history_item_sid"])
        history = ", ".join(history_sids)
        target_sid = str(row["item_sid"])
        last_history = history_sids[-1] if history_sids else None
        return {
            "input": f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}",
            "output": target_sid + "\n",
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
