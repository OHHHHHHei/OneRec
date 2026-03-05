from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings
from collections import Counter

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None,
        warn_limit_per_step: int = 2,
        enable_warning: bool = True,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count=0
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        self.warn_limit_per_step = max(0, int(warn_limit_per_step))
        self.enable_warning = bool(enable_warning)
        self.invalid_total = 0
        self.invalid_by_step: Dict[int, int] = {}
        self._warned_by_step: Dict[int, int] = {}
        self._invalid_hash_counter: Counter[Tuple[int, ...]] = Counter()
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    def get_diagnostics(self, top_k: int = 5) -> dict:
        top_hashes = [
            {"hash_key": list(hash_key), "count": count}
            for hash_key, count in self._invalid_hash_counter.most_common(top_k)
        ]
        return {
            "invalid_total": int(self.invalid_total),
            "invalid_by_step": dict(self.invalid_by_step),
            "top_invalid_hashes": top_hashes,
        }

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
            
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    hash_key=sent[-self.count:]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    self.invalid_total += 1
                    self.invalid_by_step[self.count] = self.invalid_by_step.get(self.count, 0) + 1
                    self._invalid_hash_counter[tuple(hash_key)] += 1
                    warned_count = self._warned_by_step.get(self.count, 0)
                    if self.enable_warning and warned_count < self.warn_limit_per_step:
                        warnings.warn(
                            f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                            f"(step_warn {warned_count + 1}/{self.warn_limit_per_step})"
                        )
                        self._warned_by_step[self.count] = warned_count + 1
                    # Force EOS token to end invalid sequence
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue 
                
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        self.count += 1

        scores = scores + mask
        return scores
