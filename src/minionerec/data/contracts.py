from typing import TypedDict


class PromptSample(TypedDict):
    prompt: str
    completion: str


class TokenizedSample(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
