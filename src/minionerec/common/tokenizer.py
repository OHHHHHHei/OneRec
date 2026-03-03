from typing import List


class WrappedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
        tokens = self.tokenizer.encode(text)
        while tokens and self.bos_id is not None and tokens[0] == self.bos_id:
            tokens = tokens[1:]
        while tokens and self.eos_id is not None and tokens[-1] == self.eos_id:
            tokens = tokens[:-1]
        if bos and self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if eos and self.eos_id is not None:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
