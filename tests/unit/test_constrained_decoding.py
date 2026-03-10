import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from onerec.evaluate.constrained_decoding import ConstrainedLogitsProcessor


@unittest.skipUnless(HAS_TORCH, "torch is required for constrained decoding tests")
class ConstrainedDecodingTest(unittest.TestCase):
    def test_prompt_prefix_length_uses_generated_tail(self):
        prefixes = {
            "101-102-103": [201],
            "201": [301],
            "201-301": [401],
        }

        processor = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=lambda _batch_id, tokens: prefixes.get("-".join(str(token) for token in tokens), []),
            num_beams=2,
            base_model="qwen",
            eos_token_id=999,
            prompt_prefix_length=6,
            warn_limit_per_step=0,
            enable_warning=False,
        )

        input_ids = torch.tensor(
            [
                [0, 0, 0, 101, 102, 103],
                [0, 9, 0, 101, 102, 103],
            ],
            dtype=torch.long,
        )
        scores = torch.zeros((2, 1200), dtype=torch.float32)
        processed = processor(input_ids, scores)
        self.assertGreater(processed[0, 201].item(), float("-inf"))
        self.assertEqual(processed[0, 999].item(), float("-inf"))

        next_input_ids = torch.tensor(
            [
                [0, 0, 0, 101, 102, 103, 201],
                [0, 9, 0, 101, 102, 103, 201],
            ],
            dtype=torch.long,
        )
        next_scores = torch.zeros((2, 1200), dtype=torch.float32)
        next_processed = processor(next_input_ids, next_scores)
        self.assertGreater(next_processed[0, 301].item(), float("-inf"))
        self.assertEqual(processor.get_diagnostics()["invalid_total"], 0)


if __name__ == "__main__":
    unittest.main()
