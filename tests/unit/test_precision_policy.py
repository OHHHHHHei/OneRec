import sys
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from onerec.evaluate.pipeline import _resolve_precision as resolve_eval_precision
    from onerec.rl.pipeline import _resolve_precision as resolve_rl_precision
    from onerec.sft.pipeline import _resolve_precision as resolve_sft_precision


@unittest.skipUnless(HAS_TORCH, "torch is required for precision policy tests")
class PrecisionPolicyTest(unittest.TestCase):
    def test_sft_uses_bf16_on_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            dtype, use_bf16, use_fp16 = resolve_sft_precision()
        self.assertEqual(dtype, torch.bfloat16)
        self.assertTrue(use_bf16)
        self.assertFalse(use_fp16)

    def test_rl_uses_bf16_on_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            dtype, use_bf16, use_fp16 = resolve_rl_precision()
        self.assertEqual(dtype, torch.bfloat16)
        self.assertTrue(use_bf16)
        self.assertFalse(use_fp16)

    def test_eval_uses_bf16_on_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            dtype = resolve_eval_precision()
        self.assertEqual(dtype, torch.bfloat16)

    def test_cpu_fallback_stays_fp32(self):
        with patch("torch.cuda.is_available", return_value=False):
            sft_dtype, sft_bf16, sft_fp16 = resolve_sft_precision()
            rl_dtype, rl_bf16, rl_fp16 = resolve_rl_precision()
            eval_dtype = resolve_eval_precision()
        self.assertEqual(sft_dtype, torch.float32)
        self.assertFalse(sft_bf16)
        self.assertFalse(sft_fp16)
        self.assertEqual(rl_dtype, torch.float32)
        self.assertFalse(rl_bf16)
        self.assertFalse(rl_fp16)
        self.assertEqual(eval_dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
