import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from onerec.rl.deepspeed_compat import (
    _patch_bf16_optimizer_destroy_cls,
    _patch_engine_destroy_cls,
    _patch_zero_optimizer_destroy_cls,
)


class _FakeOptimizer:
    def __init__(self, param_group_count):
        self.param_groups = [{} for _ in range(param_group_count)]


class _FakeBF16Optimizer:
    def __init__(self, using_real_optimizer, param_group_count, bf16_group_count):
        self.using_real_optimizer = using_real_optimizer
        self.optimizer = _FakeOptimizer(param_group_count)
        self.bf16_groups = [object() for _ in range(bf16_group_count)]
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1
        for i in range(len(self.optimizer.param_groups)):
            _ = self.bf16_groups[i]


class _FakeZeroOptimizer:
    def __init__(self):
        self.destroy_calls = 0

    def print_rank_0(self, *_args, **_kwargs):
        return None

    def destroy(self):
        self.destroy_calls += 1
        raise AssertionError("DeepSpeed backend not set, please initialize it using init_process_group()")


class _FakeEngine:
    def __init__(self):
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1
        raise AssertionError("DeepSpeed backend not set, please initialize it using init_process_group()")


class DeepSpeedCompatTest(unittest.TestCase):
    def test_patch_skips_dummy_optimizer_destroy_path(self):
        class FakeBF16(_FakeBF16Optimizer):
            pass

        self.assertTrue(_patch_bf16_optimizer_destroy_cls(FakeBF16))
        optimizer = FakeBF16(using_real_optimizer=False, param_group_count=2, bf16_group_count=0)
        optimizer.destroy()
        self.assertEqual(optimizer.destroy_calls, 0)

    def test_patch_skips_misaligned_group_cleanup(self):
        class FakeBF16(_FakeBF16Optimizer):
            pass

        self.assertTrue(_patch_bf16_optimizer_destroy_cls(FakeBF16))
        optimizer = FakeBF16(using_real_optimizer=True, param_group_count=2, bf16_group_count=1)
        optimizer.destroy()
        self.assertEqual(optimizer.destroy_calls, 0)

    def test_patch_keeps_real_destroy_when_state_is_consistent(self):
        class FakeBF16(_FakeBF16Optimizer):
            pass

        self.assertTrue(_patch_bf16_optimizer_destroy_cls(FakeBF16))
        optimizer = FakeBF16(using_real_optimizer=True, param_group_count=2, bf16_group_count=2)
        optimizer.destroy()
        self.assertEqual(optimizer.destroy_calls, 1)

    def test_zero_optimizer_patch_skips_destroy_when_backend_is_gone(self):
        class FakeZero(_FakeZeroOptimizer):
            pass

        self.assertTrue(_patch_zero_optimizer_destroy_cls(FakeZero))
        optimizer = FakeZero()
        with patch("onerec.rl.deepspeed_compat._distributed_backend_ready", return_value=False):
            optimizer.destroy()
        self.assertEqual(optimizer.destroy_calls, 0)

    def test_zero_optimizer_patch_swallows_missing_backend_assertion(self):
        class FakeZero(_FakeZeroOptimizer):
            pass

        self.assertTrue(_patch_zero_optimizer_destroy_cls(FakeZero))
        optimizer = FakeZero()
        with patch("onerec.rl.deepspeed_compat._distributed_backend_ready", return_value=True):
            optimizer.destroy()
        self.assertEqual(optimizer.destroy_calls, 1)

    def test_engine_patch_swallows_missing_backend_assertion(self):
        class FakeEngine(_FakeEngine):
            pass

        self.assertTrue(_patch_engine_destroy_cls(FakeEngine))
        engine = FakeEngine()
        engine.destroy()
        self.assertEqual(engine.destroy_calls, 1)


if __name__ == "__main__":
    unittest.main()
