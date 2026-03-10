import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from onerec.rl.deepspeed_compat import _patch_bf16_optimizer_destroy_cls


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


if __name__ == "__main__":
    unittest.main()
