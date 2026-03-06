import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from onerec.config import SFTConfig, load_config


class ConfigLoaderTest(unittest.TestCase):
    def test_load_default_sft_config(self):
        config = load_config(SFTConfig, "config/sft.yaml")
        self.assertEqual(config.data.category, "Industrial_and_Scientific")
        self.assertTrue(config.model.base_model)

    def test_apply_overrides(self):
        config = load_config(
            SFTConfig,
            "config/sft.yaml",
            overrides=["training.seed=7", "output.output_dir=./output/test_sft"],
        )
        self.assertEqual(config.training.seed, 7)
        self.assertEqual(config.output.output_dir, "./output/test_sft")


if __name__ == "__main__":
    unittest.main()
