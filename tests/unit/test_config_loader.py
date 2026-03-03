import unittest

from minionerec.config.loader import load_config
from minionerec.config.schema import SFTConfig


class ConfigLoaderTest(unittest.TestCase):
    def test_load_default_sft_config(self):
        config = load_config(SFTConfig, "configs/stages/sft/default.yaml")
        self.assertEqual(config.data.category, "Industrial_and_Scientific")
        self.assertTrue(config.model.base_model)

    def test_apply_overrides(self):
        config = load_config(
            SFTConfig,
            "configs/stages/sft/default.yaml",
            overrides=["training.seed=7", "output.output_dir=./output/test_sft"],
        )
        self.assertEqual(config.training.seed, 7)
        self.assertEqual(config.output.output_dir, "./output/test_sft")


if __name__ == "__main__":
    unittest.main()
