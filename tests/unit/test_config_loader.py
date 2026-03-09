import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from onerec.config import SFTConfig, load_config
from onerec.utils.config_templates import render_config_file


class ConfigLoaderTest(unittest.TestCase):
    def test_load_default_sft_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rendered = render_config_file(
                REPO_ROOT / "config" / "sft.yaml",
                REPO_ROOT / "config" / "datasets.yaml",
                dataset_key="industrial",
                output_path=Path(tmpdir) / "sft.yaml",
            )
            config = load_config(SFTConfig, str(rendered))
            self.assertEqual(config.data.category, "Industrial_and_Scientific")
            self.assertTrue(config.model.base_model)

    def test_apply_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rendered = render_config_file(
                REPO_ROOT / "config" / "sft.yaml",
                REPO_ROOT / "config" / "datasets.yaml",
                dataset_key="industrial",
                output_path=Path(tmpdir) / "sft.yaml",
            )
            config = load_config(
                SFTConfig,
                str(rendered),
                overrides=["training.seed=7", "output.output_dir=./output/test_sft"],
            )
            self.assertEqual(config.training.seed, 7)
            self.assertEqual(config.output.output_dir, "./output/test_sft")


if __name__ == "__main__":
    unittest.main()
