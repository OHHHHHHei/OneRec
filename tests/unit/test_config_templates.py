import tempfile
import unittest
from pathlib import Path

from onerec.utils.config_templates import render_config_file, render_config_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_PATH = REPO_ROOT / "config" / "datasets.yaml"


class ConfigTemplateTest(unittest.TestCase):
    def test_render_sft_industrial(self):
        payload = render_config_payload(
            config_path=REPO_ROOT / "config" / "sft.yaml",
            datasets_path=DATASETS_PATH,
            dataset_key="industrial",
        )
        self.assertEqual(payload["data"]["category"], "Industrial_and_Scientific")
        self.assertTrue(
            payload["data"]["train_file"].endswith("Industrial_and_Scientific_5_2016-10-2018-11.csv")
        )
        self.assertEqual(payload["output"]["output_dir"], "./output/sft_Industrial_and_Scientific_refactor")

    def test_render_evaluate_rl_office(self):
        payload = render_config_payload(
            config_path=REPO_ROOT / "config" / "evaluate.yaml",
            datasets_path=DATASETS_PATH,
            dataset_key="office",
            eval_model_stage="rl",
        )
        self.assertEqual(payload["data"]["category"], "Office_Products")
        self.assertEqual(
            payload["model"]["base_model"],
            "./output/rl_Office_Products_refactor/final_checkpoint",
        )
        self.assertEqual(
            payload["output"]["output_dir"],
            "./results/final_result_rl_Office_Products.json",
        )

    def test_render_to_temp_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rendered.yaml"
            rendered = render_config_file(
                config_path=REPO_ROOT / "config" / "rl.yaml",
                datasets_path=DATASETS_PATH,
                dataset_key="industrial",
                output_path=output_path,
            )
            self.assertEqual(rendered, output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
