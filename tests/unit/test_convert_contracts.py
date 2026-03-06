import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from onerec.config import ConvertConfig, load_config
from onerec.convert.pipeline import run_convert


class ConvertContractsTest(unittest.TestCase):
    def test_convert_uses_existing_contracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "converted")
            fixture_root = Path("tests/fixtures/sample_dataset").resolve()
            config = load_config(
                ConvertConfig,
                "config/convert.yaml",
                overrides=[
                    f"data.data_dir={fixture_root}",
                    "data.dataset_name=Sample",
                    "data.category=Sample",
                    f"data.output_dir={output_dir}",
                    f"info_path={os.path.join(output_dir, 'info', 'Sample_5_2016-10-2018-11.txt')}",
                ],
            )
            result = run_convert(config)
            self.assertEqual(result, output_dir)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "train")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "valid")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "test")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "info", "Sample_5_2016-10-2018-11.txt")))


if __name__ == "__main__":
    unittest.main()
