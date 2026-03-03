import os
import tempfile
import unittest
from pathlib import Path

from minionerec.cli.convert import run_convert_cli


class ConvertContractsTest(unittest.TestCase):
    def test_convert_cli_uses_existing_contracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "converted")
            fixture_root = Path("tests/fixtures/sample_dataset").resolve()
            result = run_convert_cli(
                None,
                overrides=[
                    f"data.data_dir={fixture_root}",
                    "data.dataset_name=Sample",
                    "data.category=Sample",
                    f"data.output_dir={output_dir}",
                    f"info_path={os.path.join(output_dir, 'info', 'Sample_5_2016-10-2018-11.txt')}",
                ],
            )
            self.assertEqual(result, output_dir)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "train")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "valid")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "test")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "info", "Sample_5_2016-10-2018-11.txt")))


if __name__ == "__main__":
    unittest.main()
