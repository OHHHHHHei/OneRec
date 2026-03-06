import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from onerec.evaluate.merge import merge
from onerec.evaluate.split_merge import split


class EvaluateSplitMergeTest(unittest.TestCase):
    def test_split_accepts_comma_separated_cuda_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "input.csv"
            output_dir = tmp / "shards"
            pd.DataFrame({"x": list(range(8))}).to_csv(input_csv, index=False)

            split(str(input_csv), str(output_dir), "4,5,6,7")

            for shard in ["4.csv", "5.csv", "6.csv", "7.csv"]:
                self.assertTrue((output_dir / shard).exists(), shard)
            self.assertFalse((output_dir / ",.csv").exists())

    def test_merge_accepts_comma_separated_cuda_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "jsons"
            input_dir.mkdir()
            output_json = tmp / "merged.json"
            for gpu in ["4", "5", "6", "7"]:
                with open(input_dir / f"{gpu}.json", "w", encoding="utf-8") as handle:
                    json.dump([{"gpu": gpu}], handle)

            merge(str(input_dir), str(output_json), "4,5,6,7")

            with open(output_json, "r", encoding="utf-8") as handle:
                merged = json.load(handle)
            self.assertEqual(len(merged), 4)
            self.assertEqual({row["gpu"] for row in merged}, {"4", "5", "6", "7"})


if __name__ == "__main__":
    unittest.main()
