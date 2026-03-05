import subprocess
import sys
import unittest
from pathlib import Path


class FlowLayoutTest(unittest.TestCase):
    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def test_flow_default_configs_exist(self):
        self.assertTrue((self.repo_root / "flows" / "sft" / "default.yaml").exists())
        self.assertTrue((self.repo_root / "flows" / "rl" / "default.yaml").exists())
        self.assertTrue((self.repo_root / "flows" / "evaluate" / "default.yaml").exists())

    def test_no_legacy_import_in_main_src(self):
        root = self.repo_root / "src" / "minionerec"
        for path in root.rglob("*.py"):
            rel = path.relative_to(self.repo_root).as_posix()
            if rel.startswith("src/minionerec/compat/"):
                continue
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("from legacy", content, msg=f"legacy import found in {rel}")
            self.assertNotIn("import legacy", content, msg=f"legacy import found in {rel}")

    def test_parity_check_script(self):
        proc = subprocess.run(
            [sys.executable, "parity_check.py"],
            capture_output=True,
            text=True,
            check=False,
            cwd=self.repo_root,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
        self.assertIn("Parity check passed.", proc.stdout)


if __name__ == "__main__":
    unittest.main()
