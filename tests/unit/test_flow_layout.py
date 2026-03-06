import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FlowLayoutTest(unittest.TestCase):
    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    def test_stage_configs_exist(self):
        self.assertTrue((self.repo_root / "config" / "sft.yaml").exists())
        self.assertTrue((self.repo_root / "config" / "rl.yaml").exists())
        self.assertTrue((self.repo_root / "config" / "evaluate.yaml").exists())

    def test_no_legacy_or_minionerec_import_in_main_src(self):
        root = self.repo_root / "src" / "onerec"
        for path in root.rglob("*.py"):
            rel = path.relative_to(self.repo_root).as_posix()
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("from legacy", content, msg=f"legacy import found in {rel}")
            self.assertNotIn("import legacy", content, msg=f"legacy import found in {rel}")
            self.assertNotIn("minionerec", content, msg=f"minionerec reference found in {rel}")

    def test_removed_old_layouts(self):
        self.assertFalse((self.repo_root / "flows").exists())
        self.assertFalse((self.repo_root / "scripts").exists())
        self.assertFalse((self.repo_root / "legacy").exists())
        self.assertFalse((self.repo_root / "src" / "minionerec").exists())

    def test_parity_check_script(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.run(
            [sys.executable, "parity_check.py"],
            capture_output=True,
            text=True,
            check=False,
            cwd=self.repo_root,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
        self.assertIn("Parity check passed.", proc.stdout)


if __name__ == "__main__":
    unittest.main()
