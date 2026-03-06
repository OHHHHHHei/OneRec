import os
import subprocess
import sys
import unittest
from pathlib import Path


class CliHelpTest(unittest.TestCase):
    def test_main_help(self):
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.run(
            [sys.executable, "-m", "onerec.main", "--help"],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_root,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OneRec stage runner", proc.stdout)


if __name__ == "__main__":
    unittest.main()
