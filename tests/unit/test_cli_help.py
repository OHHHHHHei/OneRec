import subprocess
import sys
import unittest


class CliHelpTest(unittest.TestCase):
    def test_main_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "minionerec.cli.main", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("MiniOneRec unified CLI", proc.stdout)


if __name__ == "__main__":
    unittest.main()
