import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from onerec.evaluate.semantic_id import canonicalize_semantic_id


class SemanticIdTest(unittest.TestCase):
    def test_extracts_canonical_semantic_id_from_wrapped_text(self):
        self.assertEqual(
            canonicalize_semantic_id('  "<a_106><b_157><c_116></s>"  '),
            "<a_106><b_157><c_116>",
        )

    def test_returns_stripped_text_when_no_semantic_id_exists(self):
        self.assertEqual(canonicalize_semantic_id(" plain-text "), "plain-text")


if __name__ == "__main__":
    unittest.main()
