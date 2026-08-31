from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ml_utils import precision_at_k


class PrecisionAtKTests(unittest.TestCase):
    def test_untied_scores_use_the_top_k_rows(self) -> None:
        result = precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], 2)

        self.assertEqual(result, 0.5)

    def test_cutoff_tie_uses_expected_precision(self) -> None:
        result = precision_at_k([1, 0, 1, 0], [1.0, 0.0, 0.0, 0.0], 2)

        self.assertTrue(math.isclose(result, 2 / 3))

    def test_cutoff_tie_is_independent_of_input_order(self) -> None:
        first = precision_at_k([1, 0, 1, 0], [1.0, 0.0, 0.0, 0.0], 2)
        reordered = precision_at_k([1, 1, 0, 0], [1.0, 0.0, 0.0, 0.0], 2)

        self.assertTrue(math.isclose(first, reordered))

    def test_k_larger_than_the_input_returns_the_base_rate(self) -> None:
        result = precision_at_k([1, 0, 1], [0.8, 0.4, 0.1], 10)

        self.assertTrue(math.isclose(result, 2 / 3))

    def test_empty_or_non_positive_k_returns_zero(self) -> None:
        self.assertEqual(precision_at_k([], [], 5), 0.0)
        self.assertEqual(precision_at_k([1], [0.5], 0), 0.0)


if __name__ == "__main__":
    unittest.main()
