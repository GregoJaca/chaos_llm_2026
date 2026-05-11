import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from chaos_llm.analysis.divergence import divergence_any_pair, divergence_pairwise, divergence_vs_baseline


class TestDivergence(unittest.TestCase):
    def test_divergence_vs_baseline(self) -> None:
        prompt_len = 2
        baseline = np.array([10, 11, 1, 2, 3], dtype=np.int32)
        perturbed = np.array(
            [
                [10, 11, 1, 9, 3],
                [10, 11, 1, 2, 3],
            ],
            dtype=np.int32,
        )
        lengths = np.array([5, 5], dtype=np.int32)

        out = divergence_vs_baseline(
            perturbed_ids=perturbed,
            lengths=lengths,
            baseline_ids=baseline,
            prompt_len=prompt_len,
            index_reference="generated",
        )
        self.assertEqual(out.tolist(), [1, -1])

    def test_divergence_any_pair(self) -> None:
        prompt_len = 2
        perturbed = np.array(
            [
                [10, 11, 1, 2, 3],
                [10, 11, 1, 9, 3],
            ],
            dtype=np.int32,
        )
        lengths = np.array([5, 5], dtype=np.int32)
        out = divergence_any_pair(
            perturbed_ids=perturbed,
            lengths=lengths,
            prompt_len=prompt_len,
            include_baseline=False,
            baseline_ids=None,
            index_reference="generated",
        )
        self.assertEqual(out, 1)

    def test_divergence_pairwise(self) -> None:
        prompt_len = 2
        perturbed = np.array(
            [
                [10, 11, 1, 2, 3, 0],
                [10, 11, 1, 9, 3, 0],
                [10, 11, 1, 2, 3, 4],
            ],
            dtype=np.int32,
        )
        lengths = np.array([5, 5, 6], dtype=np.int32)
        out = divergence_pairwise(
            perturbed_ids=perturbed,
            lengths=lengths,
            prompt_len=prompt_len,
            index_reference="generated",
            max_pairs=None,
        )
        self.assertEqual(out.tolist(), [1, 3, 1])


if __name__ == "__main__":
    unittest.main()
