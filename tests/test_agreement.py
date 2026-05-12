import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from chaos_llm.analysis.agreement import agreement_all_pairs, agreement_with_baseline


class TestAgreement(unittest.TestCase):
    def test_agreement_with_baseline(self) -> None:
        prompt_len = 1
        baseline = np.array([5, 7, 9], dtype=np.int32)
        perturbed = np.array([[5, 7, 9], [5, 8, 9]], dtype=np.int32)
        lengths = np.array([3, 3], dtype=np.int32)

        steps, rates = agreement_with_baseline(
            perturbed_ids=perturbed,
            lengths=lengths,
            baseline_ids=baseline,
            prompt_len=prompt_len,
            max_steps=None,
        )
        self.assertEqual(steps.tolist(), [0, 1])
        self.assertTrue(np.allclose(rates, [0.5, 1.0]))

    def test_agreement_all_pairs(self) -> None:
        prompt_len = 1
        perturbed = np.array([[5, 7, 9], [5, 7, 8], [5, 8, 8]], dtype=np.int32)
        lengths = np.array([3, 3, 3], dtype=np.int32)

        steps, rates = agreement_all_pairs(
            perturbed_ids=perturbed,
            lengths=lengths,
            prompt_len=prompt_len,
            max_steps=None,
        )
        self.assertEqual(steps.tolist(), [0, 1])
        self.assertTrue(np.allclose(rates, [1.0 / 3.0, 1.0 / 3.0]))


if __name__ == "__main__":
    unittest.main()
