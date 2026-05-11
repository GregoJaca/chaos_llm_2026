import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from chaos_llm.analysis.data import load_tokens
from chaos_llm.analysis.divergence import divergence_any_pair, divergence_pairwise, divergence_vs_baseline


class TestAnalysisE2E(unittest.TestCase):
    def test_npz_roundtrip_and_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run_4096_0.1_test")
            os.makedirs(run_dir, exist_ok=True)

            baseline_ids = np.array([10, 11, 1, 2, 3], dtype=np.int32)
            perturbed_ids = np.array(
                [
                    [10, 11, 1, 2, 3],
                    [10, 11, 1, 9, 3],
                ],
                dtype=np.int32,
            )
            lengths = np.array([5, 5], dtype=np.int32)
            divergence_index = np.array([-1, -1], dtype=np.int32)
            prompt_len = np.array(2, dtype=np.int32)

            np.savez_compressed(
                os.path.join(run_dir, "tokens.npz"),
                baseline_ids=baseline_ids,
                perturbed_ids=perturbed_ids,
                perturbed_lengths=lengths,
                divergence_index=divergence_index,
                prompt_len=prompt_len,
            )

            data = load_tokens(run_dir, mmap_mode=None)
            any_pair = divergence_any_pair(
                perturbed_ids=data["perturbed_ids"],
                lengths=data["perturbed_lengths"],
                prompt_len=int(data["prompt_len"]),
                include_baseline=False,
                baseline_ids=None,
                index_reference="generated",
            )
            self.assertEqual(any_pair, 1)

            vs_base = divergence_vs_baseline(
                perturbed_ids=data["perturbed_ids"],
                lengths=data["perturbed_lengths"],
                baseline_ids=data["baseline_ids"],
                prompt_len=int(data["prompt_len"]),
                index_reference="generated",
            )
            self.assertEqual(vs_base.tolist(), [-1, 1])

            pairwise = divergence_pairwise(
                perturbed_ids=data["perturbed_ids"],
                lengths=data["perturbed_lengths"],
                prompt_len=int(data["prompt_len"]),
                index_reference="generated",
            )
            self.assertEqual(pairwise.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
