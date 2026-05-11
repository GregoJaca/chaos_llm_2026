import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from chaos_llm.perturbations import build_simplex


class TestSimplex(unittest.TestCase):
    def test_simplex_equidistant_1000_in_1536(self) -> None:
        num_points = 1000
        embed_dim = 1536

        vertices = build_simplex(num_points)
        padded = np.zeros((num_points, embed_dim), dtype=np.float32)
        padded[:, :num_points] = vertices

        gram = padded @ padded.T
        diag = np.diag(gram)
        self.assertTrue(np.allclose(diag, 1.0, atol=1e-4))

        expected_dot = -1.0 / (num_points - 1)
        mask = ~np.eye(num_points, dtype=bool)
        off_vals = gram[mask]
        self.assertTrue(np.allclose(off_vals, expected_dot, atol=1e-4))

        dist_sq = 2.0 - 2.0 * gram
        expected_dist = 2.0 - 2.0 * expected_dot
        dist_vals = dist_sq[mask]
        self.assertTrue(np.allclose(dist_vals, expected_dist, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
