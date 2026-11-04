"""
Unit test for SafeTune Orthogonal Subspace Surgery mathematical invariants.
Validates:
  1. U_S^T @ P_perp == 0
  2. U_S^T @ W_down* == 0 for any arbitrary matrix W_down
  3. Associative grouping equivalence: (I - U_S @ U_S^T) @ W == W - U_S @ (U_S^T @ W)
  4. Representational capacity preservation: dim(U_S) / d == 4 / 4096 == 0.09765%
"""

import unittest
import numpy as np


class TestSafeTuneLinearAlgebra(unittest.TestCase):
    def setUp(self):
        self.d = 4096       # Hidden size of Aya Expanse 8B
        self.d_mlp = 14336  # Intermediate MLP size of Aya Expanse 8B
        self.k = 4          # Subspace rank

        # Generate orthonormal basis U_S via QR decomposition of random matrix
        np.random.seed(42)
        A = np.random.randn(self.d, self.k)
        self.U_S, _ = np.linalg.qr(A)

    def test_orthonormality(self):
        # U_S^T @ U_S == I_k
        gram = self.U_S.T @ self.U_S
        I_k = np.eye(self.k)
        max_diff = np.max(np.abs(gram - I_k))
        self.assertLess(max_diff, 1e-12)

    def test_nullification_invariant(self):
        # W_down: shape (d, d_mlp)
        W_down = np.random.randn(self.d, self.d_mlp).astype(np.float64)

        # Associative grouping: W* = W - U_S @ (U_S^T @ W)
        u_proj = self.U_S.T @ W_down
        null_comp = self.U_S @ u_proj
        W_down_star = W_down - null_comp

        # Projection check: U_S^T @ W_down* must be 0
        residual = self.U_S.T @ W_down_star
        max_error = np.max(np.abs(residual))

        # Must satisfy tolerance < 1e-10
        self.assertLess(max_error, 1e-10)

    def test_associative_grouping_identity(self):
        # For smaller d to test explicit P_perp matrix identity
        d_small = 128
        d_mlp_small = 256
        k_small = 4

        A = np.random.randn(d_small, k_small)
        U, _ = np.linalg.qr(A)
        W = np.random.randn(d_small, d_mlp_small)

        # Full matrix formulation: P_perp = I - U @ U^T
        P_perp = np.eye(d_small) - U @ U.T
        W_star_full = P_perp @ W

        # Associative formulation: W - U @ (U^T @ W)
        W_star_assoc = W - U @ (U.T @ W)

        diff = np.max(np.abs(W_star_full - W_star_assoc))
        self.assertLess(diff, 1e-12)

    def test_capacity_ratio(self):
        ratio = self.k / self.d
        expected = 4 / 4096
        self.assertAlmostEqual(ratio, expected, places=7)
        self.assertLess(ratio, 0.001)  # Less than 0.1% capacity removed


if __name__ == "__main__":
    unittest.main()

