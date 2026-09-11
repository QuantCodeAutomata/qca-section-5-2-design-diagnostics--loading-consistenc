"""
Unit tests for src/matrix_utils.py

Tests SVD pseudoinverse, rank computation, projection matrices, and QC validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.matrix_utils import (
    pinv_svd, rank_svd, proj_matrix, frobenius_norm, l2_norm,
    support_proj, validate_numerical_qc, _TOL, _QC_TOL,
)


# -------------------------------------------------------------------
# pinv_svd tests
# -------------------------------------------------------------------

class TestPinvSvd:

    def test_square_full_rank_matrix(self):
        """Pseudoinverse of full-rank square matrix equals regular inverse."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5))
        A_plus = pinv_svd(A)
        A_inv = np.linalg.inv(A)
        assert A_plus.shape == (5, 5)
        assert np.allclose(A_plus, A_inv, atol=1e-10), \
            "Pseudoinverse should equal inverse for full-rank square matrix"

    def test_tall_matrix(self):
        """Pseudoinverse of tall matrix: A^+ A = I_m."""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((10, 4))
        A_plus = pinv_svd(A)
        assert A_plus.shape == (4, 10)
        # Left inverse property
        prod = A_plus @ A
        assert np.allclose(prod, np.eye(4), atol=1e-10), \
            "A^+ A should be identity for full column-rank A"

    def test_wide_matrix(self):
        """Pseudoinverse of wide matrix: A A^+ = I_m (projection)."""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((4, 10))
        A_plus = pinv_svd(A)
        assert A_plus.shape == (10, 4)
        prod = A @ A_plus
        assert np.allclose(prod, np.eye(4), atol=1e-10), \
            "A A^+ should be identity for full row-rank A"

    def test_rank_deficient_matrix(self):
        """Pseudoinverse handles rank-deficient matrix correctly."""
        # Rank-2 matrix of size 5x4
        U = np.random.default_rng(2).standard_normal((5, 2))
        V = np.random.default_rng(3).standard_normal((2, 4))
        A = U @ V
        A_plus = pinv_svd(A, tol=_TOL)
        # Verify Moore-Penrose conditions
        assert np.allclose(A @ A_plus @ A, A, atol=1e-8), \
            "Moore-Penrose condition 1: A A^+ A = A"
        assert np.allclose(A_plus @ A @ A_plus, A_plus, atol=1e-8), \
            "Moore-Penrose condition 2: A^+ A A^+ = A^+"

    def test_zero_matrix(self):
        """Pseudoinverse of zero matrix is zero."""
        A = np.zeros((5, 3))
        A_plus = pinv_svd(A)
        assert A_plus.shape == (3, 5)
        assert np.allclose(A_plus, 0), "Pseudoinverse of zero matrix should be zero"

    def test_empty_matrix(self):
        """Handles empty matrix."""
        A = np.zeros((0, 5))
        A_plus = pinv_svd(A)
        assert A_plus.shape == (5, 0)

    def test_scalar_matrix(self):
        """Pseudoinverse of 1x1 scalar matrix."""
        A = np.array([[3.0]])
        A_plus = pinv_svd(A)
        assert np.allclose(A_plus, np.array([[1.0 / 3.0]]))

    def test_symmetric_positive_definite(self):
        """Pseudoinverse of SPD matrix matches inverse."""
        rng = np.random.default_rng(5)
        X = rng.standard_normal((10, 5))
        A = X.T @ X / 10  # PSD
        A += 0.1 * np.eye(5)  # Make PD
        A_plus = pinv_svd(A)
        A_inv = np.linalg.inv(A)
        assert np.allclose(A_plus, A_inv, atol=1e-8)

    def test_tolerance_truncation(self):
        """Very small singular values are truncated."""
        # Matrix with one tiny singular value
        U, _, Vt = np.linalg.svd(np.random.default_rng(6).standard_normal((5, 5)))
        s = np.array([100.0, 10.0, 1.0, 0.01, 1e-12])
        A = U @ np.diag(s) @ Vt
        # With tol=1e-10, singular value 1e-12 should be truncated (1e-12 < 1e-10 * 100 = 1e-8)
        A_plus = pinv_svd(A, tol=1e-10)
        # Verify rank (should be 4, not 5)
        rk = rank_svd(A, tol=1e-10)
        assert rk == 4, f"Expected rank 4, got {rk}"

    def test_matches_numpy_pinv(self):
        """Results should match numpy.linalg.pinv for well-conditioned matrices."""
        rng = np.random.default_rng(7)
        for shape in [(5, 3), (3, 5), (4, 4)]:
            A = rng.standard_normal(shape)
            A_plus_ours = pinv_svd(A, tol=1e-14)  # Very permissive tolerance
            A_plus_numpy = np.linalg.pinv(A)
            assert np.allclose(A_plus_ours, A_plus_numpy, atol=1e-8), \
                f"Mismatch with numpy for shape {shape}"


# -------------------------------------------------------------------
# rank_svd tests
# -------------------------------------------------------------------

class TestRankSvd:

    def test_full_rank(self):
        rng = np.random.default_rng(10)
        A = rng.standard_normal((5, 5))
        assert rank_svd(A) == 5

    def test_rank_deficient(self):
        U = np.random.default_rng(11).standard_normal((6, 3))
        V = np.random.default_rng(12).standard_normal((3, 4))
        A = U @ V
        assert rank_svd(A, tol=1e-10) == 3

    def test_zero_matrix(self):
        assert rank_svd(np.zeros((5, 5))) == 0

    def test_single_element(self):
        assert rank_svd(np.array([[5.0]])) == 1

    def test_empty(self):
        assert rank_svd(np.zeros((0, 5))) == 0


# -------------------------------------------------------------------
# proj_matrix tests
# -------------------------------------------------------------------

class TestProjMatrix:

    def test_idempotency(self):
        """Projection matrix satisfies P^2 = P."""
        rng = np.random.default_rng(20)
        A = rng.standard_normal((8, 3))
        P = proj_matrix(A)
        assert np.allclose(P @ P, P, atol=1e-10), "Projection must be idempotent"

    def test_symmetry(self):
        """Projection matrix is symmetric."""
        rng = np.random.default_rng(21)
        A = rng.standard_normal((8, 3))
        P = proj_matrix(A)
        assert np.allclose(P, P.T, atol=1e-10), "Projection must be symmetric"

    def test_column_in_range(self):
        """Vectors in column space of A map to themselves under P."""
        rng = np.random.default_rng(22)
        A = rng.standard_normal((8, 3))
        v = A @ rng.standard_normal(3)  # v in column space of A
        P = proj_matrix(A)
        assert np.allclose(P @ v, v, atol=1e-10), "P should fix vectors in col(A)"

    def test_orthogonal_complement(self):
        """Vectors orthogonal to col(A) map to zero under P."""
        rng = np.random.default_rng(23)
        A = rng.standard_normal((8, 3))
        P = proj_matrix(A)
        # Find vector orthogonal to col(A) using QR
        Q, _ = np.linalg.qr(A, mode="complete")
        v_perp = Q[:, 3]  # orthogonal to first 3 columns
        assert np.allclose(P @ v_perp, 0, atol=1e-10), \
            "P should annihilate orthogonal complement vectors"

    def test_eigenvalues(self):
        """Eigenvalues of projection matrix are 0 or 1."""
        rng = np.random.default_rng(24)
        A = rng.standard_normal((6, 3))
        P = proj_matrix(A)
        eigvals = np.linalg.eigvalsh(P)
        for lam in eigvals:
            assert abs(lam) < 1e-8 or abs(lam - 1) < 1e-8, \
                f"Eigenvalue {lam} is neither 0 nor 1"


# -------------------------------------------------------------------
# Frobenius and L2 norm tests
# -------------------------------------------------------------------

class TestNorms:

    def test_frobenius_norm_identity(self):
        """Frobenius norm of n x n identity = sqrt(n)."""
        n = 5
        assert abs(frobenius_norm(np.eye(n)) - np.sqrt(n)) < 1e-12

    def test_frobenius_zero(self):
        assert frobenius_norm(np.zeros((3, 3))) == 0.0

    def test_l2_norm(self):
        v = np.array([3.0, 4.0])
        assert abs(l2_norm(v) - 5.0) < 1e-12

    def test_l2_zero(self):
        assert l2_norm(np.zeros(5)) == 0.0


# -------------------------------------------------------------------
# validate_numerical_qc tests
# -------------------------------------------------------------------

class TestNumericalQC:

    def test_perfect_match(self):
        ld_mw = np.array([0.1, 0.2, 0.3])
        lg = np.array([0.1, 0.2, 0.3])
        diff = validate_numerical_qc(ld_mw, lg)
        assert diff < 1e-15

    def test_with_small_differences(self):
        ld_mw = np.array([0.1, 0.2])
        lg = np.array([0.1 + 1e-14, 0.2 - 1e-14])
        diff = validate_numerical_qc(ld_mw, lg)
        assert diff < 1e-13

    def test_empty_arrays(self):
        diff = validate_numerical_qc(np.array([]), np.array([]))
        assert diff == 0.0
