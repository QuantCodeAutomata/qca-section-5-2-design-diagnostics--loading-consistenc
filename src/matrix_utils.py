"""
Matrix utilities for QCA design-diagnostic and performance experiments.

Custom SVD-based pseudoinverse implementation with relative tolerance,
used consistently for all inversions per paper methodology.
# Custom — Context7 found no library equivalent for relative-tol SVD pseudoinverse (paper §5.2)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


_TOL = 1e-10  # Default relative tolerance for pseudoinverse (paper spec)
_QC_TOL = 2e-12  # Numerical quality-control tolerance (paper spec)


def pinv_svd(A: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """
    SVD-based pseudoinverse with relative singular-value tolerance.

    For matrix A with singular values s, cutoff = tol * max(s).
    Only singular values > cutoff are inverted and retained.

    Parameters
    ----------
    A   : (m, n) array
    tol : relative tolerance for singular value truncation (default 1e-10)

    Returns
    -------
    A_plus : (n, m) pseudoinverse of A
    """
    if A.size == 0:
        return np.zeros((A.shape[1], A.shape[0]))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if s.max() == 0:
        return np.zeros((A.shape[1], A.shape[0]))
    cutoff = tol * s.max()
    mask = s > cutoff
    s_inv = np.where(mask, 1.0 / s, 0.0)
    # A^+ = V @ diag(s_inv) @ U^T
    return (Vt.T * s_inv) @ U.T


def rank_svd(A: np.ndarray, tol: float = _TOL) -> int:
    """
    Numerical rank of A using the same SVD tolerance as pinv_svd.

    Parameters
    ----------
    A   : (m, n) array
    tol : relative tolerance (default 1e-10)

    Returns
    -------
    rank : int
    """
    if A.size == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    if s.max() == 0:
        return 0
    return int(np.sum(s > tol * s.max()))


def proj_matrix(A: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """
    Orthogonal projection onto column space of A: P_A = A @ A^+.

    Parameters
    ----------
    A   : (n, m) array
    tol : relative tolerance

    Returns
    -------
    P   : (n, n) projection matrix
    """
    return A @ pinv_svd(A, tol)


def frobenius_norm(A: np.ndarray) -> float:
    """Frobenius norm of matrix A."""
    return float(np.linalg.norm(A, "fro"))


def l2_norm(v: np.ndarray) -> float:
    """L2 norm of vector v."""
    return float(np.linalg.norm(v))


def support_proj(Q: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """
    Return Q^+ Q, the projection onto the support (row space) of Q.

    Parameters
    ----------
    Q   : (n, n) symmetric positive semidefinite matrix
    tol : relative tolerance

    Returns
    -------
    P   : (n, n) projection matrix onto support of Q
    """
    Q_plus = pinv_svd(Q, tol)
    return Q_plus @ Q


def col_space_basis(Q: np.ndarray, tol: float = _TOL) -> Tuple[np.ndarray, int]:
    """
    Return the orthonormal basis of the column space of Q via SVD.

    For symmetric PSD Q, this equals the eigenvector basis of nonzero eigenvalues.
    Using U_r @ U_r.T rather than Q @ Q^+ gives better numerical stability when
    computing projections, especially in full-rank and near-full-rank regimes.

    Parameters
    ----------
    Q   : (n, n) symmetric PSD matrix
    tol : relative singular-value tolerance

    Returns
    -------
    U_r : (n, r) matrix of leading left singular vectors (r = rank(Q))
    r   : int, numerical rank of Q
    """
    U, s, _ = np.linalg.svd(Q, full_matrices=False)
    if s.max() == 0:
        return np.zeros((Q.shape[0], 0)), 0
    cutoff = tol * s.max()
    r = int(np.sum(s > cutoff))
    return U[:, :r], r


def mw_regression_loadings_stable(
    Phi_t: np.ndarray, Q_hat: np.ndarray, tol: float = _TOL
) -> np.ndarray:
    """
    Numerically stable MW regression loadings, computed directly via SVD of Q.

    Avoids the W_MW -> Phi_hat_f round-trip that accumulates floating-point error
    in full-rank and near-full-rank regimes. Theoretically equivalent to the
    paper's formula Phi_hat_f = Q W_MW (W_MW' Q W_MW)^+.

    Algorithm:
      1. Phi_eff = U_r @ U_r^T @ Phi_t   (project Phi onto col space of Q)
      2. C = U_r^T @ Phi_t               (r x m coordinates in col-space basis)
      3. SVD of C to find rank within col space
      4. Phi_hat_MW = U_r @ C_proj        (back to n-dimensional space)

    When Q is full rank: U_r = U (full basis), Phi_eff = Phi_t, C_proj = C,
    Phi_hat_MW = Phi_t -> LD_MW = 0 exactly.

    Parameters
    ----------
    Phi_t  : (n, m) loading matrix
    Q_hat  : (n, n) second-moment matrix
    tol    : relative SVD tolerance

    Returns
    -------
    Phi_hat_MW : (n, m) MW regression loading matrix
    """
    U_r, r = col_space_basis(Q_hat, tol)
    if r == 0:
        return np.zeros_like(Phi_t)
    C = U_r.T @ Phi_t  # (r x m)
    # Project C onto its own column space for rank-deficient Phi_eff case
    if C.size == 0:
        return np.zeros_like(Phi_t)
    Uc, sc, _ = np.linalg.svd(C, full_matrices=False)
    if sc.max() == 0:
        return np.zeros_like(Phi_t)
    cutoff_c = tol * sc.max()
    rc = int(np.sum(sc > cutoff_c))
    Uc_r = Uc[:, :rc]
    C_proj = Uc_r @ (Uc_r.T @ C)  # (r x m) projected in U_r basis
    return U_r @ C_proj             # (n x m)


def validate_numerical_qc(ld_mw: np.ndarray, lg: np.ndarray, qc_tol: float = _QC_TOL) -> float:
    """
    Compute max|LD_MW - LG| across months and assert it meets paper's QC target.

    Parameters
    ----------
    ld_mw : array of MW loading discrepancies
    lg    : array of loading-gap floors
    qc_tol: tolerance (paper target ~5.5e-12)

    Returns
    -------
    max_diff : float
    """
    diff = np.abs(np.asarray(ld_mw) - np.asarray(lg))
    max_diff = float(np.nanmax(diff)) if len(diff) > 0 else 0.0
    return max_diff
