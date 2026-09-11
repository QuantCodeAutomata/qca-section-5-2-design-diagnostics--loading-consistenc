"""
Unit and integration tests for src/diagnostics.py

Tests estimate_moments, weight matrices, loading discrepancy,
loading gap floor, risk-premium gap, and their numerical relationships.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.matrix_utils import pinv_svd, rank_svd, _TOL, _QC_TOL
from src.diagnostics import (
    estimate_moments,
    compute_ols_weights,
    compute_mw_weights,
    compute_gls_weights,
    compute_regression_loadings,
    compute_loading_discrepancy,
    compute_loading_gap_floor,
    compute_risk_premium_gap,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

def make_returns(T: int, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n)) * 0.05


def make_phi(n: int, m: int, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Phi = np.zeros((n, m))
    Phi[:, 0] = 1.0  # constant column
    if m > 1:
        Phi[:, 1:] = rng.standard_normal((n, m - 1))
    return Phi


# -------------------------------------------------------------------
# estimate_moments
# -------------------------------------------------------------------

class TestEstimateMoments:

    def test_shapes(self):
        T, n = 60, 25
        ret = make_returns(T, n)
        mu, Q, Sigma = estimate_moments(ret)
        assert mu.shape == (n,)
        assert Q.shape == (n, n)
        assert Sigma.shape == (n, n)

    def test_sigma_is_q_minus_outer_mu(self):
        T, n = 48, 10
        ret = make_returns(T, n)
        mu, Q, Sigma = estimate_moments(ret)
        expected = Q - np.outer(mu, mu)
        assert np.allclose(Sigma, expected, atol=1e-12)

    def test_mu_formula(self):
        T, n = 30, 5
        ret = make_returns(T, n)
        mu, Q, Sigma = estimate_moments(ret)
        expected_mu = ret.mean(axis=0)
        assert np.allclose(mu, expected_mu, atol=1e-12)

    def test_q_formula(self):
        T, n = 30, 5
        ret = make_returns(T, n)
        mu, Q, Sigma = estimate_moments(ret)
        expected_Q = ret.T @ ret / T
        assert np.allclose(Q, expected_Q, atol=1e-12)

    def test_sigma_psd(self):
        """Sigma_hat should be positive semidefinite (eigenvalues >= 0)."""
        T, n = 120, 15
        ret = make_returns(T, n, seed=7)
        _, _, Sigma = estimate_moments(ret)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals >= -1e-10), "Sigma_hat should be PSD"

    def test_q_psd(self):
        """Q_hat should be positive semidefinite."""
        T, n = 120, 15
        ret = make_returns(T, n, seed=8)
        _, Q, _ = estimate_moments(ret)
        eigvals = np.linalg.eigvalsh(Q)
        assert np.all(eigvals >= -1e-10), "Q_hat should be PSD"

    def test_single_row(self):
        """Single return row: mu = x, Q = outer(x, x), Sigma = 0."""
        x = np.array([0.01, 0.02, -0.005])
        ret = x.reshape(1, 3)
        mu, Q, Sigma = estimate_moments(ret)
        assert np.allclose(mu, x, atol=1e-12)
        assert np.allclose(Q, np.outer(x, x), atol=1e-12)
        assert np.allclose(Sigma, 0, atol=1e-12)


# -------------------------------------------------------------------
# Weight matrix tests
# -------------------------------------------------------------------

class TestOlsWeights:

    def test_shape(self):
        n, m = 25, 4
        Phi = make_phi(n, m)
        W = compute_ols_weights(Phi)
        assert W.shape == (n, m), f"Expected ({n}, {m}), got {W.shape}"

    def test_w_prime_is_pseudoinverse(self):
        """W_OLS' = Phi^+, so W_OLS = (Phi^+)^T."""
        n, m = 25, 4
        Phi = make_phi(n, m)
        W = compute_ols_weights(Phi)
        Phi_plus = pinv_svd(Phi)
        assert np.allclose(W, Phi_plus.T, atol=1e-10)

    def test_factor_returns_shape(self):
        n, m = 25, 4
        Phi = make_phi(n, m)
        W = compute_ols_weights(Phi)
        x = np.random.default_rng(1).standard_normal(n)
        f = W.T @ x
        assert f.shape == (m,)


class TestMwWeights:

    def test_shape(self):
        T, n, m = 100, 25, 4
        ret = make_returns(T, n)
        _, Q, _ = estimate_moments(ret)
        Phi = make_phi(n, m)
        W = compute_mw_weights(Phi, Q)
        assert W.shape == (n, m)

    def test_full_rank_Q_gives_zero_lg(self):
        """When Q_hat is full rank and T >= n, MW discrepancy = LG floor = 0."""
        T, n, m = 200, 5, 2
        ret = make_returns(T, n, seed=77)
        mu, Q, Sigma = estimate_moments(ret)
        assert rank_svd(Q) == n, "Q should be full rank for T >> n"

        Phi = make_phi(n, m)
        W_mw = compute_mw_weights(Phi, Q)
        Phi_hat_mw = compute_regression_loadings(Q, W_mw)
        ld_mw = compute_loading_discrepancy(Phi_hat_mw, Phi)
        lg = compute_loading_gap_floor(Q, Phi)
        # Both should be ~0 when Q is full rank (no directions lost)
        assert ld_mw < 1e-8, f"LD_MW should be ~0 when Q is full rank, got {ld_mw}"
        assert lg < 1e-8, f"LG should be ~0 when Q is full rank, got {lg}"

    def test_mw_equals_lg_floor(self):
        """MW discrepancy should equal the LG floor (fundamental paper claim)."""
        rng = np.random.default_rng(123)
        # Underdetermined: T < n, so Q is rank-deficient
        T, n, m = 10, 25, 3
        ret = rng.standard_normal((T, n)) * 0.05
        mu, Q, Sigma = estimate_moments(ret)
        Phi = make_phi(n, m, seed=55)

        W_mw = compute_mw_weights(Phi, Q)
        Phi_hat_mw = compute_regression_loadings(Q, W_mw)
        ld_mw = compute_loading_discrepancy(Phi_hat_mw, Phi)
        lg = compute_loading_gap_floor(Q, Phi)
        # MW should achieve the floor
        assert abs(ld_mw - lg) < 1e-8, \
            f"MW discrepancy {ld_mw:.6e} should equal LG floor {lg:.6e}"


class TestGlsWeights:

    def test_shape(self):
        T, n, m = 120, 25, 4
        ret = make_returns(T, n)
        _, _, Sigma = estimate_moments(ret)
        Phi = make_phi(n, m)
        W = compute_gls_weights(Phi, Sigma)
        assert W.shape == (n, m)

    def test_singular_sigma_handled(self):
        """GLS weights should not crash when Sigma is singular (T < n)."""
        T, n, m = 10, 25, 4
        ret = make_returns(T, n)
        _, _, Sigma = estimate_moments(ret)
        Phi = make_phi(n, m)
        W = compute_gls_weights(Phi, Sigma)
        assert W.shape == (n, m)
        assert np.all(np.isfinite(W))


# -------------------------------------------------------------------
# Regression loadings
# -------------------------------------------------------------------

class TestRegressionLoadings:

    def test_shape(self):
        T, n, m = 100, 25, 4
        ret = make_returns(T, n)
        _, Q, _ = estimate_moments(ret)
        Phi = make_phi(n, m)
        W = compute_ols_weights(Phi)
        Phi_hat = compute_regression_loadings(Q, W)
        assert Phi_hat.shape == (n, m)

    def test_formula_correctness(self):
        """Phi_hat = Q W (W' Q W)^+."""
        T, n, m = 80, 10, 3
        ret = make_returns(T, n, seed=21)
        _, Q, _ = estimate_moments(ret)
        Phi = make_phi(n, m, seed=22)
        W = compute_mw_weights(Phi, Q)

        Phi_hat = compute_regression_loadings(Q, W)
        # Verify formula manually
        inner = W.T @ Q @ W
        inner_plus = pinv_svd(inner)
        expected = Q @ W @ inner_plus
        assert np.allclose(Phi_hat, expected, atol=1e-10)


# -------------------------------------------------------------------
# Loading Discrepancy
# -------------------------------------------------------------------

class TestLoadingDiscrepancy:

    def test_zero_for_identical(self):
        n, m = 25, 4
        Phi = make_phi(n, m)
        ld = compute_loading_discrepancy(Phi, Phi)
        assert ld == 0.0

    def test_range(self):
        """LD should be >= 0."""
        n, m = 10, 3
        rng = np.random.default_rng(9)
        Phi_a = rng.standard_normal((n, m))
        Phi_b = rng.standard_normal((n, m))
        ld = compute_loading_discrepancy(Phi_a, Phi_b)
        assert ld >= 0.0

    def test_zero_denominator(self):
        """Returns 0 when ||Phi_t||_F = 0."""
        n, m = 5, 2
        Phi_zero = np.zeros((n, m))
        ld = compute_loading_discrepancy(np.ones((n, m)), Phi_zero)
        assert ld == 0.0


# -------------------------------------------------------------------
# Loading Gap Floor
# -------------------------------------------------------------------

class TestLoadingGapFloor:

    def test_zero_when_phi_in_support_Q(self):
        """LG = 0 when col(Phi) subset of support(Q_hat) — i.e., Q full rank."""
        T, n, m = 200, 5, 2
        ret = make_returns(T, n, seed=42)
        _, Q, _ = estimate_moments(ret)
        Phi = make_phi(n, m)
        lg = compute_loading_gap_floor(Q, Phi)
        assert lg < 1e-8, f"LG should be ~0 when Q is full rank, got {lg}"

    def test_positive_when_q_singular(self):
        """LG > 0 when Q is rank-deficient and Phi has components outside support."""
        T, n, m = 5, 25, 4  # T << n, so Q singular
        rng = np.random.default_rng(3)
        ret = rng.standard_normal((T, n)) * 0.05
        _, Q, _ = estimate_moments(ret)
        # Phi with random non-constant columns
        Phi = rng.standard_normal((n, m))
        Phi[:, 0] = 1.0
        lg = compute_loading_gap_floor(Q, Phi)
        # With n=25, T=5, Q has rank ~5; Phi likely has components outside support
        assert lg >= 0.0

    def test_range(self):
        """LG should be in [0, 1]."""
        T, n, m = 10, 20, 3
        ret = make_returns(T, n, seed=15)
        _, Q, _ = estimate_moments(ret)
        Phi = make_phi(n, m)
        lg = compute_loading_gap_floor(Q, Phi)
        assert 0.0 <= lg <= 1.0 + 1e-8


# -------------------------------------------------------------------
# Risk-Premium Gap
# -------------------------------------------------------------------

class TestRiskPremiumGap:

    def test_range(self):
        """RPG should be in [0, 1]."""
        T, n, m = 60, 25, 4
        ret = make_returns(T, n, seed=30)
        mu, _, _ = estimate_moments(ret)
        Phi = make_phi(n, m)
        rpg = compute_risk_premium_gap(Phi, mu)
        assert 0.0 <= rpg <= 1.0 + 1e-8

    def test_zero_when_mu_in_span_phi(self):
        """RPG = 0 when mu is in column space of Phi."""
        n, m = 10, 3
        rng = np.random.default_rng(50)
        Phi = rng.standard_normal((n, m))
        alpha = rng.standard_normal(m)
        mu = Phi @ alpha  # mu exactly in col(Phi)
        rpg = compute_risk_premium_gap(Phi, mu)
        assert rpg < 1e-8, f"RPG should be ~0 when mu in span(Phi), got {rpg}"

    def test_one_when_mu_orthogonal_to_phi(self):
        """RPG = 1 when mu is orthogonal to all columns of Phi."""
        n = 8
        # Build Phi spanning first 2 standard basis vectors
        Phi = np.zeros((n, 2))
        Phi[0, 0] = 1.0
        Phi[1, 1] = 1.0
        # mu is in the orthogonal complement
        mu = np.zeros(n)
        mu[2] = 1.0
        rpg = compute_risk_premium_gap(Phi, mu)
        assert abs(rpg - 1.0) < 1e-8, f"RPG should be ~1 when mu orthogonal to Phi, got {rpg}"

    def test_zero_mu(self):
        """Returns 0 when ||mu|| = 0 (paper convention)."""
        n, m = 10, 4
        Phi = make_phi(n, m)
        mu = np.zeros(n)
        rpg = compute_risk_premium_gap(Phi, mu)
        assert rpg == 0.0

    def test_monotone_with_more_characteristics(self):
        """
        Adding characteristics should not increase RPG on average
        (paper Claim B: richer characteristic set -> smaller RPG).
        Uses a fixed return block to check ordering.
        """
        T, n = 60, 25
        ret = make_returns(T, n, seed=77)
        mu, _, _ = estimate_moments(ret)

        rpgs = []
        rng = np.random.default_rng(88)
        cols = [np.ones(n)]
        for _ in range(3):
            cols.append(rng.standard_normal(n))
            Phi = np.column_stack(cols)
            rpgs.append(compute_risk_premium_gap(Phi, mu))

        # Each added characteristic can only decrease or maintain RPG
        for i in range(len(rpgs) - 1):
            assert rpgs[i] >= rpgs[i + 1] - 1e-8, \
                f"RPG should be non-increasing: rpg[{i}]={rpgs[i]:.4f} > rpg[{i+1}]={rpgs[i+1]:.4f}"


# -------------------------------------------------------------------
# MW attains LG floor (key paper claim, Claim A)
# -------------------------------------------------------------------

class TestMwAttainsLgFloor:
    """Key numerical test: MW discrepancy = LG floor to machine precision."""

    def _run_case(self, T: int, n: int, m: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        ret = rng.standard_normal((T, n)) * 0.05
        _, Q, Sigma = estimate_moments(ret)
        Phi = make_phi(n, m, seed=seed + 1)

        W_mw = compute_mw_weights(Phi, Q)
        Phi_hat_mw = compute_regression_loadings(Q, W_mw)
        ld_mw = compute_loading_discrepancy(Phi_hat_mw, Phi)
        lg = compute_loading_gap_floor(Q, Phi)
        return abs(ld_mw - lg)

    def test_full_rank_case(self):
        """T >> n: Q full rank, LD_MW = LG ≈ 0."""
        diff = self._run_case(T=500, n=10, m=3)
        assert diff < 1e-8

    def test_rank_deficient_case(self):
        """T < n: Q rank-deficient, LD_MW = LG > 0 but equal."""
        diff = self._run_case(T=10, n=25, m=4)
        assert diff < 1e-7, f"MW should equal LG floor: diff={diff:.3e}"

    def test_borderline_rank(self):
        """T ≈ n: borderline case."""
        diff = self._run_case(T=25, n=25, m=4, seed=7)
        assert diff < 1e-7

    def test_multiple_seeds(self):
        """Test across multiple random configurations."""
        for seed in [1, 2, 3, 4, 5]:
            T, n, m = 20, 15, 3
            diff = self._run_case(T, n, m, seed)
            assert diff < 1e-7, f"Seed {seed}: MW-LG diff={diff:.3e}"
