"""
Unit tests for src/performance.py

Tests LW shrinkage, strategy weight builders, benchmark strategies,
portfolio mapping, and Sharpe ratio computation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.performance import (
    apply_ledoit_wolf,
    compute_xstar_weights,
    compute_cmve_weights,
    compute_equal_weights,
    compute_pc_weights,
    factor_to_single_portfolio,
    build_strategy_weights,
    annualized_sharpe,
    ex_ante_sharpe,
)
from src.diagnostics import estimate_moments
from src.matrix_utils import pinv_svd, _TOL


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

def make_returns(T: int, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n)) * 0.05


def make_phi(n: int, m: int, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Phi = np.zeros((n, m))
    Phi[:, 0] = 1.0
    if m > 1:
        Phi[:, 1:] = rng.standard_normal((n, m - 1))
    return Phi


# -------------------------------------------------------------------
# Ledoit-Wolf shrinkage
# -------------------------------------------------------------------

class TestLedoitWolf:

    def test_output_shape(self):
        T, n = 60, 25
        ret = make_returns(T, n)
        mu = ret.mean(axis=0)
        Sigma_lw, Q_lw = apply_ledoit_wolf(ret, mu)
        assert Sigma_lw.shape == (n, n)
        assert Q_lw.shape == (n, n)

    def test_sigma_symmetric(self):
        T, n = 60, 25
        ret = make_returns(T, n)
        mu = ret.mean(axis=0)
        Sigma_lw, _ = apply_ledoit_wolf(ret, mu)
        assert np.allclose(Sigma_lw, Sigma_lw.T, atol=1e-10)

    def test_q_lw_reconstruction(self):
        """Q_lw should equal Sigma_lw + mu mu'."""
        T, n = 60, 25
        ret = make_returns(T, n)
        mu = ret.mean(axis=0)
        Sigma_lw, Q_lw = apply_ledoit_wolf(ret, mu)
        expected_Q = Sigma_lw + np.outer(mu, mu)
        assert np.allclose(Q_lw, expected_Q, atol=1e-10)

    def test_sigma_psd(self):
        """LW covariance should be PSD."""
        T, n = 60, 25
        ret = make_returns(T, n)
        mu = ret.mean(axis=0)
        Sigma_lw, _ = apply_ledoit_wolf(ret, mu)
        eigvals = np.linalg.eigvalsh(Sigma_lw)
        assert np.all(eigvals >= -1e-8), f"Min eigenvalue: {eigvals.min()}"

    def test_shrinkage_increases_min_eigenvalue(self):
        """LW shrinkage should increase minimum eigenvalue vs raw covariance."""
        T, n = 30, 25  # Near singular: T ≈ n
        ret = make_returns(T, n, seed=7)
        mu, Q, Sigma = estimate_moments(ret)
        Sigma_lw, _ = apply_ledoit_wolf(ret, mu)
        min_raw = np.linalg.eigvalsh(Sigma).min()
        min_lw = np.linalg.eigvalsh(Sigma_lw).min()
        assert min_lw >= min_raw - 1e-10, \
            "LW should not decrease minimum eigenvalue"


# -------------------------------------------------------------------
# Benchmark strategies
# -------------------------------------------------------------------

class TestXstarWeights:

    def test_shape(self):
        T, n = 120, 25
        ret = make_returns(T, n)
        mu, Q, _ = estimate_moments(ret)
        w = compute_xstar_weights(Q, mu)
        assert w.shape == (n,)

    def test_finite(self):
        T, n = 120, 25
        ret = make_returns(T, n)
        mu, Q, _ = estimate_moments(ret)
        w = compute_xstar_weights(Q, mu)
        assert np.all(np.isfinite(w))

    def test_formula(self):
        """w = Q^+ mu."""
        T, n = 120, 10
        ret = make_returns(T, n, seed=3)
        mu, Q, _ = estimate_moments(ret)
        w = compute_xstar_weights(Q, mu)
        expected = pinv_svd(Q) @ mu
        assert np.allclose(w, expected, atol=1e-10)


class TestCmveWeights:

    def test_shape(self):
        T, n = 120, 25
        ret = make_returns(T, n)
        mu, _, Sigma = estimate_moments(ret)
        w = compute_cmve_weights(Sigma, mu)
        assert w.shape == (n,)

    def test_proportional_to_sigma_inv_mu(self):
        """CMVE is proportional to Sigma^+ mu."""
        T, n = 120, 10
        ret = make_returns(T, n, seed=5)
        mu, _, Sigma = estimate_moments(ret)
        w = compute_cmve_weights(Sigma, mu)
        expected = pinv_svd(Sigma) @ mu
        # Should be equal (no normalization)
        assert np.allclose(w, expected, atol=1e-10)


class TestEqualWeights:

    def test_shape(self):
        n = 25
        w = compute_equal_weights(n)
        assert w.shape == (n,)

    def test_sums_to_one(self):
        n = 25
        w = compute_equal_weights(n)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_all_equal(self):
        n = 10
        w = compute_equal_weights(n)
        assert np.allclose(w, 1.0 / n)


class TestPcWeights:

    def test_shape(self):
        T, n = 120, 25
        ret = make_returns(T, n, seed=6)
        mu, Q, Sigma = estimate_moments(ret)
        w = compute_pc_weights(Q, mu, Sigma, m=4)
        assert w.shape == (n,)

    def test_fewer_pcs_than_assets(self):
        T, n, m = 120, 25, 4
        ret = make_returns(T, n)
        mu, Q, Sigma = estimate_moments(ret)
        w = compute_pc_weights(Q, mu, Sigma, m=m)
        assert np.all(np.isfinite(w))


# -------------------------------------------------------------------
# factor_to_single_portfolio
# -------------------------------------------------------------------

class TestFactorToSinglePortfolio:

    def test_output_shape(self):
        n, m = 25, 4
        rng = np.random.default_rng(10)
        W_t = rng.standard_normal((n, m))
        mu_hat = rng.standard_normal(n)
        T = 120
        ret = make_returns(T, n, seed=11)
        _, _, Sigma = estimate_moments(ret)
        w_t = factor_to_single_portfolio(W_t, mu_hat, Sigma)
        assert w_t.shape == (n,)

    def test_finite_output(self):
        n, m = 10, 3
        rng = np.random.default_rng(20)
        W_t = rng.standard_normal((n, m))
        mu_hat = rng.standard_normal(n)
        T = 80
        ret = make_returns(T, n)
        _, _, Sigma = estimate_moments(ret)
        w_t = factor_to_single_portfolio(W_t, mu_hat, Sigma)
        assert np.all(np.isfinite(w_t))


# -------------------------------------------------------------------
# build_strategy_weights
# -------------------------------------------------------------------

class TestBuildStrategyWeights:

    def setup_method(self):
        T, n, m = 120, 25, 4
        self.T = T
        self.n = n
        ret = make_returns(T, n, seed=50)
        self.ret_block = ret
        self.mu, self.Q, self.Sigma = estimate_moments(ret)
        self.Phi = make_phi(n, m)

    def _check_basic(self, method: str, shrink: bool = False):
        w = build_strategy_weights(
            method=method,
            Phi_t=self.Phi,
            mu_hat=self.mu,
            Sigma_hat=self.Sigma,
            Q_hat=self.Q,
            shrink=shrink,
            returns_block=self.ret_block if shrink else None,
        )
        assert w.shape == (self.n,), f"Shape mismatch for {method}"
        assert np.all(np.isfinite(w)), f"Non-finite values for {method}"
        return w

    def test_ols(self):
        self._check_basic("ols")

    def test_mw(self):
        self._check_basic("mw")

    def test_gls(self):
        self._check_basic("gls")

    def test_xstar(self):
        self._check_basic("xstar")

    def test_cmve(self):
        self._check_basic("cmve")

    def test_equal(self):
        self._check_basic("equal")

    def test_pc(self):
        self._check_basic("pc")

    def test_mw_lw(self):
        self._check_basic("mw", shrink=True)

    def test_gls_lw(self):
        self._check_basic("gls", shrink=True)

    def test_xstar_lw(self):
        self._check_basic("xstar", shrink=True)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy method"):
            build_strategy_weights(
                method="unknown_xyz",
                Phi_t=self.Phi,
                mu_hat=self.mu,
                Sigma_hat=self.Sigma,
                Q_hat=self.Q,
            )

    def test_equal_sums_to_one(self):
        w = self._check_basic("equal")
        assert abs(w.sum() - 1.0) < 1e-10


# -------------------------------------------------------------------
# Performance metrics
# -------------------------------------------------------------------

class TestAnnualizedSharpe:

    def test_positive_returns(self):
        """Positive mean returns give positive Sharpe."""
        returns = np.full(120, 0.01)  # 1% per month, no variation -> std=0 -> SR=0
        # Add tiny noise
        returns = returns + np.random.default_rng(1).standard_normal(120) * 0.001
        sr = annualized_sharpe(returns)
        assert sr > 0, "Positive mean with positive std should give positive Sharpe"

    def test_annualization_factor(self):
        """Annual Sharpe = sqrt(12) * monthly mean / monthly std."""
        rng = np.random.default_rng(2)
        returns = rng.standard_normal(240) * 0.04 + 0.005
        expected = np.sqrt(12) * returns.mean() / returns.std(ddof=1)
        result = annualized_sharpe(returns)
        assert abs(result - expected) < 1e-10

    def test_zero_std_returns_zero(self):
        returns = np.full(50, 0.005)
        sr = annualized_sharpe(returns)
        assert sr == 0.0

    def test_empty_returns(self):
        sr = annualized_sharpe(np.array([]))
        assert sr == 0.0

    def test_single_value(self):
        sr = annualized_sharpe(np.array([0.01]))
        assert sr == 0.0  # std=0 for single obs

    def test_value_range(self):
        """Reasonable Sharpe ratios should be in plausible range."""
        rng = np.random.default_rng(3)
        returns = rng.standard_normal(240) * 0.05 + 0.008
        sr = annualized_sharpe(returns)
        assert -10 < sr < 10, f"Sharpe {sr} outside plausible range"


class TestExAnteSharpe:

    def test_positive_mu_gives_positive_ea_sr(self):
        n = 10
        rng = np.random.default_rng(4)
        w = np.ones(n) / n
        mu = np.full(n, 0.01)  # All positive
        X = rng.standard_normal((120, n)) * 0.05
        Sigma = (X.T @ X) / 120
        Sigma += 0.01 * np.eye(n)  # Make PD
        ea = ex_ante_sharpe(w, mu, Sigma)
        assert ea > 0, "Positive mu should give positive ex-ante Sharpe"

    def test_zero_variance_returns_zero(self):
        n = 5
        w = np.ones(n) / n
        mu = np.ones(n) * 0.01
        # Sigma = 0 (degenerate)
        ea = ex_ante_sharpe(w, mu, np.zeros((n, n)))
        assert ea == 0.0

    def test_finite_output(self):
        n = 25
        rng = np.random.default_rng(5)
        w = rng.standard_normal(n)
        mu = rng.standard_normal(n)
        X = rng.standard_normal((120, n))
        Sigma = (X.T @ X) / 120 + 0.01 * np.eye(n)
        ea = ex_ante_sharpe(w, mu, Sigma)
        assert np.isfinite(ea)
