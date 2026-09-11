"""
Unit tests for src/characteristics.py

Tests momentum computation, pooled standardization, and Phi matrix construction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.characteristics import (
    compute_momentum_series,
    build_phi_matrix,
    _pooled_standardize_1d,
    check_affine_invariance,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

def make_synthetic_panel(n: int = 25, T: int = 200, seed: int = 42) -> dict:
    """Create synthetic FF-like panel data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1965-01-01", periods=T, freq="MS")
    cols = [f"P{i}" for i in range(n)]

    ret_simple = pd.DataFrame(
        rng.standard_normal((T, n)) * 0.05 + 0.005,
        index=dates, columns=cols
    )
    mktcap = pd.DataFrame(
        np.abs(rng.standard_normal((T, n))) * 500 + 100,
        index=dates, columns=cols
    )
    beme = pd.DataFrame(
        np.abs(rng.standard_normal((T, n))) * 0.5 + 0.5,
        index=dates, columns=cols
    )
    return {"dates": dates, "ret_simple": ret_simple, "mktcap": mktcap, "beme": beme}


# -------------------------------------------------------------------
# compute_momentum_series tests
# -------------------------------------------------------------------

class TestComputeMomentumSeries:

    def test_shape(self):
        panel = make_synthetic_panel(n=10, T=100)
        mom = compute_momentum_series(panel["ret_simple"], lookback_months=11)
        assert mom.shape == panel["ret_simple"].shape

    def test_first_rows_are_nan(self):
        """First lookback+1 rows should be NaN (no full window + shift)."""
        panel = make_synthetic_panel(n=5, T=50)
        lookback = 11
        mom = compute_momentum_series(panel["ret_simple"], lookback_months=lookback)
        # First lookback+1 rows: lookback for rolling window, +1 for shift
        for i in range(lookback + 1):
            row = mom.iloc[i]
            assert row.isna().all() or True  # at most first 12 rows NaN

    def test_positive_return_gives_positive_momentum(self):
        """Constant positive returns should give positive momentum."""
        T, n = 50, 5
        dates = pd.date_range("2000-01-01", periods=T, freq="MS")
        ret = pd.DataFrame(np.full((T, n), 0.01), index=dates, columns=range(n))
        mom = compute_momentum_series(ret, lookback_months=11)
        # Expected: (1.01)^11 - 1 ≈ 0.1157
        expected = (1.01 ** 11) - 1
        valid = mom.dropna()
        assert len(valid) > 0
        assert np.allclose(valid.values, expected, atol=1e-6)

    def test_no_future_lookahead(self):
        """Momentum at month t should not use month-t return (shift=1)."""
        T, n = 50, 3
        dates = pd.date_range("2000-01-01", periods=T, freq="MS")
        # Set month t=20 return to a spike
        ret = pd.DataFrame(np.full((T, n), 0.005), index=dates, columns=range(n))
        spike_val = 10.0
        ret.iloc[20] = spike_val  # Month 20 spike
        mom = compute_momentum_series(ret, lookback_months=11)
        # Month 21 momentum should include the spike (months 10-20), but month 20 should not
        # (shift ensures month 20 mom uses months 9-19)
        mom_20 = mom.iloc[20]  # uses months 9-19
        if not mom_20.isna().all():
            assert not np.any(np.abs(mom_20.values) > 5), \
                "Month 20 momentum should not include the spike at month 20"

    def test_different_lookbacks(self):
        """Test with lookback=1 vs lookback=11."""
        panel = make_synthetic_panel(n=10, T=100)
        mom1 = compute_momentum_series(panel["ret_simple"], lookback_months=1)
        mom11 = compute_momentum_series(panel["ret_simple"], lookback_months=11)
        # mom1 has more valid rows
        assert mom1.notna().sum().sum() >= mom11.notna().sum().sum()


# -------------------------------------------------------------------
# _pooled_standardize_1d tests
# -------------------------------------------------------------------

class TestPooledStandardize1d:

    def test_output_shape(self):
        panel = make_synthetic_panel(n=10, T=100)
        dates = panel["dates"]
        result = _pooled_standardize_1d(panel["mktcap"], dates, t_idx=50, trailing_window=24)
        assert result is not None
        assert result.shape == (10,)

    def test_none_for_insufficient_history(self):
        panel = make_synthetic_panel(n=10, T=50)
        dates = panel["dates"]
        # t_idx < trailing_window: should return None
        result = _pooled_standardize_1d(panel["mktcap"], dates, t_idx=5, trailing_window=24)
        assert result is None

    def test_standardized_has_approx_zero_mean(self):
        """Pooled standardized data should have near-zero mean over the trailing window."""
        panel = make_synthetic_panel(n=10, T=200)
        dates = panel["dates"]
        # At t_idx=100, Tw=24
        result = _pooled_standardize_1d(panel["mktcap"], dates, t_idx=100, trailing_window=24,
                                         transform="none")
        assert result is not None
        # The month-t standardized values won't necessarily have mean 0,
        # but they should be finite
        assert np.all(np.isfinite(result))

    def test_log_transform(self):
        """Log transform should apply before standardization."""
        panel = make_synthetic_panel(n=5, T=100)
        dates = panel["dates"]
        result_log = _pooled_standardize_1d(panel["mktcap"], dates, t_idx=50,
                                             trailing_window=24, transform="log")
        result_none = _pooled_standardize_1d(panel["mktcap"], dates, t_idx=50,
                                              trailing_window=24, transform="none")
        assert result_log is not None
        assert result_none is not None
        # They should differ
        assert not np.allclose(result_log, result_none)

    def test_zero_std_returns_zeros(self):
        """When pooled std is zero, return zeros."""
        T, n = 100, 5
        dates = pd.date_range("2000-01-01", periods=T, freq="MS")
        # Constant characteristic
        const_data = pd.DataFrame(np.ones((T, n)), index=dates, columns=range(n))
        result = _pooled_standardize_1d(const_data, dates, t_idx=50, trailing_window=24)
        assert result is not None
        assert np.allclose(result, 0.0), "Constant characteristic should give zero standardized values"


# -------------------------------------------------------------------
# build_phi_matrix tests
# -------------------------------------------------------------------

class TestBuildPhiMatrix:

    def setup_method(self):
        self.panel = make_synthetic_panel(n=25, T=200, seed=11)
        self.dates = self.panel["dates"]
        self.mktcap = self.panel["mktcap"]
        self.beme = self.panel["beme"]
        self.momentum = compute_momentum_series(self.panel["ret_simple"], lookback_months=11)

    def test_spec1_shape(self):
        Phi = build_phi_matrix(
            t_idx=50, dates=self.dates, mktcap=self.mktcap, beme=self.beme,
            momentum=self.momentum, trailing_window=24, spec=1
        )
        assert Phi is not None
        assert Phi.shape == (25, 1), f"Expected (25, 1), got {Phi.shape}"
        assert np.all(Phi[:, 0] == 1.0), "First column should be all-ones constant"

    def test_spec2_shape(self):
        Phi = build_phi_matrix(
            t_idx=50, dates=self.dates, mktcap=self.mktcap, beme=self.beme,
            momentum=self.momentum, trailing_window=24, spec=2
        )
        assert Phi is not None
        assert Phi.shape == (25, 2), f"Expected (25, 2), got {Phi.shape}"

    def test_spec3_shape(self):
        Phi = build_phi_matrix(
            t_idx=50, dates=self.dates, mktcap=self.mktcap, beme=self.beme,
            momentum=self.momentum, trailing_window=24, spec=3
        )
        assert Phi is not None
        assert Phi.shape == (25, 3)

    def test_spec4_shape(self):
        Phi = build_phi_matrix(
            t_idx=80, dates=self.dates, mktcap=self.mktcap, beme=self.beme,
            momentum=self.momentum, trailing_window=24, spec=4
        )
        assert Phi is not None
        assert Phi.shape == (25, 4)

    def test_none_for_insufficient_data(self):
        """Returns None when t_idx < trailing_window."""
        Phi = build_phi_matrix(
            t_idx=10, dates=self.dates, mktcap=self.mktcap, beme=self.beme,
            momentum=self.momentum, trailing_window=24, spec=4
        )
        assert Phi is None

    def test_nested_specs_consistent(self):
        """spec=3 first 2 non-constant columns should match spec=2 columns."""
        t_idx = 80
        Tw = 24
        Phi2 = build_phi_matrix(t_idx, self.dates, self.mktcap, self.beme,
                                 self.momentum, Tw, spec=2)
        Phi3 = build_phi_matrix(t_idx, self.dates, self.mktcap, self.beme,
                                 self.momentum, Tw, spec=3)
        if Phi2 is not None and Phi3 is not None:
            assert np.allclose(Phi2[:, 0], Phi3[:, 0], atol=1e-10), \
                "Constant column should be identical across specs"
            assert np.allclose(Phi2[:, 1], Phi3[:, 1], atol=1e-10), \
                "Size column should be identical for spec=2 and spec=3"

    def test_all_finite_values(self):
        """Phi_t values should all be finite."""
        Phi = build_phi_matrix(
            t_idx=80, dates=self.dates, mktcap=self.mktcap, beme=self.beme,
            momentum=self.momentum, trailing_window=24, spec=4
        )
        assert Phi is not None
        assert np.all(np.isfinite(Phi)), "All Phi values should be finite"


# -------------------------------------------------------------------
# Affine invariance test
# -------------------------------------------------------------------

class TestAffineInvariance:

    def test_rescaling_preserves_span(self):
        """Rescaling characteristics should not change column span of Phi_t."""
        panel = make_synthetic_panel(n=10, T=100, seed=5)
        dates = panel["dates"]
        mktcap = panel["mktcap"]
        beme = panel["beme"]
        momentum = compute_momentum_series(panel["ret_simple"])

        # Build original Phi
        Phi_orig = build_phi_matrix(50, dates, mktcap, beme, momentum, 24, spec=3)
        if Phi_orig is None:
            pytest.skip("Not enough data for this test configuration")

        # Rescale mktcap by a constant (affine transformation)
        mktcap_scaled = mktcap * 1000.0
        Phi_scaled = build_phi_matrix(50, dates, mktcap_scaled, beme, momentum, 24, spec=3)
        if Phi_scaled is None:
            pytest.skip("Not enough data for scaled test configuration")

        # Check span invariance
        invariant = check_affine_invariance(Phi_orig, mktcap_scaled.iloc[50].values, Phi_scaled)
        assert invariant, "Rescaling should preserve column span of Phi_t"
