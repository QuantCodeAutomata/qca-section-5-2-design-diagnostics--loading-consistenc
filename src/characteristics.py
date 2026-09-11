"""
Build and standardize portfolio characteristics for FF25/FF100 panels.

Characteristic construction rules (from paper §5.2):
  1. constant (always 1)
  2. log(market cap)
  3. log(book-to-market)
  4. prior 2-12 month compounded simple return (11 months ending at t-1)

Standardization: pooled mean and std over all (asset, month) pairs in trailing
T_w months ending at t-1, applied to the month-t cross section.
# Custom — Context7 found no library equivalent for pooled trailing standardization (paper §5.2)
"""

from __future__ import annotations
from typing import Optional, List

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Momentum construction
# -------------------------------------------------------------------

def compute_momentum_series(
    returns_simple: pd.DataFrame,
    lookback_months: int = 11,
) -> pd.DataFrame:
    """
    Compute prior 2-12 month compounded simple return for each portfolio.

    For month t the momentum characteristic is:
        mom_t = prod_{k=t-lookback}^{t-1} (1 + r_k^simple) - 1
    where lookback_months = 11 (months t-11 through t-1 inclusive).

    Uses SIMPLE returns (not excess returns) per paper spec.

    Parameters
    ----------
    returns_simple : (T x n) DataFrame of monthly simple returns in decimal
    lookback_months: number of months to compound (default 11, for 2-12 month momentum)

    Returns
    -------
    mom : (T x n) DataFrame; NaN for months without enough history
    """
    log1r = np.log1p(returns_simple)  # log(1 + r)
    # Rolling sum of log returns over lookback_months, shifted by 1 (ending at t-1)
    log_cumret = log1r.rolling(window=lookback_months, min_periods=lookback_months).sum().shift(1)
    mom = np.expm1(log_cumret)  # exp(sum of log(1+r)) - 1
    return mom


# -------------------------------------------------------------------
# Characteristic matrix builder
# -------------------------------------------------------------------

def build_phi_matrix(
    t_idx: int,
    dates: pd.DatetimeIndex,
    mktcap: pd.DataFrame,
    beme: pd.DataFrame,
    momentum: pd.DataFrame,
    trailing_window: int,
    spec: int = 4,
) -> Optional[np.ndarray]:
    """
    Build the n x m loading matrix Phi_t for a given month t.

    Specifications (nested):
      spec=1: [constant]
      spec=2: [constant, log(ME)]
      spec=3: [constant, log(ME), log(BE/ME)]
      spec=4: [constant, log(ME), log(BE/ME), momentum]

    Characteristics are standardized using pooled (asset x month) statistics
    from trailing T_w months ending at t-1 (i.e., months t-T_w through t-1).

    Parameters
    ----------
    t_idx           : integer index into `dates` for month t
    dates           : DatetimeIndex of all evaluation months
    mktcap          : (T x n) DataFrame of portfolio market caps
    beme            : (T x n) DataFrame of portfolio B/M ratios
    momentum        : (T x n) DataFrame of pre-computed momentum values
    trailing_window : T_w rolling window length
    spec            : number of characteristics (1-4)

    Returns
    -------
    Phi_t : (n x spec) array, or None if data unavailable
    """
    t_date = dates[t_idx]
    n = mktcap.shape[1]

    # Always need constant
    phi_cols: list[np.ndarray] = [np.ones(n)]

    if spec < 1:
        return None

    if spec >= 2:
        # log(ME) at month t
        me_t = mktcap.loc[t_date].values.astype(float)
        if np.any(me_t <= 0):
            me_t = np.where(me_t <= 0, np.nan, me_t)
        log_me_t = np.log(me_t)

        # Trailing pooled standardization: T_w months ending at t-1
        log_me_std = _pooled_standardize_1d(
            mktcap, dates, t_idx, trailing_window, transform="log"
        )
        if log_me_std is None:
            return None
        phi_cols.append(log_me_std)

    if spec >= 3:
        # log(BE/ME) at month t
        bm_t = beme.loc[t_date].values.astype(float)
        if np.any(bm_t <= 0):
            bm_t = np.where(bm_t <= 0, np.nan, bm_t)

        log_bm_std = _pooled_standardize_1d(
            beme, dates, t_idx, trailing_window, transform="log"
        )
        if log_bm_std is None:
            return None
        phi_cols.append(log_bm_std)

    if spec >= 4:
        # Momentum at month t
        mom_std = _pooled_standardize_1d(
            momentum, dates, t_idx, trailing_window, transform="none"
        )
        if mom_std is None:
            return None
        phi_cols.append(mom_std)

    Phi_t = np.column_stack(phi_cols)  # (n x spec)
    return Phi_t


def _pooled_standardize_1d(
    char_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    t_idx: int,
    trailing_window: int,
    transform: str = "none",
) -> Optional[np.ndarray]:
    """
    Standardize month-t characteristic cross section using pooled trailing statistics.

    Pool all n * T_w asset-month observations from months t-T_w through t-1.
    Compute pooled mean and pooled std. Apply to month-t values.

    Parameters
    ----------
    char_df         : (T x n) DataFrame of raw characteristic values
    dates           : DatetimeIndex matching char_df rows
    t_idx           : current month index
    trailing_window : T_w
    transform       : 'log' applies log() before standardization, 'none' uses raw values

    Returns
    -------
    standardized : (n,) array for month t, or None if data unavailable
    """
    if t_idx < trailing_window:
        return None
    t_date = dates[t_idx]

    # Trailing months: t-T_w through t-1 (inclusive), i.e., T_w months
    trailing_dates = dates[t_idx - trailing_window: t_idx]
    if len(trailing_dates) < trailing_window:
        return None

    # Get trailing block: (T_w x n)
    trailing_vals = char_df.reindex(trailing_dates).values.astype(float)
    if transform == "log":
        trailing_vals = np.where(trailing_vals > 0, np.log(trailing_vals), np.nan)

    # Pooled statistics from n * T_w observations
    pooled_flat = trailing_vals.flatten()
    pooled_flat = pooled_flat[np.isfinite(pooled_flat)]
    if len(pooled_flat) == 0:
        return None
    pooled_mean = float(np.mean(pooled_flat))
    pooled_std = float(np.std(pooled_flat, ddof=0))

    # Month-t cross section
    t_vals = char_df.reindex([t_date]).values.astype(float).flatten()
    if transform == "log":
        t_vals = np.where(t_vals > 0, np.log(t_vals), np.nan)

    if pooled_std == 0.0:
        # If std is zero, set standardized values to zero (paper convention)
        return np.zeros_like(t_vals)

    standardized = (t_vals - pooled_mean) / pooled_std
    # Replace NaN with 0 (conservative; assets with missing chars get 0)
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    return standardized


# -------------------------------------------------------------------
# Validate affine-standardization invariance (paper check)
# -------------------------------------------------------------------

def check_affine_invariance(
    Phi_t_original: np.ndarray,
    char_vals_scaled: np.ndarray,
    Phi_t_scaled: np.ndarray,
    tol: float = 1e-8,
) -> bool:
    """
    Check that rescaling/recentering raw characteristics before standardization
    does not change the column span of Phi_t beyond numerical tolerance.

    Parameters
    ----------
    Phi_t_original : (n x m) original loading matrix
    char_vals_scaled: (n,) scaled raw characteristic values
    Phi_t_scaled    : (n x m) loading matrix after scaling/recentering
    tol             : tolerance

    Returns
    -------
    invariant : bool
    """
    from src.matrix_utils import proj_matrix
    P_orig = proj_matrix(Phi_t_original)
    P_scaled = proj_matrix(Phi_t_scaled)
    diff = np.linalg.norm(P_orig - P_scaled, "fro")
    return bool(diff < tol)
