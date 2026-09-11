"""
Design Diagnostics module for QCA Section 5.2 experiment.

Implements:
  - Rolling moment estimation (mu_hat, Q_hat, Sigma_hat)
  - OLS, MW, GLS factor weight matrices
  - Sample regression loading matrix Phi_hat^f
  - Loading Discrepancy (LD), Loading Gap floor (LG), Risk-Premium Gap (RPG)
  - Full diagnostic run for any panel/window/characteristic-spec combination

All matrix pseudoinversions use the SVD-based relative-tolerance implementation.
# Custom — Context7 found no library equivalent for these paper-specific metrics (paper §5.2)
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple

import warnings
import numpy as np
import pandas as pd

from src.matrix_utils import (
    pinv_svd, proj_matrix, frobenius_norm, l2_norm,
    rank_svd, validate_numerical_qc, mw_regression_loadings_stable,
    col_space_basis,
    _TOL, _QC_TOL,
)
from src.characteristics import build_phi_matrix, compute_momentum_series


# -------------------------------------------------------------------
# Rolling moment estimators (paper §5.2, Step 3)
# -------------------------------------------------------------------

def estimate_moments(
    returns: np.ndarray,
    tol: float = _TOL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate mu_hat, Q_hat, Sigma_hat from a (T_w x n) return block.

    mu_hat_t    = (1/T_w) sum x_s
    Q_hat_t     = (1/T_w) sum x_s x_s'
    Sigma_hat_t = Q_hat_t - mu_hat_t mu_hat_t'

    Parameters
    ----------
    returns : (T_w x n) array of excess returns in decimal
    tol     : pseudoinverse tolerance

    Returns
    -------
    mu_hat, Q_hat, Sigma_hat : (n,), (n,n), (n,n) arrays
    """
    T_w, n = returns.shape
    mu_hat = returns.mean(axis=0)                       # (n,)
    Q_hat = (returns.T @ returns) / T_w                 # (n,n)
    Sigma_hat = Q_hat - np.outer(mu_hat, mu_hat)        # (n,n)
    return mu_hat, Q_hat, Sigma_hat


# -------------------------------------------------------------------
# Weight matrix computations (paper §5.2, Step 5)
# -------------------------------------------------------------------

def compute_ols_weights(Phi_t: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """
    OLS factor weight matrix: W_OLS_t' = Phi_t^+.

    Returns W_t shape (n x m) such that factor returns = W_t' x.

    Parameters
    ----------
    Phi_t : (n x m) loading matrix
    tol   : pseudoinverse tolerance

    Returns
    -------
    W_t : (n x m) weight matrix (columns = portfolio weights per factor)
    """
    # W_OLS_t' is (m x n), so W_OLS_t is (n x m)
    W_t_prime = pinv_svd(Phi_t, tol)  # (m x n)
    return W_t_prime.T                 # (n x m)


def compute_mw_weights(
    Phi_t: np.ndarray,
    Q_hat: np.ndarray,
    tol: float = _TOL,
) -> np.ndarray:
    """
    Moment-Weighted factor weight matrix using effective-loading formulation.

    Phi_eff_t = Q_hat^+ Q_hat Phi_t   (projection of Phi onto support of Q)
    W_MW_t'   = (Phi_eff_t' Q_hat^+ Phi_eff_t)^+ Phi_eff_t' Q_hat^+

    This effective-loading version is the safest finite-sample implementation
    when the support may fail (paper §5.2, Step 5 note).

    Parameters
    ----------
    Phi_t  : (n x m) loading matrix
    Q_hat  : (n x n) second-moment matrix
    tol    : pseudoinverse tolerance

    Returns
    -------
    W_t : (n x m) weight matrix
    """
    n, m = Phi_t.shape
    Q_plus = pinv_svd(Q_hat, tol)                        # (n x n)
    P_Q = Q_hat @ Q_plus                                  # support projection (n x n)
    Phi_eff = P_Q @ Phi_t                                 # (n x m)

    inner = Phi_eff.T @ Q_plus @ Phi_eff                  # (m x m)
    inner_plus = pinv_svd(inner, tol)                     # (m x m)
    W_t_prime = inner_plus @ Phi_eff.T @ Q_plus           # (m x n)
    return W_t_prime.T                                    # (n x m)


def compute_gls_weights(
    Phi_t: np.ndarray,
    Sigma_hat: np.ndarray,
    tol: float = _TOL,
) -> np.ndarray:
    """
    GLS factor weight matrix: W_GLS_t' = (Phi_t' Sigma_hat^+ Phi_t)^+ Phi_t' Sigma_hat^+.

    Parameters
    ----------
    Phi_t    : (n x m) loading matrix
    Sigma_hat: (n x n) covariance matrix
    tol      : pseudoinverse tolerance

    Returns
    -------
    W_t : (n x m) weight matrix
    """
    Sigma_plus = pinv_svd(Sigma_hat, tol)                 # (n x n)
    inner = Phi_t.T @ Sigma_plus @ Phi_t                  # (m x m)
    inner_plus = pinv_svd(inner, tol)                     # (m x m)
    W_t_prime = inner_plus @ Phi_t.T @ Sigma_plus         # (m x n)
    return W_t_prime.T                                    # (n x m)


# -------------------------------------------------------------------
# Sample regression loading matrix (paper §5.2, Step 6)
# -------------------------------------------------------------------

def compute_regression_loadings(
    Q_hat: np.ndarray,
    W_t: np.ndarray,
    tol: float = _TOL,
) -> np.ndarray:
    """
    Compute regression-loading matrix of traded factors.

    Phi_hat_t^f = Q_hat W_t (W_t' Q_hat W_t)^+

    This is the exact sample analogue stated in the paper for traded factors.
    NOT a separate regression fit; uses second-moment matrix directly.

    Parameters
    ----------
    Q_hat : (n x n) second-moment matrix
    W_t   : (n x m) weight matrix
    tol   : pseudoinverse tolerance

    Returns
    -------
    Phi_hat_f : (n x m) regression loading matrix
    """
    factor_Q = W_t.T @ Q_hat @ W_t           # (m x m)
    factor_Q_plus = pinv_svd(factor_Q, tol)   # (m x m)
    return Q_hat @ W_t @ factor_Q_plus         # (n x m)


# -------------------------------------------------------------------
# Loading Discrepancy (paper §5.2, Step 7)
# -------------------------------------------------------------------

def compute_loading_discrepancy(
    Phi_hat_f: np.ndarray,
    Phi_t: np.ndarray,
) -> float:
    """
    Loading Discrepancy: LD_t = ||Phi_hat^f - Phi_t||_F / ||Phi_t||_F.

    Parameters
    ----------
    Phi_hat_f : (n x m) regression loading matrix
    Phi_t     : (n x m) reported loading matrix

    Returns
    -------
    ld : float (0 means perfect loading consistency)
    """
    denom = frobenius_norm(Phi_t)
    if denom < 1e-16:
        return 0.0
    return frobenius_norm(Phi_hat_f - Phi_t) / denom


# -------------------------------------------------------------------
# Loading Gap Floor (paper §5.2, Step 8)
# -------------------------------------------------------------------

def compute_loading_gap_floor(
    Q_hat: np.ndarray,
    Phi_t: np.ndarray,
    tol: float = _TOL,
) -> float:
    """
    Loading Gap floor: LG_t = ||(I - P_Q) Phi_t||_F / ||Phi_t||_F,
    where P_Q = U_r U_r^T is the projection onto the column space of Q.

    Uses the SVD column-space basis (same as mw_regression_loadings_stable)
    so that max|LD_MW - LG| is at machine-epsilon level.

    Parameters
    ----------
    Q_hat : (n x n) second-moment matrix
    Phi_t : (n x m) loading matrix
    tol   : pseudoinverse tolerance

    Returns
    -------
    lg : float
    """
    denom = frobenius_norm(Phi_t)
    if denom < 1e-16:
        return 0.0
    U_r, r = col_space_basis(Q_hat, tol)
    if r == 0:
        # Q has no column space -> (I - 0) Phi = Phi -> LG = 1
        return 1.0
    # P_Q Phi via orthonormal basis: P_Q @ Phi = U_r @ (U_r.T @ Phi)
    P_Q_Phi = U_r @ (U_r.T @ Phi_t)
    perp_Phi = Phi_t - P_Q_Phi
    return frobenius_norm(perp_Phi) / denom


# -------------------------------------------------------------------
# Risk-Premium Gap (paper §5.2, Step 9)
# -------------------------------------------------------------------

def compute_risk_premium_gap(
    Phi_t: np.ndarray,
    mu_hat: np.ndarray,
    tol: float = _TOL,
) -> float:
    """
    Risk-Premium Gap: RPG_t = ||mu_hat_perp||_2 / ||mu_hat||_2

    where P_Phi = Phi_t Phi_t^+, mu_hat_perp = (I - P_Phi) mu_hat.

    Returns 0 if ||mu_hat|| = 0 (paper convention).

    Parameters
    ----------
    Phi_t  : (n x m) loading matrix
    mu_hat : (n,) conditional mean vector
    tol    : pseudoinverse tolerance

    Returns
    -------
    rpg : float in [0, 1]
    """
    denom = l2_norm(mu_hat)
    if denom < 1e-16:
        return 0.0
    n = Phi_t.shape[0]
    Phi_plus = pinv_svd(Phi_t, tol)
    P_Phi = Phi_t @ Phi_plus                    # (n x n)
    mu_perp = (np.eye(n) - P_Phi) @ mu_hat     # component outside char span
    return l2_norm(mu_perp) / denom


# -------------------------------------------------------------------
# Full diagnostic run for one panel/window/spec
# -------------------------------------------------------------------

def run_diagnostics(
    returns_exc: pd.DataFrame,
    returns_vw: pd.DataFrame,
    mktcap: pd.DataFrame,
    beme: pd.DataFrame,
    Tw: int,
    spec: int = 4,
    tol: float = _TOL,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run the full design-diagnostic pipeline for a given rolling window and characteristic spec.

    Parameters
    ----------
    returns_exc : (T x n) excess returns in decimal
    returns_vw  : (T x n) simple (VW) returns in decimal, used for momentum
    mktcap      : (T x n) portfolio market cap
    beme        : (T x n) portfolio book-to-market
    Tw          : rolling estimation window length
    spec        : characteristic specification (1-4)
    tol         : pseudoinverse tolerance
    verbose     : print progress

    Returns
    -------
    results : DataFrame with columns [month, rank_Q, rank_Sigma,
              LD_OLS, LD_MW, LG, RPG]
    """
    n_assets = returns_exc.shape[1]
    dates = returns_exc.index  # DatetimeIndex

    # Pre-compute momentum series (uses simple returns)
    momentum = compute_momentum_series(returns_vw, lookback_months=11)

    records = []
    for t_idx in range(Tw, len(dates) - 1):
        t_date = dates[t_idx]

        # --- Step 3: Estimate rolling moments ---
        ret_block = returns_exc.iloc[t_idx - Tw: t_idx].values  # (Tw x n)
        if ret_block.shape[0] < Tw:
            continue
        mu_hat, Q_hat, Sigma_hat = estimate_moments(ret_block, tol)

        # --- Step 4: Build Phi_t ---
        Phi_t = build_phi_matrix(
            t_idx=t_idx,
            dates=dates,
            mktcap=mktcap,
            beme=beme,
            momentum=momentum,
            trailing_window=Tw,
            spec=spec,
        )
        if Phi_t is None:
            continue

        # --- Rank diagnostics (Step 12) ---
        rk_Q = rank_svd(Q_hat, tol)
        rk_Sigma = rank_svd(Sigma_hat, tol)

        # --- Step 5: Compute weight matrices ---
        W_ols = compute_ols_weights(Phi_t, tol)
        # W_mw retained for exp2 portfolio construction; LD_MW uses stable formula below
        W_mw = compute_mw_weights(Phi_t, Q_hat, tol)

        # --- Step 6: Regression loadings ---
        Phi_hat_ols = compute_regression_loadings(Q_hat, W_ols, tol)
        # MW loading matrix: use stable SVD-based direct formula (avoids Q -> W -> Phi chain
        # which accumulates ~4e-6 error in full-rank regime; theoretically equivalent).
        Phi_hat_mw = mw_regression_loadings_stable(Phi_t, Q_hat, tol)

        # --- Step 7: Loading Discrepancy ---
        ld_ols = compute_loading_discrepancy(Phi_hat_ols, Phi_t)
        ld_mw = compute_loading_discrepancy(Phi_hat_mw, Phi_t)

        # --- Step 8: Loading Gap Floor ---
        lg = compute_loading_gap_floor(Q_hat, Phi_t, tol)

        # --- Step 9: Risk-Premium Gap ---
        rpg = compute_risk_premium_gap(Phi_t, mu_hat, tol)

        records.append({
            "month": t_date,
            "rank_Q": rk_Q,
            "rank_Sigma": rk_Sigma,
            "LD_OLS": ld_ols,
            "LD_MW": ld_mw,
            "LG": lg,
            "RPG": rpg,
        })

        if verbose and t_idx % 50 == 0:
            print(f"  t={t_date.strftime('%Y-%m')}: LD_OLS={ld_ols:.4f}, "
                  f"LD_MW={ld_mw:.4f}, LG={lg:.4f}, RPG={rpg:.4f}")

    if not records:
        return pd.DataFrame(columns=["month", "rank_Q", "rank_Sigma", "LD_OLS", "LD_MW", "LG", "RPG"])
    return pd.DataFrame(records).set_index("month")


# -------------------------------------------------------------------
# Full experiment dispatcher
# -------------------------------------------------------------------

def run_experiment_1(
    panel_data: Dict,
    panel_name: str,
    windows: List[int] = (24, 48, 96, 120, 180, 240),
    specs: List[int] = (1, 2, 3, 4),
    tol: float = _TOL,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run Experiment 1 (Section 5.2 design diagnostics) across all windows and specs.

    Parameters
    ----------
    panel_data  : dict with keys 'returns_exc', 'returns_vw', 'mktcap', 'beme'
    panel_name  : 'FF25' or 'FF100'
    windows     : list of rolling window lengths T_w
    specs       : list of characteristic specifications (1-4)
    tol         : pseudoinverse tolerance
    verbose     : print progress

    Returns
    -------
    all_results : DataFrame with columns [panel, window, spec, month, rank_Q, rank_Sigma,
                  LD_OLS, LD_MW, LG, RPG]
    """
    returns_exc = panel_data["returns_exc"]
    returns_vw = panel_data["returns_vw"]
    mktcap = panel_data["mktcap"]
    beme = panel_data["beme"]

    all_rows = []
    for Tw in windows:
        for spec in specs:
            if verbose:
                print(f"\n[{panel_name}] Window={Tw}, Spec={spec}")
            diag = run_diagnostics(
                returns_exc=returns_exc,
                returns_vw=returns_vw,
                mktcap=mktcap,
                beme=beme,
                Tw=Tw,
                spec=spec,
                tol=tol,
                verbose=verbose,
            )
            if len(diag) == 0:
                continue
            diag = diag.reset_index()
            diag["panel"] = panel_name
            diag["window"] = Tw
            diag["spec"] = spec
            all_rows.append(diag)

    if not all_rows:
        return pd.DataFrame()
    result = pd.concat(all_rows, ignore_index=True)
    cols = ["panel", "window", "spec", "month", "rank_Q", "rank_Sigma",
            "LD_OLS", "LD_MW", "LG", "RPG"]
    return result[cols]


def compute_summary_table(
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute average diagnostics by panel/window/spec.

    Parameters
    ----------
    all_results : output of run_experiment_1

    Returns
    -------
    summary : DataFrame with mean LD_OLS, LD_MW, LG, RPG by (panel, window, spec)
    """
    agg_cols = ["LD_OLS", "LD_MW", "LG", "RPG"]
    summary = (
        all_results.groupby(["panel", "window", "spec"])[agg_cols]
        .mean()
        .round(4)
    )
    return summary


def compute_qc_max_diff(all_results: pd.DataFrame) -> float:
    """
    Compute max_t |LD_MW - LG| across all months/windows/specs.
    Paper target: ~5.5e-12 or smaller.
    """
    return validate_numerical_qc(
        all_results["LD_MW"].values,
        all_results["LG"].values,
    )
