"""
Out-of-Sample Performance module for QCA Section 5.3 experiment.

Implements characteristic-based strategies (OLS, MW, GLS), benchmark strategies
(CMVE, 1/N, X*, principal-components), and Ledoit-Wolf shrinkage variants.

Strategy mapping from multi-factor to single portfolio (paper ambiguity resolution):
  - Form factor returns f_{t+1} = W_t' x_{t+1}
  - Estimate mu_f = W_t' mu_hat, Sigma_f = W_t' Sigma_hat W_t
  - Combination weights a_t = Sigma_f^+ mu_f  (Sharpe-optimal in factor space)
  - Asset weights w_t = W_t a_t

Results labeled as 'approximation' unless verified against authors' code.

Ledoit-Wolf: pure-NumPy implementation matching sklearn's analytical formula
(avoids sklearn validation overhead in the inner monthly loop).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.matrix_utils import pinv_svd, rank_svd, _TOL
from src.characteristics import build_phi_matrix, compute_momentum_series
from src.diagnostics import (
    estimate_moments,
    compute_ols_weights,
    compute_mw_weights,
    compute_gls_weights,
)


# -------------------------------------------------------------------
# Ledoit-Wolf shrinkage — pure-NumPy (paper §5.3, Step 3)
# -------------------------------------------------------------------

def _lw_shrinkage_numpy(X_centered: np.ndarray) -> float:
    """
    Compute Ledoit-Wolf (2004) analytic shrinkage intensity.

    Implements the same formula as sklearn.covariance.ledoit_wolf_shrinkage
    but avoids the check_array / validate_data overhead by operating directly
    on a pre-centred float64 array.

    Parameters
    ----------
    X_centered : (T x n) centred returns (mean already removed)

    Returns
    -------
    shrinkage : float in [0, 1]
    """
    T, n = X_centered.shape
    X2 = X_centered ** 2
    # Diagonal of sample covariance S = X'X / T
    emp_cov_trace = X2.sum(axis=0) / T          # (n,)  trace elements
    mu = emp_cov_trace.sum() / n                 # trace(S) / n

    # beta_: sum_{i,j} (sum_t x_it^2 * x_jt^2)  — no block needed (n <= 1000)
    beta_ = float((X2.T @ X2).sum())            # scalar
    # delta_: sum_{i,j} (sum_t x_it * x_jt)^2 / T^2 = ||S||_F^2
    delta_ = float(((X_centered.T @ X_centered) ** 2).sum()) / (T ** 2)

    # delta = ||S - mu*I||_F^2 / n
    delta = (delta_ - 2.0 * mu * emp_cov_trace.sum() + n * mu ** 2) / n
    # beta = (1 / (n*T)) * (E_cross/T - ||S||_F^2)
    beta = float(1.0 / (n * T) * (beta_ / T - delta_))
    beta = min(beta, delta)                      # prevent over-shrinkage
    shrinkage = 0.0 if (delta == 0.0) else (beta / delta)
    return float(np.clip(shrinkage, 0.0, 1.0))


def apply_ledoit_wolf(
    returns_block: np.ndarray,
    mu_hat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Ledoit-Wolf (2004) linear shrinkage to Sigma_hat.

    Shrinkage target is scaled identity: F = trace(S)/n * I.
    Sigma_lw = (1 - rho)*S + rho * trace(S)/n * I
    Q_lw     = Sigma_lw + mu_hat mu_hat'

    Parameters
    ----------
    returns_block : (T_w x n) excess returns array
    mu_hat        : (n,) sample mean vector

    Returns
    -------
    Sigma_lw : (n x n) LW-shrunk covariance matrix
    Q_lw     : (n x n) reconstructed second-moment matrix
    """
    T, n = returns_block.shape
    demeaned = returns_block - returns_block.mean(axis=0)
    S = (demeaned.T @ demeaned) / T              # biased sample covariance
    rho = _lw_shrinkage_numpy(demeaned)
    mu_S = np.trace(S) / n
    Sigma_lw = (1.0 - rho) * S + rho * mu_S * np.eye(n)
    Q_lw = Sigma_lw + np.outer(mu_hat, mu_hat)
    return Sigma_lw, Q_lw


# -------------------------------------------------------------------
# Single-portfolio mapping from multi-factor representation
# -------------------------------------------------------------------

def factor_to_single_portfolio(
    W_t: np.ndarray,
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    tol: float = _TOL,
    map_method: str = "sigma_optimal",
) -> np.ndarray:
    """
    Map a multi-factor weight matrix W_t to a single portfolio weight vector w_t.

    Default map_method='sigma_optimal' (Sharpe-optimal combination in factor space):
      mu_f    = W_t' mu_hat          (m,)
      Sigma_f = W_t' Sigma_hat W_t   (m x m)
      a_t     = Sigma_f^+ mu_f       (m,)
      w_t     = W_t a_t              (n,)

    Parameters
    ----------
    W_t        : (n x m) factor weight matrix
    mu_hat     : (n,) conditional mean
    Sigma_hat  : (n x n) conditional covariance
    tol        : pseudoinverse tolerance
    map_method : 'sigma_optimal' (only supported option currently)

    Returns
    -------
    w_t : (n,) single portfolio weights
    """
    mu_f = W_t.T @ mu_hat                       # (m,)
    Sigma_f = W_t.T @ Sigma_hat @ W_t            # (m x m)
    Sigma_f_plus = pinv_svd(Sigma_f, tol)        # (m x m)
    a_t = Sigma_f_plus @ mu_f                    # (m,)
    w_t = W_t @ a_t                              # (n,)
    return w_t


# -------------------------------------------------------------------
# Benchmark strategies
# -------------------------------------------------------------------

def compute_xstar_weights(
    Q_hat: np.ndarray,
    mu_hat: np.ndarray,
    tol: float = _TOL,
) -> np.ndarray:
    """X* strategy: w_t = Q_hat^+ mu_hat."""
    return pinv_svd(Q_hat, tol) @ mu_hat


def compute_cmve_weights(
    Sigma_hat: np.ndarray,
    mu_hat: np.ndarray,
    tol: float = _TOL,
) -> np.ndarray:
    """CMVE strategy: w_t proportional to Sigma_hat^+ mu_hat."""
    return pinv_svd(Sigma_hat, tol) @ mu_hat


def compute_equal_weights(n: int) -> np.ndarray:
    """1/N equal-weight strategy."""
    return np.ones(n) / n


def compute_pc_weights(
    Q_hat: np.ndarray,
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    m: int = 4,
    tol: float = _TOL,
    map_method: str = "sigma_optimal",
) -> np.ndarray:
    """
    Principal-components benchmark: leading m eigenvectors of Q_hat form the
    loading matrix V_t; then apply factor-to-single mapping.

    Parameters
    ----------
    Q_hat     : (n x n) second-moment matrix
    mu_hat    : (n,) mean vector
    Sigma_hat : (n x n) covariance matrix
    m         : number of principal components (default 4)
    tol       : pseudoinverse tolerance
    map_method: factor-to-portfolio mapping

    Returns
    -------
    w_t : (n,) single portfolio weights
    """
    n = Q_hat.shape[0]
    # Leading m eigenvectors by descending eigenvalue
    eigvals, eigvecs = np.linalg.eigh(Q_hat)
    idx = np.argsort(eigvals)[::-1]
    V_t = eigvecs[:, idx[:m]]  # (n x m)
    # Use MW weights on principal-component loading matrix
    W_pc = compute_mw_weights(V_t, Q_hat, tol)
    w_t = factor_to_single_portfolio(W_pc, mu_hat, Sigma_hat, tol, map_method)
    return w_t


# -------------------------------------------------------------------
# Strategy weight builder
# -------------------------------------------------------------------

def build_strategy_weights(
    method: str,
    Phi_t: Optional[np.ndarray],
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    Q_hat: np.ndarray,
    shrink: bool = False,
    returns_block: Optional[np.ndarray] = None,
    tol: float = _TOL,
    map_method: str = "sigma_optimal",
    n_pc: int = 4,
    Sigma_lw: Optional[np.ndarray] = None,
    Q_lw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build single portfolio asset weights for a given strategy method.

    Parameters
    ----------
    method        : one of 'ols', 'mw', 'gls', 'xstar', 'cmve', 'equal', 'pc'
    Phi_t         : (n x m) loading matrix (required for 'ols', 'mw', 'gls')
    mu_hat        : (n,) conditional mean
    Sigma_hat     : (n x n) raw conditional covariance
    Q_hat         : (n x n) raw second-moment matrix
    shrink        : use LW-shrunk moments if True
    returns_block : (T_w x n) returns (used only when Sigma_lw/Q_lw absent)
    tol           : pseudoinverse tolerance
    map_method    : factor combination method
    n_pc          : number of principal components for 'pc' strategy
    Sigma_lw      : pre-computed LW covariance (avoids recomputing per method)
    Q_lw          : pre-computed LW second-moment

    Returns
    -------
    w_t : (n,) portfolio weights
    """
    n = len(mu_hat)

    # Resolve moments: use LW if shrink requested
    if shrink:
        if Sigma_lw is not None and Q_lw is not None:
            Sigma_use, Q_use = Sigma_lw, Q_lw
        elif returns_block is not None:
            Sigma_use, Q_use = apply_ledoit_wolf(returns_block, mu_hat)
        else:
            Sigma_use, Q_use = Sigma_hat, Q_hat
    else:
        Sigma_use, Q_use = Sigma_hat, Q_hat

    if method == "ols":
        assert Phi_t is not None, "Phi_t required for OLS"
        W_t = compute_ols_weights(Phi_t, tol)
        return factor_to_single_portfolio(W_t, mu_hat, Sigma_use, tol, map_method)

    elif method == "mw":
        assert Phi_t is not None, "Phi_t required for MW"
        W_t = compute_mw_weights(Phi_t, Q_use, tol)
        return factor_to_single_portfolio(W_t, mu_hat, Sigma_use, tol, map_method)

    elif method == "gls":
        assert Phi_t is not None, "Phi_t required for GLS"
        W_t = compute_gls_weights(Phi_t, Sigma_use, tol)
        return factor_to_single_portfolio(W_t, mu_hat, Sigma_use, tol, map_method)

    elif method == "xstar":
        return compute_xstar_weights(Q_use, mu_hat, tol)

    elif method == "cmve":
        return compute_cmve_weights(Sigma_use, mu_hat, tol)

    elif method == "equal":
        return compute_equal_weights(n)

    elif method == "pc":
        return compute_pc_weights(Q_use, mu_hat, Sigma_use, n_pc, tol, map_method)

    else:
        raise ValueError(f"Unknown strategy method: {method}")


# -------------------------------------------------------------------
# Performance metrics
# -------------------------------------------------------------------

def annualized_sharpe(returns: np.ndarray) -> float:
    """
    Annualized Sharpe ratio: sqrt(12) * mean(r) / std(r).

    Returns 0 if std is 0 or NaN (e.g., single observation with ddof=1).
    """
    if len(returns) == 0:
        return 0.0
    mean_r = np.nanmean(returns)
    std_r = np.nanstd(returns, ddof=1)
    if not np.isfinite(std_r) or std_r < 1e-16:
        return 0.0
    return float(np.sqrt(12) * mean_r / std_r)


def ex_ante_sharpe(
    w_t: np.ndarray,
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
) -> float:
    """
    Ex-ante Sharpe ratio: (w_t' mu_hat) / sqrt(w_t' Sigma_hat w_t).

    Returns 0 if ex-ante variance is non-positive.
    """
    num = float(w_t @ mu_hat)
    var = float(w_t @ Sigma_hat @ w_t)
    if var <= 0:
        return 0.0
    return num / np.sqrt(var)


# -------------------------------------------------------------------
# Full out-of-sample backtest
# -------------------------------------------------------------------

def run_performance_backtest(
    returns_exc: pd.DataFrame,
    returns_vw: pd.DataFrame,
    mktcap: pd.DataFrame,
    beme: pd.DataFrame,
    Tw: int,
    spec: int = 4,
    methods: Tuple[str, ...] = ("ols", "mw", "mw_lw", "gls", "gls_lw",
                                 "xstar", "xstar_lw", "cmve", "cmve_lw",
                                 "equal", "pc"),
    tol: float = _TOL,
    n_pc: int = 4,
    verbose: bool = False,
) -> Dict[str, pd.Series]:
    """
    Run out-of-sample monthly-rebalanced backtest for all strategies.

    Parameters
    ----------
    returns_exc : (T x n) excess returns in decimal
    returns_vw  : (T x n) simple returns for momentum
    mktcap      : (T x n) market caps
    beme        : (T x n) book-to-market ratios
    Tw          : rolling window length
    spec        : characteristic specification (default 4)
    methods     : list of strategy identifiers
    tol         : pseudoinverse tolerance
    n_pc        : number of PCs for pc strategy
    verbose     : print progress

    Returns
    -------
    realized_returns : dict {method_name: pd.Series of monthly returns}
    """
    dates = returns_exc.index
    momentum = compute_momentum_series(returns_vw, lookback_months=11)

    realized_rets: Dict[str, list] = {m: [] for m in methods}
    ex_ante_sharpes: Dict[str, list] = {m: [] for m in methods}
    months_out: list = []

    # Determine which methods need LW moments
    lw_methods_needed = any(m.endswith("_lw") for m in methods)

    for t_idx in range(Tw, len(dates) - 1):
        t_date = dates[t_idx]

        # Trailing returns block
        ret_block = returns_exc.iloc[t_idx - Tw: t_idx].values  # (Tw x n)
        if ret_block.shape[0] < Tw:
            continue

        # Raw moments
        mu_hat, Q_hat, Sigma_hat = estimate_moments(ret_block, tol)

        # LW moments — computed ONCE per month (cached for all LW strategies)
        if lw_methods_needed:
            Sigma_lw, Q_lw = apply_ledoit_wolf(ret_block, mu_hat)
        else:
            Sigma_lw, Q_lw = None, None

        # Next-period realized returns
        x_next = returns_exc.iloc[t_idx + 1].values  # (n,)

        # Build Phi_t once per month
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

        months_out.append(t_date)

        # Loop over strategies using cached moments
        for method in methods:
            use_lw = method.endswith("_lw")
            base_method = method.replace("_lw", "")
            try:
                w_t = build_strategy_weights(
                    method=base_method,
                    Phi_t=Phi_t,
                    mu_hat=mu_hat,
                    Sigma_hat=Sigma_hat,
                    Q_hat=Q_hat,
                    shrink=use_lw,
                    tol=tol,
                    n_pc=n_pc,
                    Sigma_lw=Sigma_lw,   # pass cached LW — no recomputation
                    Q_lw=Q_lw,
                )
                r_next = float(w_t @ x_next)
                # Ex-ante Sharpe — reuse cached Sigma
                Sigma_ea = Sigma_lw if use_lw and Sigma_lw is not None else Sigma_hat
                ea_sr = ex_ante_sharpe(w_t, mu_hat, Sigma_ea)
            except Exception as e:
                r_next = np.nan
                ea_sr = np.nan
                if verbose:
                    print(f"  [{method}] t={t_date}: error: {e}")

            realized_rets[method].append(r_next)
            ex_ante_sharpes[method].append(ea_sr)

        if verbose and t_idx % 100 == 0:
            print(f"  t={t_date.strftime('%Y-%m')}: processed")

    idx = pd.DatetimeIndex(months_out)
    realized = {m: pd.Series(vals, index=idx, name=m) for m, vals in realized_rets.items()}
    exante = {m: pd.Series(vals, index=idx, name=m) for m, vals in ex_ante_sharpes.items()}
    return realized, exante


def run_experiment_2(
    panel_data: Dict,
    panel_name: str,
    windows: List[int] = (24, 48, 96, 120, 180, 240),
    spec: int = 4,
    methods: Tuple[str, ...] = ("ols", "mw", "mw_lw", "gls", "gls_lw",
                                 "xstar", "xstar_lw", "cmve", "equal", "pc"),
    tol: float = _TOL,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Experiment 2 (Section 5.3 out-of-sample performance) across all windows.

    Returns
    -------
    sharpe_table : DataFrame (method x window) of annualized Sharpe ratios
    all_returns  : long-format DataFrame with monthly returns per method/window
    """
    returns_exc = panel_data["returns_exc"]
    returns_vw = panel_data["returns_vw"]
    mktcap = panel_data["mktcap"]
    beme = panel_data["beme"]

    sharpe_rows = []
    all_ret_frames = []

    for Tw in windows:
        if verbose:
            print(f"\n[{panel_name}] Window={Tw}")
        realized, exante = run_performance_backtest(
            returns_exc=returns_exc,
            returns_vw=returns_vw,
            mktcap=mktcap,
            beme=beme,
            Tw=Tw,
            spec=spec,
            methods=methods,
            tol=tol,
            verbose=verbose,
        )
        row = {"panel": panel_name, "window": Tw}
        for m, series in realized.items():
            row[m] = annualized_sharpe(series.dropna().values)
        sharpe_rows.append(row)

        # Store long-format returns
        for m, series in realized.items():
            df = series.to_frame(name="return")
            df["panel"] = panel_name
            df["window"] = Tw
            df["method"] = m
            all_ret_frames.append(df)

    sharpe_table = pd.DataFrame(sharpe_rows).set_index(["panel", "window"])
    all_returns = pd.concat(all_ret_frames) if all_ret_frames else pd.DataFrame()
    return sharpe_table, all_returns
