"""
Visualization functions for QCA Section 5.2 and 5.3 experiments.

Figure 2: Loading discrepancy (OLS vs MW) as 24-month moving averages
Figure 3: Risk-Premium Gap across characteristic specs, 24-month moving averages
Figure 4: Realized and ex-ante Sharpe ratios as 24-month rolling windows

Uses matplotlib for all plots.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = Path(__file__).parent.parent / "results"
MA_WINDOW = 24  # 24-month moving average for paper-style plots


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _rolling_mean(series: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    """24-month trailing moving average."""
    return series.rolling(window=window, min_periods=window // 2).mean()


def _save_fig(fig: plt.Figure, filename: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# -------------------------------------------------------------------
# Figure 2: Loading Discrepancy (OLS vs MW)
# -------------------------------------------------------------------

def plot_figure2(
    all_results: pd.DataFrame,
    panels: List[str] = ("FF25", "FF100"),
    windows: List[int] = (24, 48, 120),
    spec: int = 4,
    filename: str = "figure2_loading_discrepancy.png",
) -> None:
    """
    Replicate Figure 2: OLS vs MW loading discrepancy as 24-month moving averages.

    Parameters
    ----------
    all_results : output of run_experiment_1 (columns include panel, window, spec, month, LD_OLS, LD_MW)
    panels      : list of panel names
    windows     : list of T_w values to plot
    spec        : characteristic specification to use (default 4, full set)
    filename    : output file name
    """
    n_rows = len(panels)
    n_cols = len(windows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle("Figure 2: Loading Discrepancy (OLS vs MW), 24-month Moving Average", fontsize=13)

    for r, panel in enumerate(panels):
        for c, Tw in enumerate(windows):
            ax = axes[r][c]
            mask = (
                (all_results["panel"] == panel) &
                (all_results["window"] == Tw) &
                (all_results["spec"] == spec)
            )
            sub = all_results[mask].sort_values("month").set_index("month")
            if len(sub) == 0:
                ax.set_title(f"{panel}, T_w={Tw}\n(no data)")
                continue

            ols_ma = _rolling_mean(sub["LD_OLS"])
            mw_ma = _rolling_mean(sub["LD_MW"])
            lg_ma = _rolling_mean(sub["LG"])

            ax.plot(ols_ma.index, ols_ma.values, label="OLS", color="tab:blue", lw=1.5)
            ax.plot(mw_ma.index, mw_ma.values, label="MW", color="tab:orange", lw=1.5)
            ax.plot(lg_ma.index, lg_ma.values, label="LG floor", color="tab:green",
                    lw=1.0, linestyle="--")
            ax.axhline(0, color="black", lw=0.5, linestyle=":")
            ax.set_title(f"{panel}, $T_w$={Tw}", fontsize=10)
            ax.set_xlabel("Month")
            ax.set_ylabel("Relative Discrepancy")
            ax.legend(fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, filename)


# -------------------------------------------------------------------
# Figure 3: Risk-Premium Gap
# -------------------------------------------------------------------

def plot_figure3(
    all_results: pd.DataFrame,
    panels: List[str] = ("FF25", "FF100"),
    windows: List[int] = (24, 48, 120),
    specs: List[int] = (1, 2, 3, 4),
    filename: str = "figure3_risk_premium_gap.png",
) -> None:
    """
    Replicate Figure 3: RPG across nested characteristic specs, 24-month MAs.
    """
    n_rows = len(panels)
    n_cols = len(windows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle("Figure 3: Risk-Premium Gap by Characteristic Set, 24-month Moving Average",
                 fontsize=13)

    spec_labels = {1: "Const", 2: "+Size", 3: "+B/M", 4: "+Mom"}
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]

    for r, panel in enumerate(panels):
        for c, Tw in enumerate(windows):
            ax = axes[r][c]
            for i, spec in enumerate(specs):
                mask = (
                    (all_results["panel"] == panel) &
                    (all_results["window"] == Tw) &
                    (all_results["spec"] == spec)
                )
                sub = all_results[mask].sort_values("month").set_index("month")
                if len(sub) == 0:
                    continue
                rpg_ma = _rolling_mean(sub["RPG"])
                ax.plot(rpg_ma.index, rpg_ma.values,
                        label=spec_labels[spec], color=colors[i], lw=1.5)

            ax.set_title(f"{panel}, $T_w$={Tw}", fontsize=10)
            ax.set_xlabel("Month")
            ax.set_ylabel("RPG")
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, filename)


# -------------------------------------------------------------------
# Figure 4: Realized and ex-ante Sharpe (performance experiment)
# -------------------------------------------------------------------

def plot_figure4(
    all_returns: pd.DataFrame,
    exante_dict: Dict,
    panel: str = "FF25",
    Tw: int = 120,
    filename: str = "figure4_performance.png",
) -> None:
    """
    Figure 4-style: realized 24-month rolling Sharpe and ex-ante Sharpe (24-month MA).
    """
    methods_plot = ["ols", "mw", "gls", "xstar", "cmve", "equal"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:gray"]
    labels = ["OLS", "MW", "GLS", "X*", "CMVE", "1/N"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Figure 4: Out-of-Sample Performance ({panel}, $T_w$={Tw})", fontsize=13)

    # Panel A: Realized 24-month rolling Sharpe
    ax = axes[0]
    sub = all_returns[(all_returns["panel"] == panel) & (all_returns["window"] == Tw)]
    for m, color, label in zip(methods_plot, colors, labels):
        msub = sub[sub["method"] == m].sort_values(level=0) if isinstance(sub.index, pd.MultiIndex) \
               else sub[sub["method"] == m]
        if len(msub) == 0:
            continue
        ret_series = msub["return"].dropna()
        roll_sr = ret_series.rolling(window=24, min_periods=12).apply(
            lambda x: float(np.sqrt(12) * x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0,
            raw=True,
        )
        ax.plot(roll_sr.index, roll_sr.values, label=label, color=color, lw=1.2)
    ax.axhline(0, color="black", lw=0.5, linestyle=":")
    ax.set_title("Realized 24-month Rolling Sharpe")
    ax.set_xlabel("Month")
    ax.set_ylabel("Ann. Sharpe Ratio")
    ax.legend(fontsize=8)

    # Panel B: Ex-ante Sharpe (24-month MA)
    ax = axes[1]
    if exante_dict:
        for m, color, label in zip(methods_plot, colors, labels):
            if m in exante_dict:
                ea = exante_dict[m]
                if hasattr(ea, "values"):
                    ea_ma = _rolling_mean(ea) * np.sqrt(12)  # annualize
                    ax.plot(ea_ma.index, ea_ma.values, label=label, color=color, lw=1.2)
    ax.axhline(0, color="black", lw=0.5, linestyle=":")
    ax.set_title("Ex-ante Sharpe (24-month MA, annualized)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Ann. Sharpe Ratio")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, filename)


# -------------------------------------------------------------------
# Summary table plots
# -------------------------------------------------------------------

def plot_summary_heatmap(
    summary: pd.DataFrame,
    title: str,
    filename: str,
    value_col: str = "LD_OLS",
) -> None:
    """Plot a heatmap of average diagnostic values by window and panel."""
    pivot_data = {}
    for panel in summary.index.get_level_values("panel").unique():
        for spec in summary.index.get_level_values("spec").unique():
            try:
                col_data = summary.loc[(panel, slice(None), spec), value_col]
                col_data.index = col_data.index.droplevel([0, 2])  # keep window
                pivot_data[f"{panel} spec={spec}"] = col_data
            except Exception:
                pass

    if not pivot_data:
        return
    df_plot = pd.DataFrame(pivot_data)

    fig, ax = plt.subplots(figsize=(max(6, len(df_plot.columns) * 1.5), 4))
    im = ax.imshow(df_plot.values.T, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(df_plot.index)))
    ax.set_xticklabels(df_plot.index, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df_plot.columns)))
    ax.set_yticklabels(df_plot.columns, fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    _save_fig(fig, filename)
