"""
Experiment 2: Section 5.3 Out-of-Sample Performance Comparison.

Compares OLS, MW (raw/LW), GLS (raw/LW), X* (raw/LW), CMVE, 1/N, PC strategies.
Produces Table 1-style Sharpe ratio summary and Figure 4-style plots.

Note: OLS/MW/GLS strategy results labeled as 'approximation' because the exact
factor-to-single-portfolio mapping from the paper is unspecified in the paper text.
Default mapping: Sigma-optimal combination in factor space (see performance.py).
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from data.download_ff_data import load_ff25, load_ff100, CACHE_DIR
from src.performance import run_experiment_2, run_performance_backtest, annualized_sharpe
from src.characteristics import compute_momentum_series
from src.diagnostics import estimate_moments
from src.visualization import plot_figure4, RESULTS_DIR

RESULTS_DIR_PATH = Path(__file__).parent.parent / "results"
RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)

WINDOWS = [24, 48, 96, 120, 180, 240]
METHODS = ("ols", "mw", "mw_lw", "gls", "gls_lw", "xstar", "xstar_lw", "cmve", "equal", "pc")
SPEC = 4  # Full characteristic set


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 2: Section 5.3 Out-of-Sample Performance")
    print("=" * 70)
    t0 = time.time()

    print("\n[1] Loading FF25 data...")
    ff25 = load_ff25(CACHE_DIR)

    print("\n[2] Loading FF100 data...")
    ff100 = load_ff100(CACHE_DIR)

    all_sharpe_tables = []
    all_returns_frames = []

    for panel_name, panel_data in [("FF25", ff25), ("FF100", ff100)]:
        print(f"\n[3] Running {panel_name} performance backtest...")
        sharpe_table, all_returns = run_experiment_2(
            panel_data=panel_data,
            panel_name=panel_name,
            windows=WINDOWS,
            spec=SPEC,
            methods=METHODS,
            verbose=True,
        )
        all_sharpe_tables.append(sharpe_table)
        all_returns_frames.append(all_returns)

    # Combine Sharpe tables
    combined_sharpe = pd.concat(all_sharpe_tables)
    combined_returns = pd.concat(all_returns_frames) if all_returns_frames else pd.DataFrame()

    # -------------------------------------------------------------------
    # Table 1-style output
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TABLE 1-STYLE: Annualized Out-of-Sample Sharpe Ratios (spec=4, full characteristics)")
    print("NOTE: OLS/MW/GLS values are APPROXIMATION (strategy mapping per paper approximation)")
    print("=" * 70)
    print(combined_sharpe.round(3).to_string())

    # -------------------------------------------------------------------
    # Paper comparison (qualitative targets from paper text)
    # -------------------------------------------------------------------
    print("\n--- Qualitative comparison with paper targets (FF25, full spec) ---")
    paper_targets_ff25 = {
        # (Tw, method): (expected_range_low, expected_range_high)
        (24, "ols"):  (0.40, 0.70),
        (48, "ols"):  (0.40, 0.70),
        (120, "ols"): (0.70, 1.00),
        (24, "mw"):   (0.00, 0.30),
        (24, "mw_lw"): (0.50, 0.90),
        (120, "xstar_lw"): (1.00, 1.40),
    }
    try:
        ff25_sharpe = combined_sharpe.loc["FF25"]
        for (Tw, m), (lo, hi) in paper_targets_ff25.items():
            if Tw in ff25_sharpe.index and m in ff25_sharpe.columns:
                val = ff25_sharpe.loc[Tw, m]
                flag = "✓" if lo <= val <= hi else "~"
                print(f"  FF25 Tw={Tw} {m}: {val:.3f} (target [{lo:.2f}, {hi:.2f}]) {flag}")
    except Exception as e:
        print(f"  Comparison error: {e}")

    # -------------------------------------------------------------------
    # Figure 4-style plots for Tw=120
    # -------------------------------------------------------------------
    print("\n[4] Generating Figure 4-style plots...")
    for panel_name, panel_data in [("FF25", ff25), ("FF100", ff100)]:
        Tw = 120
        try:
            realized, exante = run_performance_backtest(
                returns_exc=panel_data["returns_exc"],
                returns_vw=panel_data["returns_vw"],
                mktcap=panel_data["mktcap"],
                beme=panel_data["beme"],
                Tw=Tw,
                spec=SPEC,
                methods=METHODS,
                verbose=False,
            )
            # Prepare exante dict with annualized series
            exante_ann = {m: s * np.sqrt(12) for m, s in exante.items()}

            # Build returns DataFrame for plotting
            ret_rows = []
            for m, series in realized.items():
                df = series.to_frame(name="return")
                df["panel"] = panel_name
                df["window"] = Tw
                df["method"] = m
                ret_rows.append(df)
            ret_df = pd.concat(ret_rows) if ret_rows else pd.DataFrame()

            plot_figure4(
                all_returns=ret_df,
                exante_dict={m: s for m, s in exante_ann.items()},
                panel=panel_name,
                Tw=Tw,
                filename=f"figure4_performance_{panel_name.lower()}.png",
            )
        except Exception as e:
            print(f"  Warning: Figure 4 for {panel_name}: {e}")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    print("\n[5] Saving results...")
    sharpe_path = RESULTS_DIR_PATH / "performance_sharpe_table.csv"
    combined_sharpe.round(4).reset_index().to_csv(sharpe_path, index=False)
    print(f"    Saved: {sharpe_path}")

    if len(combined_returns) > 0:
        returns_path = RESULTS_DIR_PATH / "performance_monthly_returns.csv"
        combined_returns.reset_index().to_csv(returns_path, index=False)
        print(f"    Saved: {returns_path}")

    elapsed = time.time() - t0
    print(f"\nExperiment 2 complete in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
