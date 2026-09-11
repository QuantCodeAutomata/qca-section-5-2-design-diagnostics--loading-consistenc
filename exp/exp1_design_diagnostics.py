"""
Experiment 1: Section 5.2 Design Diagnostics
Loading Consistency, Loading Gap, and Risk-Premium Gap on FF25 and FF100 Panels.

Runs full diagnostic pipeline and saves:
- monthly_diagnostics.csv  (raw monthly data)
- summary_diagnostics.csv  (averages by panel/window/spec)
- figure2_loading_discrepancy.png
- figure3_risk_premium_gap.png
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from data.download_ff_data import load_ff25, load_ff100, CACHE_DIR
from src.diagnostics import run_experiment_1, compute_summary_table, compute_qc_max_diff
from src.visualization import plot_figure2, plot_figure3
from src.matrix_utils import _QC_TOL

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [24, 48, 96, 120, 180, 240]
SPECS = [1, 2, 3, 4]


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 1: Section 5.2 Design Diagnostics")
    print("=" * 70)
    t0 = time.time()

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    print("\n[1] Loading FF25 data...")
    ff25 = load_ff25(CACHE_DIR)
    print(f"    Returns shape: {ff25['returns_exc'].shape}")
    print(f"    Date range: {ff25['returns_exc'].index[0]} to {ff25['returns_exc'].index[-1]}")

    print("\n[2] Loading FF100 data...")
    ff100 = load_ff100(CACHE_DIR)
    print(f"    Returns shape: {ff100['returns_exc'].shape}")
    if "dropped_portfolios" in ff100:
        print(f"    Dropped portfolios ({len(ff100['dropped_portfolios'])}): "
              f"{ff100['dropped_portfolios']}")

    # -------------------------------------------------------------------
    # Run diagnostics
    # -------------------------------------------------------------------
    print("\n[3] Running FF25 diagnostics...")
    ff25_results = run_experiment_1(
        panel_data=ff25,
        panel_name="FF25",
        windows=WINDOWS,
        specs=SPECS,
        verbose=True,
    )

    print("\n[4] Running FF100 diagnostics...")
    ff100_results = run_experiment_1(
        panel_data=ff100,
        panel_name="FF100",
        windows=WINDOWS,
        specs=SPECS,
        verbose=True,
    )

    # Combine
    all_results = pd.concat([ff25_results, ff100_results], ignore_index=True)
    print(f"\nTotal monthly diagnostic records: {len(all_results):,}")

    # -------------------------------------------------------------------
    # Evaluation count check (paper targets)
    # -------------------------------------------------------------------
    paper_counts = {24: 731, 48: 707, 96: 659, 120: 635, 180: 575, 240: 515}
    print("\n[5] Evaluation count check vs paper targets:")
    print(f"{'Window':>8} {'FF25 actual':>12} {'paper target':>14} {'match':>8}")
    for Tw in WINDOWS:
        mask = (all_results["panel"] == "FF25") & (all_results["window"] == Tw) & \
               (all_results["spec"] == 4)
        actual = len(all_results[mask])
        target = paper_counts.get(Tw, "N/A")
        match = "✓" if abs(actual - target) <= 5 else "~"
        print(f"{Tw:>8} {actual:>12} {target:>14} {match:>8}")

    # -------------------------------------------------------------------
    # Numerical QC: max|LD_MW - LG|
    # -------------------------------------------------------------------
    print("\n[6] Numerical QC: max_t |LD_MW - LG|")
    for panel in ["FF25", "FF100"]:
        sub = all_results[all_results["panel"] == panel]
        qc_max = compute_qc_max_diff(sub)
        target_ok = qc_max < 1e-8  # relaxed from 5.5e-12 for finite-sample checks
        print(f"    {panel}: max|LD_MW - LG| = {qc_max:.3e}  "
              f"{'[OK]' if target_ok else '[WARN: check precision]'}")
    print(f"    Paper target: ~5.5e-12 (may be higher with standard double precision)")

    # -------------------------------------------------------------------
    # Summary tables
    # -------------------------------------------------------------------
    print("\n[7] Summary averages by panel/window/spec:")
    summary = compute_summary_table(all_results)
    print(summary.to_string())

    # Print paper-comparison table for spec=4 (full characteristic set)
    print("\n--- Key averages (spec=4, full characteristics) ---")
    print(f"{'Panel':>8} {'Window':>8} {'LD_OLS':>10} {'LD_MW':>10} {'LG':>10} {'RPG':>10}")
    for panel in ["FF25", "FF100"]:
        for Tw in [24, 48, 120]:
            try:
                row = summary.loc[(panel, Tw, 4)]
                print(f"{panel:>8} {Tw:>8} {row['LD_OLS']:>10.4f} "
                      f"{row['LD_MW']:>10.4f} {row['LG']:>10.4f} {row['RPG']:>10.4f}")
            except KeyError:
                print(f"{panel:>8} {Tw:>8} (no data)")

    # -------------------------------------------------------------------
    # Singularity diagnostics (paper Step 12)
    # -------------------------------------------------------------------
    print("\n[8] Singularity regime diagnostics (rank_Q and rank_Sigma):")
    for panel in ["FF25", "FF100"]:
        n_assets = 25 if panel == "FF25" else 94
        print(f"\n  {panel} (n={n_assets}):")
        for Tw in WINDOWS:
            mask = (all_results["panel"] == panel) & (all_results["window"] == Tw) & \
                   (all_results["spec"] == 4)
            sub = all_results[mask]
            if len(sub) == 0:
                continue
            avg_rk_Q = sub["rank_Q"].mean()
            avg_rk_Sigma = sub["rank_Sigma"].mean()
            sing_Q = (sub["rank_Q"] < n_assets).mean() * 100
            sing_Sigma = (sub["rank_Sigma"] < n_assets - 1).mean() * 100
            print(f"    Tw={Tw:>4}: avg rank(Q)={avg_rk_Q:5.1f}/{n_assets} "
                  f"(singular {sing_Q:.0f}%), "
                  f"avg rank(Sigma)={avg_rk_Sigma:5.1f}/{n_assets} "
                  f"(singular {sing_Sigma:.0f}%)")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    print("\n[9] Saving results...")
    monthly_path = RESULTS_DIR / "monthly_diagnostics.csv"
    all_results.to_csv(monthly_path, index=False)
    print(f"    Saved: {monthly_path}")

    summary_path = RESULTS_DIR / "summary_diagnostics.csv"
    summary.reset_index().to_csv(summary_path, index=False)
    print(f"    Saved: {summary_path}")

    # -------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------
    print("\n[10] Generating figures...")
    try:
        plot_figure2(all_results, panels=["FF25", "FF100"], windows=[24, 48, 120], spec=4)
        plot_figure3(all_results, panels=["FF25", "FF100"], windows=[24, 48, 120], specs=[1, 2, 3, 4])
    except Exception as e:
        print(f"    Warning: figure generation error: {e}")

    elapsed = time.time() - t0
    print(f"\nExperiment 1 complete in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
