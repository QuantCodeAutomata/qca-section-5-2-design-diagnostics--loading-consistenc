# QCA Section 5.2 Design Diagnostics & Section 5.3 Performance

Replication of the design-diagnostic module and out-of-sample performance
comparison from a quantitative factor-model paper, using the Fama-French 25
and 100 portfolio panels.

## Overview

### Experiment 1 — Section 5.2: Design Diagnostics
Tests whether characteristic-based OLS factor portfolios satisfy **loading
consistency**, measures the **loading gap floor** (minimum achievable discrepancy,
attained by the moment-weighted construction), and quantifies the **risk-premium
gap** (fraction of conditional mean outside the characteristic span).

Key claims verified:
- **Claim A**: OLS loading discrepancy is materially positive; MW attains the theoretical minimum floor to machine precision.
- **Claim B**: Adding characteristics monotonically reduces the risk-premium gap.

### Experiment 2 — Section 5.3: Out-of-Sample Performance
Compares OLS, MW, GLS, CMVE, 1/N, X*, and principal-components strategies on
monthly-rebalanced out-of-sample portfolios, with and without Ledoit-Wolf
covariance shrinkage, across rolling windows Tₓ ∈ {24, 48, 96, 120, 180, 240}.

## Repository Structure

```
├── data/                   # Data download and caching
│   ├── download_ff_data.py # Kenneth French Data Library loaders
│   └── cache/              # Cached CSVs (auto-generated, gitignored)
├── src/                    # Core implementation
│   ├── matrix_utils.py     # SVD pseudoinverse, projection, norms
│   ├── characteristics.py  # Momentum, standardization, Phi_t builder
│   ├── diagnostics.py      # LD, LG floor, RPG computations (Exp 1)
│   ├── performance.py      # Strategy weights, Sharpe ratios (Exp 2)
│   └── visualization.py    # Figures 2, 3, 4
├── exp/                    # Experiment runner scripts
│   ├── exp1_design_diagnostics.py
│   └── exp2_performance.py
├── tests/                  # pytest unit tests
│   ├── test_matrix_utils.py
│   ├── test_characteristics.py
│   ├── test_diagnostics.py
│   └── test_performance.py
├── results/                # Generated outputs
│   └── RESULTS.md          # Summary of all findings
└── README.md
```

## Setup

```bash
pip install -r requirements.txt   # if present
# or install directly:
pip install numpy pandas scipy scikit-learn matplotlib requests pandas-datareader
```

No API keys required — data is downloaded from Kenneth French's public library.

## Running Experiments

```bash
# Experiment 1: Design Diagnostics
python exp/exp1_design_diagnostics.py

# Experiment 2: Out-of-Sample Performance
python exp/exp2_performance.py

# Run all tests
pytest tests/ -v
```

## Key Methodology Notes

- **Pseudoinverse**: All matrix inversions use SVD with relative tolerance `tol=1e-10`.
- **Characteristic standardization**: Pooled mean/std from all n × Tₓ asset-month observations in the trailing window ending at t-1 (not same-month cross-sectional).
- **Momentum**: 11-month compounded simple return (months t-11 to t-1), no excess return.
- **Moment-weighted construction**: Uses effective-loading form `Phi_eff = Q Q⁺ Phi` for finite-sample stability.
- **Ledoit-Wolf shrinkage**: Applied to Σ̂ using scikit-learn's `LedoitWolf` with `assume_centered=True`.
- **Strategy mapping (Exp 2)**: Multi-factor → single portfolio via Sigma-optimal combination in factor space (approximation; labeled accordingly).

## Numerical QC

The key validation target from the paper:

```
max_t |LD_MW(t) - LG(t)| ≈ 5.5e-12
```

This is verified automatically at the end of Experiment 1.

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for a summary of all findings,
diagnostic averages, and Sharpe ratio tables.
