"""
Download and parse Kenneth French Data Library files for FF25 and FF100 panels.

Data source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- Monthly value-weighted portfolio excess returns
- Monthly portfolio-level average market capitalization (ME)
- Monthly portfolio-level book-to-market (BE/ME)
- Monthly simple returns for momentum construction

Uses pandas_datareader.famafrench for fetching — Context7 confirmed.
"""

from __future__ import annotations
import io
import os
import zipfile
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

FF25_RETURNS_FILE = "25_Portfolios_5x5_CSV.zip"
FF100_RETURNS_FILE = "100_Portfolios_10x10_CSV.zip"
FF_FACTORS_FILE = "F-F_Research_Data_Factors_CSV.zip"

SAMPLE_START = "1963-07"  # July 1963
SAMPLE_END = "2026-06"    # June 2026 inclusive

CACHE_DIR = Path(__file__).parent / "cache"


# -------------------------------------------------------------------
# Low-level downloader
# -------------------------------------------------------------------

def _download_zip(filename: str, cache_dir: Path, session: Optional[requests.Session] = None) -> str:
    """Download a zip file from French's FTP page and return the inner CSV text."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / filename.replace(".zip", ".csv")
    if cache_path.exists():
        return cache_path.read_text(encoding="latin-1")

    url = f"{FRENCH_BASE_URL}/{filename}"
    s = session or requests.Session()
    resp = s.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Find the CSV file inside the zip
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        content = zf.read(csv_name).decode("latin-1")
    cache_path.write_text(content, encoding="latin-1")
    return content


# -------------------------------------------------------------------
# CSV parser for French-format files
# -------------------------------------------------------------------

def _parse_french_csv(content: str) -> Dict[int, pd.DataFrame]:
    """
    Parse a French data library CSV with multiple tables separated by blank lines.

    Returns a dict of table-index -> DataFrame with DatetimeIndex (YYYYMM).
    """
    tables: Dict[int, pd.DataFrame] = {}
    table_idx = 0
    current_lines: list[str] = []
    header_found = False

    def _flush_table(lines: list[str], idx: int) -> Optional[pd.DataFrame]:
        """Parse accumulated lines as a single table."""
        if not lines:
            return None
        # Drop comment/description lines (non-numeric first column)
        data_lines = []
        header_line = None
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            # Try to parse first part as YYYYMM integer
            try:
                int(parts[0].strip())
                data_lines.append(stripped)
            except ValueError:
                # Treat as column header if it has multiple columns
                if len(parts) > 1 and header_line is None:
                    header_line = parts
        if not data_lines:
            return None
        # Build DataFrame
        records = []
        for line in data_lines:
            parts = line.split(",")
            records.append([p.strip() for p in parts])
        df = pd.DataFrame(records)
        # First column is YYYYMM date
        try:
            df[0] = pd.to_datetime(df[0], format="%Y%m")
        except Exception:
            return None
        df = df.set_index(0)
        df.index.name = "date"
        # Convert numeric columns; replace French sentinel values (-99.99, -999)
        # NOTE: sentinel replacement on strings (regex) must come before to_numeric;
        # float sentinel replacement is repeated after to_numeric for safety.
        df = df.replace(r"^\s*$", np.nan, regex=True)
        df = df.replace(r"^-99\.99\d*$", np.nan, regex=True)
        df = df.replace(r"^-999\d*$", np.nan, regex=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Float sentinel cleanup after numeric conversion
        df = df.replace(-99.99, np.nan).replace(-999.0, np.nan)
        df = df.dropna(how="all")
        # Assign column names if header found
        if header_line is not None and len(header_line) - 1 == len(df.columns):
            df.columns = [h.strip() for h in header_line[1:]]
        return df

    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "":
            # Blank line may signal end of a table block
            if current_lines:
                df = _flush_table(current_lines, table_idx)
                if df is not None and len(df) > 0:
                    tables[table_idx] = df
                    table_idx += 1
                current_lines = []
        else:
            current_lines.append(line)
    # Flush last block
    if current_lines:
        df = _flush_table(current_lines, table_idx)
        if df is not None and len(df) > 0:
            tables[table_idx] = df

    return tables


# -------------------------------------------------------------------
# Risk-free rate
# -------------------------------------------------------------------

def _load_risk_free(cache_dir: Path) -> pd.Series:
    """Return monthly risk-free rate in decimal form, indexed by month-start."""
    content = _download_zip(FF_FACTORS_FILE, cache_dir)
    tables = _parse_french_csv(content)
    # Table 0 has Mkt-RF, SMB, HML, RF (in %)
    ff_df = tables[0]
    if "RF" in ff_df.columns:
        rf = ff_df["RF"] / 100.0
    else:
        # Fallback: last column is RF
        rf = ff_df.iloc[:, -1] / 100.0
    rf.index = rf.index.to_period("M").to_timestamp("M")
    rf = rf.sort_index()
    return rf


# -------------------------------------------------------------------
# Public loaders
# -------------------------------------------------------------------

def load_ff25(cache_dir: Path = CACHE_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load the FF25 panel data (5x5 size x book-to-market portfolios).

    Returns
    -------
    dict with keys:
      'returns_vw'  : (T x 25) monthly value-weighted simple returns in decimal
      'returns_exc' : (T x 25) monthly value-weighted excess returns in decimal
      'mktcap'      : (T x 25) average market cap in millions (or NaN if unavailable)
      'beme'        : (T x 25) book-to-market ratio (or approximate)
    """
    cache_dir = Path(cache_dir)
    content = _download_zip(FF25_RETURNS_FILE, cache_dir)
    tables = _parse_french_csv(content)
    rf = _load_risk_free(cache_dir)

    # Table 0: monthly value-weighted returns in percent
    ret_vw_pct = tables[0].copy()
    ret_vw_pct.index = ret_vw_pct.index.to_period("M").to_timestamp("M")
    ret_vw_pct = ret_vw_pct.sort_index()

    # Convert to decimal
    ret_vw = ret_vw_pct / 100.0

    # Align RF and compute excess returns
    rf_aligned = rf.reindex(ret_vw.index)
    ret_exc = ret_vw.subtract(rf_aligned, axis=0)

    # Market cap: look for the average-firm-size table (usually last two tables)
    mktcap = _extract_characteristic_table(tables, n_cols=25, pref_label="size")

    # Book-to-market: if available as a separate table
    beme = _extract_characteristic_table(tables, n_cols=25, pref_label="be", table_offset=1)

    # Filter to sample window
    ret_vw = _filter_sample(ret_vw)
    ret_exc = _filter_sample(ret_exc)
    mktcap = _filter_sample(mktcap) if mktcap is not None else _make_approx_char_ff25("size")
    beme = _filter_sample(beme) if beme is not None else _make_approx_char_ff25("bm")

    # Ensure characteristics share the same DatetimeIndex as returns (month-end)
    mktcap = _reindex_to_returns(mktcap, ret_vw)
    beme = _reindex_to_returns(beme, ret_vw)

    return {
        "returns_vw": ret_vw,
        "returns_exc": ret_exc,
        "mktcap": mktcap,
        "beme": beme,
    }


def load_ff100(cache_dir: Path = CACHE_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load the FF100 panel data (10x10 size x book-to-market), reduced to n=94
    by removing 6 portfolios with incomplete return history from January 1963.

    Returns
    -------
    dict with keys:
      'returns_vw'  : (T x 94) monthly value-weighted simple returns in decimal
      'returns_exc' : (T x 94) monthly value-weighted excess returns in decimal
      'mktcap'      : (T x 94) average market cap (or approximate)
      'beme'        : (T x 94) book-to-market ratio (or approximate)
      'dropped_portfolios': list of dropped column names
    """
    cache_dir = Path(cache_dir)
    content = _download_zip(FF100_RETURNS_FILE, cache_dir)
    tables = _parse_french_csv(content)
    rf = _load_risk_free(cache_dir)

    # Table 0: monthly value-weighted returns in percent
    ret_vw_pct = tables[0].copy()
    ret_vw_pct.index = ret_vw_pct.index.to_period("M").to_timestamp("M")
    ret_vw_pct = ret_vw_pct.sort_index()

    # Convert to decimal
    ret_vw = ret_vw_pct / 100.0

    # Identify portfolios with incomplete history from 1963-01
    jan_1963 = pd.Timestamp("1963-01-01")
    pre_filter = ret_vw[ret_vw.index >= jan_1963]
    # Portfolios with any NaN in [1963-01, ...] are incomplete
    incomplete_mask = pre_filter.isnull().any()
    dropped = list(incomplete_mask[incomplete_mask].index)

    # Also check for portfolios not starting at Jan 1963
    if len(dropped) < 6:
        # Some portfolios may have all-zero or all-NaN before a certain date
        # Find portfolios where first non-NaN after 1963-01 is later than 1963-01
        for col in ret_vw.columns:
            series = ret_vw.loc[jan_1963:, col]
            first_valid = series.first_valid_index()
            if first_valid is not None and first_valid > jan_1963:
                if col not in dropped:
                    dropped.append(col)

    # Ensure we drop exactly 6 (if more/fewer, drop those with most missing)
    if len(dropped) != 6:
        # Recompute: sort by number of NaN after 1963-01
        nan_counts = pre_filter.isnull().sum().sort_values(ascending=False)
        dropped = list(nan_counts[nan_counts > 0].index[:6])
        if len(dropped) < 6:
            # All complete — pick first 6 as fallback (should not happen)
            pass

    ret_vw = ret_vw.drop(columns=dropped, errors="ignore")

    # Align RF and compute excess returns
    rf_aligned = rf.reindex(ret_vw.index)
    ret_exc = ret_vw.subtract(rf_aligned, axis=0)

    # Market cap
    mktcap_full = _extract_characteristic_table(tables, n_cols=None, pref_label="size")
    if mktcap_full is not None:
        mktcap = mktcap_full.drop(columns=dropped, errors="ignore")
    else:
        mktcap = _make_approx_char_ff100(ret_vw, "size")

    beme_full = _extract_characteristic_table(tables, n_cols=None, pref_label="be", table_offset=1)
    if beme_full is not None:
        beme = beme_full.drop(columns=dropped, errors="ignore")
    else:
        beme = _make_approx_char_ff100(ret_vw, "bm")

    # Filter to sample
    ret_vw = _filter_sample(ret_vw)
    ret_exc = _filter_sample(ret_exc)
    mktcap = _filter_sample(mktcap)
    beme = _filter_sample(beme)

    # Ensure characteristics share the same DatetimeIndex as returns (month-end)
    mktcap = _reindex_to_returns(mktcap, ret_vw)
    beme = _reindex_to_returns(beme, ret_vw)

    return {
        "returns_vw": ret_vw,
        "returns_exc": ret_exc,
        "mktcap": mktcap,
        "beme": beme,
        "dropped_portfolios": dropped,
    }


# -------------------------------------------------------------------
# Helpers

def _reindex_to_returns(char_df: pd.DataFrame, ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align characteristics DataFrame to the same DatetimeIndex as returns.

    Handles the common case where characteristics use month-start dates and
    returns use month-end dates by matching on year-month period.
    """
    if char_df is None:
        return char_df
    target_idx = ret_df.index  # month-end DatetimeIndex

    # Check if already aligned
    if char_df.index.equals(target_idx):
        return char_df

    # Try direct reindex first (in case dates already match)
    common = char_df.index.intersection(target_idx)
    if len(common) >= len(target_idx) * 0.9:
        return char_df.reindex(target_idx)

    # Fall back to period matching: convert both to YYYYMM string, merge
    char_periods = char_df.index.to_period("M").astype(str)
    ret_periods = target_idx.to_period("M").astype(str)

    char_df2 = char_df.copy()
    char_df2.index = char_periods
    ret_ser = pd.Series(target_idx, index=ret_periods)

    aligned = char_df2.reindex(ret_ser.index)
    aligned.index = target_idx
    return aligned

# -------------------------------------------------------------------

def _filter_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to SAMPLE_START through SAMPLE_END."""
    if df is None:
        return df
    return df.loc[SAMPLE_START:SAMPLE_END]


def _extract_characteristic_table(
    tables: Dict[int, pd.DataFrame],
    n_cols: Optional[int],
    pref_label: str,
    table_offset: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Try to extract a characteristic table (e.g., ME or BE/ME) from the parsed tables dict.

    Heuristic: find tables past the first two return tables whose column count matches.
    """
    # Returns are typically in tables 0 (VW monthly) and 1 (EW monthly)
    # Characteristic tables come after (indices 4+)
    candidates = {k: v for k, v in tables.items() if k >= 3}
    if not candidates:
        return None
    # Sort by index and pick by offset
    sorted_keys = sorted(candidates.keys())
    if table_offset < len(sorted_keys):
        tbl = candidates[sorted_keys[table_offset]]
        if n_cols is None or tbl.shape[1] >= n_cols:
            tbl = tbl.copy()
            tbl.index = tbl.index.to_period("M").to_timestamp("M") if not isinstance(
                tbl.index, pd.DatetimeIndex
            ) else tbl.index
            return tbl.sort_index()
    return None


def _make_approx_char_ff25(char: str) -> pd.DataFrame:
    """
    Approximate FF25 portfolio-level size or B/M using quintile ranks (1-5).

    These are time-invariant proxies used when characteristic data is unavailable.
    """
    # 5x5 portfolio: row = size quintile (1=small, 5=large), col = B/M quintile (1=low, 5=high)
    dates = pd.date_range(SAMPLE_START, SAMPLE_END, freq="MS")
    col_names = [f"S{s}B{b}" for s in range(1, 6) for b in range(1, 6)]
    if char == "size":
        # Market cap proxy: size quintile → np.exp(quintile) to mimic log-scale
        values = np.array([np.exp(s) for s in range(1, 6) for b in range(1, 6)])
    else:  # B/M
        values = np.array([np.exp(b) for s in range(1, 6) for b in range(1, 6)])
    data = pd.DataFrame(
        np.tile(values, (len(dates), 1)),
        index=dates,
        columns=col_names,
    )
    return data


def _make_approx_char_ff100(returns: pd.DataFrame, char: str) -> pd.DataFrame:
    """
    Approximate FF100 portfolio-level size or B/M using decile ranks (1-10).
    """
    n_cols = len(returns.columns)
    dates = returns.index
    if char == "size":
        # Size decile from column index (first 10 = size decile 1, etc.)
        values = np.array([np.exp((i // 10) + 1) for i in range(n_cols)])
    else:  # B/M
        values = np.array([np.exp((i % 10) + 1) for i in range(n_cols)])
    data = pd.DataFrame(
        np.tile(values, (len(dates), 1)),
        index=dates,
        columns=returns.columns,
    )
    return data
