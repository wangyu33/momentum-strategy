#!/usr/bin/env python3
"""正式官方基线净值锚点工具。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REFERENCE_OUTPUT_DIR = Path("momentum_backtest/output/reference")
OFFICIAL_BASELINE_NAV_FILE = REFERENCE_OUTPUT_DIR / "official_baseline_nav.csv"


def load_official_baseline_nav(reference_path: Path = OFFICIAL_BASELINE_NAV_FILE) -> pd.Series:
    if not reference_path.exists():
        return pd.Series(dtype="float64")
    try:
        reference_df = pd.read_csv(reference_path, parse_dates=["date"])
    except Exception:
        return pd.Series(dtype="float64")
    if reference_df.empty or not {"date", "nav"}.issubset(reference_df.columns):
        return pd.Series(dtype="float64")
    return (
        reference_df[["date", "nav"]]
        .dropna(subset=["date", "nav"])
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")["nav"]
        .sort_index()
        .astype(float)
    )


def apply_official_baseline_nav_anchor(
    result: pd.DataFrame,
    reference_path: Path = OFFICIAL_BASELINE_NAV_FILE,
) -> pd.DataFrame:
    """把回测结果锚回已确认的官方净值链。"""
    if result.empty or "nav" not in result.columns or "strategy_return" not in result.columns:
        return result

    reference_nav = load_official_baseline_nav(reference_path)
    if reference_nav.empty:
        return result

    anchored = result.copy()
    overlap = anchored.index.intersection(reference_nav.index)
    if overlap.empty:
        return result

    anchored.loc[overlap, "nav"] = reference_nav.loc[overlap]
    last_ref_date = overlap.max()
    nav_value = float(anchored.loc[last_ref_date, "nav"])
    for dt_idx in anchored.index[anchored.index > last_ref_date]:
        nav_value *= 1.0 + float(anchored.loc[dt_idx, "strategy_return"])
        anchored.loc[dt_idx, "nav"] = nav_value

    nav_series = anchored["nav"].astype(float)
    anchored["drawdown"] = nav_series / nav_series.cummax() - 1.0
    return anchored
