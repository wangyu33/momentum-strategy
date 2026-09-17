#!/usr/bin/env python3
"""Explore multi-horizon trend consistency against the current single-window leader."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    FRONTIER_OUTPUT_DIR,
    code_to_theme_map,
    ensure_frontier_output_dir,
    load_selected_prices,
    render_markdown_table,
    save_markdown,
    write_dataframe_csv_atomic,
)


WINDOWS = [10, 20, 40, 60, 120]
FORWARD_HORIZONS = [10, 20]


def cross_sectional_rank_pct(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, method="average")


def build_multihorizon_features(prices: pd.DataFrame) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    momentums = {window: prices / prices.shift(window) - 1 for window in WINDOWS}
    rank_stack = [cross_sectional_rank_pct(momentums[window]) for window in WINDOWS]
    rank_mean = sum(rank_stack) / len(rank_stack)
    positive_agreement = sum((momentums[window] > 0).astype(float) for window in WINDOWS) / len(WINDOWS)
    score = rank_mean * positive_agreement
    return momentums, positive_agreement, score


def build_latest_snapshot(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    momentums: dict[int, pd.DataFrame],
    positive_agreement: pd.DataFrame,
    score: pd.DataFrame,
) -> pd.DataFrame:
    latest = prices.index[-1]
    snapshot = selected[["code", "theme", "name"]].copy()
    for window, frame in momentums.items():
        snapshot[f"mom_{window}"] = snapshot["code"].map(frame.loc[latest].to_dict())
    snapshot["positive_agreement"] = snapshot["code"].map(positive_agreement.loc[latest].to_dict())
    snapshot["mh_score"] = snapshot["code"].map(score.loc[latest].to_dict())
    return snapshot.sort_values(["mh_score", "positive_agreement"], ascending=False).reset_index(drop=True)


def build_forward_leader_comparison(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    momentums: dict[int, pd.DataFrame],
    score: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mom20 = momentums[20]
    raw20_leader = mom20.idxmax(axis=1, skipna=True)
    mh_leader = score.idxmax(axis=1, skipna=True)
    same_leader = raw20_leader == mh_leader

    rows = []
    detailed = []
    theme_map = code_to_theme_map(selected)
    for horizon in FORWARD_HORIZONS:
        raw_returns = []
        mh_returns = []
        diffs = []
        dates = []
        for dt in prices.index:
            loc = prices.index.get_loc(dt)
            if loc + horizon >= len(prices.index):
                continue
            raw_code = raw20_leader.loc[dt]
            mh_code = mh_leader.loc[dt]
            if pd.isna(raw_code) or pd.isna(mh_code):
                continue
            raw_base = prices.loc[dt, raw_code]
            mh_base = prices.loc[dt, mh_code]
            raw_future = prices.iloc[loc + horizon][raw_code]
            mh_future = prices.iloc[loc + horizon][mh_code]
            if pd.isna(raw_base) or pd.isna(mh_base) or pd.isna(raw_future) or pd.isna(mh_future):
                continue
            raw_ret = float(raw_future / raw_base - 1)
            mh_ret = float(mh_future / mh_base - 1)
            raw_returns.append(raw_ret)
            mh_returns.append(mh_ret)
            diffs.append(mh_ret - raw_ret)
            dates.append(dt)
            detailed.append(
                {
                    "date": dt.date().isoformat(),
                    "horizon": horizon,
                    "raw20_leader": raw_code,
                    "raw20_theme": theme_map.get(str(raw_code), str(raw_code)),
                    "mh_leader": mh_code,
                    "mh_theme": theme_map.get(str(mh_code), str(mh_code)),
                    "same_leader": bool(raw_code == mh_code),
                    "raw20_forward_ret": raw_ret,
                    "mh_forward_ret": mh_ret,
                    "mh_minus_raw20": mh_ret - raw_ret,
                }
            )
        series_raw = pd.Series(raw_returns, dtype="float64")
        series_mh = pd.Series(mh_returns, dtype="float64")
        diff_series = pd.Series(diffs, dtype="float64")
        rows.append(
            {
                "horizon": horizon,
                "samples": int(len(series_raw)),
                "same_leader_ratio_pct": float(same_leader.reindex(dates).mean() * 100) if dates else None,
                "raw20_avg_ret_pct": float(series_raw.mean() * 100) if len(series_raw) else None,
                "mh_avg_ret_pct": float(series_mh.mean() * 100) if len(series_mh) else None,
                "mh_minus_raw20_avg_pct": float(diff_series.mean() * 100) if len(diff_series) else None,
                "raw20_win_rate_pct": float((series_raw > 0).mean() * 100) if len(series_raw) else None,
                "mh_win_rate_pct": float((series_mh > 0).mean() * 100) if len(series_mh) else None,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(detailed)


def build_markdown(latest_snapshot: pd.DataFrame, summary: pd.DataFrame) -> str:
    preview_cols = ["code", "theme", "mom_10", "mom_20", "mom_40", "mom_60", "mom_120", "positive_agreement", "mh_score"]
    lines = [
        "# Multi-Horizon Trend Consistency",
        "",
        "Windows used: `10 / 20 / 40 / 60 / 120`.",
        "",
        "Scoring logic:",
        "",
        "- cross-sectional rank percentile is computed for each window",
        "- the percentiles are averaged across windows",
        "- the average is multiplied by positive-sign agreement across windows",
        "",
        "## Forward Comparison Against Current Single-Window Leader",
        "",
        render_markdown_table(summary),
        "",
        "## Latest Snapshot",
        "",
        render_markdown_table(latest_snapshot[preview_cols]),
    ]
    return "\n".join(lines)


def main() -> int:
    ensure_frontier_output_dir()
    selected, prices = load_selected_prices()
    prices = prices[selected["code"].astype(str).tolist()].apply(pd.to_numeric, errors="coerce")
    momentums, positive_agreement, score = build_multihorizon_features(prices)
    latest_snapshot = build_latest_snapshot(selected, prices, momentums, positive_agreement, score)
    summary, detailed = build_forward_leader_comparison(prices, selected, momentums, score)

    for window, frame in momentums.items():
        write_dataframe_csv_atomic(frame, FRONTIER_OUTPUT_DIR / f"multihorizon_momentum_{window}.csv")
    write_dataframe_csv_atomic(latest_snapshot, FRONTIER_OUTPUT_DIR / "multihorizon_latest_snapshot.csv", index=False)
    write_dataframe_csv_atomic(summary, FRONTIER_OUTPUT_DIR / "multihorizon_forward_summary.csv", index=False)
    write_dataframe_csv_atomic(detailed, FRONTIER_OUTPUT_DIR / "multihorizon_forward_detail.csv", index=False)
    save_markdown(
        FRONTIER_OUTPUT_DIR / "multihorizon_consistency.md",
        build_markdown(latest_snapshot, summary),
    )
    print(summary.to_string(index=False))
    print()
    print(latest_snapshot.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
