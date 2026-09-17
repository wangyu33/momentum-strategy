#!/usr/bin/env python3
"""Explore price-based crowding proxies for the current ETF universe."""

from __future__ import annotations

import pandas as pd

from common import (
    FRONTIER_OUTPUT_DIR,
    build_asset_own_momentum_percentile,
    code_to_theme_map,
    ensure_frontier_output_dir,
    load_selected_prices,
    render_markdown_table,
    save_markdown,
    write_dataframe_csv_atomic,
)


LOOKBACK = 20
FORWARD_HORIZON = 10


def build_pairwise_average_corr(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    output = pd.DataFrame(index=prices.index, columns=prices.columns, dtype="float64")
    returns = prices.pct_change()
    for end_idx in range(window, len(returns)):
        end_date = returns.index[end_idx]
        sample = returns.iloc[end_idx - window + 1 : end_idx + 1]
        corr = sample.corr()
        for code in corr.columns:
            peers = corr.loc[code].drop(labels=[code], errors="ignore").dropna()
            output.loc[end_date, code] = float(peers.mean()) if not peers.empty else float("nan")
    return output


def build_leader_crowding_table(selected: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mom20 = prices / prices.shift(20) - 1
    leader = mom20.idxmax(axis=1, skipna=True)
    leader_gap = mom20.max(axis=1, skipna=True) - mom20.apply(lambda row: row.dropna().nlargest(2).iloc[-1] if row.dropna().size >= 2 else float("nan"), axis=1)
    avg_corr20 = build_pairwise_average_corr(prices, LOOKBACK)
    own_pct = build_asset_own_momentum_percentile(prices, lookback=20, state_lookback=756, min_periods=120)
    theme_map = code_to_theme_map(selected)

    rows = []
    for dt in prices.index:
        code = leader.loc[dt]
        if pd.isna(code):
            continue
        code = str(code)
        rows.append(
            {
                "date": dt,
                "leader_code": code,
                "leader_theme": theme_map.get(code, code),
                "leader_mom20": float(mom20.loc[dt, code]) if pd.notna(mom20.loc[dt, code]) else None,
                "leader_gap20": float(leader_gap.loc[dt]) if pd.notna(leader_gap.loc[dt]) else None,
                "leader_avg_corr20": float(avg_corr20.loc[dt, code]) if pd.notna(avg_corr20.loc[dt, code]) else None,
                "leader_mom20_pct": float(own_pct.loc[dt, code]) if code in own_pct.columns and pd.notna(own_pct.loc[dt, code]) else None,
            }
        )
    leader_df = pd.DataFrame(rows).set_index("date")
    leader_df["crowding_proxy"] = leader_df["leader_avg_corr20"] * leader_df["leader_mom20_pct"]

    detailed_rows = []
    for dt in leader_df.index:
        loc = prices.index.get_loc(dt)
        if loc + FORWARD_HORIZON >= len(prices.index):
            continue
        code = leader_df.loc[dt, "leader_code"]
        base = prices.loc[dt, code]
        future = prices.iloc[loc + FORWARD_HORIZON][code]
        if pd.isna(base) or pd.isna(future):
            continue
        detailed_rows.append(
            {
                "date": dt.date().isoformat(),
                "leader_code": code,
                "leader_theme": leader_df.loc[dt, "leader_theme"],
                "leader_mom20": leader_df.loc[dt, "leader_mom20"],
                "leader_gap20": leader_df.loc[dt, "leader_gap20"],
                "leader_avg_corr20": leader_df.loc[dt, "leader_avg_corr20"],
                "leader_mom20_pct": leader_df.loc[dt, "leader_mom20_pct"],
                "crowding_proxy": leader_df.loc[dt, "crowding_proxy"],
                "forward_10d_ret": float(future / base - 1),
            }
        )
    return leader_df.reset_index(), pd.DataFrame(detailed_rows)


def build_bucket_summary(detailed: pd.DataFrame) -> pd.DataFrame:
    usable = detailed.dropna(subset=["crowding_proxy", "forward_10d_ret"]).copy()
    if usable.empty:
        return pd.DataFrame()
    usable["crowding_bucket"] = pd.qcut(usable["crowding_proxy"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    summary = usable.groupby("crowding_bucket").agg(
        samples=("forward_10d_ret", "size"),
        avg_forward_10d_ret_pct=("forward_10d_ret", lambda s: s.mean() * 100),
        median_forward_10d_ret_pct=("forward_10d_ret", lambda s: s.median() * 100),
        win_rate_pct=("forward_10d_ret", lambda s: (s > 0).mean() * 100),
        avg_leader_gap20_pct=("leader_gap20", lambda s: s.mean() * 100),
        avg_leader_corr20=("leader_avg_corr20", "mean"),
    ).reset_index()
    return summary


def build_latest_snapshot(leader_snapshot: pd.DataFrame) -> pd.DataFrame:
    if leader_snapshot.empty:
        return leader_snapshot
    return leader_snapshot.tail(20).sort_values("date", ascending=False).reset_index(drop=True)


def build_markdown(bucket_summary: pd.DataFrame, latest_snapshot: pd.DataFrame) -> str:
    lines = [
        "# Crowding Proxy Exploration",
        "",
        "Proxy used here is intentionally simple and price-only:",
        "",
        "- leader average 20-day correlation to the rest of the universe",
        "- multiplied by the leader's own 20-day momentum percentile",
        "",
        "This is not true crowding data. It is only a local proxy that can be computed from the current repo cache.",
        "",
        "## Bucket Summary",
        "",
        render_markdown_table(bucket_summary),
        "",
        "## Latest Leader Snapshot",
        "",
        render_markdown_table(latest_snapshot),
    ]
    return "\n".join(lines)


def main() -> int:
    ensure_frontier_output_dir()
    selected, prices = load_selected_prices()
    prices = prices[selected["code"].astype(str).tolist()].apply(pd.to_numeric, errors="coerce")

    leader_snapshot, detailed = build_leader_crowding_table(selected, prices)
    bucket_summary = build_bucket_summary(detailed)
    latest_snapshot = build_latest_snapshot(leader_snapshot)

    write_dataframe_csv_atomic(leader_snapshot, FRONTIER_OUTPUT_DIR / "leader_crowding_snapshot.csv", index=False)
    write_dataframe_csv_atomic(detailed, FRONTIER_OUTPUT_DIR / "leader_crowding_forward_detail.csv", index=False)
    write_dataframe_csv_atomic(bucket_summary, FRONTIER_OUTPUT_DIR / "leader_crowding_bucket_summary.csv", index=False)
    save_markdown(
        FRONTIER_OUTPUT_DIR / "crowding_proxies.md",
        build_markdown(bucket_summary, latest_snapshot),
    )
    print(bucket_summary.to_string(index=False))
    print()
    print(latest_snapshot.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
