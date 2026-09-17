#!/usr/bin/env python3
"""Backtest frontier-factor-inspired strategy variants against the current baseline."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    FRONTIER_OUTPUT_DIR,
    ensure_frontier_output_dir,
    load_selected_prices,
    render_markdown_table,
    save_markdown,
    write_dataframe_csv_atomic,
)

try:
    from compare_strategy_refinements import run_signal_strategy
    from run_backtest import (
        CORE_OUTPUT_DIR,
        DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
        DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        DEFENSIVE_CODES,
        RISK_CODES,
        build_asset_own_momentum_percentile,
        build_strategy_summary,
        normalize_code,
    )
except ImportError:  # pragma: no cover - package fallback
    from momentum_backtest.compare_strategy_refinements import run_signal_strategy
    from momentum_backtest.run_backtest import (
        CORE_OUTPUT_DIR,
        DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
        DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        DEFENSIVE_CODES,
        RISK_CODES,
        build_asset_own_momentum_percentile,
        build_strategy_summary,
        normalize_code,
    )


WINDOWS = [10, 20, 40, 60, 120]
CORR_WINDOW = 20
FORWARD_CAP_SPECS = [
    ("crowding_cap70", 0.70, 0.80, 0.90),
    ("crowding_cap50", 0.50, 0.80, 0.95),
]


def load_baseline_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pd.read_csv(CORE_OUTPUT_DIR / "backtest_nav.csv", parse_dates=["date"]).set_index("date")
    trades = pd.read_csv(CORE_OUTPUT_DIR / "trades.csv", parse_dates=["date"])
    return result, trades


def build_multihorizon_score(prices: pd.DataFrame) -> pd.DataFrame:
    score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype="float64")
    valid = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype="float64")
    for window in WINDOWS:
        momentum = prices / prices.shift(window) - 1
        rank_pct = momentum.rank(axis=1, pct=True, method="average")
        positive = (momentum > 0).astype(float)
        score = score.add(rank_pct.fillna(0.0) * positive.fillna(0.0), fill_value=0.0)
        valid = valid.add(momentum.notna().astype(float), fill_value=0.0)
    return score.divide(valid.where(valid > 0))


def build_multihorizon_variant(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score = build_multihorizon_score(prices)
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    defensive_codes = [code for code in DEFENSIVE_CODES if code in prices.columns]
    risk_score = score[risk_codes]
    defensive_score = score[defensive_codes]
    raw_mom20 = prices / prices.shift(20) - 1

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    threshold = DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD

    for dt in prices.index:
        risk_row = risk_score.loc[dt].dropna().sort_values(ascending=False)
        def_row = defensive_score.loc[dt].dropna().sort_values(ascending=False)
        r_asset = str(risk_row.index[0]) if not risk_row.empty else None
        d_asset = str(def_row.index[0]) if not def_row.empty else None
        r_mom = float(raw_mom20.loc[dt, r_asset]) if r_asset and pd.notna(raw_mom20.loc[dt, r_asset]) else float("nan")
        d_mom = float(raw_mom20.loc[dt, d_asset]) if d_asset and pd.notna(raw_mom20.loc[dt, d_asset]) else float("nan")

        if pd.notna(r_mom) and r_mom > threshold and r_asset is not None:
            signal.loc[dt] = r_asset
            target_exposure.loc[dt] = 1.0
            current_momentum.loc[dt] = r_mom
        elif pd.notna(r_mom) and r_mom > 0 and d_asset is not None:
            signal.loc[dt] = d_asset
            target_exposure.loc[dt] = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT
            current_momentum.loc[dt] = r_mom
        elif pd.notna(d_mom) and d_asset is not None:
            signal.loc[dt] = d_asset
            target_exposure.loc[dt] = 1.0 if d_mom > 0 else 0.0
            current_momentum.loc[dt] = d_mom
        else:
            signal.loc[dt] = pd.NA
            target_exposure.loc[dt] = 0.0
            current_momentum.loc[dt] = float("nan")

    return run_signal_strategy(
        prices,
        selected,
        signal.map(normalize_code),
        target_exposure,
        current_momentum,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
    )


def build_pairwise_average_corr(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = prices.pct_change()
    out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype="float64")
    for end_idx in range(window, len(prices.index)):
        sample = returns.iloc[end_idx - window + 1 : end_idx + 1]
        corr = sample.corr()
        dt = prices.index[end_idx]
        for code in corr.columns:
            peers = corr.loc[code].drop(labels=[code], errors="ignore").dropna()
            out.loc[dt, code] = float(peers.mean()) if not peers.empty else float("nan")
    return out


def build_crowding_overlay_variant(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    baseline_result: pd.DataFrame,
    *,
    cap_value: float,
    corr_quantile: float,
    momentum_pct_cut: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    avg_corr20 = build_pairwise_average_corr(prices, CORR_WINDOW)
    own_pct = build_asset_own_momentum_percentile(prices, lookback=20, state_lookback=756, min_periods=120)
    baseline_signal = baseline_result["signal"].map(normalize_code)
    baseline_target_exposure = pd.to_numeric(baseline_result["target_exposure"], errors="coerce").fillna(0.0)
    baseline_momentum = pd.to_numeric(baseline_result["current_momentum"], errors="coerce")

    corr_values = []
    for dt in prices.index:
        code = baseline_signal.loc[dt]
        if code is None or code not in avg_corr20.columns:
            continue
        value = avg_corr20.loc[dt, code]
        if pd.notna(value):
            corr_values.append(float(value))
    corr_cut = float(pd.Series(corr_values).quantile(corr_quantile)) if corr_values else math.inf

    adjusted_target = baseline_target_exposure.copy()
    event_rows = []
    risk_codes = {code for code in RISK_CODES if code in prices.columns}
    for dt in prices.index:
        code = baseline_signal.loc[dt]
        if code is None or code not in prices.columns:
            continue
        corr_value = avg_corr20.loc[dt, code] if code in avg_corr20.columns else float("nan")
        pct_value = own_pct.loc[dt, code] if code in own_pct.columns else float("nan")
        triggered = (
            code in risk_codes
            and pd.notna(corr_value)
            and pd.notna(pct_value)
            and float(corr_value) >= corr_cut
            and float(pct_value) >= momentum_pct_cut
            and float(adjusted_target.loc[dt]) > cap_value
        )
        if triggered:
            adjusted_target.loc[dt] = cap_value
        event_rows.append(
            {
                "date": dt.date().isoformat(),
                "signal_code": code,
                "signal_theme": str(selected.loc[selected["code"] == code, "theme"].iloc[0]) if (selected["code"] == code).any() else code,
                "avg_corr20": float(corr_value) if pd.notna(corr_value) else None,
                "momentum_pct20": float(pct_value) if pd.notna(pct_value) else None,
                "baseline_target_exposure": float(baseline_target_exposure.loc[dt]),
                "adjusted_target_exposure": float(adjusted_target.loc[dt]),
                "triggered": bool(triggered),
            }
        )

    result, trades = run_signal_strategy(
        prices,
        selected,
        baseline_signal,
        adjusted_target,
        baseline_momentum,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
    )
    return result, trades, pd.DataFrame(event_rows)


def summarize_variant(name: str, result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, object]:
    summary = build_strategy_summary(
        result,
        trades,
        selected=selected,
        exposure_series=result["exposure"],
        include_max_drawdown_integral=True,
    )
    summary["strategy"] = name
    return summary


def build_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Frontier Factor Backtests",
        "",
        "Variants included:",
        "",
        "- `baseline_official`: current cached official baseline from `output/core`",
        "- `multihorizon_consistency`: price-only multi-window trend consistency signal",
        "- `crowding_cap70`: baseline signal plus price-based crowding exposure cap to 70%",
        "- `crowding_cap50`: baseline signal plus price-based crowding exposure cap to 50%",
        "- `carry_term_structure`: not backtested yet because the repo still lacks carry data inputs",
        "",
        "## Summary",
        "",
        render_markdown_table(summary),
    ]
    return "\n".join(lines)


def main() -> int:
    ensure_frontier_output_dir()
    selected, prices = load_selected_prices()
    prices = prices[selected["code"].astype(str).tolist()].apply(pd.to_numeric, errors="coerce")
    baseline_result, baseline_trades = load_baseline_outputs()

    rows = [
        summarize_variant("baseline_official", baseline_result, baseline_trades, selected),
    ]

    mh_result, mh_trades = build_multihorizon_variant(prices, selected)
    rows.append(summarize_variant("multihorizon_consistency", mh_result, mh_trades, selected))

    event_frames = []
    for name, cap_value, corr_quantile, momentum_pct_cut in FORWARD_CAP_SPECS:
        result, trades, events = build_crowding_overlay_variant(
            prices,
            selected,
            baseline_result,
            cap_value=cap_value,
            corr_quantile=corr_quantile,
            momentum_pct_cut=momentum_pct_cut,
        )
        event_frames.append(events.assign(strategy=name))
        rows.append(summarize_variant(name, result, trades, selected))

    summary = pd.DataFrame(rows)
    baseline_row = summary.loc[summary["strategy"] == "baseline_official"].iloc[0]
    for col in [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_rf0",
        "max_drawdown",
        "max_drawdown_integral",
        "trade_count",
    ]:
        summary[f"{col}_diff_vs_base"] = summary[col] - float(baseline_row[col])
    summary = summary[
        [
            "strategy",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_rf0",
            "max_drawdown",
            "max_drawdown_integral",
            "trade_count",
            "avg_exposure",
            "total_return_diff_vs_base",
            "annualized_return_diff_vs_base",
            "annualized_volatility_diff_vs_base",
            "sharpe_rf0_diff_vs_base",
            "max_drawdown_diff_vs_base",
            "max_drawdown_integral_diff_vs_base",
            "trade_count_diff_vs_base",
        ]
    ].sort_values("strategy")

    event_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    write_dataframe_csv_atomic(summary, FRONTIER_OUTPUT_DIR / "frontier_factor_backtest_summary.csv", index=False)
    if not event_df.empty:
        write_dataframe_csv_atomic(event_df, FRONTIER_OUTPUT_DIR / "frontier_crowding_overlay_events.csv", index=False)
    save_markdown(FRONTIER_OUTPUT_DIR / "frontier_factor_backtests.md", build_markdown(summary))

    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
