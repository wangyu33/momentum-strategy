#!/usr/bin/env python3
"""Compare multi-horizon composite momentum signals against the current baseline."""

from __future__ import annotations

import argparse
import json

try:
    from ..runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from run_backtest import (
    CORE_OUTPUT_DIR,
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    DEFENSIVE_CODES,
    RESEARCH_OUTPUT_DIR,
    RISK_CODES,
    annualized_return,
    build_benchmark_nav,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    max_drawdown,
    run_threshold_dual_strategy,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_multihorizon_combo"


MULTIHORIZON_SPECS: list[dict[str, object]] = [
    {"name": "baseline_k25", "kind": "baseline", "lookback": 25},
    {"name": "mh_20_25_30", "kind": "combo", "windows": [20, 25, 30], "weights": [0.3, 0.4, 0.3]},
    {"name": "mh_15_25_60", "kind": "combo", "windows": [15, 25, 60], "weights": [0.25, 0.5, 0.25]},
    {"name": "mh_20_60_120", "kind": "combo", "windows": [20, 60, 120], "weights": [0.4, 0.35, 0.25]},
    {"name": "mh_10_20_25", "kind": "combo", "windows": [10, 20, 25], "weights": [0.2, 0.4, 0.4]},
    {"name": "mh_25_60", "kind": "combo", "windows": [25, 60], "weights": [0.6, 0.4]},
    {"name": "mh_30_60_120", "kind": "combo", "windows": [30, 60, 120], "weights": [0.4, 0.35, 0.25]},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multi-horizon momentum combinations.")
    parser.add_argument("--years", type=int, default=10, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--absolute-threshold", type=float, default=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD, help="Risk absolute momentum threshold.")
    parser.add_argument("--thresholds", default="", help="Optional comma-separated thresholds for multi-horizon combos.")
    parser.add_argument("--weak-trend-defensive-weight", type=float, default=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT, help="Defensive exposure when risk trend is weak but positive.")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh histories instead of using local cached output.")
    return parser.parse_args()


def summarize(result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    daily_ret = result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")

    latest_holding = None
    if not result["holding"].dropna().empty:
        latest_holding = str(result["holding"].dropna().iloc[-1])

    latest_theme = ""
    latest_name = ""
    if latest_holding:
        row = selected[selected["code"] == latest_holding]
        if not row.empty:
            latest_theme = str(row.iloc[0]["theme"])
            latest_name = str(row.iloc[0]["name"])

    return {
        "total_return": float(result["nav"].iloc[-1] - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown(result["nav"]),
        "trade_count": int(len(trades)),
        "avg_exposure": float(result["exposure"].mean()),
        "latest_holding_code": latest_holding or "",
        "latest_holding_theme": latest_theme,
        "latest_holding_name": latest_name,
        "latest_momentum": float(result["current_momentum"].dropna().iloc[-1]),
        "latest_exposure": float(result["exposure"].iloc[-1]),
    }


def run_signal_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    signal: pd.Series,
    target_exposure: pd.Series,
    signal_momentum: pd.Series,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change()
    holding = signal.shift(1)
    exposure = target_exposure.shift(1).fillna(0.0).rename("exposure")
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)
    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")
    per_side_cost = fee_rate + slippage_rate

    for dt_idx in prices.index:
        asset = holding.loc[dt_idx]
        prev_asset = prev_holding.loc[dt_idx]
        weight = float(exposure.loc[dt_idx]) if pd.notna(exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0

        gross_ret = 0.0
        if pd.notna(asset) and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = weight * float(returns.loc[dt_idx, asset])

        if pd.isna(asset) and pd.isna(prev_asset):
            day_turnover = abs(weight - prev_weight)
        elif pd.notna(asset) and pd.notna(prev_asset) and asset == prev_asset:
            day_turnover = abs(weight - prev_weight)
        else:
            day_turnover = prev_weight + weight

        turnover.loc[dt_idx] = day_turnover
        cost_rate = day_turnover * per_side_cost
        trade_cost_rate.loc[dt_idx] = cost_rate
        strategy_ret.loc[dt_idx] = (1 + gross_ret) * (1 - cost_rate) - 1

    nav = (1 + strategy_ret.fillna(0.0)).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1

    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": signal_momentum,
            "signal": signal,
            "holding": holding,
            "exposure": exposure,
            "target_exposure": target_exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )

    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    trades: list[dict[str, object]] = []
    for dt_idx in prices.index:
        asset = str(holding.loc[dt_idx]) if pd.notna(holding.loc[dt_idx]) else None
        prev_asset = str(prev_holding.loc[dt_idx]) if pd.notna(prev_holding.loc[dt_idx]) else None
        weight = float(exposure.loc[dt_idx]) if pd.notna(exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0

        if prev_asset == asset and abs(weight - prev_weight) < 1e-12:
            continue

        if prev_asset and (prev_asset != asset or prev_weight > weight):
            trades.append(
                {
                    "date": dt_idx,
                    "action": "SELL" if prev_asset != asset else "REDUCE",
                    "code": prev_asset,
                    "theme": code_to_theme.get(prev_asset, ""),
                    "name": code_to_name.get(prev_asset, ""),
                    "from_exposure": prev_weight,
                    "to_exposure": weight if prev_asset == asset else 0.0,
                    "nav": float(result.loc[dt_idx, "nav"]),
                }
            )
        if asset and (prev_asset != asset or weight > prev_weight):
            trades.append(
                {
                    "date": dt_idx,
                    "action": "BUY" if prev_asset != asset else "ADD",
                    "code": asset,
                    "theme": code_to_theme.get(asset, ""),
                    "name": code_to_name.get(asset, ""),
                    "from_exposure": prev_weight if prev_asset == asset else 0.0,
                    "to_exposure": weight,
                    "nav": float(result.loc[dt_idx, "nav"]),
                }
            )

    return result, pd.DataFrame(trades)


def build_composite_score(prices: pd.DataFrame, windows: list[int], weights: list[float]) -> pd.DataFrame:
    score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    valid = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for window, weight in zip(windows, weights):
        momentum = prices / prices.shift(window) - 1
        usable = momentum.notna().astype(float)
        score = score.add(momentum.fillna(0.0) * weight, fill_value=0.0)
        valid = valid.add(usable * weight, fill_value=0.0)
    score = score.divide(valid.where(valid > 0))
    return score


def run_multihorizon_threshold_dual_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    windows: list[int],
    weights: list[float],
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    composite = build_composite_score(prices, windows, weights)
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    defensive_codes = [code for code in DEFENSIVE_CODES if code in prices.columns]
    risk_score = composite[risk_codes]
    defensive_score = composite[defensive_codes]
    risk_winner = risk_score.idxmax(axis=1, skipna=True)
    risk_best = risk_score.max(axis=1, skipna=True)
    defensive_winner = defensive_score.idxmax(axis=1, skipna=True)
    defensive_best = defensive_score.max(axis=1, skipna=True)

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    signal_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        r_asset = risk_winner.loc[dt_idx]
        r_score = risk_best.loc[dt_idx]
        d_asset = defensive_winner.loc[dt_idx]
        d_score = defensive_best.loc[dt_idx]

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            signal_momentum.loc[dt_idx] = float("nan")
        elif pd.notna(r_score) and r_score > absolute_threshold:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            signal_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = weak_trend_defensive_weight
            signal_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            signal_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            signal_momentum.loc[dt_idx] = float("nan")

    return run_signal_strategy(
        prices,
        selected,
        signal,
        target_exposure,
        signal_momentum,
        fee_rate,
        slippage_rate,
    )


def main() -> int:
    args = parse_args()
    selected = load_fixed_etf_pool()
    if args.refresh:
        prices = fetch_histories(selected, years=args.years)
    else:
        prices = pd.read_csv(CORE_OUTPUT_DIR / "prices.csv", parse_dates=["date"]).set_index("date")

    thresholds = [args.absolute_threshold]
    if args.thresholds.strip():
        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    compare_df = pd.DataFrame(index=prices.index)
    compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")

    for spec in MULTIHORIZON_SPECS:
        active_thresholds = [args.absolute_threshold] if spec["kind"] == "baseline" else thresholds
        for threshold in active_thresholds:
            if spec["kind"] == "baseline":
                result, trades = run_threshold_dual_strategy(
                    prices,
                    selected,
                    lookback=int(spec["lookback"]),
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                    absolute_threshold=threshold,
                    weak_trend_defensive_weight=args.weak_trend_defensive_weight,
                )
                strategy_name = str(spec["name"])
            else:
                result, trades = run_multihorizon_threshold_dual_strategy(
                    prices,
                    selected,
                    windows=list(spec["windows"]),
                    weights=list(spec["weights"]),
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                    absolute_threshold=threshold,
                    weak_trend_defensive_weight=args.weak_trend_defensive_weight,
                )
                strategy_name = f"{spec['name']}_thr{int(round(threshold * 100)):02d}"

            row = {
                "strategy": strategy_name,
                "base_strategy": str(spec["name"]),
                "threshold": threshold,
                "windows": json.dumps(spec.get("windows", [spec.get("lookback")]), ensure_ascii=False),
                "weights": json.dumps(spec.get("weights", [1.0]), ensure_ascii=False),
                **summarize(result, trades, selected),
            }
            rows.append(row)
            compare_df[f"{strategy_name}_nav"] = result["nav"]

    summary = pd.DataFrame(rows).sort_values(["sharpe_rf0", "annualized_return"], ascending=[False, False]).reset_index(drop=True)
    baseline = summary[summary["strategy"] == "baseline_k25"].iloc[0]
    summary["return_diff"] = summary["total_return"] - float(baseline["total_return"])
    summary["annualized_diff"] = summary["annualized_return"] - float(baseline["annualized_return"])
    summary["sharpe_diff"] = summary["sharpe_rf0"] - float(baseline["sharpe_rf0"])
    summary["mdd_diff"] = summary["max_drawdown"] - float(baseline["max_drawdown"])
    summary["trade_diff"] = summary["trade_count"] - int(baseline["trade_count"])

    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(summary, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    top_rows = summary.head(4)
    for _, row in top_rows.iterrows():
        strategy = str(row["strategy"])
        ax.plot(compare_df.index, compare_df[f"{strategy}_nav"], linewidth=2.0, label=strategy)
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.6, linestyle="--", label="HS300")
    ax.set_title("Multi-Horizon Momentum Comparison", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
