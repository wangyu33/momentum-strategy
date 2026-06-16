#!/usr/bin/env python3
"""Compare threshold/breadth dual momentum variants against single momentum."""

from __future__ import annotations

import argparse

try:
    from ..runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    CORE_OUTPUT_DIR,
    RESEARCH_OUTPUT_DIR,
    annualized_return,
    build_benchmark_nav,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    max_drawdown,
    run_strategy,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


RISK_CODES = ["510300", "159949", "159941", "513650", "513880"]
DEFENSIVE_CODES = ["511580", "518880", "512890"]
OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_dual_momentum_variants"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dual momentum variants.")
    parser.add_argument("--years", type=int, default=10, help="Backtest years.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh histories instead of using local cached output.")
    return parser.parse_args()


def normalize_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(CORE_OUTPUT_DIR / "selected_etfs.csv", dtype={"code": str})
    prices = pd.read_csv(CORE_OUTPUT_DIR / "prices.csv", parse_dates=["date"]).set_index("date")
    return selected, prices


def build_trades(result: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    holding = result["holding"]
    exposure = result["exposure"].fillna(0.0)
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)
    trades = []
    for dt_idx in result.index:
        asset = normalize_code(holding.loc[dt_idx])
        prev_asset = normalize_code(prev_holding.loc[dt_idx])
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
    return pd.DataFrame(trades)


def summarize(result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    daily_ret = result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    holding_series = result["holding"].dropna()
    latest_holding = normalize_code(holding_series.iloc[-1]) if not holding_series.empty else ""
    latest_theme = ""
    latest_name = ""
    if latest_holding:
        row = selected[selected["code"] == latest_holding].iloc[0]
        latest_theme = row["theme"]
        latest_name = row["name"]
    exposure_series = result["exposure"] if "exposure" in result.columns else result["holding"].notna().astype(float)
    return {
        "total_return": float(result["nav"].iloc[-1] - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown(result["nav"]),
        "trade_count": int(len(trades)),
        "avg_exposure": float(exposure_series.mean()),
        "latest_holding_code": latest_holding,
        "latest_holding_theme": latest_theme,
        "latest_holding_name": latest_name,
        "latest_momentum": float(result["current_momentum"].dropna().iloc[-1]),
        "latest_exposure": float(exposure_series.iloc[-1]),
    }


def run_threshold_dual_momentum(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change()
    momentum = prices / prices.shift(lookback) - 1

    risk_codes = [c for c in RISK_CODES if c in prices.columns]
    defensive_codes = [c for c in DEFENSIVE_CODES if c in prices.columns]
    risk_mom = momentum[risk_codes]
    def_mom = momentum[defensive_codes]

    risk_winner = risk_mom.idxmax(axis=1, skipna=True)
    risk_best = risk_mom.max(axis=1, skipna=True)
    def_winner = def_mom.idxmax(axis=1, skipna=True)
    def_best = def_mom.max(axis=1, skipna=True)

    signal = pd.Series(index=prices.index, dtype="object")
    exposure = pd.Series(index=prices.index, dtype="float64")
    for dt_idx in prices.index:
        r_asset = risk_winner.loc[dt_idx]
        r_score = risk_best.loc[dt_idx]
        d_asset = def_winner.loc[dt_idx]
        d_score = def_best.loc[dt_idx]
        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            exposure.loc[dt_idx] = 0.0
            continue
        if pd.notna(r_score) and r_score > 0.05:
            signal.loc[dt_idx] = r_asset
            exposure.loc[dt_idx] = 1.0
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            # Weak trend: still use defensive winner and half exposure.
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = 0.5
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
        else:
            signal.loc[dt_idx] = pd.NA
            exposure.loc[dt_idx] = 0.0

    return run_exposure_strategy(prices, selected, signal, exposure, momentum, fee_rate, slippage_rate)


def run_breadth_dual_momentum(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change()
    momentum = prices / prices.shift(lookback) - 1

    risk_codes = [c for c in RISK_CODES if c in prices.columns]
    defensive_codes = [c for c in DEFENSIVE_CODES if c in prices.columns]
    risk_mom = momentum[risk_codes]
    def_mom = momentum[defensive_codes]

    risk_winner = risk_mom.idxmax(axis=1, skipna=True)
    def_winner = def_mom.idxmax(axis=1, skipna=True)
    negative_count = (risk_mom <= 0).sum(axis=1)

    signal = pd.Series(index=prices.index, dtype="object")
    exposure = pd.Series(index=prices.index, dtype="float64")
    for dt_idx in prices.index:
        neg = int(negative_count.loc[dt_idx]) if pd.notna(negative_count.loc[dt_idx]) else len(risk_codes)
        r_asset = risk_winner.loc[dt_idx]
        d_asset = def_winner.loc[dt_idx]
        if neg >= 3 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = 1.0
        elif neg >= 2 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = 0.5
        elif pd.notna(r_asset):
            signal.loc[dt_idx] = r_asset
            exposure.loc[dt_idx] = 1.0
        else:
            signal.loc[dt_idx] = pd.NA
            exposure.loc[dt_idx] = 0.0

    return run_exposure_strategy(prices, selected, signal, exposure, momentum, fee_rate, slippage_rate)


def run_exposure_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    signal: pd.Series,
    target_exposure: pd.Series,
    momentum: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change()
    holding = signal.shift(1)
    exposure = target_exposure.shift(1).fillna(0.0).rename("exposure")
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)

    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
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

        winner = signal.loc[dt_idx]
        if pd.notna(winner) and pd.notna(momentum.loc[dt_idx, winner]):
            current_momentum.loc[dt_idx] = float(momentum.loc[dt_idx, winner])

    nav = (1 + strategy_ret.fillna(0.0)).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1

    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": current_momentum,
            "signal": signal,
            "holding": holding,
            "exposure": exposure,
            "target_exposure": target_exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )
    return result, build_trades(result, selected)


def main() -> int:
    args = parse_args()
    if args.refresh:
        selected = load_fixed_etf_pool()
        prices = fetch_histories(selected, years=args.years)
    else:
        selected, prices = load_cached_data()

    base_result, base_trades = run_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    threshold_result, threshold_trades = run_threshold_dual_momentum(
        prices, selected, args.lookback, args.fee_rate, args.slippage_rate
    )
    breadth_result, breadth_trades = run_breadth_dual_momentum(
        prices, selected, args.lookback, args.fee_rate, args.slippage_rate
    )

    summary = pd.DataFrame(
        [
            {"strategy": "single_momentum", **summarize(base_result, base_trades, selected)},
            {"strategy": "threshold_dual_momentum", **summarize(threshold_result, threshold_trades, selected)},
            {"strategy": "breadth_dual_momentum", **summarize(breadth_result, breadth_trades, selected)},
        ]
    )
    base = summary.iloc[0]
    summary["return_diff"] = summary["total_return"] - base["total_return"]
    summary["annualized_diff"] = summary["annualized_return"] - base["annualized_return"]
    summary["sharpe_diff"] = summary["sharpe_rf0"] - base["sharpe_rf0"]
    summary["mdd_diff"] = summary["max_drawdown"] - base["max_drawdown"]
    summary["trade_diff"] = summary["trade_count"] - base["trade_count"]

    compare_df = pd.DataFrame(index=prices.index)
    compare_df["single_momentum_nav"] = base_result["nav"]
    compare_df["threshold_dual_momentum_nav"] = threshold_result["nav"]
    compare_df["breadth_dual_momentum_nav"] = breadth_result["nav"]
    compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")

    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    compare_df.to_csv(OUTPUT_DIR / "nav_compare.csv")
    threshold_result.to_csv(OUTPUT_DIR / "threshold_dual_momentum_nav.csv")
    threshold_trades.to_csv(OUTPUT_DIR / "threshold_dual_momentum_trades.csv", index=False)
    breadth_result.to_csv(OUTPUT_DIR / "breadth_dual_momentum_nav.csv")
    breadth_trades.to_csv(OUTPUT_DIR / "breadth_dual_momentum_trades.csv", index=False)

    with (OUTPUT_DIR / "rules.txt").open("w", encoding="utf-8") as fp:
        fp.write("Threshold dual momentum:\n")
        fp.write("- risk winner momentum > 5%: hold risk winner at 100%\n")
        fp.write("- 0% < risk winner momentum <= 5%: hold defensive winner at 50%\n")
        fp.write("- risk winner momentum <= 0%: hold defensive winner at 100% if its momentum > 0, else cash\n\n")
        fp.write("Breadth dual momentum:\n")
        fp.write("- if >=3 risk assets have non-positive momentum: hold defensive winner at 100%\n")
        fp.write("- if 2 risk assets have non-positive momentum: hold defensive winner at 50%\n")
        fp.write("- otherwise: hold risk winner at 100%\n")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["single_momentum_nav"], linewidth=2.2, label="Single Momentum")
    ax.plot(compare_df.index, compare_df["threshold_dual_momentum_nav"], linewidth=1.8, label="Threshold Dual")
    ax.plot(compare_df.index, compare_df["breadth_dual_momentum_nav"], linewidth=1.8, label="Breadth Dual")
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.6, linestyle="--", label="HS300")
    ax.set_title("Dual Momentum Variants Comparison", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
