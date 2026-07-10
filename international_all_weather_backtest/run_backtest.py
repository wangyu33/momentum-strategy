#!/usr/bin/env python3
"""国际全天候策略回测入口。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from momentum_backtest.runtime_env import configure_matplotlib_env, scrub_user_site_packages

scrub_user_site_packages()
configure_matplotlib_env()
warnings.filterwarnings("ignore")

import akshare as ak
import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = Path("international_all_weather_backtest/output")
CORE_OUTPUT_DIR = OUTPUT_DIR / "core"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"

DEFAULT_YEARS = 15
WARMUP_DAYS = 60
DEFAULT_COST_RATE = 0.0005
STRATEGY_NAME = "global_all_weather_lite_monthly"

ASSETS = [
    {"role": "growth", "ticker": "SPY", "theme": "US Equity", "name": "SPDR S&P 500 ETF Trust", "target_weight": 0.20},
    {"role": "growth", "ticker": "EFA", "theme": "Developed ex-US Equity", "name": "iShares MSCI EAFE ETF", "target_weight": 0.10},
    {"role": "growth", "ticker": "EEM", "theme": "Emerging Equity", "name": "iShares MSCI Emerging Markets ETF", "target_weight": 0.05},
    {"role": "defense", "ticker": "IEF", "theme": "US 7-10Y Treasury", "name": "iShares 7-10 Year Treasury Bond ETF", "target_weight": 0.35},
    {"role": "inflation", "ticker": "GLD", "theme": "Gold", "name": "SPDR Gold Shares", "target_weight": 0.15},
    {"role": "inflation", "ticker": "DBC", "theme": "Commodities", "name": "Invesco DB Commodity Index Tracking Fund", "target_weight": 0.10},
    {"role": "real_asset", "ticker": "VNQ", "theme": "US REITs", "name": "Vanguard Real Estate ETF", "target_weight": 0.05},
]
CASH_PROXY = {"role": "cash", "ticker": "BIL", "theme": "Cash Proxy", "name": "SPDR Bloomberg 1-3 Month T-Bill ETF", "target_weight": 0.0}


def write_dataframe_csv_atomic(df: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f"{path.stem}_",
            dir=str(path.parent),
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_path = Path(handle.name)
            df.to_csv(handle, index=index)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"{path.stem}_",
            dir=str(path.parent),
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, CORE_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def fetch_one_us_etf_history(symbol: str, start_date: pd.Timestamp) -> pd.Series:
    errors: list[str] = []
    for attempt in range(4):
        try:
            df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
            if df is None or df.empty:
                errors.append(f"empty attempt={attempt + 1}")
            else:
                df = df[["date", "close"]].copy()
                df["date"] = pd.to_datetime(df["date"])
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df[(df["date"] >= start_date) & df["close"].notna()].copy()
                if not df.empty:
                    return df.sort_values("date").set_index("date")["close"].rename(symbol)
                errors.append(f"filtered empty attempt={attempt + 1}")
        except Exception as exc:
            errors.append(f"attempt={attempt + 1}: {exc}")
        time.sleep(1.0)
    raise RuntimeError(f"failed to load {symbol}: {' | '.join(errors)}")


def fetch_price_panel(years: int) -> pd.DataFrame:
    analysis_end = pd.Timestamp.today().normalize()
    analysis_start = analysis_end - pd.DateOffset(years=years)
    fetch_start = analysis_start - pd.Timedelta(days=WARMUP_DAYS)

    symbols = [asset["ticker"] for asset in ASSETS] + [CASH_PROXY["ticker"]]
    frames = [fetch_one_us_etf_history(symbol, fetch_start) for symbol in symbols]
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="all")
    prices.index.name = "date"
    return prices.loc[prices.index >= fetch_start].copy()


def build_monthly_targets(prices: pd.DataFrame) -> pd.DataFrame:
    monthly_index = prices.resample("M").last().index
    target_weights = pd.DataFrame(0.0, index=monthly_index, columns=prices.columns, dtype="float64")
    for asset in ASSETS:
        target_weights.loc[:, asset["ticker"]] = float(asset["target_weight"])
    target_weights.loc[:, CASH_PROXY["ticker"]] = float(CASH_PROXY["target_weight"])
    return target_weights


def build_daily_target_weights(prices: pd.DataFrame, monthly_targets: pd.DataFrame, analysis_start: pd.Timestamp) -> pd.DataFrame:
    daily_index = prices.loc[prices.index >= analysis_start].index
    daily_targets = pd.DataFrame(float("nan"), index=daily_index, columns=prices.columns, dtype="float64")

    first_day_by_month = daily_index.to_series().groupby(daily_index.to_period("M")).min()
    prev_month_targets = monthly_targets.loc[monthly_targets.index < daily_index.min()]
    if not prev_month_targets.empty:
        daily_targets.iloc[0] = prev_month_targets.iloc[-1]
    else:
        daily_targets.iloc[0] = monthly_targets.iloc[0]

    for rebalance_dt, weights in monthly_targets.iterrows():
        next_period = rebalance_dt.to_period("M") + 1
        if next_period not in first_day_by_month.index:
            continue
        effective_dt = pd.Timestamp(first_day_by_month.loc[next_period])
        if effective_dt not in daily_targets.index:
            continue
        daily_targets.loc[effective_dt] = weights

    return daily_targets.ffill().fillna(0.0)


def simulate_monthly_rebalance(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change().fillna(0.0)
    current_weights = target_weights.iloc[0].astype("float64").copy()
    nav = 1.0

    nav_list: list[float] = []
    strategy_return_list: list[float] = []
    turnover_list: list[float] = []
    trade_cost_list: list[float] = []
    weight_rows: list[dict[str, float]] = []
    trade_rows: list[dict[str, object]] = []

    prev_period = None
    for dt_idx in prices.index:
        current_period = dt_idx.to_period("M")
        desired_weights = target_weights.loc[dt_idx].astype("float64")
        turnover = 0.0
        if prev_period is None or current_period != prev_period:
            turnover = float((desired_weights - current_weights).abs().sum())
            for symbol in target_weights.columns:
                curr_w = float(current_weights[symbol])
                next_w = float(desired_weights[symbol])
                if abs(next_w - curr_w) < 1e-12:
                    continue
                trade_rows.append(
                    {
                        "date": dt_idx,
                        "action": "BUY" if next_w > curr_w else "SELL",
                        "symbol": symbol,
                        "from_weight": curr_w,
                        "to_weight": next_w,
                        "nav": nav,
                    }
                )
            current_weights = desired_weights.copy()

        trade_cost_rate = turnover * cost_rate
        gross_return = float((current_weights * returns.loc[dt_idx]).sum())
        strategy_return = (1.0 + gross_return) * (1.0 - trade_cost_rate) - 1.0
        nav *= 1.0 + strategy_return

        after_close = current_weights * (1.0 + returns.loc[dt_idx])
        total_weight = float(after_close.sum())
        if total_weight > 0:
            current_weights = after_close / total_weight

        nav_list.append(nav)
        strategy_return_list.append(strategy_return)
        turnover_list.append(turnover)
        trade_cost_list.append(trade_cost_rate)
        weight_rows.append({f"weight_{symbol}": float(current_weights[symbol]) for symbol in target_weights.columns})
        prev_period = current_period

    result = pd.DataFrame(index=prices.index)
    result["nav"] = nav_list
    result["strategy_return"] = strategy_return_list
    result["turnover"] = turnover_list
    result["trade_cost_rate"] = trade_cost_list
    result = pd.concat([result, pd.DataFrame(weight_rows, index=prices.index)], axis=1)
    trades = pd.DataFrame(trade_rows)
    return result, trades


def build_yearly_returns(result: pd.DataFrame) -> pd.DataFrame:
    nav = result["nav"]
    benchmark = result["benchmark_nav"]
    prior_nav = nav.shift(1)
    prior_benchmark = benchmark.shift(1)
    rows: list[dict[str, object]] = []
    for year in sorted(nav.index.year.unique()):
        period_index = nav.index[nav.index.year == year]
        if period_index.empty:
            continue
        start_idx = period_index[0]
        end_idx = period_index[-1]
        nav_start = float(prior_nav.loc[start_idx]) if pd.notna(prior_nav.loc[start_idx]) else 1.0
        bench_start = float(prior_benchmark.loc[start_idx]) if pd.notna(prior_benchmark.loc[start_idx]) else 1.0
        rows.append(
            {
                "year": year,
                "start_date": str(start_idx.date()),
                "end_date": str(end_idx.date()),
                "strategy_return": float(nav.loc[end_idx] / nav_start - 1.0),
                "benchmark_return": float(benchmark.loc[end_idx] / bench_start - 1.0),
                "excess_return": float(nav.loc[end_idx] / nav_start - benchmark.loc[end_idx] / bench_start),
            }
        )
    return pd.DataFrame(rows)


def build_summary(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | str]:
    nav = result["nav"]
    returns = result["strategy_return"]
    drawdown = result["drawdown"]
    benchmark_nav = result["benchmark_nav"]
    benchmark_returns = result["benchmark_return"]
    benchmark_drawdown = benchmark_nav / benchmark_nav.cummax() - 1.0
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)

    annualized_return = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
    annualized_vol = float(returns.std(ddof=0) * math.sqrt(252))
    benchmark_annualized_return = float(benchmark_nav.iloc[-1] ** (1.0 / years) - 1.0)
    benchmark_annualized_vol = float(benchmark_returns.std(ddof=0) * math.sqrt(252))
    latest_weights = {
        asset["ticker"]: float(result[f"weight_{asset['ticker']}"] .iloc[-1])
        for asset in ASSETS
    }
    latest_top3 = sorted(latest_weights.items(), key=lambda item: item[1], reverse=True)[:3]

    return {
        "strategy": STRATEGY_NAME,
        "start_date": str(nav.index[0].date()),
        "end_date": str(nav.index[-1].date()),
        "total_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_rf0": float(annualized_return / annualized_vol) if annualized_vol > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_integral": float((-drawdown.clip(upper=0.0)).sum()),
        "benchmark_total_return": float(benchmark_nav.iloc[-1] - 1.0),
        "benchmark_annualized_return": benchmark_annualized_return,
        "benchmark_annualized_volatility": benchmark_annualized_vol,
        "benchmark_sharpe_rf0": float(benchmark_annualized_return / benchmark_annualized_vol)
        if benchmark_annualized_vol > 0
        else float("nan"),
        "benchmark_max_drawdown": float(benchmark_drawdown.min()),
        "trade_count": int(len(trades)),
        "latest_top1_symbol": latest_top3[0][0],
        "latest_top1_weight": latest_top3[0][1],
        "latest_top2_symbol": latest_top3[1][0],
        "latest_top2_weight": latest_top3[1][1],
        "latest_top3_symbol": latest_top3[2][0],
        "latest_top3_weight": latest_top3[2][1],
    }


def save_plots(result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result.index, result["nav"], label="All Weather Lite NAV", color="#1f4f82", linewidth=2.2)
    ax.plot(result.index, result["benchmark_nav"], label="Equal-Weight Multi-Asset", color="#d2691e", linewidth=1.8)
    ax.set_title("International All Weather Lite vs Equal-Weight Benchmark", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_OUTPUT_DIR / "nav_vs_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result.index, result["drawdown"], color="#b22222", linewidth=1.8)
    ax.fill_between(result.index, result["drawdown"].values, 0, color="#f4a6a6", alpha=0.6)
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("Historical Drawdown", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(ANALYSIS_OUTPUT_DIR / "max_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    weight_columns = [f"weight_{asset['ticker']}" for asset in ASSETS]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(
        result.index,
        [result[column].values for column in weight_columns],
        labels=[column.replace("weight_", "") for column in weight_columns],
        alpha=0.9,
    )
    ax.set_title("Static Strategic Weights", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(ANALYSIS_OUTPUT_DIR / "weights_stack.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_strategy(prices: pd.DataFrame, years: int, cost_rate: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    analysis_end = prices.index.max()
    analysis_start = max(prices.index.min(), analysis_end - pd.DateOffset(years=years))
    analysis_prices = prices.loc[prices.index >= analysis_start].copy()
    core_symbols = [asset["ticker"] for asset in ASSETS]

    monthly_targets = build_monthly_targets(prices)
    daily_targets = build_daily_target_weights(prices, monthly_targets, analysis_start=analysis_start)
    daily_targets = daily_targets.reindex(analysis_prices.index).ffill().fillna(0.0)

    returns = analysis_prices.pct_change().fillna(0.0)
    benchmark_return = returns[core_symbols].mean(axis=1)
    benchmark_nav = (1.0 + benchmark_return).cumprod()
    benchmark_nav.iloc[0] = 1.0

    result, trades = simulate_monthly_rebalance(analysis_prices, daily_targets, cost_rate=cost_rate)
    drawdown = result["nav"] / result["nav"].cummax() - 1.0

    result["benchmark_nav"] = benchmark_nav.reindex(result.index).ffill()
    result["benchmark_return"] = benchmark_return.reindex(result.index).fillna(0.0)
    result["drawdown"] = drawdown

    monthly_snapshot = monthly_targets.copy()
    monthly_snapshot.columns = [f"target_{column}" for column in monthly_snapshot.columns]
    summary = build_summary(result, trades)
    monthly_snapshot = monthly_snapshot.loc[monthly_snapshot.index >= (analysis_start - pd.DateOffset(months=1))].copy()
    return result, trades, monthly_snapshot, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行国际全天候策略回测。")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="分析最近多少年历史数据。")
    parser.add_argument("--cost-rate", type=float, default=DEFAULT_COST_RATE, help="按换手收取的一次性总交易成本率。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()

    prices = fetch_price_panel(years=max(args.years, DEFAULT_YEARS))
    result, trades, monthly_snapshot, summary = run_strategy(prices, years=args.years, cost_rate=args.cost_rate)

    prices_to_save = prices.loc[prices.index >= (result.index.min() - pd.Timedelta(days=WARMUP_DAYS))].copy()
    prices_to_save.index.name = "date"
    result_to_save = result.copy()
    result_to_save.index.name = "date"

    write_dataframe_csv_atomic(pd.DataFrame([*ASSETS, CASH_PROXY]), CORE_OUTPUT_DIR / "selected_assets.csv", index=False)
    write_dataframe_csv_atomic(prices_to_save, CORE_OUTPUT_DIR / "prices.csv")
    write_dataframe_csv_atomic(monthly_snapshot, CORE_OUTPUT_DIR / "monthly_weights.csv")
    write_dataframe_csv_atomic(result_to_save, CORE_OUTPUT_DIR / "backtest_nav.csv")
    write_dataframe_csv_atomic(trades, CORE_OUTPUT_DIR / "trades.csv", index=False)
    write_dataframe_csv_atomic(build_yearly_returns(result), CORE_OUTPUT_DIR / "yearly_returns.csv", index=False)
    write_dataframe_csv_atomic(pd.DataFrame([summary]), CORE_OUTPUT_DIR / "summary.csv", index=False)
    write_json_atomic(CORE_OUTPUT_DIR / "summary.json", summary)
    save_plots(result)

    print(pd.DataFrame([summary]).to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
