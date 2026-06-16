#!/usr/bin/env python3
"""对比把紫金矿业加入当前正式基线候选池后的效果。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import warnings
warnings.filterwarnings("ignore")

import akshare as ak
import matplotlib
import pandas as pd

from run_backtest import (
    DEFAULT_HISTORY_START,
    DEFAULT_YEARS,
    RESEARCH_OUTPUT_DIR,
    build_benchmark_nav,
    build_default_strategy_params,
    build_strategy_summary,
    ensure_output_dirs,
    fetch_histories,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)
from compare_goal_optimizations import load_market_volume_proxy

matplotlib.use("Agg")

DEFAULT_STOCK = {
    "theme": "紫金矿业",
    "code": "601899",
    "name": "紫金矿业",
    "asset_type": "stock",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比把紫金矿业加入当前正式基线候选池后的效果。")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="抓取历史数据年数。")
    parser.add_argument("--start-date", type=str, default="2012-01-01", help="分析起始日期。")
    parser.add_argument("--stock-code", type=str, default=DEFAULT_STOCK["code"], help="候选股票代码。")
    parser.add_argument("--stock-name", type=str, default=DEFAULT_STOCK["name"], help="候选股票名称。")
    parser.add_argument("--stock-theme", type=str, default=DEFAULT_STOCK["theme"], help="候选股票主题展示名。")
    return parser.parse_args()


def fetch_stock_histories(selected: pd.DataFrame, years: int, start_date: pd.Timestamp) -> pd.DataFrame:
    end_date = pd.Timestamp.today().normalize()
    frames: list[pd.Series] = []

    for row in selected.itertuples(index=False):
        symbol = f"sh{row.code}" if str(row.code).startswith('6') else f"sz{row.code}"
        df = ak.stock_zh_a_daily(symbol=symbol, adjust="hfq")
        if df is None or df.empty:
            raise RuntimeError(f"empty stock history for {row.code} {row.name}")
        if "date" not in df.columns or "close" not in df.columns:
            raise RuntimeError(f"unexpected stock history columns for {row.code}: {list(df.columns)}")
        df = df[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[(df["date"] >= start_date) & df["close"].notna()].copy()
        if df.empty:
            raise RuntimeError(f"no {years}y stock history for {row.code} {row.name}")
        frames.append(df.set_index("date")["close"].rename(str(row.code)))

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.index.name = "date"
    return prices


def fetch_mixed_histories(selected: pd.DataFrame, years: int) -> pd.DataFrame:
    selected = selected.copy()
    if "asset_type" not in selected.columns:
        selected["asset_type"] = "etf"
    selected["asset_type"] = selected["asset_type"].fillna("etf").astype(str)

    end_date = pd.Timestamp.today().normalize()
    start_date = max(end_date - pd.DateOffset(years=years), DEFAULT_HISTORY_START)

    etf_selected = selected[selected["asset_type"] == "etf"].copy()
    stock_selected = selected[selected["asset_type"] == "stock"].copy()

    frames: list[pd.DataFrame] = []
    if not etf_selected.empty:
        etf_prices = fetch_histories(etf_selected, years=years)
        etf_prices = etf_prices.loc[etf_prices.index >= start_date]
        frames.append(etf_prices)
    if not stock_selected.empty:
        stock_prices = fetch_stock_histories(stock_selected, years=years, start_date=start_date)
        frames.append(stock_prices)

    if not frames:
        raise RuntimeError("no assets selected")

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="any")
    prices.index.name = "date"
    return prices


def summarize(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, object]:
    latest_row = result.iloc[-1]
    return {
        "start_date": result.index[0].date().isoformat(),
        "end_date": result.index[-1].date().isoformat(),
        **build_strategy_summary(result, trades),
        "latest_signal": str(latest_row.get("signal")) if pd.notna(latest_row.get("signal")) else "",
        "latest_holding": str(latest_row.get("holding")) if pd.notna(latest_row.get("holding")) else "",
        "latest_exposure": float(latest_row.get("exposure", 0.0)),
        "latest_momentum": float(latest_row.get("current_momentum")) if pd.notna(latest_row.get("current_momentum")) else float("nan"),
    }


def main() -> int:
    args = parse_args()
    ensure_output_dirs()

    candidate_stock = {
        "theme": args.stock_theme,
        "code": str(args.stock_code),
        "name": args.stock_name,
        "asset_type": "stock",
    }

    base_selected = load_default_strategy_backtest_pool().copy()
    base_selected["asset_type"] = "etf"
    plus_selected = pd.concat([base_selected, pd.DataFrame([candidate_stock])], ignore_index=True)

    base_params = build_default_strategy_params()
    plus_params = build_default_strategy_params(risk_codes=list(base_params["risk_codes"]) + [candidate_stock["code"]])

    prices = fetch_mixed_histories(plus_selected, years=args.years)
    prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)].copy()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)

    base_result, base_trades = run_default_strategy_with_params(
        prices=prices[base_selected["code"].tolist()],
        selected=base_selected,
        params=base_params,
        market_proxy=market_proxy,
    )
    plus_result, plus_trades = run_default_strategy_with_params(
        prices=prices[plus_selected["code"].tolist()],
        selected=plus_selected,
        params=plus_params,
        market_proxy=market_proxy,
    )

    candidate_tag = f"plus_{candidate_stock['code']}_{candidate_stock['name']}"
    base_summary = {"candidate": "base_current_baseline", **summarize(base_result, base_trades)}
    plus_summary = {"candidate": candidate_tag, **summarize(plus_result, plus_trades)}
    summary_df = pd.DataFrame([base_summary, plus_summary])
    metric_cols = [
        "total_return",
        "annualized_return",
        "sharpe_rf0",
        "max_drawdown",
        "max_drawdown_integral",
        "max_drawdown_integral_annualized",
        "trade_count",
        "latest_exposure",
        "latest_momentum",
    ]
    base_row = summary_df.iloc[0]
    for col in metric_cols:
        if col in summary_df.columns:
            summary_df[f"{col}_diff_vs_base"] = summary_df[col] - base_row[col]

    nav_compare = pd.DataFrame(index=base_result.index)
    nav_compare["base_current_baseline"] = base_result["nav"]
    nav_compare[candidate_tag] = plus_result["nav"].reindex(nav_compare.index)
    nav_compare["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300").reindex(nav_compare.index)

    out_base = RESEARCH_OUTPUT_DIR / 'archive_flat' / f"compare_zijin_candidate_{candidate_stock['code']}"
    out_base.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(summary_df, out_base / "summary.csv", index=False)
    write_dataframe_csv_atomic(nav_compare, out_base / "nav_compare.csv")
    write_dataframe_csv_atomic(plus_trades, out_base / f"plus_{candidate_stock['code']}_trades.csv", index=False)

    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
