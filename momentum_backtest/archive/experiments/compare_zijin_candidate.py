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

from archive_data_loaders import build_archive_flat_output_dir, load_selected_prices_from_start

try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

from archive_strategy_common import (
    DEFAULT_HISTORY_START,
    DEFAULT_YEARS,
    build_default_strategy_params,
    fetch_histories,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
)
from goal_optimization_common import load_market_volume_proxy
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    filter_available_plot_lines,
    filter_available_metric_mappings,
    finalize_baseline_diff_summary,
    save_variant_compare_artifacts,
    summarize_variant_result,
)

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
    parser.add_argument("--refresh", action="store_true", help="重新抓取 ETF 历史，而不是优先复用 core 缓存。")
    parser.add_argument("--base-only", action="store_true", help="只运行当前正式基线，不拉股票历史。")
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


def fetch_mixed_histories(selected: pd.DataFrame, years: int, *, refresh: bool) -> pd.DataFrame:
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
        if refresh:
            etf_prices = fetch_histories(etf_selected, years=years)
            etf_prices = etf_prices.loc[etf_prices.index >= start_date]
        else:
            etf_prices = load_selected_prices_from_start(etf_selected, start_date)
        frames.append(etf_prices)
    if not stock_selected.empty:
        try:
            stock_prices = fetch_stock_histories(stock_selected, years=years, start_date=start_date)
        except Exception as exc:
            codes = ", ".join(stock_selected["code"].astype(str).tolist())
            raise RuntimeError(
                f"failed to load stock history for {codes}; rerun with network access"
            ) from exc
        frames.append(stock_prices)

    if not frames:
        raise RuntimeError("no assets selected")

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="any")
    prices.index.name = "date"
    return prices


def main() -> int:
    args = parse_args()

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

    start_ts = pd.Timestamp(args.start_date)
    base_prices = load_selected_prices_from_start(base_selected, start_ts)
    prices = base_prices.copy()
    if not args.base_only:
        prices = fetch_mixed_histories(plus_selected, years=args.years, refresh=args.refresh)
        prices = prices.loc[prices.index >= start_ts].copy()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)

    base_result, base_trades = run_default_strategy_with_params(
        prices=prices[base_selected["code"].tolist()],
        selected=base_selected,
        params=base_params,
        market_proxy=market_proxy,
    )
    base_result = apply_official_baseline_nav_anchor(base_result)
    candidate_tag = f"plus_{candidate_stock['code']}_{candidate_stock['name']}"
    rows: list[dict[str, object]] = []
    nav_compare = build_compare_frame(prices.loc[base_result.index])
    named_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "base_current_baseline": (base_result, base_trades),
    }
    append_variant_result(
        rows,
        nav_compare,
        None,
        strategy="base_current_baseline",
        result=base_result,
        summary=summarize_variant_result(
            base_result,
            base_trades,
            include_window_dates=True,
            include_latest_signal_holding=True,
        ),
        strategy_field="candidate",
        nav_column="base_current_baseline",
    )
    if not args.base_only:
        plus_result, plus_trades = run_default_strategy_with_params(
            prices=prices[plus_selected["code"].tolist()],
            selected=plus_selected,
            params=plus_params,
            market_proxy=market_proxy,
        )
        append_variant_result(
            rows,
            nav_compare,
            None,
            strategy=candidate_tag,
            result=plus_result,
            summary=summarize_variant_result(
                plus_result,
                plus_trades,
                include_window_dates=True,
                include_latest_signal_holding=True,
            ),
            strategy_field="candidate",
            nav_column=candidate_tag,
        )
        named_outputs[f"plus_{candidate_stock['code']}"] = (plus_result, plus_trades)
    metric_mappings = [
        ("total_return", "total_return_diff_vs_base"),
        ("annualized_return", "annualized_return_diff_vs_base"),
        ("sharpe_rf0", "sharpe_rf0_diff_vs_base"),
        ("max_drawdown", "max_drawdown_diff_vs_base"),
        ("max_drawdown_integral", "max_drawdown_integral_diff_vs_base"),
        ("max_drawdown_integral_annualized", "max_drawdown_integral_annualized_diff_vs_base"),
        ("trade_count", "trade_count_diff_vs_base"),
        ("latest_exposure", "latest_exposure_diff_vs_base"),
        ("latest_momentum", "latest_momentum_diff_vs_base"),
    ]
    metric_mappings = filter_available_metric_mappings(rows, metric_mappings)
    summary_df = finalize_baseline_diff_summary(
        rows,
        baseline_field="candidate",
        baseline_value="base_current_baseline",
        metric_mappings=metric_mappings,
    )

    plot_lines = [
        ("base_current_baseline", "Base Current Baseline", 2.2),
        (candidate_tag, f"+{candidate_stock['code']}", 1.8),
    ]
    available_plot_lines = filter_available_plot_lines(nav_compare, plot_lines)

    out_base = build_archive_flat_output_dir(f"compare_zijin_candidate_{candidate_stock['code']}")
    save_variant_compare_artifacts(
        out_base,
        summary_df,
        nav_compare,
        named_outputs=named_outputs,
        plot_filename="comparison.png",
        title=f"Zijin Candidate Comparison ({candidate_stock['code']})",
        lines=available_plot_lines,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
