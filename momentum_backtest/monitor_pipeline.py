#!/usr/bin/env python3
"""daily_monitor 的运行编排辅助层。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

try:
    from .monitor_snapshot import build_market_volume_proxy_for_monitor
    from .core.config import DEFAULT_FEE_RATE, DEFAULT_LOOKBACK, DEFAULT_SLIPPAGE_RATE
    from .core.data import fetch_histories
    from .core.strategy import run_default_strategy
    from .core.reporting import save_outputs
    from .core.io import write_dataframe_csv_atomic
except ImportError:
    from monitor_snapshot import build_market_volume_proxy_for_monitor
    from core.config import DEFAULT_FEE_RATE, DEFAULT_LOOKBACK, DEFAULT_SLIPPAGE_RATE
    from core.data import fetch_histories
    from core.strategy import run_default_strategy
    from core.reporting import save_outputs
    from core.io import write_dataframe_csv_atomic

MONITOR_HISTORY_START = pd.Timestamp("2012-01-01")


def build_price_panel(
    today: date,
    strategy_config: dict[str, object],
    *,
    include_realtime: bool,
    allow_same_day_close: bool = False,
    base_prices: pd.DataFrame | None = None,
    spot_prices: dict[str, float] | None = None,
    raw_closes: dict[str, float] | None = None,
    log_fn=None,
) -> pd.DataFrame:
    selected_pool = list(strategy_config["selected_pool"])
    selected = pd.DataFrame(selected_pool)
    if base_prices is None:
        output_dir = Path(strategy_config["backtest_nav_file"]).parent
        prices_path = output_dir / "prices.csv"
        today_ts = pd.Timestamp(today)

        prices = pd.DataFrame()
        if prices_path.exists():
            try:
                prices = pd.read_csv(prices_path, parse_dates=["date"]).set_index("date")
            except Exception:
                prices = pd.DataFrame()

        years = max(pd.Timestamp(today).year - MONITOR_HISTORY_START.year + 1, 15)
        needs_refresh = prices.empty
        if not prices.empty and prices.index.min() > MONITOR_HISTORY_START:
            needs_refresh = True
        if not prices.empty and prices.index.max().normalize() < today_ts.normalize():
            needs_refresh = True

        if needs_refresh:
            try:
                prices = fetch_histories(selected, years=years)
            except Exception as exc:
                if log_fn is not None:
                    log_fn(f"fallback to cached prices because history refresh failed: {exc}")

        prices = prices.loc[prices.index >= MONITOR_HISTORY_START].copy()
    else:
        prices = base_prices.copy()

    today_ts = pd.Timestamp(today)
    if not allow_same_day_close and today_ts in prices.index:
        prices = prices.loc[prices.index < today_ts].copy()

    if include_realtime:
        if spot_prices is None:
            try:
                from .daily_monitor import fetch_realtime_prices  # local import to avoid cycles at module import time
            except ImportError:
                from daily_monitor import fetch_realtime_prices
            try:
                spot_prices = fetch_realtime_prices()
            except Exception as exc:
                if log_fn is not None:
                    log_fn(f"failed to fetch realtime ETF prices, use cached closes: {exc}")
                spot_prices = {}
        if raw_closes is None:
            raw_closes = {}
        for code in prices.columns:
            if code in spot_prices and pd.notna(spot_prices[code]):
                latest_spot = float(spot_prices[code])
                prev_raw_close = raw_closes.get(code)
                history_before_today = prices.loc[prices.index < today_ts, code].dropna()
                prev_adjusted_close = float(history_before_today.iloc[-1]) if not history_before_today.empty else None
                if (
                    prev_adjusted_close is not None
                    and prev_raw_close is not None
                    and pd.notna(prev_raw_close)
                    and abs(float(prev_raw_close)) > 1e-12
                ):
                    prices.loc[today_ts, code] = prev_adjusted_close * latest_spot / float(prev_raw_close)
                else:
                    prices.loc[today_ts, code] = latest_spot

        if today_ts in prices.index:
            for code in prices.columns:
                if pd.notna(prices.loc[today_ts, code]):
                    continue
                history_before_today = prices.loc[prices.index < today_ts, code].dropna()
                if not history_before_today.empty:
                    prices.loc[today_ts, code] = float(history_before_today.iloc[-1])

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    if log_fn is not None and not prices.empty:
        latest_prices = []
        for item in selected_pool:
            code = item["code"]
            if code in prices.columns and pd.notna(prices.iloc[-1][code]):
                latest_prices.append(f"{code}={float(prices.iloc[-1][code]):.4f}")
        panel_kind = "realtime" if include_realtime else "close"
        log_fn(f"latest {panel_kind} price panel date={prices.index[-1].date()}, prices: {', '.join(latest_prices)}")
    return prices


def run_strategy_snapshot(
    prices: pd.DataFrame,
    strategy_config: dict[str, object],
):
    selected_pool = strategy_config["selected_pool"]
    selected = pd.DataFrame(selected_pool)
    market_proxy_context: dict[str, float | str | None] = {
        "mode": None,
        "progress": None,
        "market_amount_ratio_20_60": None,
        "market_amount_ratio_5_20": None,
        "market_breadth_proxy": None,
    }
    market_proxy, market_proxy_context = build_market_volume_proxy_for_monitor(prices)
    result, trades = run_default_strategy(
        prices,
        selected,
        fee_rate=DEFAULT_FEE_RATE,
        slippage_rate=DEFAULT_SLIPPAGE_RATE,
        market_proxy=market_proxy,
    )
    if "target_exposure" not in result.columns and "exposure" in result.columns:
        result["target_exposure"] = result["exposure"].copy()
    if "base_target_exposure" in result.columns:
        base_target_exposure = result["base_target_exposure"].copy()
    else:
        base_target_exposure = result["target_exposure"].copy()
    return result, trades, base_target_exposure, market_proxy_context


def persist_strategy_outputs(
    prices: pd.DataFrame,
    strategy_config: dict[str, object],
    result: pd.DataFrame,
    trades: pd.DataFrame,
) -> None:
    selected = pd.DataFrame(strategy_config["selected_pool"])
    if strategy_config["strategy_id"] == "default":
        save_outputs(
            selected=selected,
            prices=prices,
            result=result,
            trades=trades,
            lookback=DEFAULT_LOOKBACK,
            strategy_name=str(strategy_config["strategy_label"]),
        )
        return

    output_dir = Path(strategy_config["backtest_nav_file"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prices_to_save = prices.copy()
    prices_to_save.index.name = "date"
    result_to_save = result.copy()
    result_to_save.index.name = "date"
    write_dataframe_csv_atomic(selected, output_dir / "selected_etfs.csv", index=False)
    write_dataframe_csv_atomic(prices_to_save, output_dir / "prices.csv")
    write_dataframe_csv_atomic(result_to_save, output_dir / "backtest_nav.csv")
    write_dataframe_csv_atomic(trades, output_dir / "trades.csv", index=False)
    write_dataframe_csv_atomic(
        result_to_save[["nav"]].rename(columns={"nav": "historical_nav"}),
        output_dir / "historical_nav.csv",
    )
