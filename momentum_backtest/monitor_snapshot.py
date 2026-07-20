#!/usr/bin/env python3
"""daily_monitor 的 snapshot 计算辅助层。"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from .monitor_render import drawdown_bucket, momentum_bucket
except ImportError:
    from monitor_render import drawdown_bucket, momentum_bucket

REGIME_MIN_SAMPLE_COUNT = 20
MONITOR_HISTORY_START = pd.Timestamp("2012-01-01")


def build_forward_regime_stats(nav_df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    required_cols = {"nav", "current_momentum", "drawdown"}
    if not required_cols.issubset(nav_df.columns):
        return pd.DataFrame()

    df = nav_df[["nav", "current_momentum", "drawdown"]].dropna().copy()
    if df.empty or len(df) <= horizon:
        return pd.DataFrame()

    df["fwd_ret"] = df["nav"].shift(-horizon) / df["nav"] - 1
    nav_values = df["nav"].to_numpy()
    future_max_drawdowns: list[float] = []
    for idx in range(len(df)):
        if idx + horizon >= len(df):
            future_max_drawdowns.append(float("nan"))
            continue
        future_path = nav_values[idx : idx + horizon + 1]
        running_peak = pd.Series(future_path).cummax().to_numpy()
        future_drawdown = future_path / running_peak - 1
        future_max_drawdowns.append(float(future_drawdown.min()))
    df["fwd_mdd"] = future_max_drawdowns
    df = df.dropna(subset=["fwd_ret", "fwd_mdd"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["mom_bucket"] = df["current_momentum"].apply(lambda x: momentum_bucket(float(x)))
    df["dd_bucket"] = df["drawdown"].apply(lambda x: drawdown_bucket(float(x)))
    grouped = (
        df.groupby(["mom_bucket", "dd_bucket"], dropna=False)
        .agg(
            count=("fwd_ret", "size"),
            win_rate_60=("fwd_ret", lambda s: float((s > 0).mean())),
            avg_ret_60=("fwd_ret", "mean"),
            avg_mdd_60=("fwd_mdd", "mean"),
        )
        .reset_index()
    )
    return grouped


def load_backtest_reference_context(
    backtest_nav_file: Path,
    result: pd.DataFrame,
) -> dict[str, object]:
    max_drawdown = None
    current_drawdown = None
    current_nav = None
    nav_date = None
    peak_nav_date = None
    peak_nav_value = None
    hist_momentum = pd.Series(dtype="float64")
    regime_stats = pd.DataFrame()
    latest_result_nav_date = None

    if "nav" in result.columns and not result["nav"].dropna().empty:
        latest_result_nav_date = pd.Timestamp(result["nav"].dropna().index[-1]).normalize()

    if backtest_nav_file.exists():
        backtest_nav = pd.read_csv(backtest_nav_file, parse_dates=["date"])
        if "date" in backtest_nav.columns and not result.empty:
            snapshot_end = pd.Timestamp(result.index.max()).normalize()
            backtest_nav = backtest_nav.loc[backtest_nav["date"] <= snapshot_end].copy()
        latest_backtest_nav_date = None
        if "nav" in backtest_nav.columns and not backtest_nav["nav"].dropna().empty and "date" in backtest_nav.columns:
            latest_backtest_nav_date = pd.Timestamp(backtest_nav["date"].iloc[-1]).normalize()
        if "drawdown" in backtest_nav.columns and not backtest_nav["drawdown"].dropna().empty:
            max_drawdown = float(backtest_nav["drawdown"].min())
            current_drawdown = float(backtest_nav["drawdown"].dropna().iloc[-1])
        if "nav" in backtest_nav.columns and not backtest_nav["nav"].dropna().empty:
            current_nav = float(backtest_nav["nav"].dropna().iloc[-1])
            nav_date = str(pd.Timestamp(backtest_nav["date"].iloc[-1]).date())
            peak_idx = backtest_nav["nav"].idxmax()
            peak_nav_value = float(backtest_nav.loc[peak_idx, "nav"])
            peak_nav_date = str(pd.Timestamp(backtest_nav.loc[peak_idx, "date"]).date())
        if "current_momentum" in backtest_nav.columns:
            hist_momentum = backtest_nav["current_momentum"].dropna()
        regime_stats = build_forward_regime_stats(backtest_nav, horizon=60)

        backtest_is_stale = (
            latest_result_nav_date is not None
            and latest_backtest_nav_date is not None
            and latest_backtest_nav_date < latest_result_nav_date
        )
        if backtest_is_stale:
            max_drawdown = None
            current_drawdown = None
            current_nav = None
            nav_date = None
            peak_nav_date = None
            peak_nav_value = None
            hist_momentum = pd.Series(dtype="float64")
            regime_stats = pd.DataFrame()

    if max_drawdown is None and "drawdown" in result.columns and not result["drawdown"].dropna().empty:
        max_drawdown = float(result["drawdown"].min())
    if current_drawdown is None and "drawdown" in result.columns and not result["drawdown"].dropna().empty:
        current_drawdown = float(result["drawdown"].dropna().iloc[-1])
    if current_nav is None and "nav" in result.columns and not result["nav"].dropna().empty:
        current_nav = float(result["nav"].dropna().iloc[-1])
    if nav_date is None and "nav" in result.columns and not result["nav"].dropna().empty:
        nav_date = str(pd.Timestamp(result["nav"].dropna().index[-1]).date())
    if peak_nav_value is None and "nav" in result.columns and not result["nav"].dropna().empty:
        peak_idx = result["nav"].idxmax()
        peak_nav_value = float(result.loc[peak_idx, "nav"])
        peak_nav_date = str(pd.Timestamp(peak_idx).date())
    if hist_momentum.empty and "current_momentum" in result.columns:
        hist_momentum = result["current_momentum"].dropna()
    if regime_stats.empty:
        regime_stats = build_forward_regime_stats(result, horizon=60)

    return {
        "max_drawdown": max_drawdown,
        "current_drawdown": current_drawdown,
        "current_nav": current_nav,
        "nav_date": nav_date,
        "peak_nav_date": peak_nav_date,
        "peak_nav_value": peak_nav_value,
        "hist_momentum": hist_momentum,
        "regime_stats": regime_stats,
    }


def build_entry_risk_context(
    current_mom: float | None,
    current_drawdown: float | None,
    max_drawdown: float | None,
    hist_momentum: pd.Series,
    regime_stats: pd.DataFrame,
) -> dict[str, object]:
    entry_risk_score = None
    entry_risk_level = None
    historical_regime_label = None
    historical_regime_count = None
    historical_win_rate_60 = None
    historical_win_rate_60_percentile = None
    historical_avg_ret_60 = None
    historical_avg_mdd_60 = None
    historical_avg_ret_60_percentile = None
    entry_advice = None

    momentum_percentile = None
    drawdown_buffer_ratio = None
    if current_mom is not None and not hist_momentum.empty:
        momentum_percentile = float((hist_momentum <= current_mom).mean())
    if current_drawdown is not None and max_drawdown is not None and max_drawdown < 0:
        drawdown_buffer_ratio = float(min(abs(current_drawdown) / abs(max_drawdown), 1.0))

    if momentum_percentile is not None and drawdown_buffer_ratio is not None:
        drawdown_risk = 1.0 - drawdown_buffer_ratio
        entry_risk_score = 100.0 * (0.55 * drawdown_risk + 0.45 * momentum_percentile)
        if entry_risk_score >= 70:
            entry_risk_level = "高"
        elif entry_risk_score >= 40:
            entry_risk_level = "中"
        else:
            entry_risk_level = "低"

    current_mom_bucket = momentum_bucket(current_mom)
    current_dd_bucket = drawdown_bucket(current_drawdown)
    if current_mom_bucket and current_dd_bucket and not regime_stats.empty:
        matched = regime_stats[
            (regime_stats["mom_bucket"] == current_mom_bucket) & (regime_stats["dd_bucket"] == current_dd_bucket)
        ]
        if not matched.empty:
            row = matched.sort_values("count", ascending=False).iloc[0]
            historical_regime_count = int(row["count"])
            historical_win_rate_60 = float(row["win_rate_60"])
            historical_avg_ret_60 = float(row["avg_ret_60"])
            historical_avg_mdd_60 = float(row["avg_mdd_60"])
            valid_stats = regime_stats.loc[regime_stats["count"] >= REGIME_MIN_SAMPLE_COUNT].copy()
            valid_avg_ret = valid_stats["avg_ret_60"].dropna()
            if not valid_avg_ret.empty:
                historical_avg_ret_60_percentile = float((valid_avg_ret <= historical_avg_ret_60).mean())
            valid_win_rate = valid_stats["win_rate_60"].dropna()
            if not valid_win_rate.empty:
                historical_win_rate_60_percentile = float((valid_win_rate <= historical_win_rate_60).mean())
            if historical_regime_count < REGIME_MIN_SAMPLE_COUNT:
                historical_regime_label = "样本少"
            elif historical_win_rate_60 >= 0.75 and historical_avg_mdd_60 >= -0.03:
                historical_regime_label = "低"
            elif historical_win_rate_60 >= 0.60 and historical_avg_mdd_60 >= -0.05:
                historical_regime_label = "中"
            else:
                historical_regime_label = "高"

    if entry_risk_level == "高":
        entry_advice = "不宜追高"
    elif historical_regime_count is not None and historical_regime_count < REGIME_MIN_SAMPLE_COUNT:
        entry_advice = "样本不足，谨慎参考"
    elif historical_regime_label == "低" and entry_risk_level in {"低", "中"}:
        entry_advice = "可建仓"
    elif historical_regime_label == "中" or entry_risk_level == "中":
        entry_advice = "谨慎建仓"
    elif historical_regime_label == "高":
        entry_advice = "不宜追高"
    elif entry_risk_level == "低":
        entry_advice = "可建仓"

    return {
        "entry_risk_score": entry_risk_score,
        "entry_risk_level": entry_risk_level,
        "historical_regime_label": historical_regime_label,
        "historical_regime_count": historical_regime_count,
        "historical_win_rate_60": historical_win_rate_60,
        "historical_win_rate_60_percentile": historical_win_rate_60_percentile,
        "historical_avg_ret_60": historical_avg_ret_60,
        "historical_avg_mdd_60": historical_avg_mdd_60,
        "historical_avg_ret_60_percentile": historical_avg_ret_60_percentile,
        "momentum_percentile": momentum_percentile,
        "drawdown_buffer_ratio": drawdown_buffer_ratio,
        "entry_advice": entry_advice,
    }


def build_asset_price_context(
    asset_code_norm: str | None,
    prices: pd.DataFrame,
    spot_prices: dict[str, float],
    raw_closes: dict[str, float],
    session_label: str,
) -> dict[str, float | None]:
    current_price = None
    previous_close_price = None
    intraday_price_return = None
    current_price_from_spot = False
    if asset_code_norm is not None and asset_code_norm in prices.columns:
        latest_price = spot_prices.get(asset_code_norm)
        history = prices[asset_code_norm].dropna()
        latest_history_date = pd.Timestamp(history.index[-1]).normalize() if not history.empty else None
        has_same_day_snapshot = latest_history_date is not None and latest_history_date == pd.Timestamp(prices.index[-1]).normalize()

        if pd.notna(latest_price):
            current_price = float(latest_price)
            current_price_from_spot = True
        elif session_label == "收盘后" and not history.empty:
            current_price = float(history.iloc[-1])

        prev_close = raw_closes.get(asset_code_norm)
        if pd.notna(prev_close):
            previous_close_price = float(prev_close)
        elif not current_price_from_spot and current_price is not None and has_same_day_snapshot and len(history) >= 2:
            previous_close_price = float(history.iloc[-2])
        elif not current_price_from_spot and not history.empty:
            previous_close_price = float(history.iloc[-1])

        if current_price is not None and previous_close_price not in (None, 0.0):
            intraday_price_return = current_price / previous_close_price - 1
    return {
        "current_price": current_price,
        "previous_close_price": previous_close_price,
        "intraday_price_return": intraday_price_return,
    }


def estimate_live_nav(
    close_result: pd.DataFrame,
    close_nav: float | None,
    spot_prices: dict[str, float],
    raw_closes: dict[str, float],
) -> tuple[float | None, float | None]:
    if close_nav is None:
        return None, None
    if close_result.empty:
        return close_nav, 1.0
    latest_idx = close_result.index[-1]
    weight_cols = [col for col in close_result.columns if str(col).startswith("weight_")]
    if not weight_cols:
        return confirmed_nav, None

    invested_return = 0.0
    covered_weight = 0.0
    for col in weight_cols:
        code = str(col).removeprefix("weight_")
        weight = close_result.loc[latest_idx, col]
        if pd.isna(weight):
            continue
        weight = float(weight)
        if abs(weight) < 1e-12:
            continue
        prev_close = raw_closes.get(code)
        current_price = spot_prices.get(code)
        if prev_close in (None, 0.0) or current_price is None or pd.isna(current_price):
            continue
        intraday_ret = float(current_price) / float(prev_close) - 1.0
        invested_return += weight * intraday_ret
        covered_weight += weight

    total_weight = float(close_result.loc[latest_idx, weight_cols].fillna(0.0).sum())
    if total_weight <= 1e-12:
        return close_nav, 1.0
    live_nav = close_nav * (1.0 + invested_return)
    coverage = min(covered_weight / total_weight, 1.0) if total_weight > 0 else 1.0
    return live_nav, coverage


def classify_market_session(now: datetime) -> str:
    minutes = now.hour * 60 + now.minute + now.second / 60.0
    morning_start = 9 * 60 + 30
    morning_end = 11 * 60 + 30
    afternoon_start = 13 * 60
    afternoon_end = 15 * 60
    if minutes < morning_start:
        return "开盘前"
    if morning_start <= minutes < morning_end:
        return "盘中"
    if morning_end <= minutes < afternoon_start:
        return "午间休市"
    if afternoon_start <= minutes < afternoon_end:
        return "盘中"
    return "收盘后"


def should_include_realtime_snapshot(session_label: str) -> bool:
    return session_label in {"盘中", "午间休市"}


def build_market_volume_proxy_for_monitor(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | str | None]]:
    try:
        from .goal_optimization_common import load_market_volume_proxy
    except ImportError:
        from goal_optimization_common import load_market_volume_proxy

    years = max(int(math.ceil((prices.index.max() - prices.index.min()).days / 365.25)) + 1, 15)
    proxy = load_market_volume_proxy(years=years, refresh=False).copy()
    context: dict[str, float | str | None] = {
        "mode": "上一交易日收盘量能",
        "progress": None,
        "market_amount_ratio_20_60": None,
        "market_amount_ratio_5_20": None,
        "market_breadth_proxy": None,
    }
    if proxy.empty:
        return proxy, context

    latest_proxy = proxy.reindex(prices.index).ffill().iloc[-1]
    for key in ("market_amount_ratio_20_60", "market_amount_ratio_5_20", "market_breadth_proxy"):
        value = latest_proxy.get(key)
        context[key] = float(value) if pd.notna(value) else None
    return proxy, context
