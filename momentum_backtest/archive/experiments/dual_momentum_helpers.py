"""Shared helpers for archived dual momentum experiments."""

from __future__ import annotations

from typing import Mapping, Union

import pandas as pd

from archive_data_loaders import load_core_cached_data
from archive_strategy_common import annualized_return, max_drawdown


RISK_CODES = ["510300", "159949", "159941", "513650", "513880"]
DEFENSIVE_CODES = ["511580", "518880", "512890"]
SummaryValue = Union[float, int, str]


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
    return load_core_cached_data()


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


def summarize_variant_result(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    include_window_dates: bool = False,
    extra_fields: Mapping[str, SummaryValue] | None = None,
) -> dict[str, SummaryValue]:
    """为 exposure 类 archive 脚本补统一摘要包装。"""
    summary: dict[str, SummaryValue] = summarize(result, trades, selected)
    if include_window_dates:
        summary["start_date"] = result.index[0].date().isoformat()
        summary["end_date"] = result.index[-1].date().isoformat()
    if extra_fields:
        summary.update(dict(extra_fields))
    return summary


def run_exposure_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    signal: pd.Series,
    target_exposure: pd.Series,
    momentum: pd.DataFrame | None,
    fee_rate: float,
    slippage_rate: float,
    *,
    current_momentum: pd.Series | None = None,
    normalize_signal_series: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if normalize_signal_series:
        signal = signal.map(normalize_code)

    returns = prices.pct_change()
    holding = signal.shift(1)
    exposure = target_exposure.shift(1).fillna(0.0).rename("exposure")
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)

    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    current_momentum_series = (
        current_momentum.reindex(prices.index).astype("float64").rename("current_momentum")
        if current_momentum is not None
        else pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    )
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")
    per_side_cost = fee_rate + slippage_rate

    for dt_idx in prices.index:
        asset = holding.loc[dt_idx]
        prev_asset = prev_holding.loc[dt_idx]
        weight = float(exposure.loc[dt_idx]) if pd.notna(exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0

        gross_ret = 0.0
        if pd.notna(asset) and asset in returns.columns and pd.notna(returns.loc[dt_idx, asset]):
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

        if current_momentum is None and momentum is not None:
            winner = signal.loc[dt_idx]
            if pd.notna(winner) and winner in momentum.columns and pd.notna(momentum.loc[dt_idx, winner]):
                current_momentum_series.loc[dt_idx] = float(momentum.loc[dt_idx, winner])

    nav = (1 + strategy_ret.fillna(0.0)).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1

    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": current_momentum_series,
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
