#!/usr/bin/env python3
"""候选池实验共用的标的清单与基础策略 helper。"""

from __future__ import annotations

import pandas as pd

from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
    DEFAULT_OVERHEAT_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    build_strategy_summary,
    normalize_code,
)


ETF_588000 = {"theme": "科创50", "code": "588000", "name": "科创50ETF华夏", "sina_symbol": "sh588000"}
ETF_513180 = {"theme": "恒生科技", "code": "513180", "name": "恒生科技指数ETF", "sina_symbol": "sh513180"}
ETF_511260 = {"theme": "10年国债", "code": "511260", "name": "十年国债ETF", "sina_symbol": "sh511260"}
ETF_513030 = {"theme": "德国ETF", "code": "513030", "name": "德国ETF", "sina_symbol": "sh513030"}
ETF_513080 = {"theme": "法国ETF", "code": "513080", "name": "法国CAC40ETF", "sina_symbol": "sh513080"}
ETF_513050 = {"theme": "中概互联", "code": "513050", "name": "中概互联网ETF", "sina_symbol": "sh513050"}
ETF_511090 = {"theme": "30年国债", "code": "511090", "name": "30年国债ETF", "sina_symbol": "sh511090"}
ETF_512480 = {"theme": "半导体", "code": "512480", "name": "半导体ETF", "sina_symbol": "sh512480"}
ETF_515790 = {"theme": "光伏", "code": "515790", "name": "光伏ETF", "sina_symbol": "sh515790"}
ETF_511380 = {"theme": "可转债", "code": "511380", "name": "可转债ETF", "sina_symbol": "sh511380"}
ETF_510500 = {"theme": "中证500", "code": "510500", "name": "中证500ETF", "sina_symbol": "sh510500"}
ETF_512100 = {"theme": "中证1000", "code": "512100", "name": "中证1000ETF", "sina_symbol": "sh512100"}
ETF_510900 = {"theme": "H股", "code": "510900", "name": "H股ETF", "sina_symbol": "sh510900"}
ETF_510050 = {"theme": "上证50", "code": "510050", "name": "上证50ETF", "sina_symbol": "sh510050"}
ETF_510230 = {"theme": "金融", "code": "510230", "name": "金融ETF", "sina_symbol": "sh510230"}
ETF_510880 = {"theme": "红利ETF", "code": "510880", "name": "红利ETF", "sina_symbol": "sh510880"}
ETF_510880_SH_DIVIDEND = {"theme": "上证红利", "code": "510880", "name": "上证红利ETF", "sina_symbol": "sh510880"}
ETF_510410 = {"theme": "资源", "code": "510410", "name": "资源ETF", "sina_symbol": "sh510410"}
ETF_508000 = {"theme": "REITs", "code": "508000", "name": "REITsETF", "sina_symbol": "sh508000"}
ETF_164824 = {"theme": "印度", "code": "164824", "name": "印度基金LOF", "sina_symbol": "sz164824"}
ETF_159930 = {"theme": "能源", "code": "159930", "name": "能源ETF", "sina_symbol": "sz159930"}
ETF_159985 = {"theme": "豆粕", "code": "159985", "name": "豆粕ETF", "sina_symbol": "sz159985"}
ETF_515220 = {"theme": "煤炭", "code": "515220", "name": "煤炭ETF", "sina_symbol": "sh515220"}
ETF_513400 = {"theme": "道琼斯", "code": "513400", "name": "道琼斯ETF鹏华", "sina_symbol": "sh513400"}
ETF_513300 = {"theme": "海外红利", "code": "513300", "name": "海外红利ETF", "sina_symbol": "sh513300"}
ETF_513660 = {"theme": "港股红利低波", "code": "513660", "name": "港股红利低波ETF", "sina_symbol": "sh513660"}

BASE_RISK_CODES = ["510300", "159949", "159941", "513650", "513880"]
BASE_DEFENSIVE_CODES = ["511580", "518880", "512890"]


def run_custom_threshold_dual_with_overheat(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    risk_codes: list[str],
    defensive_codes: list[str],
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """运行候选池实验里共用的阈值双动量 + 过热限仓基线。"""
    momentum = prices / prices.shift(lookback) - 1
    returns = prices.pct_change()
    risk_codes = [code for code in risk_codes if code in prices.columns]
    defensive_codes = [code for code in defensive_codes if code in prices.columns]
    risk_mom = momentum[risk_codes]
    defensive_mom = momentum[defensive_codes]

    risk_winner = risk_mom.idxmax(axis=1, skipna=True)
    risk_best = risk_mom.max(axis=1, skipna=True)
    defensive_winner = defensive_mom.idxmax(axis=1, skipna=True)
    defensive_best = defensive_mom.max(axis=1, skipna=True)

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        r_asset = risk_winner.loc[dt_idx]
        r_score = risk_best.loc[dt_idx]
        d_asset = defensive_winner.loc[dt_idx]
        d_score = defensive_best.loc[dt_idx]

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
        elif pd.notna(r_score) and r_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT
            current_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            current_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")

    base_holding = signal.shift(1)
    base_exposure = target_exposure.shift(1).fillna(0.0)
    base_strategy_ret = pd.Series(0.0, index=prices.index, name="base_strategy_return")
    per_side_cost = fee_rate + slippage_rate

    prev_holding = base_holding.shift(1)
    prev_exposure = base_exposure.shift(1).fillna(0.0)
    for dt_idx in prices.index:
        asset = normalize_code(base_holding.loc[dt_idx])
        prev_asset = normalize_code(prev_holding.loc[dt_idx])
        weight = float(base_exposure.loc[dt_idx]) if pd.notna(base_exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0
        gross_ret = 0.0
        if asset and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = weight * float(returns.loc[dt_idx, asset])
        if not asset and not prev_asset:
            turnover = abs(weight - prev_weight)
        elif asset and prev_asset and asset == prev_asset:
            turnover = abs(weight - prev_weight)
        else:
            turnover = prev_weight + weight
        base_strategy_ret.loc[dt_idx] = (1 + gross_ret) * (1 - turnover * per_side_cost) - 1

    base_nav = (1 + base_strategy_ret.fillna(0.0)).cumprod()
    base_nav.iloc[0] = 1.0
    base_drawdown = base_nav / base_nav.cummax() - 1

    reduce_mask = (
        signal.isin(risk_codes)
        & (target_exposure > DEFAULT_OVERHEAT_MAX_EXPOSURE)
        & (base_drawdown >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_MOMENTUM_CUT)
    )
    capped_exposure = target_exposure.copy()
    capped_exposure.loc[reduce_mask] = DEFAULT_OVERHEAT_MAX_EXPOSURE
    high_reduce_mask = (
        signal.isin(risk_codes)
        & (capped_exposure > DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE)
        & (base_drawdown >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT)
    )
    capped_exposure.loc[high_reduce_mask] = DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE

    holding = signal.shift(1)
    exposure = capped_exposure.shift(1).fillna(0.0).rename("exposure")
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)
    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")

    for dt_idx in prices.index:
        asset = normalize_code(holding.loc[dt_idx])
        prev_asset = normalize_code(prev_holding.loc[dt_idx])
        weight = float(exposure.loc[dt_idx]) if pd.notna(exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0
        gross_ret = 0.0
        if asset and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = weight * float(returns.loc[dt_idx, asset])
        if not asset and not prev_asset:
            day_turnover = abs(weight - prev_weight)
        elif asset and prev_asset and asset == prev_asset:
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
            "current_momentum": current_momentum,
            "signal": signal,
            "holding": holding,
            "exposure": exposure,
            "target_exposure": capped_exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )

    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    trades: list[dict[str, object]] = []
    for dt_idx in prices.index:
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
    return result, pd.DataFrame(trades)


def summarize(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | str]:
    """输出候选池实验共用的基础摘要字段。"""
    return {
        "start_date": result.index[0].date().isoformat(),
        "end_date": result.index[-1].date().isoformat(),
        **build_strategy_summary(result, trades),
    }
