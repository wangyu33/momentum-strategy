"""回测引擎。

执行纪律
--------
- 调仓日 T 收盘后用已生效财报选股，**T+1 收盘价成交**，杜绝用当日收盘价自成交。
- 收益用**后复权价**计算（等价于分红再投资）。
- 持有期内份额固定，市值随价格漂移；调仓时按「双边换手 × 对应费率」扣成本。
- 停牌 / 退市：价格按前值延续（价值冻结），不参与当次换手。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_rebalance_dates(
    index: pd.DatetimeIndex,
    months: tuple[int, ...] = (5,),
    start: str | None = None,
    end: str | None = None,
) -> list[pd.Timestamp]:
    """取每年指定月份的**第一个交易日**作为调仓日。"""
    idx = index
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx <= pd.Timestamp(end)]
    out = []
    for year, g in pd.Series(idx).groupby(idx.year):
        for m in months:
            sub = g[g.dt.month == m]
            if len(sub):
                out.append(pd.Timestamp(sub.iloc[0]))
    return sorted(set(out))


def _target_weights(codes: list[str], max_weight: float | None = None) -> pd.Series:
    if not codes:
        return pd.Series(dtype=float)
    w = pd.Series(1.0 / len(codes), index=pd.Index(codes, name="code"))
    if max_weight is not None and w.max() > max_weight:
        # 超出上限的部分按其余权重比例摊派
        for _ in range(20):
            over = w > max_weight + 1e-12
            if not over.any():
                break
            excess = float((w[over] - max_weight).sum())
            w[over] = max_weight
            rest = ~over
            if not rest.any() or w[rest].sum() <= 0:
                break
            w[rest] = w[rest] + excess * w[rest] / w[rest].sum()
        w = w / w.sum()
    return w


def run_portfolio(
    close_hfq: pd.DataFrame,
    picks: dict[pd.Timestamp, list[str]],
    buy_cost: float,
    sell_cost: float,
    max_weight: float | None = None,
    start: str | None = None,
    end: str | None = None,
    initial_nav: float = 1.0,
) -> dict:
    """按调仓指令回测。picks: 调仓日 -> 选中的股票代码列表。"""
    px = close_hfq.copy()
    if start:
        px = px[px.index >= pd.Timestamp(start)]
    if end:
        px = px[px.index <= pd.Timestamp(end)]
    px = px.sort_index()
    px_ff = px.ffill()                      # 停牌沿用前价

    dates = px.index
    rebs = {pd.Timestamp(k): v for k, v in picks.items()}

    cols = px.columns
    shares = pd.Series(0.0, index=cols)
    cash = float(initial_nav)          # 建仓前以现金形式持有
    nav_list, hold_list, turn_list, cost_list = [], [], [], []
    pending: list[str] | None = None
    current: list[str] = []

    def portfolio_value(p: pd.Series) -> float:
        v = float((shares * p).sum())
        return cash + (0.0 if np.isnan(v) else v)

    for i, t in enumerate(dates):
        p = px_ff.loc[t]
        # 1) 执行上一交易日收盘后确定的调仓（T+1 成交）
        cost = 0.0
        turn = 0.0
        if pending is not None:
            val = portfolio_value(p)
            w_new = _target_weights(pending, max_weight)
            if len(w_new):
                hold_val = float((shares * p).fillna(0).sum())
                w_old = (shares * p).fillna(0) / hold_val if hold_val > 0 \
                    else pd.Series(0.0, index=cols)
                w_full = w_new.reindex(cols).fillna(0.0)
                turn = float((w_full - w_old.reindex(cols).fillna(0.0)).abs().sum())
                cost = val * turn / 2.0 * (buy_cost + sell_cost)
                after = max(val - cost, 0.0)
                shares = (w_full * after / p).fillna(0.0)
                cash = 0.0
            current = list(pending)
            pending = None

        # 2) 记录当日净值
        nav_list.append(portfolio_value(p))
        hold_list.append(shares.copy())
        turn_list.append(turn)
        cost_list.append(cost)

        # 3) 调仓日：收盘后决策，下一交易日执行
        if t in rebs:
            pending = rebs[t]

    nav = pd.Series(nav_list, index=dates, name="nav")
    holdings = pd.DataFrame(hold_list, index=dates)
    # 份额 -> 权重（用于换手与集中度分析）
    val_series = (holdings * px_ff).sum(axis=1).replace(0, np.nan)
    weights = holdings.div(val_series, axis=0).fillna(0.0)

    return {
        "nav": nav,
        "holdings": holdings,
        "weights": weights,
        "turnover": pd.Series(turn_list, index=dates),
        "cost": pd.Series(cost_list, index=dates),
        "picks": rebs,
    }


def run_equal_weight_pool(close_hfq: pd.DataFrame, start=None, end=None,
                          buy_cost=0.0, sell_cost=0.0) -> pd.Series:
    """池内等权持有（不作为策略，只作为「池子本身」的参照）。"""
    px = close_hfq.ffill()
    if start:
        px = px[px.index >= pd.Timestamp(start)]
    if end:
        px = px[px.index <= pd.Timestamp(end)]
    ret = px.pct_change(fill_method=None)
    port = ret.mean(axis=1, skipna=True).fillna(0.0)
    return (1 + port).cumprod()
