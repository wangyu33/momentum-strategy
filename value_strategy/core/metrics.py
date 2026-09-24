"""绩效指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE = 0.02


def max_drawdown(nav: pd.Series) -> dict:
    """最大回撤及区间。"""
    nav = nav.dropna()
    if nav.empty:
        return {"mdd": np.nan, "peak": None, "trough": None, "recover": None, "span": 0}
    peak = nav.cummax()
    dd = nav / peak - 1.0
    trough = dd.idxmin()
    mdd = float(dd.min())
    peak_date = nav.loc[:trough].idxmax()
    after = nav.loc[trough:]
    rec = after[after >= nav.loc[peak_date]]
    recover = rec.index[0] if len(rec) else None
    span = (recover - peak_date).days if recover is not None else (nav.index[-1] - peak_date).days
    return {"mdd": mdd, "peak": peak_date, "trough": trough, "recover": recover, "span": span}


def annualized_return(nav: pd.Series) -> float:
    nav = nav.dropna()
    if len(nav) < 2:
        return np.nan
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)


def summarize(nav: pd.Series, bench: pd.Series | None = None,
              turnover: pd.Series | None = None) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return {}

    cagr = annualized_return(nav)
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    years = (nav.index[-1] - nav.index[0]).days / 365.25

    ret = nav.pct_change().dropna()
    vol = float(ret.std() * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else np.nan
    sharpe = (cagr - RISK_FREE) / vol if vol and vol > 0 else np.nan

    dd = max_drawdown(nav)
    calmar = cagr / abs(dd["mdd"]) if dd["mdd"] and dd["mdd"] < 0 else np.nan

    out = {
        "累计收益": total,
        "年化收益": cagr,
        "年化波动": vol,
        "夏普": sharpe,
        "最大回撤": dd["mdd"],
        "Calmar": calmar,
        "回撤区间": f"{dd['peak'].date()} ~ {dd['recover'].date() if dd['recover'] is not None else '未修复'}",
        "回撤历时(日)": dd["span"],
        "年数": years,
        "期末净值": float(nav.iloc[-1]),
    }

    if turnover is not None:
        # 年度单边换手合计（每次调仓双边换手的一半）
        out["年均换手"] = float(turnover.sum() / 2 / years) if years > 0 else np.nan

    if bench is not None:
        b = bench.reindex(nav.index).ffill().dropna()
        common = nav.index.intersection(b.index)
        if len(common) > 2:
            n2, b2 = nav.loc[common], b.loc[common]
            bcagr = annualized_return(b2)
            out["基准年化"] = bcagr
            out["超额年化"] = cagr - bcagr
            # 信息比率
            er = (n2.pct_change() - b2.pct_change()).dropna()
            te = float(er.std() * np.sqrt(TRADING_DAYS))
            out["跟踪误差"] = te
            out["信息比率"] = float(er.mean() * TRADING_DAYS / te) if te > 0 else np.nan
    return out


def yearly_returns(nav: pd.Series, bench: pd.Series | None = None) -> pd.DataFrame:
    nav = nav.dropna()
    y = nav.resample("YE").last()
    r = y.pct_change()
    r.iloc[0] = y.iloc[0] / nav.iloc[0] - 1
    out = pd.DataFrame({"策略": r})
    if bench is not None:
        b = bench.reindex(nav.index).ffill().dropna().resample("YE").last()
        br = b.pct_change()
        br.iloc[0] = b.iloc[0] / bench.dropna().iloc[0] - 1
        out["基准"] = br
        out["超额"] = out["策略"] - out["基准"]
    return out
