"""第三轮因子分层检验：反转 / 动量 / 低波动 / 特质波动 / Beta / SUE。

上一轮（价值类因子）的结论是「几乎全部无效且多为反向」，这一轮换到 A 股实证
最强的几类异象上：

- 反转：东北金工(2018) 1M 最强、6M 最弱；开源 全历史多空年化 15.66%
- 低波动：Blitz-Hanauer-van Vliet(2021) A股 最低1/10 跑赢最高1/10 年化 12.9%
- 低 IVOL：中信建投(2018) 后30%等权 年化超额 7.55%，IR 1.62
- 低 Beta：Frazzini-Pedersen(2014) BAB
- SUE：东方证券(2018) RankIC 4.02%，IC_IR 3.49

设计：季度调仓（反转类因子在年度调仓下信号太旧，无意义），每期截面重算五分位，
等权 30 只，T+1 成交，成本与主线一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import backtest as BT  # noqa: E402
from core import config as cfg  # noqa: E402
from core import metrics as M  # noqa: E402
from run_backtest import build_context  # noqa: E402

TOP_N = 30
MIN_MKTCAP = 2e9

# 因子 -> (列名, ascending=True 表示值越小越好)
FACTORS = {
    "reversal_1m":     ("ret_1m", True, "1月反转"),
    "reversal_3m":     ("ret_3m", True, "3月反转"),
    "reversal_6m":     ("ret_6m", True, "6月反转"),
    "momentum_12_1":   ("mom_12_1", False, "12-1 动量（对照）"),
    "low_vol":         ("vol_1y", True, "低波动(1年)"),
    "low_vol_60d":     ("vol_60d", True, "低波动(60日)"),
    "low_ivol":        ("ivol_1y", True, "低特质波动"),
    "low_beta":        ("beta_1y", True, "低Beta"),
    "sue":             ("sue", False, "SUE 超预期"),
}


def _goodness(row: pd.DataFrame, kind: str) -> pd.Series:
    col, asc, _ = FACTORS[kind]
    v = pd.to_numeric(row.get(col), errors="coerce")
    pct = v.rank(pct=True, na_option="keep")
    return (1.0 - pct) if asc else pct


def quintile_navs(ctx, kind: str, start: str, end: str | None = None,
                  top_n: int = TOP_N) -> list[float]:
    """按因子分五档回测，返回 5 个年化收益（第 1 档 = 理论最优）。"""
    out = []
    for i in range(5):
        lo, hi = i / 5, (i + 1) / 5
        picks = {}
        for d in ctx["dates"]:
            if d < pd.Timestamp(start):
                continue
            if end is not None and d > pd.Timestamp(end):
                continue
            try:
                row = ctx["snap"].loc[d]
            except KeyError:
                continue
            mc = pd.to_numeric(row.get("mktcap"), errors="coerce").fillna(0)
            row = row[mc >= MIN_MKTCAP]
            if row.empty:
                continue
            g = _goodness(row, kind).dropna()
            if len(g) < 20:
                continue
            r = g.rank(ascending=False, pct=True, na_option="keep")
            sel = r[(r >= lo) & (r < hi)]
            if len(sel) == 0:
                continue
            picks[d] = list(sel.sort_values(ascending=True).index)[:top_n]
        if len(picks) < 2:
            out.append(np.nan)
            continue
        res = BT.run_portfolio(ctx["close_hfq"], picks, cfg.BUY_COST_RATE,
                               cfg.SELL_COST_RATE, start=start, end=end,
                               initial_nav=cfg.INITIAL_NAV)
        out.append(M.annualized_return(res["nav"]))
    return out


def _is_monotone(c: list[float]) -> str:
    v = [x for x in c if np.isfinite(x)]
    if len(v) < 5:
        return "-"
    d = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    if all(x < -0.002 for x in d):
        return "递减"
    if all(x > 0.002 for x in d):
        return "递增"
    return "非单调"


def main() -> None:
    print("构建上下文（季度调仓）...", flush=True)
    ctx = build_context(TOP_N, "Q", cfg.START_DATE, None)
    print(f"  {len(ctx['dates'])} 个调仓日", flush=True)

    spans = [("2008-05-01", "全期 2008 起"),
             ("2017-05-01", "近段 2017 起")]

    rows = []
    for start, label in spans:
        print(f"\n=== {label} —— 五分位年化（P1=理论最优档，P5=最差档）===", flush=True)
        print(f"{'因子':<20}{'P1':>8}{'P2':>8}{'P3':>8}{'P4':>8}{'P5':>8}{'P1-P5':>9}{'单调':>8}")
        for kind, (_, _, name) in FACTORS.items():
            c = quintile_navs(ctx, kind, start)
            spread = (c[0] - c[4]) if all(np.isfinite(c)) else np.nan
            mono = _is_monotone(c)
            print(f"{name:<20}" + "".join(
                f"{x*100:>8.2f}" if np.isfinite(x) else f"{'-':>8}" for x in c)
                + f"{spread*100:>+9.2f}{mono:>8}", flush=True)
            rows.append({"区间": label, "因子": name,
                         **{f"P{i+1}": c[i] for i in range(5)},
                         "P1-P5": spread, "单调": mono})

    out = pd.DataFrame(rows)
    p = cfg.REPORT_DIR / "factor_tests_round3.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
