"""第二轮因子分层检验：PB-ROE 残差 / 市赚率 / GP/A / 应计 / 资产增长。

核心问题：这些「公开经典」因子在 A 股中证800 里到底有没有单调性？
对照组是已知的 PB 五分位（实测非单调：P1 9.8% ... P4 12.6%）。

关键设计：每期在截面内重算五分位，等权 30 只，年度调仓，T+1 成交。
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


def _resid_series(row: pd.DataFrame) -> pd.Series:
    """ln(PB) 对 ROE 的截面回归残差（越小 = 相对自身盈利能力越便宜）。"""
    pb = pd.to_numeric(row.get("pb"), errors="coerce")
    roe = pd.to_numeric(row.get("roe_ttm"), errors="coerce")
    out = pd.Series(np.nan, index=row.index)
    m = (pb > 0) & roe.notna()
    if m.sum() < 20:
        return out
    X = np.column_stack([np.ones(int(m.sum())), roe[m].values])
    beta, *_ = np.linalg.lstsq(X, np.log(pb[m]).values, rcond=None)
    out[m] = np.log(pb[m]) - (beta[0] + beta[1] * roe[m])
    return out


def _goodness(row: pd.DataFrame, kind: str) -> pd.Series:
    """返回「越大越好」的截面分数。"""
    if kind == "pb_roe_resid":
        return -_resid_series(row)                      # 残差越小越好 → 取负
    col, asc = {
        "pb":        ("pb", True),
        "pr":        ("pr", True),
        "pr_adj":    ("pr_adj", True),
        "gpa":       ("gp_to_assets", False),
        "accruals":  ("accruals", True),
        "asset_growth": ("asset_growth", True),
    }[kind]
    v = pd.to_numeric(row.get(col), errors="coerce")
    pct = v.rank(pct=True, na_option="keep")
    return (1.0 - pct) if asc else pct


def quintile_navs(ctx, kind: str, start: str, end: str | None = None,
                  top_n: int = TOP_N) -> list[float]:
    """按因子分五档回测，返回 5 个年化收益（第 1 档 = 理论最优）。"""
    cagr = []
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
            # r 是「降序百分位排名」，r 越小 = g 越大 = 该档内越靠近理论最优的一侧
            r = g.rank(ascending=False, pct=True, na_option="keep")
            sel = r[(r >= lo) & (r < hi)]
            if len(sel) == 0:
                continue
            picks[d] = list(sel.sort_values(ascending=True).index)[:top_n]
        if len(picks) < 2:
            cagr.append(np.nan)
            continue
        res = BT.run_portfolio(ctx["close_hfq"], picks, cfg.BUY_COST_RATE,
                               cfg.SELL_COST_RATE, start=start, end=end,
                               initial_nav=cfg.INITIAL_NAV)
        cagr.append(M.annualized_return(res["nav"]))
    return cagr


def main() -> None:
    ctx = build_context(TOP_N, "Y", cfg.START_DATE, None)

    kinds = [
        ("pb", "PB（对照组）"),
        ("pb_roe_resid", "PB-ROE 残差"),
        ("pr", "市赚率 PR"),
        ("pr_adj", "修正市赚率 N×PR"),
        ("gpa", "GP/A（Novy-Marx）"),
        ("accruals", "应计（Sloan）"),
        ("asset_growth", "资产增长（Cooper）"),
    ]
    spans = [("2008-05-01", "全期 2008 起"),
             ("2017-05-01", "近段 2017 起")]

    rows = []
    for start, label in spans:
        print(f"\n=== {label} —— 五分位年化（第1档=理论最优，第5档=最差）===", flush=True)
        print(f"{'因子':<22}{'P1':>8}{'P2':>8}{'P3':>8}{'P4':>8}{'P5':>8}{'P1-P5':>9}{'单调':>6}")
        for kind, name in kinds:
            c = quintile_navs(ctx, kind, start)
            spread = (c[0] - c[4]) if all(np.isfinite(c)) else np.nan
            mono = _is_monotone(c)
            print(f"{name:<22}" + "".join(f"{x*100:>8.2f}" if np.isfinite(x) else f"{'-':>8}" for x in c)
                  + f"{spread*100:>+9.2f}{mono:>6}")
            rows.append({"区间": label, "因子": name,
                         **{f"P{i+1}": c[i] for i in range(5)},
                         "P1-P5": spread, "单调": mono})

    out = pd.DataFrame(rows)
    p = cfg.REPORT_DIR / "factor_tests_round2.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n已写入 {p}")


def _is_monotone(c: list[float]) -> str:
    v = [x for x in c if np.isfinite(x)]
    if len(v) < 5:
        return "NA"
    d = [v[i + 1] - v[i] for i in range(4)]
    if all(x <= 0.002 for x in d):
        return "是↓"
    if all(x >= -0.002 for x in d):
        return "是↑"
    return "否"


if __name__ == "__main__":
    main()
