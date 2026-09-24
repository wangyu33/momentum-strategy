"""价值因子的分层检验与稳健性扫描。

回答三个问题：
  Q1 A股中证800 里，低估值本身有没有超额？（PB 分档）
  Q2 在低估值组内，Piotroski F-Score 能不能区分赢家与价值陷阱？（原论文核心）
  Q3 结论对起点、持仓数、调仓频率、是否含金融股是否稳健？
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
from core import factors as F  # noqa: E402
from core import metrics as M  # noqa: E402
from core import strategies as S  # noqa: E402
from run_backtest import build_context, run_random  # noqa: E402


def group_nav(ctx, pick_fn, top_n=30, start=None, end=None):
    """按 pick_fn(row)->codes 构建组合并回测，返回 (stats, nav)。

    注意：必须限制 start，否则净值曲线会从行情起点（1991 年）开始计年数，
    建仓前的空仓期会把年化收益严重稀释。
    """
    if start is None:
        start = cfg.START_DATE
    picks = {}
    for d in ctx["dates"]:
        if d < pd.Timestamp(start):
            continue
        try:
            row = ctx["snap"].loc[d]
        except KeyError:
            continue
        codes = pick_fn(row)
        if codes:
            picks[d] = list(codes)[:top_n]
    if not picks:
        return None, None
    res = BT.run_portfolio(ctx["close_hfq"], picks,
                           buy_cost=cfg.BUY_COST_RATE, sell_cost=cfg.SELL_COST_RATE,
                           start=start, end=end, initial_nav=cfg.INITIAL_NAV)
    nav = res["nav"]
    bench = ctx["bench"].reindex(nav.index).ffill()
    return M.summarize(nav, bench=bench, turnover=res["turnover"]), nav


def q1_pb_decile(ctx, top_n=30):
    print("\n=== Q1：PB 分档（价值效应是否存在）===")
    rows = []
    for lab, lo, hi in (("P1 最低20%", 0.0, 0.20), ("P2", 0.20, 0.40),
                        ("P3", 0.40, 0.60), ("P4", 0.60, 0.80), ("P5 最高20%", 0.80, 1.01)):
        def pick(row, lo=lo, hi=hi):
            # 与策略口径一致：同样要求市值 >= 20 亿
            m = S.base_mask(row, {"min_mktcap": 2e9})
            r = row[m].copy()
            r["_pb"] = pd.to_numeric(r["pb"], errors="coerce")
            r = r[r["_pb"] > 0]
            if r.empty:
                return []
            r["_p"] = r["_pb"].rank(pct=True)
            sel = r[(r["_p"] >= lo) & (r["_p"] < hi)]
            return list(sel.sort_values("_pb").index)[:top_n]
        st, _ = group_nav(ctx, pick, top_n=top_n)
        if st:
            rows.append({"分组": lab, "年化": st["年化收益"], "最大回撤": st["最大回撤"],
                         "Calmar": st["Calmar"], "夏普": st["夏普"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return df


def q2_fscore_in_value(ctx, top_n=30):
    print("\n=== Q2：低 PB 组内按 F-Score 分层（Piotroski 核心检验）===")
    rows = []
    for lab, lo, hi in (("F 0-3（价值陷阱）", 0, 4), ("F 4-6", 4, 7), ("F 7-9（强者）", 7, 10)):
        def pick(row, lo=lo, hi=hi):
            m = S.base_mask(row, {"min_mktcap": 2e9})
            r = row[m].copy()
            r["_pb"] = pd.to_numeric(r["pb"], errors="coerce")
            r = r[r["_pb"] > 0]
            if r.empty:
                return []
            thr = r["_pb"].quantile(0.20)
            r = r[r["_pb"] <= thr]
            r["_fs"] = pd.to_numeric(r["f_score"], errors="coerce")
            sel = r[(r["_fs"] >= lo) & (r["_fs"] < hi)]
            return list(sel.sort_values(["_fs", "_pb"], ascending=[False, True]).index)[:top_n]
        st, _ = group_nav(ctx, pick, top_n=top_n)
        if st:
            rows.append({"分组": lab, "年化": st["年化收益"], "最大回撤": st["最大回撤"],
                         "Calmar": st["Calmar"], "夏普": st["夏普"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(df) >= 2:
        print(f"  F7-9 减 F0-3 年化差：{df['年化'].iloc[-1] - df['年化'].iloc[0]:+.2%}")
    return df


def q3_robustness(ctx, keys=("piotroski_f", "value_composite", "dogs_dividend", "low_pb")):
    print("\n=== Q3：稳健性（起点 / 持仓数 / 频率 / 剔除金融）===")
    base_start = cfg.START_DATE
    res = {}

    print("\n-- 不同起点 --")
    print(f"{'策略':<22}" + "".join(f"{s[:4]:>10}" for s in ("2008-05", "2013-05", "2016-05", "2019-05")))
    for k in keys:
        meta = S.STRATEGIES[k]
        line = f"{meta['name']:<22}"
        for st in ("2008-05-01", "2013-05-01", "2016-05-01", "2019-05-01"):
            picks = {}
            for d in ctx["dates"]:
                if d < pd.Timestamp(st):
                    continue
                row = ctx["snap"].loc[d]
                sel = S.select(meta["fn"](row, meta["params"]), cfg.DEFAULT_TOP_N)
                if sel:
                    picks[d] = sel
            if not picks:
                line += f"{'-':>10}"
                continue
            r = BT.run_portfolio(ctx["close_hfq"], picks, cfg.BUY_COST_RATE,
                                 cfg.SELL_COST_RATE, start=st, initial_nav=cfg.INITIAL_NAV)
            line += f"{M.annualized_return(r['nav']):>10.2%}"
        print(line)

    print("\n-- 不同持仓数（2013 起，排除早期数据稀疏）--")
    print(f"{'策略':<22}" + "".join(f"{n:>10}" for n in (10, 20, 30, 50)))
    for k in keys:
        meta = S.STRATEGIES[k]
        line = f"{meta['name']:<22}"
        for n in (10, 20, 30, 50):
            picks = {}
            for d in ctx["dates"]:
                if d < pd.Timestamp("2013-05-01"):
                    continue
                row = ctx["snap"].loc[d]
                sel = S.select(meta["fn"](row, meta["params"]), n)
                if sel:
                    picks[d] = sel
            r = BT.run_portfolio(ctx["close_hfq"], picks, cfg.BUY_COST_RATE,
                                 cfg.SELL_COST_RATE, start="2013-05-01", initial_nav=cfg.INITIAL_NAV)
            line += f"{M.annualized_return(r['nav']):>10.2%}"
        print(line)

    print("\n-- 剔除金融股（银行/证券/保险，2013 起）--")
    print(f"{'策略':<22}{'含金融':>12}{'剔金融':>12}{'差异':>12}")
    for k in keys:
        meta = S.STRATEGIES[k]
        out = []
        for ex in (False, True):
            params = dict(meta["params"]); params["exclude_financial"] = ex
            picks = {}
            for d in ctx["dates"]:
                if d < pd.Timestamp("2013-05-01"):
                    continue
                row = ctx["snap"].loc[d]
                sel = S.select(meta["fn"](row, params), cfg.DEFAULT_TOP_N)
                if sel:
                    picks[d] = sel
            if not picks:
                out.append(np.nan); continue
            r = BT.run_portfolio(ctx["close_hfq"], picks, cfg.BUY_COST_RATE,
                                 cfg.SELL_COST_RATE, start="2013-05-01", initial_nav=cfg.INITIAL_NAV)
            out.append(M.annualized_return(r["nav"]))
        diff = out[1] - out[0] if all(pd.notna(x) for x in out) else np.nan
        print(f"{meta['name']:<22}{out[0]:>12.2%}{out[1]:>12.2%}{diff:>+12.2%}")

    print("\n-- 半年调仓 vs 年度调仓（2013 起）--")
    print(f"{'策略':<22}{'年度':>12}{'半年':>12}{'差异':>12}")
    # 半年调仓日不在年度截面里，必须按各自的调仓日重新构建因子截面
    snaps = {}
    for months in ((5,), (5, 11)):
        dates = BT.make_rebalance_dates(ctx["close_hfq"].index, months, start="2013-05-01")
        sn = F.build_snapshots(ctx["fin"], dates, ctx["close_raw"], ctx["share"],
                               ctx["dividends"])
        sn["name"] = sn.index.get_level_values("code").map(
            ctx["pool"].set_index("code")["name"])
        snaps[months] = (dates, sn)

    for k in keys:
        meta = S.STRATEGIES[k]
        out = []
        for months in ((5,), (5, 11)):
            dates, sn = snaps[months]
            picks = {}
            for d in dates:
                try:
                    row = sn.loc[d]
                except KeyError:
                    continue
                sel = S.select(meta["fn"](row, meta["params"]), cfg.DEFAULT_TOP_N)
                if sel:
                    picks[d] = sel
            if not picks:
                out.append(np.nan); continue
            r = BT.run_portfolio(ctx["close_hfq"], picks, cfg.BUY_COST_RATE,
                                 cfg.SELL_COST_RATE, start="2013-05-01",
                                 initial_nav=cfg.INITIAL_NAV)
            out.append(M.annualized_return(r["nav"]))
        diff = out[1] - out[0] if all(pd.notna(x) for x in out) else np.nan
        print(f"{meta['name']:<22}{out[0]:>12.2%}{out[1]:>12.2%}{diff:>+12.2%}")


def main():
    print("构建上下文 ...", flush=True)
    ctx = build_context(cfg.DEFAULT_TOP_N, "Y", cfg.START_DATE, None)
    q1 = q1_pb_decile(ctx)
    q2 = q2_fscore_in_value(ctx)
    q3_robustness(ctx)

    out = cfg.REPORT_DIR / "factor_tests.csv"
    pd.concat([q1.assign(检验="PB分档"), q2.assign(检验="低PB内F-Score分层")]).to_csv(
        out, index=False)
    print(f"\n分层检验结果 -> {out}")


if __name__ == "__main__":
    main()
