"""第四轮因子扫描：换手率 / 流动性 / 现金流 / 盈利稳定性 / 业绩加速度 / SUE 变体。

前三轮的结论分别是：
  1. 经典价值（PE/PB/股息/F-Score/神奇公式…）在中证800 内几乎全部跑不赢随机
  2. PB-ROE / 市赚率 / GP/A 等「增强价值」同样不成立，且分层多为反向
  3. 价格类异象里只有 PEAD/SUE 稳健，低波族是降风险而非 alpha

本轮换一批**此前完全没测过的维度**，并且对每个因子同时给出三个证据：
  - RankIC：因子值与下一持有期收益的秩相关（均值 / IC_IR / 正比例）
  - 五分位分层：是否单调
  - Top30 组合：年化 / 回撤 / Calmar，与随机对照组比较

对与市值高度相关的因子（换手 -0.30、非流动性 -0.77）额外给出**市值中性化**版本，
以区分「真 alpha」与「小市值的另一种说法」。
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
from run_backtest import build_context, run_random  # noqa: E402

TOP_N = 30
MIN_MKTCAP = 2e9

# 因子 -> (列名, ascending=True 表示值越小越好, 中文名)
FACTORS: dict[str, tuple[str, bool, str]] = {
    # ---- 换手率 / 流动性（本轮重点，此前完全未测）----
    "turnover_1m":      ("turnover_1m", True, "低换手(1月)"),
    "turnover_3m":      ("turnover_3m", True, "低换手(3月)"),
    "turn_abn":         ("turn_abn", True, "低异常换手(1月/1年)"),
    "illiq":            ("illiq_1y", False, "高非流动性 Amihud"),
    "turnover_neu":     ("turnover_1m_neu", True, "低换手(市值中性)"),
    "illiq_neu":        ("illiq_1y_neu", False, "非流动性(市值中性)"),
    # ---- 现金流 / 估值 ----
    "cfp_yield":        ("cfp_yield", False, "高现金流收益率 CFP"),
    "ep_ex":            ("ep_ex", False, "EP(剔微利股)"),
    "ebit_ev":          ("ebit_ev", False, "EBIT/EV 近似"),
    # ---- 质量 ----
    "ocf_ni":           ("ocf_ni_ratio", False, "盈利质量 OCF/NI"),
    "roe_std":          ("roe_std12", True, "盈利稳定(低ROE波动)"),
    "prof_ratio":       ("prof_ratio12", False, "盈利持续性"),
    "roe_min":          ("roe_min12", False, "ROE 下限(安全)"),
    # ---- 成长加速度 / 意外 ----
    "d_ni_growth":      ("d_ni_growth", False, "净利增速加速"),
    "d_rev_growth":     ("d_rev_growth", False, "营收增速加速"),
    "sur":              ("sur", False, "营收超预期 SUR"),
    "sue_mean4":        ("sue_mean4", False, "SUE 四期均值"),
    "sue_neu":          ("sue_neu", False, "SUE(市值中性)"),
    # ---- 红利 ----
    "div_years":        ("div_years3", False, "分红连续年数"),
    # ---- 对照 ----
    "sue":              ("sue", False, "SUE 超预期(对照)"),
}

# 合成因子：[(列名, ascending, 权重), ...]
BLENDS: dict[str, tuple[list[tuple[str, bool, float]], str]] = {
    "qmj_proxy": ([("roe_ttm", False, 1.0), ("roe_std12", True, 1.0),
                   ("debt_ratio", True, 1.0), ("beta_1y", True, 1.0)],
                  "QMJ 近似(盈利/稳定/低杠杆/低Beta)"),
    "sue_turnover": ([("sue_neu", False, 1.0), ("turnover_1m_neu", True, 1.0)],
                     "SUE + 低换手"),
    "sue_lowvol":  ([("sue_neu", False, 1.0), ("vol_1y", True, 1.0)],
                    "SUE + 低波动"),
    "orth4":       ([("sue_neu", False, 1.0), ("turnover_1m_neu", True, 1.0),
                     ("vol_1y", True, 1.0), ("ep", False, 1.0)],
                    "正交四因子(SUE/换手/低波/价值)"),
    "sue_sur":     ([("sue_neu", False, 1.0), ("sur", False, 1.0)],
                    "双超预期(盈利+营收)"),
    "sur_turnover": ([("sur", False, 1.0), ("turnover_1m_neu", True, 1.0)],
                     "SUR + 低换手"),
    "sur_sue_turn": ([("sur", False, 1.0), ("sue_neu", False, 1.0),
                      ("turnover_1m_neu", True, 1.0)],
                     "三合一(营收+盈利超预期+低换手)"),
    "pead_pure":   ([("sur", False, 1.0), ("sue_neu", False, 1.0),
                     ("sue", False, 1.0)],
                     "PEAD 三口径等权"),
}


def _score_row(row: pd.DataFrame, col: str, asc: bool) -> pd.Series:
    if col not in row.columns:
        return pd.Series(np.nan, index=row.index)
    v = pd.to_numeric(row[col], errors="coerce")
    pct = v.rank(pct=True, na_option="keep")
    return (1.0 - pct) if asc else pct


def _blend_row(row: pd.DataFrame, parts: list[tuple[str, bool, float]]) -> pd.Series:
    tot_w = 0.0
    acc = pd.Series(0.0, index=row.index)
    ok = pd.Series(True, index=row.index)
    for col, asc, w in parts:
        s = _score_row(row, col, asc)
        ok &= s.notna()
        acc = acc + s.fillna(0.5) * w
        tot_w += w
    return (acc / tot_w).where(ok)


def signal(ctx, kind: str) -> callable:
    """返回 row -> score 的函数（覆盖单因子与合成因子）。"""
    if kind in FACTORS:
        col, asc, _ = FACTORS[kind]
        return lambda row: _score_row(row, col, asc)
    parts, _ = BLENDS[kind]
    return lambda row: _blend_row(row, parts)


def name_of(kind: str) -> str:
    return FACTORS[kind][2] if kind in FACTORS else BLENDS[kind][1]


# ---------------------------------------------------------------- RankIC

def rank_ic(ctx, kind: str) -> dict:
    """因子值 vs 下一持有期收益的秩相关（Spearman）。

    持有期与回测口径一致：调仓日 d 的次日成交，下一调仓日 d' 的次日卖出。
    """
    px = ctx["close_hfq"]
    dates = ctx["dates"]
    fn = signal(ctx, kind)
    ics = []
    for i in range(len(dates) - 1):
        d, dn = dates[i], dates[i + 1]
        try:
            row = ctx["snap"].loc[d]
        except KeyError:
            continue
        mc = pd.to_numeric(row.get("mktcap"), errors="coerce").fillna(0)
        row = row[mc >= MIN_MKTCAP]
        if len(row) < 50:
            continue
        f = fn(row).dropna()
        if len(f) < 50:
            continue
        codes = f.index
        p0 = px.index.searchsorted(d) + 1
        p1 = px.index.searchsorted(dn) + 1
        if p1 >= len(px.index):
            break
        r = px.iloc[p1][codes] / px.iloc[p0][codes] - 1.0
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 50:
            continue
        ic = np.corrcoef(f[common].rank(), r[common].rank())[0, 1]
        if np.isfinite(ic):
            ics.append(ic)
    a = np.array(ics)
    if len(a) < 5:
        return {"IC均值": np.nan, "IC_IR": np.nan, "IC>0占比": np.nan, "期数": len(a)}
    return {"IC均值": float(a.mean()), "IC_IR": float(a.mean() / a.std(ddof=1) * np.sqrt(len(a))),
            "IC>0占比": float((a > 0).mean()), "期数": len(a)}


# ---------------------------------------------------------------- 组合

def top_nav(ctx, kind: str, start: str, end: str | None = None,
            top_n: int = TOP_N) -> dict:
    fn = signal(ctx, kind)
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
        s = fn(row).dropna()
        if len(s) < top_n:
            continue
        picks[d] = list(s.sort_values(ascending=False).index[:top_n])
    if len(picks) < 2:
        return {"年化": np.nan, "回撤": np.nan, "Calmar": np.nan}
    res = BT.run_portfolio(ctx["close_hfq"], picks, cfg.BUY_COST_RATE,
                           cfg.SELL_COST_RATE, start=start, end=end,
                           initial_nav=cfg.INITIAL_NAV)
    st = M.summarize(res["nav"])
    return {"年化": st["年化收益"], "回撤": st["最大回撤"], "Calmar": st["Calmar"],
            "年化波动": st["年化波动"], "夏普": st["夏普"],
            "年均换手": res["turnover"].resample("YE").sum().mean()}


def quintile_navs(ctx, kind: str, start: str, top_n: int = TOP_N) -> list[float]:
    fn = signal(ctx, kind)
    out = []
    for i in range(5):
        lo, hi = i / 5, (i + 1) / 5
        picks = {}
        for d in ctx["dates"]:
            if d < pd.Timestamp(start):
                continue
            try:
                row = ctx["snap"].loc[d]
            except KeyError:
                continue
            mc = pd.to_numeric(row.get("mktcap"), errors="coerce").fillna(0)
            row = row[mc >= MIN_MKTCAP]
            if row.empty:
                continue
            g = fn(row).dropna()
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
                               cfg.SELL_COST_RATE, start=start,
                               initial_nav=cfg.INITIAL_NAV)
        out.append(M.annualized_return(res["nav"]))
    return out


def _mono(c: list[float]) -> str:
    v = [x for x in c if np.isfinite(x)]
    if len(v) < 5:
        return "-"
    d = [v[i + 1] - v[i] for i in range(4)]
    if all(x < -0.002 for x in d):
        return "递减"
    if all(x > 0.002 for x in d):
        return "递增"
    return "非单调"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑指定 key，逗号分隔")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--no-quintile", action="store_true")
    args = ap.parse_args()

    print("构建上下文（季度调仓）...", flush=True)
    ctx = build_context(TOP_N, "Q", cfg.START_DATE, None)
    print(f"  {len(ctx['dates'])} 个调仓日", flush=True)

    print("\n随机对照组 ...", flush=True)
    _, rc = run_random(ctx, TOP_N, cfg.START_DATE, None, trials=args.trials)
    rc = np.array(rc)
    rand_med = float(np.median(rc))
    print(f"  随机{TOP_N}只 × {len(rc)} 次：年化中位 {rand_med:.2%} "
          f"[p10 {np.percentile(rc,10):.2%}, p90 {np.percentile(rc,90):.2%}]", flush=True)

    kinds = list(FACTORS) + list(BLENDS)
    if args.only:
        kinds = [k for k in kinds if k in args.only.split(",")]
    rows = []
    for kind in kinds:
        ic = rank_ic(ctx, kind)
        t = top_nav(ctx, kind, cfg.START_DATE)
        t17 = top_nav(ctx, kind, "2017-05-01")
        if args.no_quintile:
            q = [np.nan] * 5
        else:
            q = quintile_navs(ctx, kind, cfg.START_DATE)
        beat = float((rc < t["年化"]).mean()) if np.isfinite(t["年化"]) else np.nan
        rows.append({
            "因子": name_of(kind), "key": kind,
            "IC均值": ic["IC均值"], "IC_IR": ic["IC_IR"], "IC>0占比": ic["IC>0占比"],
            "年化": t["年化"], "回撤": t["回撤"], "Calmar": t["Calmar"],
            "夏普": t.get("夏普", np.nan), "年均换手": t.get("年均换手", np.nan),
            "胜过随机": beat,
            "2017起年化": t17["年化"],
            **{f"P{i+1}": q[i] for i in range(5)},
            "P1-P5": (q[0] - q[4]) if all(np.isfinite(q)) else np.nan,
            "单调": _mono(q),
        })
        print(f"  {name_of(kind):<24} IC {ic['IC均值']:+.3f} "
              f"(IR {ic['IC_IR']:+.2f})  年化 {t['年化']:>7.2%}  "
              f"回撤 {t['回撤']:>7.2%}  胜过随机 {beat:>4.0%}  "
              f"分层 {q[0]*100:>6.2f}/{q[4]*100:>6.2f} {_mono(q)}", flush=True)

    out = pd.DataFrame(rows).sort_values("年化", ascending=False)
    p = cfg.REPORT_DIR / "factor_scan_round4.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")

    print(f"\n随机{TOP_N}只中位年化 {rand_med:.2%}；按年化排序：", flush=True)
    show = out[["因子", "IC均值", "IC_IR", "年化", "回撤", "Calmar", "胜过随机",
                "2017起年化", "P1-P5", "单调"]].copy()
    for c in ("IC均值", "IC_IR"):
        show[c] = show[c].apply(lambda v: f"{v:+.3f}" if pd.notna(v) else "-")
    for c in ("年化", "回撤", "2017起年化", "P1-P5"):
        show[c] = show[c].apply(lambda v: f"{v*100:>7.2f}" if pd.notna(v) else "-")
    show["Calmar"] = show["Calmar"].apply(lambda v: f"{v:5.2f}" if pd.notna(v) else "-")
    show["胜过随机"] = show["胜过随机"].apply(lambda v: f"{v:>5.0%}" if pd.notna(v) else "-")
    print(show.to_string(index=False))
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
