"""诊断：为什么「便宜」在 A 股不赚钱？

第二轮的结论是低 PB 五分位非单调（P1 9.8% ... P4 12.6%），PB-ROE 残差甚至反向
（低残差组最差）。一个自然的怀疑是：**低估值组合被银行/地产/钢铁填满了**——
它们 PB 低、ROE 高，但市场给的折价是理性的（杠杆高、资产质量存疑、增长停滞）。

本脚本回答三件事：
1. 低估值组到底装了些什么（行业关键词占比、市值中位、ROE 中位、杠杆中位）
2. 剔除银行 / 非银 / 地产 / 钢铁 后，价值因子的表现是否改善
3. 同样的剔除对 SUE 有没有影响（对照组：确认不是「剔除什么都变好」）

行业分类用股票名称关键词近似（无申万行业数据），这是本项目的已知局限之一。
"""

from __future__ import annotations

import argparse
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

# 名称关键词 -> 行业标签（粗分类，仅用于诊断）
SECTOR_KW = {
    "银行": ["银行"],
    "非银": ["证券", "保险", "信托", "期货", "金控", "国投资本"],
    "地产": ["地产", "置业", "城建", "发展", "招商蛇口", "万科", "保利"],
    "钢铁有色煤炭": ["钢铁", "钢构", "铜业", "铝业", "锌业", "铅", "煤", "矿业",
                     "有色", "稀土", "黄金", "金属"],
    "白酒食品": ["酒", "食品", "乳业", "乳品", "牧原", "温氏", "农业", "糖业"],
    "医药": ["医药", "生物", "制药", "医疗", "健康", "爱尔", "药"],
    "电力公用": ["电力", "水务", "燃气", "环保", "公用", "能源"],
    "建筑建材": ["建筑", "建材", "水泥", "玻璃", "中铁", "交建", "路桥"],
}

FACTORS = {
    "低PB": ("pb", True),
    "低PE": ("pe", True),
    "PB-ROE分位差": (None, None),          # 特殊：需要两列
    "EP剔微利": ("ep_ex", False),
    "SUE": ("sue", False),
    "EBIT/EV": ("ebit_ev", False),
}


def tag_sector(name: pd.Series) -> pd.Series:
    s = name.astype(str)
    out = pd.Series("其他", index=s.index)
    for tag, kws in SECTOR_KW.items():
        m = s.str.contains("|".join(kws), na=False)
        out[m & (out == "其他")] = tag
    return out


def _score(row: pd.DataFrame, kind: str) -> pd.Series:
    col, asc = FACTORS[kind]
    if kind == "PB-ROE分位差":
        pb = pd.to_numeric(row["pb"], errors="coerce")
        roe = pd.to_numeric(row["roe_ttm"], errors="coerce")
        g = pb.rank(pct=True) - roe.rank(pct=True)
        return -g
    v = pd.to_numeric(row[col], errors="coerce")
    pct = v.rank(pct=True, na_option="keep")
    return (1.0 - pct) if asc else pct


def composition(ctx, kind: str, exclude: list[str] | None = None) -> pd.DataFrame:
    """低估值组（分数最高的 TOP_N 只）的行业与风格暴露。"""
    rows = []
    for d in ctx["dates"]:
        try:
            row = ctx["snap"].loc[d]
        except KeyError:
            continue
        mc = pd.to_numeric(row.get("mktcap"), errors="coerce").fillna(0)
        row = row[mc >= MIN_MKTCAP]
        if exclude:
            sec = tag_sector(row.get("name", pd.Series("", index=row.index)))
            row = row[~sec.isin(exclude)]
        if row.empty:
            continue
        s = _score(row, kind).dropna()
        if len(s) < TOP_N:
            continue
        sel = row.loc[s.sort_values(ascending=False).index[:TOP_N]]
        sec = tag_sector(sel.get("name", pd.Series("", index=sel.index)))
        rows.append({
            "date": d,
            "银行%": (sec == "银行").mean(),
            "非银%": (sec == "非银").mean(),
            "地产%": (sec == "地产").mean(),
            "周期%": (sec == "钢铁有色煤炭").mean(),
            "市值中位/全池中位": sel["mktcap"].median() / row["mktcap"].median(),
            "ROE中位": pd.to_numeric(sel.get("roe_ttm"), errors="coerce").median(),
            "PB中位": pd.to_numeric(sel.get("pb"), errors="coerce").median(),
            "负债率中位": pd.to_numeric(sel.get("debt_ratio"), errors="coerce").median(),
        })
    return pd.DataFrame(rows)


def run(ctx, kind: str, exclude: list[str] | None, start: str) -> dict:
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
        if exclude:
            sec = tag_sector(row.get("name", pd.Series("", index=row.index)))
            row = row[~sec.isin(exclude)]
        if row.empty:
            continue
        s = _score(row, kind).dropna()
        if len(s) < TOP_N:
            continue
        picks[d] = list(s.sort_values(ascending=False).index[:TOP_N])
    if len(picks) < 2:
        return {"年化": np.nan, "回撤": np.nan, "Calmar": np.nan}
    res = BT.run_portfolio(ctx["close_hfq"], picks, cfg.BUY_COST_RATE,
                           cfg.SELL_COST_RATE, start=start,
                           initial_nav=cfg.INITIAL_NAV)
    st = M.summarize(res["nav"])
    return {"年化": st["年化收益"], "回撤": st["最大回撤"], "Calmar": st["Calmar"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="Q")
    args = ap.parse_args()

    ctx = build_context(TOP_N, args.freq, cfg.START_DATE, None)
    print(f"{len(ctx['dates'])} 个调仓日\n")

    print("=== 1. 各因子 TOP30 组合的成分构成（全期平均）===")
    print(f"{'因子':<14}{'银行%':>7}{'非银%':>7}{'地产%':>7}{'周期%':>7}"
          f"{'市值比':>8}{'ROE中位':>9}{'PB中位':>8}{'负债率':>8}")
    for kind in FACTORS:
        c = composition(ctx, kind)
        if c.empty:
            print(f"{kind:<14}{'无有效样本':>20}")
            continue
        print(f"{kind:<14}{c['银行%'].mean():>7.1%}{c['非银%'].mean():>7.1%}"
              f"{c['地产%'].mean():>7.1%}{c['周期%'].mean():>7.1%}"
              f"{c['市值中位/全池中位'].mean():>8.2f}"
              f"{c['ROE中位'].mean():>9.1f}{c['PB中位'].mean():>8.2f}"
              f"{c['负债率中位'].mean():>8.1f}")

    print("\n=== 2. 剔除特定行业后的 TOP30 表现（年化 / 回撤）===")
    variants = [
        ("全池", None),
        ("剔银行", ["银行"]),
        ("剔金融(银行+非银)", ["银行", "非银"]),
        ("剔金融地产", ["银行", "非银", "地产"]),
        ("剔金融地产周期", ["银行", "非银", "地产", "钢铁有色煤炭"]),
    ]
    print(f"{'因子':<14}" + "".join(f"{v[0]:>18}" for v in variants))
    rows = []
    for kind in FACTORS:
        cells = []
        rec = {"因子": kind}
        for label, ex in variants:
            r = run(ctx, kind, ex, cfg.START_DATE)
            cells.append(f"{r['年化']*100:>8.2f}/{r['回撤']*100:>7.1f}")
            rec[label] = r["年化"]
            rec[label + "_回撤"] = r["回撤"]
        print(f"{kind:<14}" + "".join(f"{c:>18}" for c in cells))
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(cfg.REPORT_DIR / "diagnose_value_trap.csv", index=False,
               encoding="utf-8-sig")
    print(f"\n已写入 {cfg.REPORT_DIR / 'diagnose_value_trap.csv'}")


if __name__ == "__main__":
    main()
