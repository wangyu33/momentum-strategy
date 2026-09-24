#!/usr/bin/env python3
"""回撤开仓点分析：在什么回撤深度买入，历史期望最划算？

背景：ETF 动量策略的净值不是单调向上的，历史上出现过多次 -12% ~ -17% 的回撤。
新资金什么时候进场，是实盘最实际的问题之一。

方法
----
1. 逐日口径：把每个交易日按「当前距历史新高的回撤深度」分桶，统计之后
   21 / 63 / 126 / 252 个交易日的收益（均值、中位数、胜率、最差 10%）。
   缺点：日频样本高度重叠，只反映「平均而言」的差异。
2. 事件口径：每次回撤事件只取**最深的那一天**作为唯一观测，消除重叠，
   样本量小但独立性强。两种口径结论一致才采信。

输出
----
- output/analysis/entry_point_by_drawdown.csv
- output/analysis/entry_point_by_drawdown.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from runtime_env import configure_matplotlib_env, prepare_local_imports  # noqa: E402

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAV_PATH = ROOT / "output" / "core" / "backtest_nav.csv"
OUT_DIR = ROOT / "output" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 回撤分桶（负数，单位 %）：上界开、下界闭
BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 13), (13, 100)]
HORIZONS = [21, 63, 126, 252]


def bin_label(lo, hi):
    if hi >= 100:
        return f"> -{lo}%"
    if hi - lo <= 1:
        return f"-{lo}%~-{hi}%"
    return f"-{lo}%~-{hi}%"


def load_nav() -> pd.DataFrame:
    df = pd.read_csv(NAV_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()
    nav = df["nav"].astype(float)
    df["peak"] = nav.cummax()
    df["dd"] = nav / df["peak"] - 1.0
    return df


def forward_stats(nav: pd.Series, dd: pd.Series) -> pd.DataFrame:
    """逐日口径：按回撤深度分桶，统计各持有期的未来收益。"""
    rows = []
    for lo, hi in BINS:
        m = (dd * -100 >= lo) & (dd * -100 < hi)
        idx = dd.index[m]
        rec = {"回撤区间": bin_label(lo, hi), "样本天数": int(m.sum()),
               "占比": float(m.mean())}
        for h in HORIZONS:
            fwd = []
            for t in idx:
                pos = nav.index.get_loc(t)
                if pos + h < len(nav):
                    fwd.append(nav.iloc[pos + h] / nav.iloc[pos] - 1.0)
            a = np.array(fwd)
            if len(a) < 5:
                rec[f"{h}日均"] = np.nan
                rec[f"{h}日中位"] = np.nan
                rec[f"{h}日胜率"] = np.nan
                rec[f"{h}日最差10%"] = np.nan
                continue
            rec[f"{h}日均"] = float(a.mean())
            rec[f"{h}日中位"] = float(np.median(a))
            rec[f"{h}日胜率"] = float((a > 0).mean())
            rec[f"{h}日最差10%"] = float(np.percentile(a, 10))
            # 买入后的最大浮亏（MAE）：持有期内最低点相对买入点的跌幅
            mae = []
            for t in idx:
                pos = nav.index.get_loc(t)
                seg = nav.iloc[pos + 1: pos + h + 1]
                if len(seg):
                    mae.append(seg.min() / nav.iloc[pos] - 1.0)
            m = np.array(mae)
            if len(m):
                rec[f"{h}日最大浮亏中位"] = float(np.median(m))
                rec[f"{h}日最大浮亏P90"] = float(np.percentile(m, 90))
        rows.append(rec)
    return pd.DataFrame(rows)


def event_stats(nav: pd.Series, dd: pd.Series) -> pd.DataFrame:
    """事件口径：每次回撤只在最深点取一个观测。"""
    # 识别回撤事件：从 dd==0 开始，到重新回到 dd==0 结束
    events = []
    in_dd = False
    start = None
    best_i = None
    for i, (t, v) in enumerate(dd.items()):
        if not in_dd and v < -1e-9:
            in_dd, start, best_i = True, i, i
        elif in_dd:
            if v < dd.iloc[best_i]:
                best_i = i
            if v >= -1e-9:          # 回到新高，事件结束
                events.append((dd.index[start], dd.index[best_i], dd.index[i],
                               dd.iloc[best_i]))
                in_dd = False
    if in_dd:                       # 结束时仍在回撤中
        events.append((dd.index[start], dd.index[best_i], None, dd.iloc[best_i]))

    rows = []
    for lo, hi in BINS:
        sel = [e for e in events if lo <= -e[3] * 100 < hi]
        if not sel:
            continue
        rec = {"回撤区间": bin_label(lo, hi), "事件数": len(sel)}
        for h in HORIZONS:
            fwd = []
            for _, trough, _, _ in sel:
                pos = nav.index.get_loc(trough)
                if pos + h < len(nav):
                    fwd.append(nav.iloc[pos + h] / nav.iloc[pos] - 1.0)
            a = np.array(fwd)
            if len(a) == 0:
                rec[f"{h}日均"] = np.nan
                rec[f"{h}日胜率"] = np.nan
                continue
            rec[f"{h}日均"] = float(a.mean())
            rec[f"{h}日中位"] = float(np.median(a))
            rec[f"{h}日胜率"] = float((a > 0).mean())
        rows.append(rec)
    return pd.DataFrame(rows), events


def recovery_stats(nav: pd.Series, dd: pd.Series) -> pd.DataFrame:
    """从各回撤深度买入，需要多久才能看到净值创新高（中位数 / 最差）。"""
    rows = []
    for lo, hi in BINS:
        m = (dd * -100 >= lo) & (dd * -100 < hi)
        idx = dd.index[m]
        waits = []
        for t in idx:
            pos = nav.index.get_loc(t)
            peak = nav.iloc[:pos + 1].max()
            future = nav.iloc[pos + 1:]
            hit = future[future >= peak]
            waits.append(len(future[:hit.index[0]]) if len(hit) else np.nan)
        a = np.array(waits, dtype=float)
        rows.append({
            "回撤区间": bin_label(lo, hi), "样本天数": int(m.sum()),
            "中位等待天数": float(np.nanmedian(a)) if len(a) else np.nan,
            "75分位等待天数": float(np.nanpercentile(a, 75)) if len(a) else np.nan,
            "1年内回本比例": float(np.isfinite(a).sum() and
                                 (a <= 252).sum() / len(a)) if len(a) else np.nan,
        })
    return pd.DataFrame(rows)


def entry_schedule_sim(nav: pd.Series, dd: pd.Series,
                       hold_days: int = 756,
                       cash_rate: float = 0.02,
                       step: int = 5) -> pd.DataFrame:
    """分批建仓规则模拟：把资金等分成若干份，各自等到指定回撤深度再投入。

    为什么必须做这一步：只看「深跌后未来收益高」会得出错误结论——
    「等到 -10% 再买」可能一年只有几天能成交，资金长期闲置的机会成本
    往往吃掉择时的收益。这里把**等待成本**显式计入（未投入部分按 cash_rate 计息）。

    做法：以每个起点为假想入场日，向后看 hold_days 个交易日，
    找出各份资金的成交日（第一个达到指定回撤深度且晚于上一笔的日子），
    用「份额 = 1/n ÷ 买入日净值」精确计算终值。
    """
    n = len(nav)
    nav_v = nav.values
    dd_v = (-dd.values) * 100.0          # 正数表示回撤百分比
    starts = list(range(0, n - hold_days, step))

    plans = {
        "立即全仓": [0.0],
        "-3%全仓": [3.0],
        "-5%全仓": [5.0],
        "-8%全仓": [8.0],
        "分3批 -3/-5/-7": [3.0, 5.0, 7.0],
        "分3批 -2/-4/-6": [2.0, 4.0, 6.0],
        "分3批 -4/-7/-10": [4.0, 7.0, 10.0],
        "分3批 -5/-8/-12": [5.0, 8.0, 12.0],
        "分4批 -2/-4/-7/-11": [2.0, 4.0, 7.0, 11.0],
        "定投(每季1/12)": None,           # 特殊：不看回撤，按时间分批
    }

    rows = []
    for name, levels in plans.items():
        finals, invested_ratio, unfilled = [], [], []
        for i0 in starts:
            end = min(i0 + hold_days, n - 1)
            horizon = end - i0
            if horizon < 60:
                continue
            if levels is None:
                # 定投：每 63 个交易日投一份，共 12 份
                slots = [i0 + k * 63 for k in range(12) if i0 + k * 63 <= end]
                k = len(slots) or 1
                val = 0.0
                for s in slots:
                    val += (1.0 / k) * nav_v[end] / nav_v[s]
                    invested_ratio.append(1.0)
                unfilled.append(0)
                finals.append(val)
                continue

            k = len(levels)
            each = 1.0 / k
            val = 0.0
            filled = 0
            cursor = i0
            for lv in levels:
                seg = dd_v[cursor:end + 1]
                hit = np.nonzero(seg >= lv)[0]
                if len(hit):
                    buy = cursor + int(hit[0])
                    val += each * nav_v[end] / nav_v[buy] \
                        * (1 + cash_rate) ** ((end - buy) / 252.0)
                    filled += 1
                    cursor = buy + 1
                    invested_ratio.append(1.0)
                else:
                    # 一直没等到：这笔钱全程闲置
                    val += each * (1 + cash_rate) ** (horizon / 252.0)
                    unfilled.append(1)
                    invested_ratio.append(0.0)
            finals.append(val)
        a = np.array(finals)
        # 年化（按 hold_days 折算）
        yrs = hold_days / 252.0
        ann = a ** (1 / yrs) - 1.0
        rows.append({
            "建仓方案": name,
            "样本起点数": len(a),
            "终值中位": float(np.median(a)),
            "年化中位": float(np.median(ann)),
            "年化均值": float(ann.mean()),
            "跑输立即全仓比例": float((ann < 0).mean()),   # 占位，稍后用基准比较
            "平均成交比例": float(np.mean(invested_ratio)) if invested_ratio else np.nan,
        })
    out = pd.DataFrame(rows)
    base = out.loc[out["建仓方案"] == "立即全仓", "年化中位"].iloc[0]
    out["vs立即全仓(pp)"] = (out["年化中位"] - base) * 100
    out = out.drop(columns=["跑输立即全仓比例"])
    return out


def main() -> None:
    df = load_nav()
    nav, dd = df["nav"], df["dd"]
    print(f"净值区间 {nav.index[0].date()} ~ {nav.index[-1].date()}，{len(nav)} 个交易日")
    print(f"当前净值 {nav.iloc[-1]:.4f}，当前回撤 {dd.iloc[-1]*100:.2f}%")
    print(f"历史最大回撤 {dd.min()*100:.2f}%\n")

    daily = forward_stats(nav, dd)
    ev_df, events = event_stats(nav, dd)
    rec = recovery_stats(nav, dd)

    print("=== 逐日口径：不同回撤深度买入后的未来收益 ===")
    cols = ["回撤区间", "样本天数"] + [f"{h}日均" for h in HORIZONS]
    d = daily[cols].copy()
    for c in cols[2:]:
        d[c] = d[c].apply(lambda v: f"{v*100:>7.2f}%" if pd.notna(v) else "      -")
    print(d.to_string(index=False))

    print("\n=== 逐日口径：胜率（持有期收益 > 0 的比例）===")
    cols2 = ["回撤区间", "样本天数"] + [f"{h}日胜率" for h in HORIZONS]
    d2 = daily[cols2].copy()
    for c in cols2[2:]:
        d2[c] = d2[c].apply(lambda v: f"{v:>7.0%}" if pd.notna(v) else "      -")
    print(d2.to_string(index=False))

    print("\n=== 买入后的最大浮亏（持有期内最糟糕时点，负数=被套幅度）===")
    cols3 = ["回撤区间", "样本天数"] + [f"{h}日最大浮亏中位" for h in HORIZONS] \
        + [f"{h}日最大浮亏P90" for h in HORIZONS]
    d3 = daily[cols3].copy()
    for c in cols3[2:]:
        d3[c] = d3[c].apply(lambda v: f"{v*100:>7.2f}%" if pd.notna(v) else "      -")
    print(d3.to_string(index=False))

    print("\n=== 持有 1 年的建仓模拟（窗口更短，等待成本更明显）===")
    s1 = entry_schedule_sim(nav, dd, hold_days=252)
    s1.to_csv(OUT_DIR / "entry_schedule_sim_1y.csv", index=False,
              encoding="utf-8-sig")
    v = s1.copy()
    for c in ("年化中位", "年化均值"):
        v[c] = v[c].apply(lambda x: f"{x*100:>7.2f}%")
    v["vs立即全仓(pp)"] = v["vs立即全仓(pp)"].apply(lambda x: f"{x:+.2f}")
    v["平均成交比例"] = v["平均成交比例"].apply(
        lambda x: f"{x:.0%}" if pd.notna(x) else "-")
    print(v[["建仓方案", "年化中位", "年化均值", "平均成交比例",
             "vs立即全仓(pp)"]].to_string(index=False))

    print("\n=== 事件口径（每次回撤只取最深点，样本独立）===")
    for c in ev_df.columns[2:]:
        ev_df[c] = ev_df[c].apply(lambda v: f"{v*100:>7.2f}%" if pd.notna(v) else "-")
    print(ev_df.to_string(index=False))

    print("\n=== 买入后多久创新高 ===")
    r = rec.copy()
    r["中位等待天数"] = r["中位等待天数"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "-")
    r["75分位等待天数"] = r["75分位等待天数"].apply(
        lambda v: f"{v:.0f}" if pd.notna(v) else "-")
    r["1年内回本比例"] = r["1年内回本比例"].apply(
        lambda v: f"{v:.0%}" if pd.notna(v) else "-")
    print(r.to_string(index=False))

    # 回撤事件深度分布
    depths = np.array([-e[3] * 100 for e in events])
    print(f"\n回撤事件数 {len(events)}；深度分布：")
    for th in (2, 5, 8, 10, 13):
        print(f"  深于 {th}%: {(depths >= th).sum()} 次 "
              f"（{(depths >= th).mean():.0%}）")

    daily.to_csv(OUT_DIR / "entry_point_by_drawdown.csv", index=False,
                 encoding="utf-8-sig")
    (OUT_DIR / "entry_point_events.csv").write_text(
        pd.DataFrame([{"开始": e[0].date(), "最深日": e[1].date(),
                       "修复日": e[2].date() if e[2] else None,
                       "深度%": round(-e[3] * 100, 2)} for e in events]
                     ).to_csv(index=False, encoding="utf-8-sig"),
        encoding="utf-8-sig")
    rec.to_csv(OUT_DIR / "entry_point_recovery.csv", index=False,
               encoding="utf-8-sig")

    # ---- 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.plot(nav.index, dd * 100, color="#c53030", linewidth=1.2)
    ax.fill_between(nav.index, dd * 100, 0, color="#c53030", alpha=0.15)
    ax.set_title("Historical drawdown (%)", loc="left", fontweight="bold")
    ax.axhline(dd.min() * 100, color="#2b6cb0", linestyle="--", linewidth=1.0,
               label=f"max {dd.min()*100:.1f}%")
    ax.legend()

    ax = axes[0, 1]
    x = np.arange(len(daily))
    w = 0.2
    for i, h in enumerate(HORIZONS):
        ax.bar(x + (i - 1.5) * w, daily[f"{h}日均"] * 100, width=w,
               label=f"{h}d")
    ax.set_xticks(x)
    ax.set_xticklabels(daily["回撤区间"], rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Avg forward return by entry drawdown (%)", loc="left",
                 fontweight="bold")
    ax.legend()

    ax = axes[1, 0]
    for i, h in enumerate(HORIZONS):
        ax.bar(x + (i - 1.5) * w, daily[f"{h}日胜率"] * 100, width=w,
               label=f"{h}d")
    ax.set_xticks(x)
    ax.set_xticklabels(daily["回撤区间"], rotation=30, ha="right")
    ax.axhline(50, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Win rate by entry drawdown (%)", loc="left", fontweight="bold")
    ax.legend()

    ax = axes[1, 1]
    ax.hist(depths, bins=np.arange(0, 18, 1), color="#2b6cb0", alpha=0.85)
    ax.set_title("Drawdown episode depth distribution", loc="left",
                 fontweight="bold")
    ax.set_xlabel("depth (%)")
    ax.set_ylabel("count")

    fig.tight_layout()
    p = OUT_DIR / "entry_point_by_drawdown.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n已写入 {OUT_DIR}")
    print(f"  图：{p}")

    # ---- 分批建仓模拟 ----
    print("\n=== 分批建仓模拟（每笔资金等分，等到指定回撤再投；"
          "未投入按年化 2% 计息，持有 3 年）===")
    sched = entry_schedule_sim(nav, dd)
    sched.to_csv(OUT_DIR / "entry_schedule_sim.csv", index=False,
                 encoding="utf-8-sig")
    s = sched.copy()
    for c in ("终值中位",):
        s[c] = s[c].apply(lambda v: f"{v:.3f}")
    for c in ("年化中位", "年化均值"):
        s[c] = s[c].apply(lambda v: f"{v*100:>6.2f}%")
    s["平均成交比例"] = s["平均成交比例"].apply(
        lambda v: f"{v:.0%}" if pd.notna(v) else "-")
    s["vs立即全仓(pp)"] = s["vs立即全仓(pp)"].apply(lambda v: f"{v:+.2f}")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
