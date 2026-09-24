"""第四轮候选因子的稳健性矩阵。

扫描只能挑出「历史上最好看」的因子，无法区分 alpha 与偏差。本脚本用正交的
六个维度同时拷问每个候选，任一维度崩溃即判为不可信：

  1. 持仓数 10 / 30 / 80     —— 是否只在极端集中时成立
  2. 成本 0 / 基准 / 3 倍     —— 是否被换手吃掉
  3. 起点 2008 / 2013 / 2017  —— 是否只靠早期样本
  4. 分段 2008-16 / 2017-21 / 2022-26 —— 是否只在某个行情阶段成立
  5. 剔除池内最小市值 0 / 30% / 50% —— 是否只是小市值的马甲（本池小市值有
     严重幸存者偏差，因此这一项对「流动性类」因子是生死线）
  6. 调仓频率 季度 / 月度     —— 月度是更苛刻的实盘口径

判定标准（与第三轮一致）：每个维度都要显著高于随机对照组中位数。
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
from experiments.factor_scan_round4 import FACTORS, BLENDS, signal, name_of  # noqa: E402
from run_backtest import build_context, run_random  # noqa: E402

TOP_N = 30
MIN_MKTCAP = 2e9

CANDIDATES = ["sur", "sue", "sue_neu", "turnover_neu", "illiq", "illiq_neu",
              "sue_turnover", "orth4", "turn_abn", "d_ni_growth",
              "turnover_1m", "small_cap_ref"]


def build_picks(ctx, kind: str, top_n: int, start: str, end: str | None,
                drop_small: float = 0.0) -> dict:
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
        mc = pd.to_numeric(row.get("mktcap"), errors="coerce")
        row = row[mc.fillna(0) >= MIN_MKTCAP]
        if row.empty:
            continue
        if drop_small > 0:
            thr = mc[row.index].quantile(drop_small)
            row = row[mc[row.index].fillna(0) >= thr]
        if len(row) < top_n:
            continue
        s = fn(row).dropna()
        if len(s) < top_n:
            continue
        picks[d] = list(s.sort_values(ascending=False).index[:top_n])
    return picks


def run(ctx, kind: str, top_n: int = TOP_N, costs=(cfg.BUY_COST_RATE, cfg.SELL_COST_RATE),
        start: str = cfg.START_DATE, end: str | None = None,
        drop_small: float = 0.0) -> dict:
    picks = build_picks(ctx, kind, top_n, start, end, drop_small)
    if len(picks) < 2:
        return {"年化": np.nan, "回撤": np.nan}
    res = BT.run_portfolio(ctx["close_hfq"], picks, costs[0], costs[1],
                           start=start, end=end, initial_nav=cfg.INITIAL_NAV)
    st = M.summarize(res["nav"])
    return {"年化": st["年化收益"], "回撤": st["最大回撤"], "Calmar": st["Calmar"]}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    print("构建上下文（季度调仓）...", flush=True)
    ctx = build_context(TOP_N, "Q", cfg.START_DATE, None)

    # 月度上下文单独构建（调仓日不同）
    print("构建月度上下文 ...", flush=True)
    from run_backtest import build_context as _bc
    ctx_m = _bc(TOP_N, "M", cfg.START_DATE, None)

    print("\n随机对照组 ...", flush=True)
    _, rc = run_random(ctx, TOP_N, cfg.START_DATE, None, trials=20)
    rc = np.array(rc)
    med = float(np.median(rc))
    _, rc_m = run_random(ctx_m, TOP_N, cfg.START_DATE, None, trials=10)
    med_m = float(np.median(np.array(rc_m)))
    print(f"  季度随机中位 {med:.2%} / 月度随机中位 {med_m:.2%}", flush=True)

    if args.only:
        kinds = [k for k in kinds if k in args.only.split(",")]
    if "small_cap_ref" in kinds:        # 小市值对照：用现成信号
        FACTORS["small_cap_ref"] = ("mktcap", True, "小市值(对照)")

    rows = []
    for kind in kinds:
        r = {"因子": name_of(kind)}
        r["基准"] = run(ctx, kind)["年化"]
        r["10只"] = run(ctx, kind, top_n=10)["年化"]
        r["80只"] = run(ctx, kind, top_n=80)["年化"]
        r["零成本"] = run(ctx, kind, costs=(0.0, 0.0))["年化"]
        r["3倍成本"] = run(ctx, kind, costs=(cfg.BUY_COST_RATE * 3,
                                             cfg.SELL_COST_RATE * 3))["年化"]
        r["2013起"] = run(ctx, kind, start="2013-05-01")["年化"]
        r["2017起"] = run(ctx, kind, start="2017-05-01")["年化"]
        r["08-16"] = run(ctx, kind, start="2008-05-01", end="2016-12-31")["年化"]
        r["17-21"] = run(ctx, kind, start="2017-01-01", end="2021-12-31")["年化"]
        r["22-26"] = run(ctx, kind, start="2022-01-01", end="2026-09-21")["年化"]
        r["剔最小30%"] = run(ctx, kind, drop_small=0.30)["年化"]
        r["剔最小50%"] = run(ctx, kind, drop_small=0.50)["年化"]
        r["月度"] = run(ctx_m, kind)["年化"]
        r["基准回撤"] = run(ctx, kind)["回撤"]
        rows.append(r)
        print(f"  {name_of(kind):<14} 基准 {r['基准']:>7.2%} | "
              f"剔小30% {r['剔最小30%']:>7.2%} 剔小50% {r['剔最小50%']:>7.2%} | "
              f"22-26 {r['22-26']:>7.2%} | 月度 {r['月度']:>7.2%}", flush=True)

    out = pd.DataFrame(rows)
    p = cfg.REPORT_DIR / "robustness_round4.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")

    print(f"\n季度随机中位 {med:.2%} / 月度随机中位 {med_m:.2%}（年化，%）")
    show = out.copy()
    for c in show.columns[1:]:
        show[c] = show[c].apply(lambda v: f"{v*100:>7.2f}" if pd.notna(v) else "-")
    print(show.to_string(index=False))
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
