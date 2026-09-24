"""运行全部价值策略回测并输出对比报告。

用法：
    python run_backtest.py                       # 全策略，年度调仓，top30
    python run_backtest.py --top-n 20 --freq H   # 半年调仓，top20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config as cfg  # noqa: E402
from core import backtest as BT  # noqa: E402
from core import data  # noqa: E402
from core import factors as F  # noqa: E402
from core import metrics as M  # noqa: E402
from core import strategies as S  # noqa: E402


FREQ_MONTHS = {
    "Y": (5,),
    "H": (5, 11),
    "Q": (2, 5, 8, 11),
    "M": tuple(range(1, 13)),
}


def build_context(top_n: int, freq: str, start: str, end: str | None):
    print("加载数据 ...", flush=True)
    close_raw = pd.read_pickle(cfg.DATA_DIR / "close_raw.pkl")
    close_hfq = pd.read_pickle(cfg.DATA_DIR / "close_hfq.pkl")
    share_path = cfg.DATA_DIR / "share.pkl"
    share = pd.read_pickle(share_path) if share_path.exists() else None
    vol_path = cfg.DATA_DIR / "volume.pkl"
    volume = pd.read_pickle(vol_path) if vol_path.exists() else None
    bench = pd.read_csv(cfg.DATA_DIR / "benchmark.csv", index_col=0, parse_dates=True).iloc[:, 0]
    pool = data.fetch_pool_codes()
    fin_raw = pd.read_csv(cfg.DATA_DIR / "financials.csv.gz")
    div_path = cfg.DATA_DIR / "dividends.csv.gz"
    dividends = pd.read_csv(div_path) if div_path.exists() else None

    print(f"  池子 {len(pool)} 只 / 财务 {fin_raw['code'].nunique()} 只", flush=True)

    print("整理财务（TTM + 生效日 + F-Score + SUE）...", flush=True)
    fin = F.prepare_financials(fin_raw)
    fin = F.add_f_score(fin)
    fin = F.add_sue(fin)
    fin = F.add_revenue_and_sur(fin)
    fin = F.add_stability(fin)
    print(f"  财务长表 {len(fin)} 行", flush=True)

    print("生成调仓日 ...", flush=True)
    dates = BT.make_rebalance_dates(close_hfq.index, FREQ_MONTHS[freq], start=start, end=end)
    print(f"  {len(dates)} 个调仓日：{dates[0].date()} ~ {dates[-1].date()}", flush=True)

    print("构建因子截面 ...", flush=True)
    snap = F.build_snapshots(fin, dates, close_raw, share, dividends,
                             close_hfq=close_hfq, volume=volume)
    name_map = pool.set_index("code")["name"]
    snap["name"] = snap.index.get_level_values("code").map(name_map)
    print(f"  截面 {len(snap)} 行", flush=True)

    return {
        "close_raw": close_raw, "close_hfq": close_hfq, "bench": bench,
        "pool": pool, "fin": fin, "snap": snap, "dates": dates,
        "share": share, "dividends": dividends, "volume": volume,
    }


def run_one(ctx, key: str, top_n: int, start: str, end: str | None,
            max_weight: float | None, **override) -> dict:
    meta = S.STRATEGIES[key]
    params = dict(meta["params"])
    params.update(override)

    snap = ctx["snap"]
    picks = {}
    diag = []
    for d in ctx["dates"]:
        try:
            row = snap.loc[d]
        except KeyError:
            continue
        score = meta["fn"](row, params)
        n_valid = int(score.notna().sum())
        sel = S.select(score, top_n)
        diag.append({"date": d, "候选数": n_valid, "选中数": len(sel)})
        if sel:
            picks[d] = sel

    if not picks:
        return {"key": key, "name": meta["name"], "ok": False, "diag": pd.DataFrame(diag)}

    res = BT.run_portfolio(
        ctx["close_hfq"], picks,
        buy_cost=cfg.BUY_COST_RATE, sell_cost=cfg.SELL_COST_RATE,
        max_weight=max_weight, start=start, end=end,
        initial_nav=cfg.INITIAL_NAV,
    )
    nav = res["nav"]
    bench = ctx["bench"].reindex(nav.index).ffill()
    bench = bench / bench.iloc[0] * cfg.INITIAL_NAV

    stats = M.summarize(nav, bench=bench, turnover=res["turnover"])
    return {
        "key": key, "name": meta["name"], "ok": True,
        "nav": nav, "bench": bench, "stats": stats,
        "picks": picks, "diag": pd.DataFrame(diag),
        "yearly": M.yearly_returns(nav, bench),
        "weights": res["weights"],
    }


def run_random(ctx, top_n: int, start: str, end: str | None,
               trials: int = 20, seed: int = 42) -> tuple[pd.Series, list[float]]:
    """随机选 top_n 只，重复 trials 次。

    价值策略只持 30 只，而池内等权持 600+ 只，两者不可直接比。
    随机组合提供了「同样集中度下的期望表现」，是判断选股有无 alpha 的正确对照组。
    """
    snap = ctx["snap"]
    rng = np.random.default_rng(seed)
    navs = []
    cagrs = []
    for _ in range(trials):
        picks = {}
        for d in ctx["dates"]:
            try:
                row = snap.loc[d]
            except KeyError:
                continue
            avail = list(row.index[row["close"].notna()])
            if not avail:
                continue
            k = min(top_n, len(avail))
            picks[d] = list(rng.choice(avail, size=k, replace=False))
        if not picks:
            continue
        res = BT.run_portfolio(ctx["close_hfq"], picks,
                               buy_cost=cfg.BUY_COST_RATE, sell_cost=cfg.SELL_COST_RATE,
                               start=start, end=end, initial_nav=cfg.INITIAL_NAV)
        navs.append(res["nav"])
        cagrs.append(M.annualized_return(res["nav"]))

    avg = pd.concat(navs, axis=1).mean(axis=1)
    avg.name = "random_avg"
    return avg, cagrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=cfg.DEFAULT_TOP_N)
    ap.add_argument("--freq", choices=list(FREQ_MONTHS), default="Y")
    ap.add_argument("--start", default=cfg.START_DATE)
    ap.add_argument("--end", default=cfg.END_DATE)
    ap.add_argument("--max-weight", type=float, default=cfg.DEFAULT_MAX_WEIGHT)
    ap.add_argument("--only", default="", help="只跑指定策略，逗号分隔")
    ap.add_argument("--random-trials", type=int, default=20, help="随机对照组重复次数")
    args = ap.parse_args()

    ctx = build_context(args.top_n, args.freq, args.start, args.end)

    keys = list(S.STRATEGIES)
    if args.only:
        keys = [k for k in keys if k in args.only.split(",")]

    print("\n运行策略 ...", flush=True)
    results = []
    for k in keys:
        r = run_one(ctx, k, args.top_n, args.start, args.end, args.max_weight)
        results.append(r)
        if r["ok"]:
            s = r["stats"]
            print(f"  {r['name']:<20} 年化 {s['年化收益']:>7.2%}  回撤 {s['最大回撤']:>7.2%}  "
                  f"Calmar {s['Calmar']:>5.2f}  超额 {s.get('超额年化', float('nan')):>7.2%}", flush=True)
        else:
            print(f"  {r['name']:<20} 无有效持仓", flush=True)

    # 池内等权（按调仓日买入全部可得股票）：与策略同池、同频率、同股息口径，
    # 用于回答「选股本身有没有 alpha」，剔除池子自身的幸存者偏差。
    snap = ctx["snap"]
    pool_picks = {}
    for d in ctx["dates"]:
        try:
            row = snap.loc[d]
        except KeyError:
            continue
        avail = row.index[row["close"].notna()]
        if len(avail):
            pool_picks[d] = list(avail)
    pool_res = BT.run_portfolio(ctx["close_hfq"], pool_picks,
                                buy_cost=cfg.BUY_COST_RATE, sell_cost=cfg.SELL_COST_RATE,
                                start=args.start, end=args.end, initial_nav=cfg.INITIAL_NAV)
    pool_nav = pool_res["nav"]

    print("\n随机对照组（同等集中度）...", flush=True)
    rand_nav, rand_cagrs = run_random(ctx, args.top_n, args.start, args.end,
                                      trials=args.random_trials)
    rc = np.array([c for c in rand_cagrs if pd.notna(c)])
    print(f"  随机{args.top_n}只 × {len(rc)} 次：年化中位数 {np.median(rc):.2%} "
          f"[p10 {np.percentile(rc,10):.2%}, p90 {np.percentile(rc,90):.2%}]", flush=True)
    print("\n各策略在随机分布中的分位：", flush=True)
    for r in results:
        if r["ok"]:
            pct = float((rc < r["stats"]["年化收益"]).mean())
            print(f"  {r['name']:<20} 年化 {r['stats']['年化收益']:>6.2%}  "
                  f"胜过 {pct:>5.0%} 的随机组合", flush=True)

    out = _assemble(ctx, results, pool_nav, args, random={"nav": rand_nav, "cagrs": rc})
    _persist(out, args)
    _print_table(out)


def _assemble(ctx, results, pool_nav, args, random=None):
    rows = []
    navs = {}
    rc = None
    if random is not None:
        rc = random["cagrs"]
        for r in results:
            if r["ok"] and len(rc):
                r["stats"]["胜过随机比例"] = float((rc < r["stats"]["年化收益"]).mean())

    for r in results:
        if not r["ok"]:
            rows.append({"策略": r["name"], "状态": "无有效持仓"})
            continue
        s = r["stats"]
        cand = r["diag"]["候选数"].mean() if len(r["diag"]) else np.nan
        rows.append({
            "策略": r["name"],
            "状态": "OK",
            "平均候选数": round(cand, 1) if pd.notna(cand) else np.nan,
            "胜过随机": s.get("胜过随机比例"),
            "年化收益": s["年化收益"],
            "超额年化": s.get("超额年化"),
            "累计收益": s["累计收益"],
            "最大回撤": s["最大回撤"],
            "Calmar": s["Calmar"],
            "夏普": s["夏普"],
            "年化波动": s["年化波动"],
            "信息比率": s.get("信息比率"),
            "年均换手": s.get("年均换手"),
            "期末净值": s["期末净值"],
        })
        navs[r["name"]] = r["nav"]

    ps = M.summarize(pool_nav, bench=ctx["bench"].reindex(pool_nav.index).ffill())
    rows.append({"策略": "池内等权(同口径参照)", "状态": "参照", "年化收益": ps["年化收益"],
                 "超额年化": ps.get("超额年化"), "累计收益": ps["累计收益"],
                 "最大回撤": ps["最大回撤"], "Calmar": ps["Calmar"], "夏普": ps["夏普"],
                 "年化波动": ps["年化波动"], "期末净值": ps["期末净值"]})
    navs["池内等权(同口径参照)"] = pool_nav

    # 随机对照组：同样集中度下的期望表现
    if random is not None:
        rs = M.summarize(random["nav"],
                         bench=ctx["bench"].reindex(random["nav"].index).ffill())
        # 用「各次随机组合年化的中位数」代表对照组，避免算术平均净值被极端样本拉高
        rs["年化收益"] = float(np.median(rc)) if len(rc) else rs["年化收益"]
        rows.append({"策略": f"随机{args.top_n}只(对照组)", "状态": "对照",
                     "年化收益": rs["年化收益"], "超额年化": rs.get("超额年化"),
                     "累计收益": rs["累计收益"], "最大回撤": rs["最大回撤"],
                     "Calmar": rs["Calmar"], "夏普": rs["夏普"],
                     "年化波动": rs["年化波动"], "期末净值": rs["期末净值"]})
        navs[f"随机{args.top_n}只(对照组)"] = random["nav"]

    bs = M.summarize(ctx["bench"].reindex(pool_nav.index).ffill().dropna())
    rows.append({"策略": f"{cfg.BENCHMARK_NAME}(基准)", "状态": "基准",
                 "年化收益": bs["年化收益"], "累计收益": bs["累计收益"],
                 "最大回撤": bs["最大回撤"], "Calmar": bs["Calmar"], "夏普": bs["夏普"],
                 "年化波动": bs["年化波动"], "期末净值": bs["期末净值"]})
    navs[f"{cfg.BENCHMARK_NAME}(基准)"] = ctx["bench"].reindex(pool_nav.index).ffill()

    return {"table": pd.DataFrame(rows), "navs": navs, "results": results,
            "pool_nav": pool_nav, "random": random}


def _persist(out, args):
    tag = f"{args.freq}_n{args.top_n}"
    tbl = out["table"].copy()
    for c in ("年化收益", "超额年化", "累计收益", "最大回撤", "年化波动"):
        if c in tbl.columns:
            tbl[c] = tbl[c].apply(lambda v: round(v * 100, 2) if pd.notna(v) else np.nan)
    for c in ("Calmar", "夏普", "信息比率", "年均换手", "期末净值"):
        if c in tbl.columns:
            tbl[c] = tbl[c].apply(lambda v: round(v, 3) if pd.notna(v) else np.nan)
    tbl.to_csv(cfg.REPORT_DIR / f"summary_{tag}.csv", index=False)

    navs = pd.DataFrame(out["navs"]).dropna(how="all")
    navs.to_csv(cfg.REPORT_DIR / f"nav_{tag}.csv")

    if out.get("random") is not None:
        pd.Series(out["random"]["cagrs"]).to_csv(
            cfg.REPORT_DIR / f"random_cagrs_{tag}.csv", index=False, header=False)

    for r in out["results"]:
        if not r["ok"]:
            continue
        r["yearly"].to_csv(cfg.REPORT_DIR / f"yearly_{r['key']}_{tag}.csv")
        pd.DataFrame(r["picks"].items() if isinstance(r["picks"], dict) else [],
                     columns=["date", "codes"]).to_csv(
            cfg.REPORT_DIR / f"holdings_{r['key']}_{tag}.csv", index=False)
    print(f"\n报告已写入 {cfg.REPORT_DIR}")


def _print_table(out):
    t = out["table"]
    show = t[["策略", "年化收益", "最大回撤", "Calmar", "夏普", "年均换手",
              "胜过随机", "平均候选数"]].copy()
    if "胜过随机" in show.columns:
        show["胜过随机"] = show["胜过随机"].apply(
            lambda v: f"{v:7.0%}" if pd.notna(v) else "      -")
    for c in ("年化收益", "最大回撤"):
        if c in show.columns:
            show[c] = show[c].apply(lambda v: f"{v:7.2%}" if pd.notna(v) else "     -")
    for c in ("Calmar", "夏普", "年均换手"):
        if c in show.columns:
            show[c] = show[c].apply(lambda v: f"{v:7.2f}" if pd.notna(v) else "     -")
    if "平均候选数" in show.columns:
        show["平均候选数"] = show["平均候选数"].apply(
            lambda v: f"{v:7.1f}" if pd.notna(v) else "      -")
    print("\n" + show.to_string(index=False))


if __name__ == "__main__":
    main()
