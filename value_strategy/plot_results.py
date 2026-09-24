"""生成回测图表。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import config as cfg  # noqa: E402

for f in ("PingFang SC", "Heiti TC", "Arial Unicode MS", "Songti SC"):
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

COLORS = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#d35400", "#16a085",
          "#7f8c8d", "#2c3e50", "#e67e22", "#c2185b", "#00838f"]


def plot_nav(nav_path: Path, out: Path, title: str):
    df = pd.read_csv(nav_path, index_col=0, parse_dates=True).dropna(how="all")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6.2))
    for i, c in enumerate(df.columns):
        s = df[c].dropna()
        lw = 2.2 if "参照" not in c and "基准" not in c and "对照" not in c else 1.5
        ls = "--" if ("参照" in c or "基准" in c or "对照" in c) else "-"
        ax.plot(s.index, s.values, label=c, linewidth=lw, linestyle=ls,
                color=COLORS[i % len(COLORS)], alpha=0.9)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel("净值（对数轴）")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=8.5, ncol=3, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _rand_median(rep: Path, tag: str) -> float:
    """取某调仓频率下随机对照组的年化中位数（%）。"""
    f = rep / f"random_cagrs_{tag}.csv"
    if not f.exists():
        return float("nan")
    return float(np.nanmedian(pd.read_csv(f, header=None).iloc[:, 0].dropna()) * 100)


def plot_round4_scan(csv: Path, rand_med: float, out: Path):
    """第四轮因子扫描：RankIC vs 年化收益散点，气泡大小=回撤深度。"""
    df = pd.read_csv(csv)
    if df.empty or "IC均值" not in df.columns:
        return
    d = df.dropna(subset=["IC均值", "年化"]).copy()
    if d.empty:
        return
    d["年化"] *= 100
    d["回撤"] = d["回撤"].abs() * 100

    fig, ax = plt.subplots(figsize=(11, 6.4))
    beat = d["年化"] >= rand_med
    ax.scatter(d.loc[beat, "IC均值"], d.loc[beat, "年化"],
               s=20 + d.loc[beat, "回撤"] * 2.2, c="#c0392b", alpha=0.75,
               edgecolor="white", linewidth=0.6, label=f"跑赢随机（≥{rand_med:.1f}%）")
    ax.scatter(d.loc[~beat, "IC均值"], d.loc[~beat, "年化"],
               s=20 + d.loc[~beat, "回撤"] * 2.2, c="#7f8c8d", alpha=0.7,
               edgecolor="white", linewidth=0.6, label="未跑赢随机")
    if np.isfinite(rand_med):
        ax.axhline(rand_med, color="#2c3e50", linestyle="--", linewidth=1.3)
        ax.text(ax.get_xlim()[0], rand_med, f" 随机中位 {rand_med:.1f}%",
                va="bottom", fontsize=9)
    for _, r in d.iterrows():
        ax.annotate(str(r["因子"]), (r["IC均值"], r["年化"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=8,
                    color="#34495e")
    ax.set_xlabel("RankIC 均值（因子值与下期收益的秩相关）")
    ax.set_ylabel("Top30 组合年化收益 (%)")
    ax.set_title("第四轮因子扫描：24 个新因子 / 8 个合成因子（中证800，季度调仓，气泡=回撤深度）",
                 fontsize=13, pad=12)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_surprise_nav(nav_csv: Path, out: Path, rand_med: float | None = None):
    """超预期族 vs 对照组净值曲线。"""
    df = pd.read_csv(nav_csv, index_col=0, parse_dates=True).dropna(how="all")
    keys = ["营收超预期 SUR", "双超预期(盈利+营收)", "PEAD 三口径等权",
            "三合一(营收+盈利+低换手)", "超预期+低换手", "PEAD 超预期（SUE）"]
    cols = [c for c in keys if c in df.columns]
    ref = [c for c in df.columns if ("随机" in c or "池内等权" in c or "基准" in c)]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(12, 6.2))
    for i, c in enumerate(cols):
        s = df[c].dropna()
        ax.plot(s.index, s.values, label=c, linewidth=2.0, color=COLORS[i % len(COLORS)])
    for c in ref:
        s = df[c].dropna()
        ax.plot(s.index, s.values, label=c, linewidth=1.4, linestyle="--",
                color="#7f8c8d", alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("超预期族（PEAD/SUR）净值：中证800，季度调仓，等权30只", fontsize=13.5, pad=12)
    ax.set_ylabel("净值（对数轴）")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=9, ncol=2, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_factor_bars(csv: Path, out: Path):
    df = pd.read_csv(csv)
    if df.empty or "检验" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (name, g) in zip(axes, df.groupby("检验")):
        x = np.arange(len(g))
        vals = g["年化"].values * 100
        bars = ax.bar(x, vals, color=["#2980b9" if v < np.mean(vals) else "#c0392b"
                                      for v in vals], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(g["分组"], rotation=18, ha="right", fontsize=9)
        ax.set_title(f"{name}：年化收益(%)", fontsize=12)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    fig.suptitle("价值因子分层检验（中证800，2008-2026，年度调仓，等权30只）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_strategy_vs_random(summary_csv: Path, rand_cagrs: list[float], out: Path):
    df = pd.read_csv(summary_csv)
    if df.empty or "胜过随机" not in df.columns:
        return
    d = df[df["胜过随机"].notna()].copy()
    if d.empty:
        return
    rc = np.array([c for c in rand_cagrs if pd.notna(c)]) * 100
    d = d.sort_values("年化收益", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    # 随机分布作为背景
    ax.hist(rc, bins=min(12, max(4, len(rc) // 2)), orientation="horizontal",
            color="#bdc3c7", alpha=0.55, label=f"随机30只 × {len(rc)} 次")
    med = np.median(rc)
    ax.axhline(med, color="#2c3e50", linestyle="--", linewidth=1.4,
               label=f"随机中位数 {med:.1f}%")

    y = np.arange(len(d))
    colors = ["#c0392b" if v >= med else "#7f8c8d" for v in d["年化收益"]]
    ax.scatter(d["年化收益"], y, s=70, c=colors, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(d["策略"], fontsize=9.5)
    ax.set_xlabel("年化收益 (%)")
    ax.set_title("各价值策略 vs 随机选股分布（同池、同持仓数、同调仓频率）", fontsize=13, pad=12)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    rep = cfg.REPORT_DIR
    nav_files = sorted(rep.glob("nav_*.csv"))
    if nav_files:
        p = nav_files[-1]
        plot_nav(p, rep / "nav_curves.png",
                 "经典价值策略回测：中证800 池，年度调仓，等权30只（2008-05 ~ 2026-09）")
        print("净值图 ->", rep / "nav_curves.png")

    ft = rep / "factor_tests.csv"
    if ft.exists():
        plot_factor_bars(ft, rep / "factor_bars.png")
        print("分层图 ->", rep / "factor_bars.png")

    sm = sorted(rep.glob("summary_*.csv"))
    rc_files = sorted(rep.glob("random_cagrs_*.csv"))
    if sm and rc_files:
        rc = pd.read_csv(rc_files[-1], header=None).iloc[:, 0].dropna().tolist()
        plot_strategy_vs_random(sm[-1], rc, rep / "strategy_vs_random.png")
        print("对照图 ->", rep / "strategy_vs_random.png")

    # 第四轮：因子扫描散点 + 超预期族净值
    r4 = rep / "factor_scan_round4.csv"
    if r4.exists():
        med = _rand_median(rep, "Q_n30")
        plot_round4_scan(r4, med, rep / "factor_scan_round4.png")
        print("第四轮扫描图 ->", rep / "factor_scan_round4.png")

    nav_q = rep / "nav_Q_n30.csv"
    if nav_q.exists():
        plot_surprise_nav(nav_q, rep / "surprise_family_nav.png",
                          _rand_median(rep, "Q_n30"))
        print("超预期族净值图 ->", rep / "surprise_family_nav.png")


if __name__ == "__main__":
    main()
