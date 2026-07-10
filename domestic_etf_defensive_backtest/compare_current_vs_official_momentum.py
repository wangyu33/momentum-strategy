#!/usr/bin/env python3
"""对比当前独立策略与正式动量策略的核心指标。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from momentum_backtest.run_backtest import (
    build_strategy_summary,
    configure_matplotlib,
    save_figure_atomic,
)

OUTPUT_DIR = Path("domestic_etf_defensive_backtest/output/research/current_vs_official")


def load_result_bundle(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = pd.read_csv(base_dir / "backtest_nav.csv", parse_dates=["date"]).set_index("date")
    trades = pd.read_csv(base_dir / "trades.csv")
    selected = pd.read_csv(base_dir / "selected_etfs.csv")
    return result, trades, selected


def format_metric(metric: str, value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if metric in {"annualized_return", "annualized_volatility", "max_drawdown", "avg_exposure"}:
        return f"{float(value):.2%}"
    if metric in {"sharpe_rf0", "max_drawdown_integral", "max_drawdown_episode_integral"}:
        return f"{float(value):.3f}"
    if metric in {"trade_count", "trade_action_count"}:
        return str(int(value))
    return f"{float(value):.4f}"


def normalize_for_score(metric: str, series: pd.Series) -> pd.Series:
    values = series.astype(float)
    if metric in {"annualized_return", "sharpe_rf0"}:
        min_v, max_v = values.min(), values.max()
        if abs(max_v - min_v) < 1e-12:
            return pd.Series(1.0, index=series.index)
        return (values - min_v) / (max_v - min_v)
    # lower is better
    min_v, max_v = values.min(), values.max()
    if abs(max_v - min_v) < 1e-12:
        return pd.Series(1.0, index=series.index)
    return (max_v - values) / (max_v - min_v)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    official_dir = Path("momentum_backtest/output/core")
    current_dir = Path("domestic_etf_defensive_backtest/output/base/core")

    official_result, official_trades, official_selected = load_result_bundle(official_dir)
    current_result, current_trades, current_selected = load_result_bundle(current_dir)

    official_summary = build_strategy_summary(official_result, official_trades, selected=official_selected, include_max_drawdown_integral=True)
    current_summary = build_strategy_summary(current_result, current_trades, selected=current_selected, include_max_drawdown_integral=True)

    metrics = [
        "annualized_return",
        "annualized_volatility",
        "sharpe_rf0",
        "max_drawdown",
        "max_drawdown_integral",
        "max_drawdown_episode_integral",
        "trade_count",
        "avg_exposure",
    ]

    compare_df = pd.DataFrame(
        {
            "metric": metrics,
            "official_momentum": [official_summary[m] for m in metrics],
            "current_strategy": [current_summary[m] for m in metrics],
        }
    )
    compare_df["official_display"] = [format_metric(m, official_summary[m]) for m in metrics]
    compare_df["current_display"] = [format_metric(m, current_summary[m]) for m in metrics]
    compare_df.to_csv(OUTPUT_DIR / "core_metrics_compare.csv", index=False)

    overlap_start = max(official_result.index.min(), current_result.index.min())
    overlap_end = min(official_result.index.max(), current_result.index.max())
    official_nav = official_result.loc[(official_result.index >= overlap_start) & (official_result.index <= overlap_end), "nav"]
    current_nav = current_result.loc[(current_result.index >= overlap_start) & (current_result.index <= overlap_end), "nav"]

    score_df = compare_df.copy()
    score_df["official_score"] = 0.0
    score_df["current_score"] = 0.0
    for metric in metrics:
        normalized = normalize_for_score(metric, score_df.loc[score_df["metric"] == metric, ["official_momentum", "current_strategy"]].iloc[0])
        score_df.loc[score_df["metric"] == metric, "official_score"] = float(normalized["official_momentum"])
        score_df.loc[score_df["metric"] == metric, "current_score"] = float(normalized["current_strategy"])

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 4])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(official_nav.index, official_nav, linewidth=2.0, label="正式动量策略")
    ax1.plot(current_nav.index, current_nav, linewidth=2.0, label="当前独立策略(base)")
    ax1.set_title("当前独立策略 vs 正式动量策略：净值对比", loc="left", fontsize=16, fontweight="bold")
    ax1.set_ylabel("净值")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    y = range(len(metrics))
    bar_h = 0.35
    ax2.barh([i - bar_h / 2 for i in y], score_df["official_score"], height=bar_h, label="正式动量策略", color="#2b6cb0")
    ax2.barh([i + bar_h / 2 for i in y], score_df["current_score"], height=bar_h, label="当前独立策略(base)", color="#d97706")
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(metrics)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("归一化评分（收益/Sharpe 越高越好，其余指标越低越好）")
    ax2.set_title("Core 指标同图对比", loc="left", fontsize=16, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, axis="x", alpha=0.3)

    for i, row in score_df.iterrows():
        ax2.text(min(row["official_score"] + 0.02, 1.02), i - bar_h / 2, row["official_display"], va="center", fontsize=9)
        ax2.text(min(row["current_score"] + 0.02, 1.02), i + bar_h / 2, row["current_display"], va="center", fontsize=9)

    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "core_metrics_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    notes = [
        "# 当前独立策略 vs 正式动量策略",
        "",
        f"对比区间：{overlap_start.date().isoformat()} ~ {overlap_end.date().isoformat()}",
        "",
        "核心指标：",
    ]
    for _, row in compare_df.iterrows():
        notes.append(f"- {row['metric']}: 正式动量 {row['official_display']} | 当前策略 {row['current_display']}")
    (OUTPUT_DIR / "README.md").write_text("\n".join(notes), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
