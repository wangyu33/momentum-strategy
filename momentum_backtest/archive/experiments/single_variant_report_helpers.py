#!/usr/bin/env python3
"""为单分支历史回测脚本生成完整导出物的共享 helper。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

try:
    from .variant_compare_helpers import (
        build_summary_frame,
        build_yearly_returns_df,
        ensure_output_dir,
        get_summary_mapping,
        print_mapping_summary,
        print_saved_summary,
        save_aux_csv_output,
    )
except ImportError:
    from variant_compare_helpers import (
        build_summary_frame,
        build_yearly_returns_df,
        ensure_output_dir,
        get_summary_mapping,
        print_mapping_summary,
        print_saved_summary,
        save_aux_csv_output,
    )

try:
    from .archive_strategy_common import (
        annualized_return,
        build_benchmark_nav,
        build_contribution_summary,
        max_drawdown,
        save_figure_atomic,
    )
except ImportError:
    from archive_strategy_common import (
        annualized_return,
        build_benchmark_nav,
        build_contribution_summary,
        max_drawdown,
        save_figure_atomic,
    )


def save_yearly_returns(output_dir: Path, result: pd.DataFrame, benchmark_nav: pd.Series) -> None:
    save_aux_csv_output(
        output_dir,
        "yearly_returns.csv",
        build_yearly_returns_df(
            result["nav"],
            benchmark_nav,
            benchmark_return_col="hs300_etf_return",
        ),
        index=False,
    )


def save_single_variant_tables(
    output_dir: Path,
    *,
    summary_df: pd.DataFrame | None = None,
    contribution_df: pd.DataFrame | None = None,
) -> None:
    """统一落盘单策略 archive 报表附加表格。"""
    if summary_df is not None:
        save_aux_csv_output(output_dir, "summary.csv", summary_df, index=False)
    if contribution_df is not None:
        save_aux_csv_output(output_dir, "contribution_summary.csv", contribution_df, index=False)


def save_and_print_single_variant_tables(
    output_dir: Path,
    *,
    summary_df: pd.DataFrame | None = None,
    contribution_df: pd.DataFrame | None = None,
) -> None:
    """统一落盘单策略 archive 附加表格，并在有 summary 时打印摘要。"""
    save_single_variant_tables(
        output_dir,
        summary_df=summary_df,
        contribution_df=contribution_df,
    )
    if summary_df is not None:
        print_saved_summary(summary_df, output_dir=output_dir)


def build_single_variant_summary_frame(
    strategy_name: str,
    summarize_fn,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    extra_fields: dict[str, object] | None = None,
) -> pd.DataFrame:
    """统一构造单策略 archive 报表的摘要表。"""
    summary = summarize_fn(result, trades)
    summary.update(
        {
            "strategy": strategy_name,
            "annualized_return_full": annualized_return(result["nav"]),
            "max_drawdown_full": max_drawdown(result["nav"]),
        }
    )
    if extra_fields:
        summary.update(extra_fields)
    return build_summary_frame([summary])


def save_and_print_single_variant_summary(
    output_dir: Path,
    *,
    strategy_name: str,
    summarize_fn,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    contribution_df: pd.DataFrame | None = None,
    summary_extra_fields: dict[str, object] | None = None,
) -> pd.DataFrame:
    """统一落盘并打印单策略 archive 摘要表。"""
    summary_df = build_single_variant_summary_frame(
        strategy_name,
        summarize_fn,
        result,
        trades,
        extra_fields=summary_extra_fields,
    )
    save_and_print_single_variant_tables(
        output_dir,
        summary_df=summary_df,
        contribution_df=contribution_df,
    )
    return summary_df


def save_and_print_single_variant_report(
    output_dir: Path,
    chart_prefix: str,
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    lookback: int,
    *,
    strategy_name: str | None = None,
    summarize_fn=None,
    summary_df: pd.DataFrame | None = None,
    contribution_df: pd.DataFrame | None = None,
    summary_extra_fields: dict[str, object] | None = None,
    nav_title: str | None = None,
    compare_title: str = "Strategy vs HS300 ETF",
    include_momentum_series: bool = True,
) -> tuple[pd.Series, pd.DataFrame | None]:
    """统一导出单策略 archive 图表、摘要与贡献表。"""
    benchmark_nav = save_outputs(
        output_dir,
        chart_prefix,
        selected,
        prices,
        result,
        trades,
        lookback,
        nav_title=nav_title,
        compare_title=compare_title,
        include_momentum_series=include_momentum_series,
    )
    if contribution_df is None and {"holding", "signal", "turnover", "position"}.issubset(result.columns):
        contribution_df = build_contribution_summary(prices, selected, result)
    if summary_df is None and strategy_name is not None and summarize_fn is not None:
        summary_df = build_single_variant_summary_frame(
            strategy_name,
            summarize_fn,
            result,
            trades,
            extra_fields=summary_extra_fields,
        )
    save_and_print_single_variant_tables(
        output_dir,
        summary_df=summary_df,
        contribution_df=contribution_df,
    )
    return benchmark_nav, summary_df


def save_outputs(
    output_dir: Path,
    chart_prefix: str,
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    lookback: int,
    *,
    nav_title: str | None = None,
    compare_title: str = "Strategy vs HS300 ETF",
    include_momentum_series: bool = True,
) -> pd.Series:
    ensure_output_dir(output_dir)
    prices_to_save = prices.copy()
    prices_to_save.index.name = "date"
    result_to_save = result.copy()
    result_to_save.index.name = "date"
    save_aux_csv_output(output_dir, "selected_etfs.csv", selected, index=False)
    save_aux_csv_output(output_dir, "prices.csv", prices_to_save, index=True)
    save_aux_csv_output(output_dir, "backtest_nav.csv", result_to_save, index=True)
    save_aux_csv_output(output_dir, "trades.csv", trades, index=False)
    save_aux_csv_output(
        output_dir,
        "historical_nav.csv",
        result_to_save[["nav"]].rename(columns={"nav": "historical_nav"}),
        index=True,
    )

    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")
    compare_df = pd.concat([result["nav"], benchmark_nav], axis=1)
    save_aux_csv_output(output_dir, "strategy_vs_hs300.csv", compare_df, index=True)
    save_yearly_returns(output_dir, result, benchmark_nav)

    buy_trades = trades[trades["action"] == "BUY"]
    sell_trades = trades[trades["action"] == "SELL"]

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result.index, result["nav"], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    if not buy_trades.empty:
        ax.scatter(buy_trades["date"], buy_trades["nav"], color="#1f9d55", marker="^", s=55, label="Buy", zorder=3)
    if not sell_trades.empty:
        ax.scatter(sell_trades["date"], sell_trades["nav"], color="#d64545", marker="v", s=55, label="Sell", zorder=3)
    ax.set_title(nav_title or f"{chart_prefix} NAV (lookback={lookback})", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, output_dir / "nav_with_trades.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if include_momentum_series:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(result.index, result["current_momentum"], color="#dd6b20", linewidth=2.0)
        ax.axhline(0.0, color="#4a5568", linestyle="--", linewidth=1.0)
        ax.set_title(f"Signal Momentum (trailing {lookback} trading days)", loc="left", fontsize=16, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Momentum")
        fig.tight_layout()
        save_figure_atomic(fig, output_dir / "momentum_series.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    drawdown = result["drawdown"]
    max_dd_date = drawdown.idxmin()
    max_dd_value = float(drawdown.loc[max_dd_date])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(drawdown.index, drawdown, color="#d64545", linewidth=1.8)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax.scatter([max_dd_date], [max_dd_value], color="#9b2c2c", s=70, zorder=3)
    ax.annotate(
        f"Max DD {max_dd_value:.2%}\n{max_dd_date.date()}",
        xy=(max_dd_date, max_dd_value),
        xytext=(12, -8),
        textcoords="offset points",
        fontsize=10,
        color="#742a2a",
        ha="left",
        va="top",
    )
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("Historical Drawdown vs Running Peak", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    save_figure_atomic(fig, output_dir / "max_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["nav"], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    ax.plot(compare_df.index, compare_df["benchmark_nav"], color="#d64545", linewidth=2.0, label="HS300 ETF")
    ax.set_title(compare_title, loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, output_dir / "strategy_vs_hs300.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return benchmark_nav


def print_summary(strategy_name: str, summarize_fn, result: pd.DataFrame, trades: pd.DataFrame) -> None:
    summary_df = build_single_variant_summary_frame(strategy_name, summarize_fn, result, trades)
    print_mapping_summary(get_summary_mapping(summary_df))
