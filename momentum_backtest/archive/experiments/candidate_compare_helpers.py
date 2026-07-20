#!/usr/bin/env python3
"""候选池对比类历史脚本共享的摘要、导出与画图 helper。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Union

import argparse
import pandas as pd
from matplotlib import pyplot as plt

try:
    from .archive_data_loaders import (
        build_missing_history_reason,
        dedupe_selected_pool,
        filter_selected_price_columns,
        load_workspace_price_cache,
    )
    from .variant_compare_helpers import (
        append_variant_result,
        build_summary_frame,
        ensure_compare_frame,
        finalize_baseline_diff_summary,
        format_table_text,
        print_saved_summary,
        print_summary_csv,
        save_aux_csv_output,
        save_variant_compare_outputs,
        summarize_variant_result,
    )
except ImportError:
    from archive_data_loaders import (
        build_missing_history_reason,
        dedupe_selected_pool,
        filter_selected_price_columns,
        load_workspace_price_cache,
    )
    from variant_compare_helpers import (
        append_variant_result,
        build_summary_frame,
        ensure_compare_frame,
        finalize_baseline_diff_summary,
        format_table_text,
        print_saved_summary,
        print_summary_csv,
        save_aux_csv_output,
        save_variant_compare_outputs,
        summarize_variant_result,
    )

try:
    from ...candidate_pool_common import run_custom_threshold_dual_with_overheat
    from .archive_strategy_common import (
        fetch_histories,
        save_figure_atomic,
    )
    from ...official_baseline import apply_official_baseline_nav_anchor
    from ...search_utils import try_join_missing_candidate_histories
except ImportError:
    from candidate_pool_common import run_custom_threshold_dual_with_overheat
    from official_baseline import apply_official_baseline_nav_anchor
    from search_utils import try_join_missing_candidate_histories
    from archive_strategy_common import (
        fetch_histories,
        save_figure_atomic,
    )


CandidateSpec = tuple[str, list[dict[str, str]], list[str], list[str]]
SummaryValue = Union[float, int, str]
SummaryFn = Callable[[pd.DataFrame, pd.DataFrame], dict[str, SummaryValue]]
PlotLine = tuple[str, str, float]
SkippedCandidate = dict[str, str]


def summarize_candidate_run(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, SummaryValue]:
    """输出候选池对比脚本共用的基础摘要字段。"""
    return summarize_variant_result(result, trades, include_window_dates=True)


def load_candidate_compare_price_cache(years: int) -> pd.DataFrame:
    """优先合并本地 core/archive 价格缓存，减少 archive 候选池脚本对实时抓数的依赖。"""
    cached = load_workspace_price_cache()
    if cached.empty:
        return cached
    start_ts = cached.index.max() - pd.DateOffset(years=years)
    return cached.loc[cached.index >= start_ts].copy()


def load_candidate_prices(
    selected: pd.DataFrame,
    *,
    years: int,
    cached_prices: pd.DataFrame,
    refresh: bool,
) -> tuple[pd.DataFrame, str | None]:
    """为单个候选池变体准备价格面板；缺少必需历史时返回可读原因。"""
    selected = dedupe_selected_pool(selected)
    if refresh or cached_prices.empty:
        try:
            prices = fetch_histories(selected, years=years)
        except Exception as exc:
            return pd.DataFrame(), str(exc)
    else:
        prices = cached_prices.copy()
        prices, missing_codes, fetch_error = try_join_missing_candidate_histories(
            prices=prices,
            candidates=selected,
            years=years,
            fetch_fn=fetch_histories,
        )
        prices = filter_selected_price_columns(prices, selected)
        if missing_codes:
            return pd.DataFrame(), build_missing_history_reason(missing_codes, fetch_error=fetch_error)

    prices = filter_selected_price_columns(prices, selected)
    prices = prices.dropna(how="any")
    if prices.empty:
        return pd.DataFrame(), "no overlapping non-null price window"
    return prices, None


def format_skipped_variant_details(skipped_variants: Sequence[SkippedCandidate]) -> str:
    """把缺历史跳过明细统一格式化为可读表格文本。"""
    skipped_df = pd.DataFrame(skipped_variants)
    if skipped_df.empty:
        return ""
    return format_table_text(skipped_df, index=False)


def run_candidate_pool_comparison(
    base_pool: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    *,
    years: int,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    name_column: str,
    summarize_fn: SummaryFn = summarize_candidate_run,
    use_official_baseline_anchor: bool = False,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[SkippedCandidate]]:
    """运行一组候选池变体，并输出摘要表与净值对比表。"""
    rows: list[dict[str, object]] = []
    compare_df: pd.DataFrame | None = None
    skipped_variants: list[SkippedCandidate] = []
    cached_prices = pd.DataFrame() if refresh else load_candidate_compare_price_cache(years)

    for name, additions, risk_codes, defensive_codes in candidates:
        selected = pd.concat([base_pool, pd.DataFrame(additions)], ignore_index=True) if additions else base_pool.copy()
        selected = dedupe_selected_pool(selected)
        prices, skip_reason = load_candidate_prices(
            selected,
            years=years,
            cached_prices=cached_prices,
            refresh=refresh,
        )
        if skip_reason is not None:
            skipped_variants.append({name_column: name, "reason": skip_reason})
            continue
        result, trades = run_custom_threshold_dual_with_overheat(
            prices=prices,
            selected=selected,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
            lookback=lookback,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        if use_official_baseline_anchor and name == "base_pool":
            result = apply_official_baseline_nav_anchor(result)
        compare_df = ensure_compare_frame(compare_df, prices, index=result.index)
        append_variant_result(
            rows,
            compare_df,
            None,
            strategy=name,
            result=result,
            summary=summarize_fn(result, trades),
            strategy_field=name_column,
            nav_column=name,
            extra_fields={
                "risk_codes": ",".join([code for code in risk_codes if code in prices.columns]),
                "defensive_codes": ",".join([code for code in defensive_codes if code in prices.columns]),
            },
        )

    if compare_df is None:
        details = format_skipped_variant_details(skipped_variants)
        if details:
            raise RuntimeError(
                "candidate comparison found no variant with usable histories; "
                "try --refresh or populate local price caches first.\n"
                f"{details}"
            )
        raise ValueError("candidate comparison requires at least one candidate with usable histories")

    summary_frame = build_summary_frame(rows)
    if "base_pool" not in summary_frame[name_column].astype(str).tolist():
        skipped_df = pd.DataFrame(skipped_variants)
        base_skip = skipped_df.loc[skipped_df[name_column].astype(str) == "base_pool"]
        if not base_skip.empty:
            details = format_skipped_variant_details(base_skip.to_dict("records"))
            raise RuntimeError(
                "base_pool variant is required but missing usable histories; "
                "try --refresh or populate local price caches first.\n"
                f"{details}"
            )
        raise RuntimeError("base_pool variant is required but missing usable histories")

    summary_df = finalize_baseline_diff_summary(
        rows,
        baseline_field=name_column,
        baseline_value="base_pool",
        metric_mappings=(
            ("total_return", "return_diff"),
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "mdd_diff"),
            ("trade_count", "trade_diff"),
        ),
    )
    return summary_df, compare_df, skipped_variants


def save_candidate_compare_outputs(output_dir: Path, summary_df: pd.DataFrame, compare_df: pd.DataFrame) -> None:
    """统一落盘候选池对比脚本的核心导出物。"""
    save_variant_compare_outputs(output_dir, summary_df, compare_df)


def save_skipped_candidate_variants(
    output_dir: Path,
    skipped_variants: Sequence[SkippedCandidate],
    *,
    name_column: str,
) -> None:
    """记录因缺失历史被跳过的候选池变体，避免 archive 脚本静默少跑。"""
    if skipped_variants:
        save_aux_csv_output(output_dir, "skipped_variants.csv", pd.DataFrame(skipped_variants), index=False)
        return
    empty = pd.DataFrame(columns=[name_column, "reason"])
    save_aux_csv_output(output_dir, "skipped_variants.csv", empty, index=False)


def save_candidate_compare_artifacts(
    output_dir: Path,
    summary_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    skipped_variants: Sequence[SkippedCandidate],
    *,
    name_column: str,
) -> None:
    """统一落盘候选池对比脚本的摘要、净值对比和缺历史记录。"""
    save_candidate_compare_outputs(output_dir, summary_df, compare_df)
    save_skipped_candidate_variants(output_dir, skipped_variants, name_column=name_column)


def filter_available_plot_lines(compare_df: pd.DataFrame, lines: Sequence[PlotLine]) -> list[PlotLine]:
    """按 compare_df 实际存在的列过滤图例配置，支持缺历史时跳过部分变体。"""
    return [line for line in lines if line[0] in compare_df.columns]


def print_candidate_compare_results(
    summary_df: pd.DataFrame,
    skipped_variants: Sequence[SkippedCandidate],
    *,
    output_dir: Path,
) -> None:
    """统一打印候选池对比脚本摘要与缺历史提示。"""
    print_saved_summary(summary_df, output_dir=output_dir)
    if skipped_variants:
        print("skipped_variants=")
        print_summary_csv(pd.DataFrame(skipped_variants))


def save_and_print_candidate_compare_artifacts(
    output_dir: Path,
    summary_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    skipped_variants: Sequence[SkippedCandidate],
    *,
    name_column: str,
) -> None:
    """统一落盘并打印候选池对比脚本的核心结果。"""
    save_candidate_compare_artifacts(
        output_dir,
        summary_df,
        compare_df,
        skipped_variants,
        name_column=name_column,
    )
    print_candidate_compare_results(summary_df, skipped_variants, output_dir=output_dir)


def run_candidate_compare_entrypoint(
    args: argparse.Namespace,
    *,
    base_pool: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    output_dir: Path,
    name_column: str,
    title: str,
    lines: Sequence[PlotLine],
    summarize_fn: SummaryFn = summarize_candidate_run,
    use_official_baseline_anchor: bool = False,
) -> int:
    """候选池 archive 脚本的统一入口：运行、落盘、画图并处理可读错误。"""
    try:
        summary_df, compare_df, skipped_variants = run_candidate_pool_comparison(
            base_pool,
            candidates,
            years=int(args.years),
            lookback=int(args.lookback),
            fee_rate=float(args.fee_rate),
            slippage_rate=float(args.slippage_rate),
            name_column=name_column,
            summarize_fn=summarize_fn,
            use_official_baseline_anchor=use_official_baseline_anchor,
            refresh=bool(getattr(args, "refresh", False)),
        )
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1

    save_and_print_candidate_compare_artifacts(
        output_dir,
        summary_df,
        compare_df,
        skipped_variants,
        name_column=name_column,
    )
    plot_candidate_compare(
        compare_df,
        output_dir / "comparison.png",
        title=title,
        lines=filter_available_plot_lines(compare_df, lines),
    )
    return 0


def plot_candidate_compare(
    compare_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    lines: Sequence[PlotLine],
    benchmark_label: str = "HS300 ETF",
    benchmark_column: str = "hs300_benchmark",
) -> None:
    """按脚本传入的图例配置画净值对比图。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    for column, label, linewidth in lines:
        ax.plot(compare_df.index, compare_df[column], linewidth=linewidth, label=label)
    ax.plot(compare_df.index, compare_df[benchmark_column], linewidth=1.6, linestyle="--", label=benchmark_label)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
