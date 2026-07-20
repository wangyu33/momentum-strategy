#!/usr/bin/env python3
"""多变体历史对比脚本共享的摘要、diff、导出与画图 helper。

这个 helper 默认服务于 archive 历史研究脚本；是否锚定正式 28.2691 官方基线
应由调用方显式决定，而不是在这里隐式处理。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import pandas as pd
from matplotlib import pyplot as plt

try:
    from ...runtime_env import write_json_atomic
    from .archive_strategy_common import (
        build_benchmark_nav,
        build_yearly_return_rows,
        build_strategy_summary,
        ensure_output_dirs,
        save_figure_atomic,
        write_dataframe_csv_atomic,
    )
    from ...search_utils import (
        load_incremental_notify_candidates,
        notify_best_candidate,
        notify_ranked_incremental_candidate,
        save_best_payload,
    )
except ImportError:
    from runtime_env import write_json_atomic
    from archive_strategy_common import (
        build_benchmark_nav,
        build_strategy_summary,
        build_yearly_return_rows,
        ensure_output_dirs,
        save_figure_atomic,
        write_dataframe_csv_atomic,
    )
    from search_utils import load_incremental_notify_candidates, notify_best_candidate, notify_ranked_incremental_candidate, save_best_payload

try:
    from ...search_utils import sort_notify_candidates
except ImportError:
    from search_utils import sort_notify_candidates


SummaryValue = Union[float, int, str]
PlotLine = tuple[str, str, float]
INTENTIONALLY_UNANCHORED_BASELINE = True
STANDARD_PREVIEW_METRIC_COLUMNS: tuple[str, ...] = (
    "annualized_return",
    "sharpe_rf0",
    "max_drawdown_integral",
)
STANDARD_PREVIEW_DIFF_COLUMNS: tuple[str, ...] = (
    "annualized_diff",
    "sharpe_diff",
    "max_drawdown_integral_diff",
)


def ensure_output_dir(output_dir: Path) -> None:
    """统一保证 archive 导出目录已创建。"""
    ensure_output_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)


def format_table_text(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    head: int | None = None,
    index: bool = False,
) -> str:
    """统一格式化 archive 表格预览文本。"""
    table = df.copy()
    if columns is not None:
        table = table[[col for col in columns if col in table.columns]]
    if head is not None:
        table = table.head(head)
    return table.to_string(index=index)


def format_csv_text(df: pd.DataFrame, *, index: bool = False) -> str:
    """统一格式化 archive CSV 风格输出文本。"""
    return df.to_csv(index=index)


def format_mapping_text(values: Mapping[str, object]) -> str:
    """统一格式化 archive 键值摘要文本。"""
    return pd.Series(dict(values)).to_string()


def print_mapping_summary(values: Mapping[str, object]) -> None:
    """统一打印 archive 键值摘要。"""
    print(format_mapping_text(values))


def summarize_variant_result(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    selected: Optional[pd.DataFrame] = None,
    include_max_drawdown_integral: bool = False,
    include_window_dates: bool = False,
    include_latest_signal_holding: bool = False,
    extra_fields: Optional[Mapping[str, SummaryValue]] = None,
) -> dict[str, SummaryValue]:
    """输出多变体对比脚本共用的基础摘要字段。"""
    summary = build_strategy_summary(
        result,
        trades,
        selected=selected,
        include_max_drawdown_integral=include_max_drawdown_integral,
    )
    if include_window_dates:
        summary["start_date"] = result.index[0].date().isoformat()
        summary["end_date"] = result.index[-1].date().isoformat()
    if include_latest_signal_holding:
        latest = result.iloc[-1]
        summary["latest_signal"] = str(latest.get("signal")) if pd.notna(latest.get("signal")) else ""
        summary["latest_holding"] = str(latest.get("holding")) if pd.notna(latest.get("holding")) else ""
    if extra_fields:
        summary.update(dict(extra_fields))
    return summary


def append_variant_result(
    rows: list[dict[str, object]],
    nav_compare: pd.DataFrame | None,
    descriptions: dict[str, str] | None,
    *,
    strategy: str,
    result: pd.DataFrame | None,
    summary: Mapping[str, object],
    description: str | None = None,
    extra_fields: Optional[Mapping[str, object]] = None,
    strategy_field: str = "strategy",
    nav_column: str | None = None,
) -> dict[str, object]:
    """统一登记 archive 变体结果到 rows/nav_compare/descriptions。"""
    record = dict(summary)
    record[strategy_field] = strategy
    if extra_fields:
        record.update(dict(extra_fields))
    rows.append(record)
    if nav_compare is not None and result is not None:
        target_column = nav_column or f"{strategy}_nav"
        nav_compare[target_column] = result["nav"].reindex(nav_compare.index)
    if descriptions is not None and description is not None:
        descriptions[strategy] = description
    return record


def add_hs300_benchmark(
    compare_df: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    benchmark_code: str = "510300",
    column: str = "hs300_benchmark",
) -> None:
    """统一向 archive 对比面板追加 HS300 基准净值列。"""
    compare_df[column] = build_benchmark_nav(prices, benchmark_code=benchmark_code).reindex(compare_df.index)


def build_compare_frame(
    prices: pd.DataFrame,
    *,
    include_hs300_benchmark: bool = True,
    benchmark_code: str = "510300",
    benchmark_column: str = "hs300_benchmark",
) -> pd.DataFrame:
    """统一构造 archive 净值对比面板，并按需补 HS300 基准列。"""
    compare_df = pd.DataFrame(index=prices.index)
    if include_hs300_benchmark:
        add_hs300_benchmark(
            compare_df,
            prices,
            benchmark_code=benchmark_code,
            column=benchmark_column,
        )
    return compare_df


def ensure_compare_frame(
    compare_df: pd.DataFrame | None,
    prices: pd.DataFrame,
    *,
    index: pd.Index | None = None,
    include_hs300_benchmark: bool = True,
    benchmark_code: str = "510300",
    benchmark_column: str = "hs300_benchmark",
) -> pd.DataFrame:
    """按需懒初始化 archive 对比面板，兼容先拿到结果索引再补基准列的链路。"""
    if compare_df is not None:
        return compare_df
    compare_index = prices.index if index is None else index
    compare_df = pd.DataFrame(index=compare_index)
    if include_hs300_benchmark:
        add_hs300_benchmark(
            compare_df,
            prices,
            benchmark_code=benchmark_code,
            column=benchmark_column,
        )
    return compare_df


def filter_available_plot_lines(
    compare_df: pd.DataFrame,
    plot_lines: Sequence[PlotLine],
) -> list[PlotLine]:
    """按当前净值对比面板列过滤可用 plot lines。"""
    return [line for line in plot_lines if line[0] in compare_df.columns]


def build_baseline_preview_columns(
    prefix_columns: Sequence[str],
    *,
    metric_columns: Sequence[str] = STANDARD_PREVIEW_METRIC_COLUMNS,
    diff_columns: Sequence[str] = STANDARD_PREVIEW_DIFF_COLUMNS,
    include_valid_change: bool = True,
) -> list[str]:
    """统一拼接 baseline preview 常用的字段列顺序。"""
    suffix_columns = ["is_valid_change"] if include_valid_change else []
    return build_preview_columns(
        prefix_columns,
        metric_columns=metric_columns,
        diff_columns=diff_columns,
        suffix_columns=suffix_columns,
    )


def build_preview_columns(
    prefix_columns: Sequence[str],
    *,
    metric_columns: Sequence[str] = (),
    diff_columns: Sequence[str] = (),
    suffix_columns: Sequence[str] = (),
) -> list[str]:
    """统一拼接 archive 预览表列顺序，兼容额外指标或筛选标记。"""
    return [*prefix_columns, *metric_columns, *diff_columns, *suffix_columns]


def build_yearly_returns_df(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    *,
    benchmark_return_col: str = "hs300_return",
    strategy: str | None = None,
) -> pd.DataFrame:
    """统一构造 archive 年度收益对比表。"""
    compare = pd.concat([strategy_nav, benchmark_nav.reindex(strategy_nav.index)], axis=1)
    compare.columns = ["strategy_nav", "benchmark_nav"]
    rows = build_yearly_return_rows(
        compare,
        strategy_col="strategy_nav",
        benchmark_col="benchmark_nav",
        benchmark_return_col=benchmark_return_col,
    )
    yearly_df = build_summary_frame(rows)
    if strategy is not None:
        yearly_df.insert(0, "strategy", strategy)
    return yearly_df


def add_diff_vs_baseline(
    summary_df: pd.DataFrame,
    *,
    baseline_mask: pd.Series,
    metric_mappings: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    """按给定基线行，为摘要表补 diff 列。"""
    baseline_rows = summary_df.loc[baseline_mask]
    if baseline_rows.empty:
        raise ValueError("baseline row is required for diff computation")
    baseline = baseline_rows.iloc[0]
    enriched = summary_df.copy()
    for source_col, diff_col in metric_mappings:
        enriched[diff_col] = enriched[source_col] - baseline[source_col]
    return enriched


def build_typed_summary_fields(
    summary: Mapping[str, object],
    *,
    float_fields: Sequence[str] = (),
    int_fields: Sequence[str] = (),
    str_fields: Sequence[str] = (),
    extra_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """按字段类型统一裁剪 archive 变体摘要，避免脚本里重复手写强转。"""
    record: dict[str, object] = {}
    for field in float_fields:
        record[field] = float(summary[field])
    for field in int_fields:
        record[field] = int(summary[field])
    for field in str_fields:
        record[field] = str(summary[field])
    if extra_fields:
        record.update(dict(extra_fields))
    return record


def build_summary_frame(
    rows: Sequence[Mapping[str, object]] | pd.DataFrame,
    *,
    sort_by: Sequence[str] | None = None,
    ascending: Sequence[bool] | bool = True,
    reset_index: bool = False,
) -> pd.DataFrame:
    """把 rows 统一转成 DataFrame，并按需排序。"""
    summary_frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if sort_by:
        summary_frame = summary_frame.sort_values(list(sort_by), ascending=ascending)
    if reset_index:
        summary_frame = summary_frame.reset_index(drop=True)
    return summary_frame


def get_summary_row(
    summary_df: pd.DataFrame,
    *,
    field: str | None = None,
    value: object | None = None,
    row_index: int = 0,
) -> pd.Series:
    """统一提取 archive 摘要里的单行记录。"""
    target_df = summary_df
    if field is not None:
        target_df = target_df.loc[target_df[field] == value]
    if target_df.empty:
        if field is None:
            raise ValueError("summary row is required")
        raise ValueError(f"summary row not found: {field}={value}")
    return target_df.iloc[row_index]


def get_summary_mapping(
    summary_df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    field: str | None = None,
    value: object | None = None,
    row_index: int = 0,
) -> dict[str, object]:
    """统一提取 archive 摘要里的单行 dict 视图。"""
    row = get_summary_row(summary_df, field=field, value=value, row_index=row_index)
    if columns is not None:
        row = row[[col for col in columns if col in row.index]]
    return row.to_dict()


def filter_available_metric_mappings(
    rows: Sequence[Mapping[str, object]] | pd.DataFrame,
    metric_mappings: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """按现有摘要列过滤可计算 diff 的指标映射。"""
    summary_frame = build_summary_frame(rows)
    return [
        (source_col, diff_col)
        for source_col, diff_col in metric_mappings
        if source_col in summary_frame.columns
    ]


def finalize_baseline_diff_summary(
    rows: Sequence[Mapping[str, object]] | pd.DataFrame,
    *,
    baseline_field: str,
    baseline_value: object,
    metric_mappings: Sequence[tuple[str, str]],
    sort_by: Sequence[str] | None = None,
    ascending: Sequence[bool] | bool = True,
    reset_index: bool = False,
) -> pd.DataFrame:
    """把 rows 汇总成 DataFrame，并按指定 baseline 字段统一补 diff 列。"""
    summary_frame = build_summary_frame(rows)
    if baseline_field not in summary_frame.columns:
        raise ValueError(f"baseline field not found: {baseline_field}")
    diff_summary = add_diff_vs_baseline(
        summary_frame,
        baseline_mask=summary_frame[baseline_field] == baseline_value,
        metric_mappings=metric_mappings,
    )
    if sort_by:
        diff_summary = diff_summary.sort_values(list(sort_by), ascending=ascending)
    if reset_index:
        diff_summary = diff_summary.reset_index(drop=True)
    return diff_summary


def select_valid_change_rows(
    summary_df: pd.DataFrame,
    *,
    valid_column: str = "is_valid_change",
    rerank_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """统一筛选 archive 摘要里命中的 valid-change 候选。"""
    valid_df = summary_df.loc[summary_df[valid_column]].copy()
    if rerank_fn is not None and not valid_df.empty:
        valid_df = rerank_fn(valid_df)
    return valid_df


def select_nonbaseline_rows(
    summary_df: pd.DataFrame,
    *,
    baseline_strategy: str,
    strategy_field: str = "strategy",
    rerank_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """统一筛选 archive 摘要里排除 baseline 的候选上下文。"""
    candidate_df = summary_df.loc[summary_df[strategy_field] != baseline_strategy].copy()
    if rerank_fn is not None and not candidate_df.empty:
        candidate_df = rerank_fn(candidate_df)
    return candidate_df


def save_variant_compare_outputs(output_dir: Path, summary_df: pd.DataFrame, compare_df: pd.DataFrame) -> None:
    """统一落盘多变体对比脚本的核心导出物。"""
    ensure_output_dir(output_dir)
    write_dataframe_csv_atomic(summary_df, output_dir / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, output_dir / "nav_compare.csv")


def save_named_nav_output(
    output_dir: Path,
    name: str,
    nav_df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    index: bool = True,
) -> None:
    """统一落盘单个变体的 `*_nav.csv` 导出。"""
    ensure_output_dir(output_dir)
    export_df = nav_df.copy()
    if columns is not None:
        export_df = export_df[[col for col in columns if col in export_df.columns]].copy()
    if index and export_df.index.name is None:
        export_df.index.name = "date"
    write_dataframe_csv_atomic(export_df, output_dir / f"{name}_nav.csv", index=index)


def save_named_nav_outputs(
    output_dir: Path,
    nav_frames: Mapping[str, pd.DataFrame],
    *,
    columns: Sequence[str] | None = None,
    index: bool = True,
) -> None:
    """统一批量落盘多个变体的 `*_nav.csv` 导出。"""
    for name, nav_df in nav_frames.items():
        save_named_nav_output(
            output_dir,
            name,
            nav_df,
            columns=columns,
            index=index,
        )


def save_named_nav_and_trades(
    output_dir: Path,
    name: str,
    nav_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    index: bool = True,
) -> None:
    """统一落盘单个变体的 `*_nav.csv` 与 `*_trades.csv`。"""
    save_named_nav_output(
        output_dir,
        name,
        nav_df,
        columns=columns,
        index=index,
    )
    save_named_trades_output(output_dir, name, trades_df)


def save_named_nav_and_trades_outputs(
    output_dir: Path,
    named_outputs: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    columns: Sequence[str] | None = None,
    index: bool = True,
) -> None:
    """统一批量落盘多个变体的 `*_nav.csv` 与 `*_trades.csv`。"""
    for name, (nav_df, trades_df) in named_outputs.items():
        save_named_nav_and_trades(
            output_dir,
            name,
            nav_df,
            trades_df,
            columns=columns,
            index=index,
        )


def save_named_trades_output(output_dir: Path, name: str, trades_df: pd.DataFrame) -> None:
    """统一落盘单个变体的 `*_trades.csv` 导出。"""
    ensure_output_dir(output_dir)
    write_dataframe_csv_atomic(trades_df, output_dir / f"{name}_trades.csv", index=False)


def save_aux_csv_output(
    output_dir: Path,
    filename: str,
    df: pd.DataFrame,
    *,
    index: bool = False,
) -> None:
    """统一落盘 archive 额外 CSV 导出物。"""
    ensure_output_dir(output_dir)
    write_dataframe_csv_atomic(df, output_dir / filename, index=index)


def save_text_output(output_dir: Path, filename: str, content: str) -> None:
    """统一落盘 archive 额外文本导出物。"""
    ensure_output_dir(output_dir)
    path = output_dir / filename
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def save_summary_output(output_dir: Path, summary_df: pd.DataFrame) -> None:
    """统一落盘仅导出摘要表的历史研究脚本结果。"""
    ensure_output_dir(output_dir)
    write_dataframe_csv_atomic(summary_df, output_dir / "summary.csv", index=False)


def save_and_print_summary_output(
    output_dir: Path,
    summary_df: pd.DataFrame,
    *,
    as_csv: bool = False,
) -> None:
    """统一落盘摘要表并打印摘要。"""
    save_summary_output(output_dir, summary_df)
    print_saved_summary(summary_df, output_dir=output_dir, as_csv=as_csv)


def print_summary_preview(
    summary_df: pd.DataFrame,
    columns: Sequence[str],
    *,
    head: int | None = None,
    output_dir: Path | None = None,
) -> None:
    """统一打印 archive 多变体脚本的摘要预览表。"""
    print(format_table_text(summary_df, columns=columns, head=head, index=False))
    if output_dir is not None:
        print(f"\noutputs saved to {output_dir}")


def save_and_print_summary_preview(
    output_dir: Path,
    summary_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    columns: Sequence[str],
    *,
    head: int | None = None,
) -> None:
    """统一落盘 compare 输出并打印摘要预览。"""
    save_variant_compare_outputs(output_dir, summary_df, compare_df)
    print_summary_preview(summary_df, columns, head=head, output_dir=output_dir)


def save_plot_and_print_summary_preview(
    output_dir: Path,
    summary_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    columns: Sequence[str],
    *,
    plot_filename: str,
    title: str,
    lines: Sequence[PlotLine],
    benchmark_label: str = "HS300",
    benchmark_column: str = "hs300_benchmark",
    head: int | None = None,
) -> None:
    """统一落盘 compare 输出、导出对比图并打印摘要预览。"""
    save_variant_compare_outputs(output_dir, summary_df, compare_df)
    plot_variant_compare(
        compare_df,
        output_dir / plot_filename,
        title=title,
        lines=lines,
        benchmark_label=benchmark_label,
        benchmark_column=benchmark_column,
    )
    print_summary_preview(summary_df, columns, head=head, output_dir=output_dir)


def print_summary_csv(summary_df: pd.DataFrame) -> None:
    """统一打印 CSV 风格摘要，便于历史脚本管道复用。"""
    print(format_csv_text(summary_df, index=False))


def print_saved_summary(
    summary_df: pd.DataFrame,
    *,
    output_dir: Path,
    as_csv: bool = False,
) -> None:
    """统一打印摘要并提示导出目录。"""
    if as_csv:
        print_summary_csv(summary_df)
    else:
        print(format_table_text(summary_df, index=False))
    print(f"\noutputs saved to {output_dir}")


def save_and_print_variant_compare_outputs(
    output_dir: Path,
    summary_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    *,
    as_csv: bool = False,
) -> None:
    """统一落盘多变体结果并打印摘要，避免脚本重复串联两步调用。"""
    save_variant_compare_outputs(output_dir, summary_df, compare_df)
    print_saved_summary(summary_df, output_dir=output_dir, as_csv=as_csv)


def print_baseline_and_preview(
    baseline_summary: Mapping[str, object],
    summary_df: pd.DataFrame,
    columns: Sequence[str],
    *,
    head: int | None = None,
    output_dir: Path | None = None,
) -> None:
    """统一打印 archive 搜参脚本的 baseline 与候选摘要预览。"""
    print("Baseline:")
    print_mapping_summary(baseline_summary)
    print("\nTop candidates:")
    print_summary_preview(summary_df, columns, head=head, output_dir=output_dir)


def save_and_print_baseline_preview(
    output_dir: Path,
    summary_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    *,
    baseline_summary: Mapping[str, object],
    columns: Sequence[str],
    head: int | None = None,
) -> None:
    """统一落盘 compare 输出并打印 baseline/候选预览。"""
    save_variant_compare_outputs(output_dir, summary_df, compare_df)
    print_baseline_and_preview(
        baseline_summary,
        summary_df,
        columns,
        head=head,
        output_dir=output_dir,
    )


def save_plot_and_print_baseline_preview(
    output_dir: Path,
    summary_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    *,
    baseline_summary: Mapping[str, object],
    columns: Sequence[str],
    plot_filename: str,
    title: str,
    lines: Sequence[PlotLine],
    benchmark_label: str = "HS300",
    benchmark_column: str = "hs300_benchmark",
    head: int | None = None,
) -> None:
    """统一落盘 compare 输出、导出对比图并打印 baseline/候选预览。"""
    save_variant_compare_outputs(output_dir, summary_df, compare_df)
    plot_variant_compare(
        compare_df,
        output_dir / plot_filename,
        title=title,
        lines=lines,
        benchmark_label=benchmark_label,
        benchmark_column=benchmark_column,
    )
    print_baseline_and_preview(
        baseline_summary,
        summary_df,
        columns,
        head=head,
        output_dir=output_dir,
    )


def save_and_print_summary_baseline_preview(
    output_dir: Path,
    summary_df: pd.DataFrame,
    *,
    baseline_summary: Mapping[str, object],
    columns: Sequence[str],
    head: int | None = None,
) -> None:
    """统一落盘摘要表并打印 baseline/候选预览。"""
    save_summary_output(output_dir, summary_df)
    print_baseline_and_preview(
        baseline_summary,
        summary_df,
        columns,
        head=head,
        output_dir=output_dir,
    )


def print_webhook_notify_result(notified: bool, detail: str | None) -> None:
    """统一打印 archive webhook 通知结果。"""
    if notified and detail:
        print(f"\nWebhook notified for {detail}")


def print_ranked_notify_result(notified: bool, detail: str | None) -> None:
    """统一打印 archive 排名式增量通知结果。"""
    if detail == "no_valid":
        print("No valid improvements to notify.")
        return
    if detail == "no_incremental":
        print("No incremental improvements to notify.")
        return
    if notified and detail:
        print(f"Notified improvement: {detail}")


def rerank_with_sort_notify_candidates(
    frame: pd.DataFrame,
    _previous: dict[str, object] | None,
) -> pd.DataFrame:
    """统一复用 archive 常见的 notify 候选排序逻辑。"""
    return sort_notify_candidates(frame)


def save_best_payload_and_notify_webhook(
    args: object,
    *,
    best_path: Path,
    notify_state_path: Path,
    baseline_summary: dict[str, object],
    improvements_df: pd.DataFrame,
    descriptions: dict[str, str],
    metric_tolerance: float,
    default_webhook: str,
    send_fn: Callable[[dict[str, object], str, dict[str, object], str, str], None],
    rerank_fn: Callable[[pd.DataFrame, dict[str, object] | None], pd.DataFrame] | None = None,
    sort_candidates: bool = True,
    suppress_exceptions: bool = True,
    default_notify_enabled: bool | None = None,
    print_fn: Callable[[str], None] | None = print,
    improvement_key: str = "valid_improvements",
) -> tuple[bool, str | None]:
    """统一处理 archive webhook 型最佳候选落盘与增量通知。"""
    save_best_payload(
        best_path,
        baseline_summary=baseline_summary,
        improvements_df=improvements_df,
        descriptions=descriptions,
        improvement_key=improvement_key,
    )
    notify_df = load_incremental_notify_candidates(
        notify_state_path,
        improvements_df,
        metric_tolerance=metric_tolerance,
        rerank_fn=rerank_fn,
    )
    notified, detail = notify_best_candidate(
        args,
        notify_df,
        descriptions=descriptions,
        baseline_summary=baseline_summary,
        default_webhook=default_webhook,
        notify_state_path=notify_state_path,
        send_fn=send_fn,
        sort_candidates=sort_candidates,
        suppress_exceptions=suppress_exceptions,
        default_notify_enabled=default_notify_enabled,
        print_fn=print_fn,
    )
    print_webhook_notify_result(notified, detail)
    return notified, detail


def save_best_payload_and_notify_ranked(
    args: object,
    *,
    best_path: Path,
    notify_state_path: Path,
    baseline_summary: dict[str, object],
    improvements_df: pd.DataFrame,
    descriptions: dict[str, str],
    metric_tolerance: float,
    default_webhook: str,
    send_fn: Callable[[dict[str, object], str, dict[str, object], str, str], None],
    improvement_key: str = "valid_improvements",
    print_result_fn: Callable[[bool, str | None], None] = print_ranked_notify_result,
) -> tuple[bool, str | None]:
    """统一处理 archive 排名型最佳候选落盘与增量通知。"""
    save_best_payload(
        best_path,
        baseline_summary=baseline_summary,
        improvements_df=improvements_df,
        descriptions=descriptions,
        improvement_key=improvement_key,
    )
    notified, detail = notify_ranked_incremental_candidate(
        args,
        improvements_df,
        descriptions=descriptions,
        baseline_summary=baseline_summary,
        metric_tolerance=metric_tolerance,
        default_webhook=default_webhook,
        notify_state_path=notify_state_path,
        send_fn=send_fn,
    )
    print_result_fn(notified, detail)
    return notified, detail


def save_best_payload_and_notify_ranked_webhook(
    args: object,
    *,
    best_path: Path,
    notify_state_path: Path,
    baseline_summary: dict[str, object],
    improvements_df: pd.DataFrame,
    descriptions: dict[str, str],
    metric_tolerance: float,
    default_webhook: str,
    send_fn: Callable[[dict[str, object], str, dict[str, object], str, str], None],
    improvement_key: str = "valid_improvements",
) -> tuple[bool, str | None]:
    """统一处理 archive 排名型通知里沿用 webhook 打印口径的收尾。"""
    return save_best_payload_and_notify_ranked(
        args,
        best_path=best_path,
        notify_state_path=notify_state_path,
        baseline_summary=baseline_summary,
        improvements_df=improvements_df,
        descriptions=descriptions,
        metric_tolerance=metric_tolerance,
        default_webhook=default_webhook,
        send_fn=send_fn,
        improvement_key=improvement_key,
        print_result_fn=print_webhook_notify_result,
    )


def append_best_payload_records(
    best_path: Path,
    *,
    key: str,
    records_df: pd.DataFrame,
    limit: int | None = None,
    print_fn: Callable[[str], None] | None = print,
    message: str | None = None,
) -> bool:
    """向既有 best.json 追加一组记录，供历史链做非通知型上下文兜底。"""
    if records_df.empty or not best_path.exists():
        return False
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    export_df = records_df if limit is None else records_df.head(limit)
    payload[key] = export_df.to_dict(orient="records")
    write_json_atomic(best_path, payload)
    if print_fn is not None and message is not None:
        print_fn(message)
    return True


def plot_variant_compare(
    compare_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    lines: Sequence[PlotLine],
    benchmark_label: str = "HS300",
    benchmark_column: str = "hs300_benchmark",
) -> None:
    """按传入配置画净值对比图。"""
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


def save_plot_and_print_variant_compare_outputs(
    output_dir: Path,
    summary_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    *,
    plot_filename: str,
    title: str,
    lines: Sequence[PlotLine],
    benchmark_label: str = "HS300",
    benchmark_column: str = "hs300_benchmark",
    as_csv: bool = False,
) -> None:
    """统一落盘多变体结果、导出对比图并打印摘要。"""
    save_and_print_variant_compare_outputs(output_dir, summary_df, compare_df, as_csv=as_csv)
    plot_variant_compare(
        compare_df,
        output_dir / plot_filename,
        title=title,
        lines=lines,
        benchmark_label=benchmark_label,
        benchmark_column=benchmark_column,
    )


def save_variant_compare_artifacts(
    output_dir: Path,
    summary_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    *,
    plot_filename: str,
    title: str,
    lines: Sequence[PlotLine],
    benchmark_label: str = "HS300",
    benchmark_column: str = "hs300_benchmark",
    as_csv: bool = False,
    named_outputs: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    yearly_returns_df: pd.DataFrame | None = None,
    yearly_returns_filename: str = "yearly_returns.csv",
    named_output_columns: Sequence[str] | None = None,
    named_output_index: bool = True,
) -> None:
    """统一落盘多变体对比脚本常见的完整导出物。"""
    if yearly_returns_df is not None:
        save_aux_csv_output(output_dir, yearly_returns_filename, yearly_returns_df, index=False)
    if named_outputs:
        save_named_nav_and_trades_outputs(
            output_dir,
            named_outputs,
            columns=named_output_columns,
            index=named_output_index,
        )
    save_plot_and_print_variant_compare_outputs(
        output_dir,
        summary_df,
        compare_df,
        plot_filename=plot_filename,
        title=title,
        lines=lines,
        benchmark_label=benchmark_label,
        benchmark_column=benchmark_column,
        as_csv=as_csv,
    )
