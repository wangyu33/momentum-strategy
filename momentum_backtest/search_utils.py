#!/usr/bin/env python3
"""本地策略搜索脚本共用的辅助函数。"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

import pandas as pd
try:
    from .runtime_env import write_json_atomic
except ImportError:
    from runtime_env import write_json_atomic


def add_notify_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_enabled: bool = False,
) -> argparse.ArgumentParser:
    """为研究脚本补齐统一的通知相关 CLI 参数。"""
    if default_enabled:
        parser.add_argument("--notify", action="store_true", help="兼容旧参数；当前默认已开启找到更优策略时的 webhook 通知。")
        parser.add_argument("--disable-notify", action="store_true", help="关闭找到更优策略时的 webhook 通知。")
    else:
        parser.add_argument("--notify", action="store_true", help="找到更优策略时发送 webhook 通知。")
    parser.add_argument("--webhook-url", type=str, default="", help="临时指定 webhook。")
    return parser


def should_send_notify(args: argparse.Namespace, *, default_enabled: bool | None = None) -> bool:
    """统一解析“默认开启，除非显式关闭”的通知开关。"""
    if default_enabled is not None:
        if default_enabled:
            return bool(getattr(args, "notify", False) or not getattr(args, "disable_notify", False))
        return bool(getattr(args, "notify", False))
    if hasattr(args, "disable_notify"):
        return bool(getattr(args, "notify", False) or not getattr(args, "disable_notify", False))
    return bool(getattr(args, "notify", False))


def resolve_webhook_url(args: argparse.Namespace, default_webhook: str) -> str:
    """统一解析研究脚本 webhook 优先级：命令行 > 环境变量 > 默认值。"""
    cli_value = str(getattr(args, "webhook_url", "") or "").strip()
    return cli_value or os.getenv("DAILY_MONITOR_WEBHOOK_URL", default_webhook)


def notify_best_candidate(
    args: argparse.Namespace,
    notify_df: pd.DataFrame,
    *,
    descriptions: dict[str, str],
    baseline_summary: dict[str, object],
    default_webhook: str,
    notify_state_path: Path,
    send_fn: Callable[[dict[str, object], str, dict[str, object], str, str], None],
    sort_candidates: bool = True,
    suppress_exceptions: bool = False,
    default_notify_enabled: bool | None = None,
    print_fn: Callable[[str], None] | None = None,
) -> tuple[bool, str | None]:
    """统一处理研究脚本的最佳候选通知、状态落盘与错误抑制。"""
    if not should_send_notify(args, default_enabled=default_notify_enabled) or notify_df.empty:
        return False, None

    ranked = sort_notify_candidates(notify_df) if sort_candidates else notify_df
    best = ranked.iloc[0]
    strategy_name = str(best["strategy"])
    description = str(descriptions.get(strategy_name, ""))
    webhook_url = resolve_webhook_url(args, default_webhook)

    try:
        send_fn(baseline_summary, strategy_name, best.to_dict(), description, webhook_url)
        save_notify_state(notify_state_path, strategy_name, best.to_dict(), description)
        return True, strategy_name
    except Exception as exc:
        if suppress_exceptions:
            detail = f"{strategy_name}: {exc}"
            if print_fn is not None:
                print_fn(f"[warn] webhook notify skipped: {detail}")
            return False, detail
        raise


def filter_candidates_against_previous_summary(
    frame: pd.DataFrame,
    previous_summary: dict[str, object] | None,
    *,
    metric_tolerance: float,
    rerank_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """按上次已通知摘要做增量过滤，只保留继续变好的候选。"""
    if frame.empty or previous_summary is None:
        return frame

    filtered = frame[
        (frame["annualized_return"] >= float(previous_summary["annualized_return"]) - metric_tolerance)
        & (frame["sharpe_rf0"] >= float(previous_summary["sharpe_rf0"]) - metric_tolerance)
        & (frame["max_drawdown_integral"] <= float(previous_summary["max_drawdown_integral"]) + metric_tolerance)
        & (
            (frame["annualized_return"] > float(previous_summary["annualized_return"]) + metric_tolerance)
            | (frame["sharpe_rf0"] > float(previous_summary["sharpe_rf0"]) + metric_tolerance)
            | (frame["max_drawdown_integral"] < float(previous_summary["max_drawdown_integral"]) - metric_tolerance)
        )
    ].copy()
    if rerank_fn is not None:
        filtered = rerank_fn(filtered)
    return filtered


def load_incremental_notify_candidates(
    notify_state_path: Path,
    frame: pd.DataFrame,
    *,
    metric_tolerance: float,
    rerank_fn: Callable[[pd.DataFrame, dict[str, object] | None], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """读取上次通知状态，并过滤出仍值得继续通知的候选。"""
    notify_state = load_notify_state(notify_state_path)
    previous_summary = extract_valid_previous_summary(notify_state)
    filtered = filter_candidates_against_previous_summary(
        frame,
        previous_summary,
        metric_tolerance=metric_tolerance,
    )
    if rerank_fn is not None:
        filtered = rerank_fn(filtered, previous_summary)
    return filtered


def save_best_payload(
    path: Path,
    *,
    baseline_summary: dict[str, object],
    improvements_df: pd.DataFrame,
    descriptions: dict[str, str],
    improvement_key: str = "valid_improvements",
) -> None:
    """统一落盘 best.json 载荷，避免研究脚本重复拼装相同结构。"""
    write_json_atomic(
        path,
        {
            "baseline": baseline_summary,
            improvement_key: improvements_df.to_dict(orient="records"),
            "descriptions": descriptions,
        },
    )


def load_notify_state(path: Path) -> dict[str, object] | None:
    """读取研究脚本上次通知状态，不存在时返回空。"""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def normalize_strategy_payload(payload: object) -> dict[str, object] | None:
    """校验并标准化策略上下文载荷，坏 schema 直接视为不可用。"""
    if not isinstance(payload, dict):
        return None
    strategy_name = str(payload.get("strategy", "")).strip()
    if not strategy_name:
        return None
    summary = payload.get("summary", {})
    if summary is None:
        summary = {}
    if not isinstance(summary, dict):
        return None
    return {
        "strategy": strategy_name,
        "summary": summary,
        "description": str(payload.get("description", "")),
    }


def load_preferred_strategy_payload(notify_path: Path, best_path: Path) -> dict[str, object] | None:
    """优先读取通知状态；若不存在则回退到 best.json 里的最佳候选或基线。"""
    payload = normalize_strategy_payload(load_notify_state(notify_path))
    if payload is not None:
        return payload

    best_payload = load_notify_state(best_path)
    if best_payload is None:
        return None

    descriptions = best_payload.get("descriptions", {})
    if isinstance(descriptions, dict):
        description_map = descriptions
    else:
        description_map = {}

    # 兼容两类研究脚本输出：
    # - valid_improvements：满足“宽松有效改进”条件的候选
    # - strict_improvements：满足“严格改进”条件的候选
    # 如果通知状态文件丢失，应该优先恢复这两类候选里的第一名，
    # 而不是错误回退到 baseline。
    for improvement_key in ("valid_improvements", "strict_improvements"):
        improvements = best_payload.get(improvement_key)
        if isinstance(improvements, list) and improvements:
            best_summary = improvements[0]
            if isinstance(best_summary, dict) and best_summary.get("strategy"):
                strategy_name = str(best_summary["strategy"])
                return normalize_strategy_payload(
                    {
                        "strategy": strategy_name,
                        "summary": best_summary,
                        "description": str(description_map.get(strategy_name, "")),
                    }
                )

    baseline = best_payload.get("baseline")
    if isinstance(baseline, dict) and baseline.get("strategy"):
        strategy_name = str(baseline["strategy"])
        return normalize_strategy_payload(
            {
                "strategy": strategy_name,
                "summary": baseline,
                "description": str(description_map.get(strategy_name, "")),
            }
        )

    return None


def iter_strategy_payload_candidates(notify_path: Path, best_path: Path) -> list[dict[str, object]]:
    """按优先级枚举可恢复的策略载荷，供要求特定字段的上下文读取方兜底挑选。"""
    candidates: list[dict[str, object]] = []

    payload = normalize_strategy_payload(load_notify_state(notify_path))
    if payload is not None:
        candidates.append(payload)

    best_payload = load_notify_state(best_path)
    if best_payload is None:
        return candidates

    descriptions = best_payload.get("descriptions", {})
    description_map = descriptions if isinstance(descriptions, dict) else {}

    for improvement_key in ("valid_improvements", "strict_improvements"):
        improvements = best_payload.get(improvement_key)
        if not isinstance(improvements, list):
            continue
        for improvement in improvements:
            if not isinstance(improvement, dict) or not improvement.get("strategy"):
                continue
            payload = normalize_strategy_payload(
                {
                    "strategy": str(improvement["strategy"]),
                    "summary": improvement,
                    "description": str(description_map.get(str(improvement["strategy"]), "")),
                }
            )
            if payload is not None:
                candidates.append(payload)

    baseline = best_payload.get("baseline")
    if isinstance(baseline, dict) and baseline.get("strategy"):
        payload = normalize_strategy_payload(
            {
                "strategy": str(baseline["strategy"]),
                "summary": baseline,
                "description": str(description_map.get(str(baseline["strategy"]), "")),
            }
        )
        if payload is not None:
            candidates.append(payload)

    return candidates


def load_required_strategy_payload(
    notify_path: Path,
    best_path: Path,
    *,
    context_name: str,
    required_summary_fields: tuple[str, ...] = (),
) -> dict[str, object]:
    """读取某条研究链上游策略上下文；通知状态丢失时自动回退到 best.json。"""
    candidates = iter_strategy_payload_candidates(notify_path, best_path)
    if not candidates:
        raise RuntimeError(f"missing {context_name}")

    missing_field_names: set[str] = set()
    for payload in candidates:
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            continue

        missing_fields = [field for field in required_summary_fields if field not in summary]
        if missing_fields:
            missing_field_names.update(missing_fields)
            continue

        return {
            "strategy": str(payload["strategy"]),
            "summary": dict(summary),
            "description": str(payload.get("description", "")),
        }

    if missing_field_names:
        missing_fields_text = ", ".join(sorted(missing_field_names))
        raise RuntimeError(f"invalid {context_name}: missing summary field {missing_fields_text}")
    raise RuntimeError(f"invalid {context_name}: bad summary payload")


def save_notify_state(path: Path, strategy_name: str, summary: dict[str, object], description: str) -> None:
    """保存研究脚本最新一次通知对应的策略摘要。"""
    payload = {"strategy": strategy_name, "summary": summary, "description": description}
    write_json_atomic(path, payload)


def sort_notify_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """按通知链真正选 best 的口径排序候选。"""
    if frame.empty:
        return frame
    return frame.sort_values(
        ["annualized_return", "sharpe_rf0", "max_drawdown_integral", "max_drawdown"],
        ascending=[False, False, True, False],
    )


def extract_valid_previous_summary(
    notify_state: dict[str, object] | None,
    required_metrics: tuple[str, ...] = ("annualized_return", "sharpe_rf0", "max_drawdown_integral"),
) -> dict[str, object] | None:
    """从通知状态里提取可用于继续比较的摘要；缺字段或值不可转 float 时直接忽略。"""
    if not isinstance(notify_state, dict):
        return None
    summary = notify_state.get("summary")
    if not isinstance(summary, dict):
        return None
    for key in required_metrics:
        try:
            float(summary[key])
        except Exception:
            return None
    return summary


def rank_valid_improvements(frame: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    """按收益、夏普和全历史回撤积分综合排序候选改进方案。"""
    if frame.empty:
        return frame
    ranked = frame.copy()
    ranked["annualized_return_gain"] = ranked["annualized_return"] - float(baseline["annualized_return"])
    ranked["sharpe_gain"] = ranked["sharpe_rf0"] - float(baseline["sharpe_rf0"])
    ranked["drawdown_integral_improvement"] = float(baseline["max_drawdown_integral"]) - ranked["max_drawdown_integral"]
    ranked["composite_improvement_score"] = (
        ranked["annualized_return_gain"] * 100
        + ranked["sharpe_gain"] * 10
        + ranked["drawdown_integral_improvement"] / 10
    )
    return ranked.sort_values(
        [
            "composite_improvement_score",
            "annualized_return_gain",
            "sharpe_gain",
            "drawdown_integral_improvement",
            "annualized_return",
            "sharpe_rf0",
            "max_drawdown_integral",
        ],
        ascending=[False, False, False, False, False, False, True],
    )


def annotate_valid_improvements(
    frame: Sequence[Mapping[str, object]] | pd.DataFrame,
    baseline_summary: dict[str, object],
    *,
    metric_tolerance: float,
    strategy_name: str,
) -> pd.DataFrame:
    """统一补齐改进 diff 字段、有效改进标记与默认排序。"""
    annotated = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    baseline_annualized = float(baseline_summary["annualized_return"])
    baseline_sharpe = float(baseline_summary["sharpe_rf0"])
    baseline_dd_integral = float(baseline_summary["max_drawdown_integral"])

    annotated["annualized_diff"] = annotated["annualized_return"] - baseline_annualized
    annotated["sharpe_diff"] = annotated["sharpe_rf0"] - baseline_sharpe
    annotated["max_drawdown_integral_diff"] = annotated["max_drawdown_integral"] - baseline_dd_integral
    annotated["is_valid_change"] = (
        (annotated["strategy"] != strategy_name)
        & (annotated["annualized_return"] >= baseline_annualized - metric_tolerance)
        & (annotated["sharpe_rf0"] >= baseline_sharpe - metric_tolerance)
        & (annotated["max_drawdown_integral"] <= baseline_dd_integral + metric_tolerance)
        & (
            (annotated["annualized_return"] > baseline_annualized + metric_tolerance)
            | (annotated["sharpe_rf0"] > baseline_sharpe + metric_tolerance)
            | (annotated["max_drawdown_integral"] < baseline_dd_integral - metric_tolerance)
        )
    )
    return annotated.sort_values(
        ["is_valid_change", "annualized_return", "sharpe_rf0", "max_drawdown_integral"],
        ascending=[False, False, False, True],
    )


def extract_ranked_valid_improvements(
    frame: Sequence[Mapping[str, object]] | pd.DataFrame,
    baseline_summary: dict[str, object],
    *,
    metric_tolerance: float,
    strategy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """统一生成带 diff 的结果表和按综合得分重排后的有效改进集。"""
    annotated = annotate_valid_improvements(
        frame,
        baseline_summary,
        metric_tolerance=metric_tolerance,
        strategy_name=strategy_name,
    )
    ranked = rank_valid_improvements(annotated[annotated["is_valid_change"]].copy(), pd.Series(baseline_summary))
    return annotated, ranked


def load_ranked_incremental_notify_candidates(
    notify_state_path: Path,
    ranked_frame: pd.DataFrame,
    *,
    metric_tolerance: float,
) -> pd.DataFrame:
    """按上次通知摘要过滤候选，并在需要时按同口径重新综合排序。"""
    return load_incremental_notify_candidates(
        notify_state_path,
        ranked_frame,
        metric_tolerance=metric_tolerance,
        rerank_fn=lambda frame, previous_summary: frame
        if previous_summary is None
        else rank_valid_improvements(frame, pd.Series(previous_summary)),
    )


def notify_ranked_incremental_candidate(
    args: argparse.Namespace,
    ranked_frame: pd.DataFrame,
    *,
    descriptions: dict[str, str],
    baseline_summary: dict[str, object],
    metric_tolerance: float,
    default_webhook: str,
    notify_state_path: Path,
    send_fn: Callable[[dict[str, object], str, dict[str, object], str, str], None],
) -> tuple[bool, str]:
    """统一处理“已按综合得分排好序”的增量通知链。"""
    if not should_send_notify(args):
        return False, "disabled"
    if ranked_frame.empty:
        return False, "no_valid"

    notify_df = load_ranked_incremental_notify_candidates(
        notify_state_path,
        ranked_frame,
        metric_tolerance=metric_tolerance,
    )
    if notify_df.empty:
        return False, "no_incremental"

    notified, detail = notify_best_candidate(
        args,
        notify_df,
        descriptions=descriptions,
        baseline_summary=baseline_summary,
        default_webhook=default_webhook,
        notify_state_path=notify_state_path,
        send_fn=send_fn,
        sort_candidates=False,
        suppress_exceptions=True,
        print_fn=print,
    )
    if not notified:
        return False, detail or "skipped"
    return True, detail or str(notify_df.iloc[0]["strategy"])


def try_join_missing_candidate_histories(
    prices: pd.DataFrame,
    candidates: pd.DataFrame,
    years: int,
    fetch_fn: Callable[[pd.DataFrame, int], pd.DataFrame],
) -> tuple[pd.DataFrame, list[str], str | None]:
    """尽量补齐缓存里缺失的候选标的历史；抓取失败时保留现有价格面板继续运行。"""
    if candidates.empty:
        return prices, [], None

    candidate_frame = candidates.copy()
    candidate_frame["code"] = candidate_frame["code"].astype(str)
    missing_codes = [code for code in candidate_frame["code"].tolist() if code not in prices.columns]
    if not missing_codes:
        return prices, [], None

    missing_candidates = candidate_frame[candidate_frame["code"].isin(missing_codes)].reset_index(drop=True)
    error_message: str | None = None
    try:
        extra = fetch_fn(missing_candidates, years=years)
    except Exception as exc:
        extra = pd.DataFrame(index=prices.index)
        error_message = str(exc)

    if not extra.empty:
        prices = prices.join(extra, how="outer")

    remaining_missing = [code for code in missing_codes if code not in prices.columns]
    return prices, remaining_missing, error_message


def raise_if_missing_required_histories(prices: pd.DataFrame, candidates: pd.DataFrame, *, context: str) -> None:
    """校验正式策略必需标的历史是否齐全；缺失时抛出可读错误，避免后续掉进 KeyError。"""
    if candidates.empty:
        return
    candidate_frame = candidates.copy()
    candidate_frame["code"] = candidate_frame["code"].astype(str)
    missing = candidate_frame[~candidate_frame["code"].isin(prices.columns)].copy()
    if missing.empty:
        return
    parts = [f"{row.code} {row.name}" for row in missing.itertuples(index=False)]
    raise RuntimeError(f"missing required {context} history: {', '.join(parts)}")
