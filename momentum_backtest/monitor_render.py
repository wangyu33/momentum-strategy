#!/usr/bin/env python3
"""daily_monitor 的渲染与格式化层。"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from .run_backtest import (
        DEFAULT_CLOSE_TOP2_GAP,
        DEFAULT_CLOSE_TOP2_RISK_CAP,
        DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
        DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
        DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
        DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
        DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_CUT,
        DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_EXPOSURE,
        DEFAULT_REGIME_MIX_PRE_OVERHEAT_START_CUT,
        DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        RISK_CODES,
        DEFENSIVE_CODES,
    )
except ImportError:
    from run_backtest import (
        DEFAULT_CLOSE_TOP2_GAP,
        DEFAULT_CLOSE_TOP2_RISK_CAP,
        DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
        DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
        DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
        DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
        DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_CUT,
        DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_EXPOSURE,
        DEFAULT_REGIME_MIX_PRE_OVERHEAT_START_CUT,
        DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        RISK_CODES,
        DEFENSIVE_CODES,
    )

ALLOCATION_DISPLAY_DIGITS = 1


def momentum_bucket(value: float | None) -> str | None:
    if value is None:
        return None
    if value <= 0:
        return "<=0%"
    if value <= 0.05:
        return "0-5%"
    if value <= 0.10:
        return "5-10%"
    if value <= 0.15:
        return "10-15%"
    if value <= 0.20:
        return "15-20%"
    return ">20%"


def drawdown_bucket(value: float | None) -> str | None:
    if value is None:
        return None
    if value <= -0.20:
        return "<=-20%"
    if value <= -0.10:
        return "-20~-10%"
    if value <= -0.05:
        return "-10~-5%"
    if value <= -0.02:
        return "-5~-2%"
    return "-2~0%"


def format_allocation_percent(value: float) -> str:
    text = f"{float(value):.{ALLOCATION_DISPLAY_DIGITS}%}"
    if text.endswith(".0%"):
        text = text[:-3] + "%"
    return text


def allocation_hidden_by_display(value: float) -> bool:
    return abs(float(value)) < 0.005 or format_allocation_percent(value) == format_allocation_percent(0.0)


def allocations_equal_for_display(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) < 0.01 or format_allocation_percent(left) == format_allocation_percent(right)


def format_exposure_transition_for_display(from_value: float, to_value: float) -> str | None:
    if allocations_equal_for_display(from_value, to_value):
        return None
    return f"{format_allocation_percent(from_value)}->{format_allocation_percent(to_value)}"


def extract_weight_allocations(row: pd.Series, prefix: str) -> list[tuple[str, float]]:
    allocations: list[tuple[str, float]] = []
    for column, value in row.items():
        if not str(column).startswith(prefix) or pd.isna(value):
            continue
        weight = float(value)
        if abs(weight) < 1e-8 or allocation_hidden_by_display(weight):
            continue
        code = str(column).removeprefix(prefix)
        allocations.append((code, weight))
    allocations.sort(key=lambda item: (-item[1], item[0]))
    return allocations


def format_portfolio_allocations(
    allocations: list[tuple[str, float]],
    theme_map: dict[str, str],
    name_map: dict[str, str],
    risk_leader: str | None = None,
    risk_leader_name: str | None = None,
    risk_leader_theme: str | None = None,
    risk_leader_momentum: float | None = None,
    defensive_leader: str | None = None,
    defensive_leader_name: str | None = None,
    defensive_leader_theme: str | None = None,
    defensive_leader_momentum: float | None = None,
) -> str:
    if not allocations:
        return "空仓"
    parts = []
    for code, weight in allocations:
        theme = theme_map.get(code, "")
        name = name_map.get(code, "")
        if theme and name:
            parts.append(f"{theme} / {name} ({code}) {format_allocation_percent(weight)}")
        elif name:
            parts.append(f"{name} ({code}) {format_allocation_percent(weight)}")
        else:
            parts.append(f"{code} {format_allocation_percent(weight)}")
    return "；".join(parts)


def build_trade_details_from_allocations(
    current_allocations: list[tuple[str, float]],
    desired_allocations: list[tuple[str, float]],
    theme_map: dict[str, str],
    name_map: dict[str, str],
) -> tuple[str, bool]:
    current_map = {code: weight for code, weight in current_allocations}
    desired_map = {code: weight for code, weight in desired_allocations}
    parts: list[str] = []

    all_codes = sorted(set(current_map) | set(desired_map))
    for code in all_codes:
        current_weight = float(current_map.get(code, 0.0))
        desired_weight = float(desired_map.get(code, 0.0))
        if abs(current_weight - desired_weight) < 1e-8 or allocations_equal_for_display(current_weight, desired_weight):
            continue
        theme = theme_map.get(code, "")
        name = name_map.get(code, "")
        label = f"{theme} / {name} ({code})" if theme and name else f"{name} ({code})" if name else code
        current_hidden = allocation_hidden_by_display(current_weight)
        desired_hidden = allocation_hidden_by_display(desired_weight)
        current_display = format_allocation_percent(current_weight)
        desired_display = format_allocation_percent(desired_weight)
        if current_hidden and not desired_hidden:
            parts.append(f"买入 {label} {current_display}->{desired_display}")
        elif not current_hidden and desired_hidden:
            parts.append(f"卖出 {label} {current_display}->{desired_display}")
        elif desired_weight > current_weight:
            parts.append(f"加仓 {label} {current_display}->{desired_display}")
        else:
            parts.append(f"减仓 {label} {current_display}->{desired_display}")

    trade_details = "；".join(parts) if parts else "无"
    return trade_details, trade_details != "无"


def build_confirmed_trade_reason(
    *,
    trade_previous_allocations: list[tuple[str, float]],
    trade_current_allocations: list[tuple[str, float]],
    strategy_id: str,
    current_momentum: float | None,
    effective_momentum: float | None,
    current_drawdown: float | None,
    base_exposure: float | None,
    desired_exposure: float | None,
    extra_cap_triggered: bool,
    extra_cap_reason: str | None,
    top2_close_cap_triggered: bool,
    top2_close_gap: float | None,
    top2_close_risk_cap: float | None,
    theme_map: dict[str, str],
    name_map: dict[str, str],
    risk_leader: str | None = None,
    risk_leader_name: str | None = None,
    risk_leader_theme: str | None = None,
    risk_leader_momentum: float | None = None,
    defensive_leader: str | None = None,
    defensive_leader_name: str | None = None,
    defensive_leader_theme: str | None = None,
    defensive_leader_momentum: float | None = None,
) -> str:
    parts: list[str] = []

    if trade_current_allocations:
        lead_code, lead_weight = trade_current_allocations[0]
        lead_theme = theme_map.get(lead_code, "")
        lead_name = name_map.get(lead_code, "")
        lead_label = f"{lead_theme} / {lead_name} ({lead_code})" if lead_theme and lead_name else lead_code
        if len(trade_current_allocations) == 1:
            if strategy_id == "default":
                parts.append(f"主信号指向 {lead_label}，因为它在风险资产里按 25 日信号质量排序位列第一，不是只看原始动量最大")
            else:
                parts.append(f"主信号指向 {lead_label}")
        else:
            support_labels = []
            support_code, support_weight = trade_current_allocations[1]
            for code, weight in trade_current_allocations[1:]:
                theme = theme_map.get(code, "")
                name = name_map.get(code, "")
                label = f"{theme} / {name} ({code})" if theme and name else code
                support_labels.append(f"{label} {format_allocation_percent(weight)}")
            if strategy_id == "default":
                parts.append(
                    f"主信号指向 {lead_label} {format_allocation_percent(lead_weight)}，因为它在风险资产里按 25 日信号质量排序位列第一，不是只看原始动量最大"
                )
                if support_code == "510300":
                    parts.append(
                        f"出现辅助仓位是因为正式基线使用 regime mix 的 core-satellite 框架，当前保留 {'、'.join(support_labels)} 作为核心仓，其余仓位分配给主信号"
                    )
                else:
                    parts.append(f"不是单一满仓，保留 {'、'.join(support_labels)} 作为辅助仓位")
            else:
                parts.append(f"主信号指向 {lead_label} {format_allocation_percent(lead_weight)}")
                parts.append(f"不是单一满仓，保留 {'、'.join(support_labels)} 作为辅助仓位")

    weak_trend_defensive_switch = (
        strategy_id == "default"
        and risk_leader is not None
        and defensive_leader is not None
        and any(code == defensive_leader for code, _ in trade_current_allocations)
        and all(code in DEFENSIVE_CODES for code, _ in trade_current_allocations)
        and risk_leader != defensive_leader
        and risk_leader_momentum is not None
        and 0 < float(risk_leader_momentum) <= DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD + 1e-12
    )
    if weak_trend_defensive_switch:
        risk_label = describe_signal(risk_leader, risk_leader_theme, risk_leader_name)
        defensive_label = describe_signal(defensive_leader, defensive_leader_theme, defensive_leader_name)
        parts = [
            f"风险池第一名是 {risk_label}，25日动量 {format_ratio_percent(risk_leader_momentum, digits=2)}，但未超过 {format_ratio_percent(DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD, digits=0)} 的风险持有阈值",
            f"因此按弱趋势规则切到防守池第一名 {defensive_label}，其25日动量 {format_ratio_percent(defensive_leader_momentum, digits=2)}，目标仓位 {format_allocation_percent(desired_exposure if desired_exposure is not None else DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT)}",
        ]
        if effective_momentum is not None:
            parts.append(f"策略热度 {format_ratio_percent(effective_momentum, digits=2)}")
        return "；".join(parts)

    if current_momentum is not None:
        parts.append(f"信号动量 {format_ratio_percent(current_momentum, digits=2)}")
    if effective_momentum is not None and (
        current_momentum is None or abs(float(effective_momentum) - float(current_momentum)) >= 1e-6
    ):
        parts.append(f"策略热度 {format_ratio_percent(effective_momentum, digits=2)}")

    if top2_close_cap_triggered:
        close_gap = top2_close_gap if top2_close_gap is not None else DEFAULT_CLOSE_TOP2_GAP
        close_cap = top2_close_risk_cap if top2_close_risk_cap is not None else DEFAULT_CLOSE_TOP2_RISK_CAP
        parts.append(
            f"当前前二风险资产信号质量差不超过 {format_ratio_percent(close_gap, digits=1)}，按近似并列规则先把总风险仓上限压到 {format_allocation_percent(close_cap)}"
        )

    if extra_cap_triggered and base_exposure is not None and desired_exposure is not None:
        drawdown_text = format_ratio_percent(current_drawdown, digits=2) if current_drawdown is not None else "N/A"
        momentum_text = format_ratio_percent(current_momentum, digits=2) if current_momentum is not None else "N/A"
        effective_text = format_ratio_percent(effective_momentum, digits=2) if effective_momentum is not None else "N/A"
        if extra_cap_reason == "pre_overheat_cap":
            parts.append(
                f"当前处于预减仓区：信号动量 {momentum_text}，策略热度 {effective_text}；为避免继续追高，把总仓位从 "
                f"{format_allocation_percent(base_exposure)} 下调到 {format_allocation_percent(desired_exposure)}"
            )
        elif extra_cap_reason == "mid_overheat_cap":
            parts.append(
                f"当前进入连续过热缩放区：信号动量 {momentum_text}，策略热度 {effective_text}，当前回撤 {drawdown_text}；因此把总仓位从 "
                f"{format_allocation_percent(base_exposure)} 下调到 {format_allocation_percent(desired_exposure)}"
            )
        elif extra_cap_reason == "overheat_cap":
            parts.append(
                f"当前命中过热压仓：信号动量 {momentum_text}，策略热度 {effective_text}，当前回撤 {drawdown_text}；因此把总仓位从 "
                f"{format_allocation_percent(base_exposure)} 下调到 {format_allocation_percent(desired_exposure)}"
            )
        elif extra_cap_reason == "overheat_high_cap":
            parts.append(
                f"当前命中极热压仓：信号动量 {momentum_text}，策略热度 {effective_text}，当前回撤 {drawdown_text}；因此把总仓位从 "
                f"{format_allocation_percent(base_exposure)} 下调到 {format_allocation_percent(desired_exposure)}"
            )
        elif extra_cap_reason == "stability_cap":
            parts.append(
                f"当前命中稳定性附加压仓：信号动量 {momentum_text}，策略热度 {effective_text}，当前回撤 {drawdown_text}；因此把总仓位从 "
                f"{format_allocation_percent(base_exposure)} 下调到 {format_allocation_percent(desired_exposure)}"
            )
        else:
            parts.append(
                describe_extra_cap_reason(
                    extra_cap_reason,
                    base_exposure=base_exposure,
                    desired_exposure=desired_exposure,
                )
            )

    return "；".join(parts) if parts else "沿用最新确认收盘组合，无额外说明"


def format_ratio_percent(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    normalized = 0.0 if abs(float(value)) < 5e-6 else float(value)
    return f"{normalized:.{digits}%}"


def format_historical_regime_summary(snapshot: Any) -> str:
    regime = snapshot.historical_regime_label if snapshot.historical_regime_label is not None else "N/A"
    regime_display_map = {
        "低": "风险低",
        "中": "风险中",
        "高": "风险高",
        "样本少": "样本少",
    }
    regime = regime_display_map.get(regime, regime)
    sample_text = str(snapshot.historical_regime_count) if snapshot.historical_regime_count is not None else "N/A"
    win_rate = format_ratio_percent(snapshot.historical_win_rate_60, digits=0)
    if getattr(snapshot, "historical_win_rate_60_percentile", None) is not None:
        win_rate += f"（百分位 {format_ratio_percent(snapshot.historical_win_rate_60_percentile, digits=1)}）"
    avg_ret = format_ratio_percent(snapshot.historical_avg_ret_60, digits=2)
    avg_mdd = format_ratio_percent(snapshot.historical_avg_mdd_60, digits=2)
    return f"同类60日: {regime}, 样本 {sample_text}, 胜率 {win_rate}, 收益 {avg_ret}, 回撤 {avg_mdd}"


def describe_signal(code: str | None, theme: str | None, name: str | None) -> str:
    if not code:
        return "空仓"
    if theme:
        return f"{theme} / {name} ({code})"
    return f"{name} ({code})" if name else str(code)


def format_exposure(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "0%"


def describe_extra_cap_reason(
    reason: str | None,
    *,
    base_exposure: float | None,
    desired_exposure: float | None,
) -> str:
    transition = ""
    if base_exposure is not None and desired_exposure is not None:
        transition = f" {format_allocation_percent(base_exposure)} -> {format_allocation_percent(desired_exposure)}"
    mapping = {
        "pre_overheat_cap": f"预减仓缩放{transition}".strip(),
        "mid_overheat_cap": f"过热连续缩放{transition}".strip(),
        "overheat_cap": f"过热压仓{transition}".strip(),
        "overheat_high_cap": f"极热压仓{transition}".strip(),
        "stability_cap": f"稳定性附加压仓{transition}".strip(),
    }
    if reason in mapping:
        return mapping[reason]
    if transition:
        return f"附加风险仓缩放{transition}"
    return "附加风险仓缩放"


def format_market_volume_summary(snapshot: Any) -> tuple[str, str]:
    mode = snapshot.market_volume_mode or "N/A"
    if snapshot.market_volume_progress is not None and 0 < snapshot.market_volume_progress < 1:
        mode += f" ({format_ratio_percent(snapshot.market_volume_progress, digits=0)})"
    ratio_20_60 = format_ratio_percent(snapshot.market_amount_ratio_20_60, digits=2)
    ratio_5_20 = format_ratio_percent(snapshot.market_amount_ratio_5_20, digits=2)
    breadth = format_ratio_percent(snapshot.market_breadth_proxy, digits=2)
    summary = f"市场量能: 20/60={ratio_20_60}, 5/20={ratio_5_20}, 广度={breadth}"
    return mode, summary


def format_market_message(snapshot: Any) -> str:
    momentum_text = format_ratio_percent(snapshot.current_momentum, digits=2)
    if snapshot.momentum_lookback_days is not None and snapshot.momentum_start_date and snapshot.momentum_end_date:
        momentum_text = f"{momentum_text}（{snapshot.momentum_lookback_days}交易日回看：{snapshot.momentum_start_date} -> {snapshot.momentum_end_date}）"
    effective_momentum_text = format_ratio_percent(snapshot.effective_momentum, digits=2)
    risk_text = (
        f"{snapshot.entry_risk_score:.1f}/100 ({snapshot.entry_risk_level})"
        if snapshot.entry_risk_score is not None and snapshot.entry_risk_level is not None
        else "N/A"
    )
    advice_text = snapshot.entry_advice if snapshot.entry_advice is not None else "N/A"
    risk_scorecard = None
    scorecard_parts: list[str] = []
    if snapshot.momentum_percentile is not None:
        scorecard_parts.append(f"动量分位 {format_ratio_percent(snapshot.momentum_percentile, digits=1)}")
    if snapshot.drawdown_buffer_ratio is not None:
        scorecard_parts.append(f"回撤进度 {format_ratio_percent(snapshot.drawdown_buffer_ratio, digits=1)}")
    if snapshot.historical_avg_ret_60 is not None:
        expectation_text = f"60日期望 {format_ratio_percent(snapshot.historical_avg_ret_60, digits=2)}"
        if snapshot.historical_avg_ret_60_percentile is not None:
            expectation_text += f" / 分位 {format_ratio_percent(snapshot.historical_avg_ret_60_percentile, digits=1)}"
        scorecard_parts.append(expectation_text)
    if scorecard_parts:
        risk_scorecard = "风险评分卡: " + "，".join(scorecard_parts)
    extra_cap_detail = (
        f"{snapshot.base_exposure:.0%} -> {snapshot.desired_exposure:.0%}"
        if snapshot.extra_cap_triggered and snapshot.base_exposure is not None and snapshot.desired_exposure is not None
        else "无"
    )
    _, market_volume_summary = format_market_volume_summary(snapshot)
    historical_regime_summary = format_historical_regime_summary(snapshot)
    if snapshot.current_holding_date and snapshot.trade_date != snapshot.current_holding_date:
        title_text = f"ETF市场热度 {snapshot.trade_date}（确认数据截至 {snapshot.current_holding_date}）"
    else:
        title_text = f"ETF市场热度 {snapshot.trade_date}"
    lines = [
        title_text,
        f"交易时段: {snapshot.market_session_label or 'N/A'}",
        f"信号动量: {momentum_text} / 策略热度: {effective_momentum_text}",
        market_volume_summary,
        f"建仓风险: {risk_text}, 建议 {advice_text}",
        historical_regime_summary,
    ]
    if risk_scorecard is not None:
        lines.append(risk_scorecard)
    weak_trend_switch_line = None
    if (
        getattr(snapshot, 'risk_leader', None)
        and getattr(snapshot, 'defensive_leader', None)
        and getattr(snapshot, 'current_signal', None) == getattr(snapshot, 'defensive_leader', None)
        and getattr(snapshot, 'risk_leader', None) != getattr(snapshot, 'current_signal', None)
        and getattr(snapshot, 'risk_leader_momentum', None) is not None
        and 0 < float(getattr(snapshot, 'risk_leader_momentum')) <= DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD + 1e-12
    ):
        risk_label = describe_signal(snapshot.risk_leader, snapshot.risk_leader_theme, snapshot.risk_leader_name)
        defensive_label = describe_signal(snapshot.defensive_leader, snapshot.defensive_leader_theme, snapshot.defensive_leader_name)
        weak_trend_switch_line = (
            f"切换说明: 风险池第一名 {risk_label} 动量 {format_ratio_percent(snapshot.risk_leader_momentum, digits=2)}，"
            f"但未超过 {format_ratio_percent(DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD, digits=0)}；"
            f"按弱趋势规则转向防守池第一名 {defensive_label} {format_ratio_percent(snapshot.defensive_leader_momentum, digits=2)}，"
            f"目标仓位 {format_allocation_percent(snapshot.desired_exposure or DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT)}"
        )
    if weak_trend_switch_line is not None:
        lines.append(weak_trend_switch_line)
    if snapshot.top2_close_cap_triggered:
        close_gap = snapshot.top2_close_gap if snapshot.top2_close_gap is not None else DEFAULT_CLOSE_TOP2_GAP
        close_cap = snapshot.top2_close_risk_cap if snapshot.top2_close_risk_cap is not None else DEFAULT_CLOSE_TOP2_RISK_CAP
        lines.append(f"命中接近降仓: 是 (前二信号质量差<={format_ratio_percent(close_gap, digits=1)}, 总风险仓上限 {format_allocation_percent(close_cap)})")
    if snapshot.extra_cap_triggered:
        lines.append(f"{snapshot.extra_cap_label}: 是 ({extra_cap_detail})")
    return "\n".join(lines)


def format_trade_message(snapshot: Any) -> str:
    current_drawdown_text = format_ratio_percent(snapshot.current_drawdown, digits=2)
    max_drawdown_text = format_ratio_percent(snapshot.max_drawdown, digits=2)
    current_nav_text = f"{snapshot.current_nav:.4f}" if snapshot.current_nav is not None else "N/A"
    live_nav_text = f"{snapshot.live_nav:.4f}" if snapshot.live_nav is not None else "N/A"
    nav_date_text = snapshot.nav_date or "N/A"
    live_nav_date_text = snapshot.live_nav_date or snapshot.trade_date
    live_nav_suffix = (
        f" (覆盖 {snapshot.live_nav_coverage:.0%})"
        if snapshot.live_nav_coverage is not None and snapshot.live_nav_coverage < 0.999
        else ""
    )
    peak_nav_text = (
        f"{snapshot.peak_nav_value:.4f} ({snapshot.peak_nav_date})"
        if snapshot.peak_nav_value is not None and snapshot.peak_nav_date is not None
        else "N/A"
    )
    current_holding_date_text = snapshot.current_holding_date or "N/A"
    today_trade_happened = (
        snapshot.confirmed_trade_date is not None
        and snapshot.confirmed_trade_date == snapshot.trade_date
        and snapshot.confirmed_trade_details != "无"
    )
    today_trade_date_text = snapshot.trade_date
    if today_trade_happened:
        today_trade_text = snapshot.confirmed_trade_details
        today_trade_reason_text = snapshot.confirmed_trade_reason
    elif snapshot.market_session_label != "收盘后" and snapshot.changed:
        today_trade_text = snapshot.trade_details
        today_trade_reason_text = snapshot.pending_trade_reason
    else:
        today_trade_text = "无"
        today_trade_reason_text = "无"
    if snapshot.current_holding_date and snapshot.trade_date != snapshot.current_holding_date:
        title_text = f"ETF交易与回撤 {snapshot.trade_date}（确认数据截至 {snapshot.current_holding_date}）"
    else:
        title_text = f"ETF交易与回撤 {snapshot.trade_date}"
    return "\n".join(
        [
            title_text,
            f"当前确认持仓({current_holding_date_text}): {snapshot.current_portfolio}",
            f"今日目标组合(按今日收盘信号推演): {snapshot.desired_portfolio}",
            f"今日交易({today_trade_date_text}): {today_trade_text}",
            f"本次操作原因: {today_trade_reason_text}",
            f"净值: 上收({nav_date_text}) {current_nav_text} / 当前({live_nav_date_text}) {live_nav_text}{live_nav_suffix}",
            f"当前回撤: {current_drawdown_text}",
            f"历史最大回撤: {max_drawdown_text}",
            f"历史最高净值: {peak_nav_text}",
        ]
    )


def build_market_card(snapshot: Any) -> dict[str, Any]:
    _, market_volume_summary = format_market_volume_summary(snapshot)
    historical_regime_summary = format_historical_regime_summary(snapshot)
    risk_text = (
        f"{snapshot.entry_risk_score:.1f}/100 ({snapshot.entry_risk_level})"
        if snapshot.entry_risk_score is not None and snapshot.entry_risk_level is not None
        else "N/A"
    )
    advice_text = snapshot.entry_advice if snapshot.entry_advice is not None else "N/A"
    signal_label = describe_signal(snapshot.current_signal, snapshot.current_signal_theme, snapshot.current_signal_name)
    momentum_percentile_text = format_ratio_percent(snapshot.momentum_percentile, digits=1)
    drawdown_buffer_text = format_ratio_percent(snapshot.drawdown_buffer_ratio, digits=1)
    expectation_text = format_ratio_percent(snapshot.historical_avg_ret_60, digits=2)
    expectation_percentile_text = format_ratio_percent(snapshot.historical_avg_ret_60_percentile, digits=1)
    risk_scorecard_parts = [
        f"动量分位 {momentum_percentile_text}",
        f"回撤进度 {drawdown_buffer_text}",
        f"60日期望 {expectation_text}",
    ]
    if snapshot.historical_avg_ret_60_percentile is not None:
        risk_scorecard_parts[-1] += f" / 分位 {expectation_percentile_text}"
    risk_scorecard = "，".join(risk_scorecard_parts)
    title_text = f"ETF市场热度 {snapshot.trade_date}"
    if snapshot.current_holding_date and snapshot.trade_date != snapshot.current_holding_date:
        title_text = f"ETF市场热度 {snapshot.trade_date}（确认截至 {snapshot.current_holding_date}）"
    template_map = {"低": "green", "中": "yellow", "高": "red"}
    template = template_map.get(snapshot.entry_risk_level, "blue")
    elements: list[dict[str, Any]] = [
        {
            "tag": "div",
            "fields": [
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**当前信号**\n{signal_label}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**交易时段**\n{snapshot.market_session_label or 'N/A'}"}},
            ],
        },
        {"tag": "markdown", "content": f"**25日主信号动量：{format_ratio_percent(snapshot.current_momentum, digits=2)}**"},
        {
            "tag": "div",
            "fields": [
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**策略热度**\n{format_ratio_percent(snapshot.effective_momentum, digits=2)}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**建仓风险**\n{risk_text}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**建议**\n{advice_text}"}},
            ],
        },
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**风险评分卡：{risk_scorecard}**"},
        {"tag": "markdown", "content": f"**历史同类区间**\n{historical_regime_summary}"},
        {"tag": "markdown", "content": f"**市场量能**\n{market_volume_summary}"},
    ]
    weak_trend_switch_line = None
    if (
        getattr(snapshot, "risk_leader", None)
        and getattr(snapshot, "defensive_leader", None)
        and getattr(snapshot, "current_signal", None) == getattr(snapshot, "defensive_leader", None)
        and getattr(snapshot, "risk_leader", None) != getattr(snapshot, "current_signal", None)
        and getattr(snapshot, "risk_leader_momentum", None) is not None
        and 0 < float(getattr(snapshot, "risk_leader_momentum")) <= DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD + 1e-12
    ):
        risk_label = describe_signal(snapshot.risk_leader, snapshot.risk_leader_theme, snapshot.risk_leader_name)
        defensive_label = describe_signal(snapshot.defensive_leader, snapshot.defensive_leader_theme, snapshot.defensive_leader_name)
        weak_trend_switch_line = (
            f"风险池第一名 {risk_label} 动量 {format_ratio_percent(snapshot.risk_leader_momentum, digits=2)}，"
            f"未超过 {format_ratio_percent(DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD, digits=0)} 的风险持有阈值；"
            f"因此按弱趋势规则转向防守池第一名 {defensive_label} {format_ratio_percent(snapshot.defensive_leader_momentum, digits=2)}，"
            f"目标仓位 {format_allocation_percent(snapshot.desired_exposure or DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT)}"
        )
    if weak_trend_switch_line is not None:
        elements.extend([
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**弱趋势切换**\n{weak_trend_switch_line}"},
        ])
    if snapshot.top2_close_cap_triggered or snapshot.extra_cap_triggered:
        cap_parts: list[str] = []
        if snapshot.top2_close_cap_triggered:
            close_gap = snapshot.top2_close_gap if snapshot.top2_close_gap is not None else DEFAULT_CLOSE_TOP2_GAP
            close_cap = snapshot.top2_close_risk_cap if snapshot.top2_close_risk_cap is not None else DEFAULT_CLOSE_TOP2_RISK_CAP
            cap_parts.append(
                f"前二信号接近，质量差<={format_ratio_percent(close_gap, digits=1)}，风险仓上限 {format_allocation_percent(close_cap)}"
            )
        if snapshot.extra_cap_triggered and snapshot.base_exposure is not None and snapshot.desired_exposure is not None:
            cap_parts.append(
                f"{snapshot.extra_cap_label}: "
                f"{describe_extra_cap_reason(snapshot.extra_cap_reason, base_exposure=snapshot.base_exposure, desired_exposure=snapshot.desired_exposure)}"
            )
        elements.extend([
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**风控动作**\n" + "\n".join(f"- {part}" for part in cap_parts)},
        ])
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": title_text},
        },
        "elements": elements,
    }


def build_trade_card(snapshot: Any) -> dict[str, Any]:
    current_drawdown_text = format_ratio_percent(snapshot.current_drawdown, digits=2)
    max_drawdown_text = format_ratio_percent(snapshot.max_drawdown, digits=2)
    current_nav_text = f"{snapshot.current_nav:.4f}" if snapshot.current_nav is not None else "N/A"
    live_nav_text = f"{snapshot.live_nav:.4f}" if snapshot.live_nav is not None else "N/A"
    nav_date_text = snapshot.nav_date or "N/A"
    live_nav_date_text = snapshot.live_nav_date or snapshot.trade_date
    peak_nav_text = (
        f"{snapshot.peak_nav_value:.4f} ({snapshot.peak_nav_date})"
        if snapshot.peak_nav_value is not None and snapshot.peak_nav_date is not None
        else "N/A"
    )
    today_trade_happened = (
        snapshot.confirmed_trade_date is not None
        and snapshot.confirmed_trade_date == snapshot.trade_date
        and snapshot.confirmed_trade_details != "无"
    )
    if today_trade_happened:
        today_trade_text = snapshot.confirmed_trade_details
        today_trade_reason_text = snapshot.confirmed_trade_reason
    elif snapshot.market_session_label != "收盘后" and snapshot.changed:
        today_trade_text = snapshot.trade_details
        today_trade_reason_text = snapshot.pending_trade_reason
    else:
        today_trade_text = "无"
        today_trade_reason_text = "无"
    title_text = f"ETF交易与回撤 {snapshot.trade_date}"
    if snapshot.current_holding_date and snapshot.trade_date != snapshot.current_holding_date:
        title_text = f"ETF交易与回撤 {snapshot.trade_date}（确认截至 {snapshot.current_holding_date}）"
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "wathet",
            "title": {"tag": "plain_text", "content": title_text},
        },
        "elements": [
            {"tag": "markdown", "content": f"**当前确认持仓**\n{snapshot.current_portfolio}"},
            {"tag": "markdown", "content": f"**今日目标组合**\n{snapshot.desired_portfolio}"},
            {"tag": "hr"},
            {"tag": "markdown", "content": f"**今日交易**\n{today_trade_text}"},
            {"tag": "markdown", "content": f"**本次操作原因**\n{today_trade_reason_text}"},
            {"tag": "hr"},
            {
                "tag": "div",
                "fields": [
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**上收净值**\n{nav_date_text} / {current_nav_text}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**当前净值**\n{live_nav_date_text} / {live_nav_text}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**当前回撤**\n{current_drawdown_text}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**历史最大回撤**\n{max_drawdown_text}"}},
                ],
            },
            {"tag": "markdown", "content": f"**历史最高净值**\n{peak_nav_text}"},
        ],
    }
