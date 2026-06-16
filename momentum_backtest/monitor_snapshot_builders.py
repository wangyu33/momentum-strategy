#!/usr/bin/env python3
"""`compute_snapshot()` 的装配辅助函数。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from .monitor_render import (
        build_confirmed_trade_reason,
        build_trade_details_from_allocations,
        extract_weight_allocations,
        format_portfolio_allocations,
    )
    from .monitor_snapshot import (
        build_asset_price_context,
        build_entry_risk_context,
        classify_market_session,
        estimate_live_nav,
        load_backtest_reference_context,
        should_include_realtime_snapshot,
    )
    from .run_backtest import (
        DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        DEFAULT_LOOKBACK,
        build_default_strategy_params,
        build_signal_quality_score,
        choose_signal_winner_with_margin,
        normalize_code,
        resolve_strategy_universe,
    )
except ImportError:
    from monitor_render import (
        build_confirmed_trade_reason,
        build_trade_details_from_allocations,
        extract_weight_allocations,
        format_portfolio_allocations,
    )
    from monitor_snapshot import (
        build_asset_price_context,
        build_entry_risk_context,
        classify_market_session,
        estimate_live_nav,
        load_backtest_reference_context,
        should_include_realtime_snapshot,
    )
    from run_backtest import (
        DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        DEFAULT_LOOKBACK,
        build_default_strategy_params,
        build_signal_quality_score,
        choose_signal_winner_with_margin,
        normalize_code,
        resolve_strategy_universe,
    )


def build_leader_context(
    *,
    latest_idx: pd.Timestamp,
    prices: pd.DataFrame,
    strategy_id: str,
    theme_map: dict[str, str],
    name_map: dict[str, str],
) -> dict[str, object]:
    if strategy_id != 'default':
        return {
            'risk_leader': None,
            'risk_leader_name': None,
            'risk_leader_theme': None,
            'risk_leader_momentum': None,
            'defensive_leader': None,
            'defensive_leader_name': None,
            'defensive_leader_theme': None,
            'defensive_leader_momentum': None,
        }

    params = build_default_strategy_params()
    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params.get('risk_codes', [])],
        defensive_codes=[str(code) for code in params.get('defensive_codes', [])],
    )
    raw_mom, score = build_signal_quality_score(
        prices,
        lookback=DEFAULT_LOOKBACK,
        method=str(params.get('signal_quality_method', 'raw')),
        slope_penalty=float(params.get('signal_slope_penalty', 0.0)),
    )
    risk_score = score[active_risk_codes]
    defensive_score = score[active_defensive_codes]
    prev_risk = None
    prev_def = None
    risk_leader = None
    def_leader = None
    for dt in prices.index:
        risk_leader = choose_signal_winner_with_margin(risk_score.loc[dt], prev_risk, float(params.get('signal_leader_margin', 0.0)))
        def_leader = choose_signal_winner_with_margin(defensive_score.loc[dt], prev_def, float(params.get('signal_leader_margin', 0.0)))
        prev_risk = str(risk_leader) if risk_leader else None
        prev_def = str(def_leader) if def_leader else None
        if dt == latest_idx:
            break

    risk_code = normalize_code(risk_leader)
    def_code = normalize_code(def_leader)
    risk_mom = None
    def_mom = None
    if risk_code and risk_code in raw_mom.columns and pd.notna(raw_mom.loc[latest_idx, risk_code]):
        risk_mom = float(raw_mom.loc[latest_idx, risk_code])
    if def_code and def_code in raw_mom.columns and pd.notna(raw_mom.loc[latest_idx, def_code]):
        def_mom = float(raw_mom.loc[latest_idx, def_code])
    return {
        'risk_leader': risk_code,
        'risk_leader_name': name_map.get(risk_code) if risk_code else None,
        'risk_leader_theme': theme_map.get(risk_code) if risk_code else None,
        'risk_leader_momentum': risk_mom,
        'defensive_leader': def_code,
        'defensive_leader_name': name_map.get(def_code) if def_code else None,
        'defensive_leader_theme': theme_map.get(def_code) if def_code else None,
        'defensive_leader_momentum': def_mom,
    }


def build_reference_and_risk_context(
    *,
    latest_idx: pd.Timestamp,
    result: pd.DataFrame,
    close_result: pd.DataFrame,
    backtest_nav_file: Path,
    prices: pd.DataFrame,
    spot_prices: dict[str, float],
    raw_closes: dict[str, float],
    now: datetime,
) -> dict[str, object]:
    current_momentum = result["current_momentum"]
    current_mom = None
    if pd.notna(current_momentum.loc[latest_idx]):
        current_mom = float(current_momentum.loc[latest_idx])
    effective_mom = None
    if "effective_momentum" in result.columns and pd.notna(result.loc[latest_idx, "effective_momentum"]):
        effective_mom = float(result.loc[latest_idx, "effective_momentum"])

    reference_context = load_backtest_reference_context(backtest_nav_file, close_result)
    max_drawdown = reference_context["max_drawdown"]
    current_drawdown = reference_context["current_drawdown"]
    current_nav = reference_context["current_nav"]
    nav_date = reference_context["nav_date"]
    peak_nav_date = reference_context["peak_nav_date"]
    peak_nav_value = reference_context["peak_nav_value"]
    hist_momentum = reference_context["hist_momentum"]
    regime_stats = reference_context["regime_stats"]

    market_session_label = classify_market_session(now)
    estimated_live_nav = None
    coverage = None
    live_nav_date = None
    live_nav = None
    live_nav_coverage = None
    if should_include_realtime_snapshot(market_session_label):
        estimated_live_nav, coverage = estimate_live_nav(close_result, current_nav, spot_prices, raw_closes)
    if estimated_live_nav is not None:
        live_nav = estimated_live_nav
        live_nav_date = str(prices.index[-1].date())
        live_nav_coverage = coverage
    elif spot_prices:
        live_nav = current_nav
        live_nav_date = nav_date
        live_nav_coverage = coverage
    elif "nav" in result.columns and not result["nav"].dropna().empty:
        live_nav = float(result["nav"].dropna().iloc[-1])
        live_nav_date = str(pd.Timestamp(result["nav"].dropna().index[-1]).date())

    can_promote_live_nav = live_nav is not None and (live_nav_coverage is None or live_nav_coverage >= 0.999)
    if can_promote_live_nav:
        reference_peak = peak_nav_value if peak_nav_value is not None else live_nav
        effective_peak = max(reference_peak, live_nav)
        if effective_peak > 0:
            current_drawdown = live_nav / effective_peak - 1.0
        if peak_nav_value is None or live_nav >= peak_nav_value - 1e-12:
            peak_nav_value = live_nav
            peak_nav_date = live_nav_date

    risk_context = build_entry_risk_context(
        current_mom=current_mom,
        current_drawdown=current_drawdown,
        max_drawdown=max_drawdown,
        hist_momentum=hist_momentum,
        regime_stats=regime_stats,
    )

    momentum_lookback_days = DEFAULT_LOOKBACK
    momentum_start_date = None
    momentum_end_date = str(pd.Timestamp(latest_idx).date())
    latest_position = result.index.get_loc(latest_idx)
    if isinstance(latest_position, slice):
        latest_position = latest_position.stop - 1
    if isinstance(latest_position, int) and latest_position >= DEFAULT_LOOKBACK:
        momentum_start_date = str(pd.Timestamp(result.index[latest_position - DEFAULT_LOOKBACK]).date())

    return {
        "current_mom": current_mom,
        "effective_mom": effective_mom,
        "max_drawdown": max_drawdown,
        "current_drawdown": current_drawdown,
        "current_nav": current_nav,
        "nav_date": nav_date,
        "peak_nav_date": peak_nav_date,
        "peak_nav_value": peak_nav_value,
        "market_session_label": market_session_label,
        "live_nav": live_nav,
        "live_nav_date": live_nav_date,
        "live_nav_coverage": live_nav_coverage,
        "risk_context": risk_context,
        "momentum_lookback_days": momentum_lookback_days,
        "momentum_start_date": momentum_start_date,
        "momentum_end_date": momentum_end_date,
    }


def build_position_and_trade_context(
    *,
    latest_idx: pd.Timestamp,
    confirmed_latest_idx: pd.Timestamp,
    confirmed_prev_idx: pd.Timestamp,
    result: pd.DataFrame,
    close_result: pd.DataFrame,
    close_trades: pd.DataFrame,
    confirmed_holding: pd.Series,
    confirmed_exposure_series: pd.Series,
    current_signal_code_norm: str | None,
    current_code_norm: str | None,
    previous_code_norm: str | None,
    desired_exposure: float,
    theme_map: dict[str, str],
    name_map: dict[str, str],
    prices: pd.DataFrame,
    spot_prices: dict[str, float],
    raw_closes: dict[str, float],
    market_session_label: str,
    base_target_exposure: pd.Series,
) -> dict[str, object]:
    current_allocations = extract_weight_allocations(close_result.loc[confirmed_latest_idx], prefix="weight_")
    previous_allocations = extract_weight_allocations(close_result.loc[confirmed_prev_idx], prefix="weight_")
    desired_allocations = extract_weight_allocations(result.loc[latest_idx], prefix="target_weight_")
    current_holding_date = str(pd.Timestamp(confirmed_latest_idx).date())
    previous_holding_date = str(pd.Timestamp(confirmed_prev_idx).date())
    current_exposure = float(confirmed_exposure_series.loc[confirmed_latest_idx]) if pd.notna(confirmed_exposure_series.loc[confirmed_latest_idx]) else None
    previous_exposure = float(confirmed_exposure_series.loc[confirmed_prev_idx]) if pd.notna(confirmed_exposure_series.loc[confirmed_prev_idx]) else None
    if not current_allocations and current_exposure not in (None, 0.0) and current_code_norm is not None:
        current_allocations = [(current_code_norm, float(current_exposure))]
    if not previous_allocations and previous_exposure not in (None, 0.0) and previous_code_norm is not None:
        previous_allocations = [(previous_code_norm, float(previous_exposure))]
    if not desired_allocations and desired_exposure > 0 and current_signal_code_norm is not None:
        desired_allocations = [(current_signal_code_norm, desired_exposure)]

    previous_portfolio = format_portfolio_allocations(previous_allocations, theme_map, name_map)
    current_portfolio = format_portfolio_allocations(current_allocations, theme_map, name_map)
    desired_portfolio = format_portfolio_allocations(desired_allocations, theme_map, name_map)
    desired_code_norm = desired_allocations[0][0] if desired_allocations else None
    price_context = build_asset_price_context(current_code_norm, prices, spot_prices, raw_closes, market_session_label)
    current_price = price_context["current_price"]
    previous_close_price = price_context["previous_close_price"]
    intraday_price_return = price_context["intraday_price_return"]
    base_exposure = float(base_target_exposure.loc[latest_idx]) if pd.notna(base_target_exposure.loc[latest_idx]) else None

    top2_close_cap_triggered = False
    top2_close_gap = None
    top2_close_risk_cap = None
    if "top2_close_risk_cap_triggered" in result.columns and latest_idx in result.index:
        top2_close_cap_triggered = bool(result.loc[latest_idx, "top2_close_risk_cap_triggered"])
        if "top2_close_gap" in result.columns and pd.notna(result.loc[latest_idx, "top2_close_gap"]):
            top2_close_gap = float(result.loc[latest_idx, "top2_close_gap"])
        if "top2_close_risk_cap" in result.columns and pd.notna(result.loc[latest_idx, "top2_close_risk_cap"]):
            top2_close_risk_cap = float(result.loc[latest_idx, "top2_close_risk_cap"])

    extra_cap_triggered = (
        current_signal_code_norm is not None
        and base_exposure is not None
        and desired_exposure is not None
        and base_exposure > desired_exposure
    )

    trade_rows = pd.DataFrame()
    confirmed_trade_date = None
    trade_previous_allocations = previous_allocations
    trade_current_allocations = current_allocations
    if not close_trades.empty and "date" in close_trades.columns:
        trade_dates = pd.to_datetime(close_trades["date"]).dt.date
        eligible_dates = sorted({d for d in trade_dates if d <= pd.Timestamp(confirmed_latest_idx).date()})
        if eligible_dates:
            latest_trade_date = eligible_dates[-1]
            confirmed_trade_date = str(latest_trade_date)
            trade_rows = close_trades.loc[trade_dates == latest_trade_date].copy()
            trade_idx = pd.Timestamp(latest_trade_date)
            if trade_idx in close_result.index:
                trade_current_allocations = extract_weight_allocations(close_result.loc[trade_idx], prefix="weight_")
                trade_position = close_result.index.get_loc(trade_idx)
                if isinstance(trade_position, slice):
                    trade_position = trade_position.stop - 1
                if isinstance(trade_position, int) and trade_position > 0:
                    prev_trade_idx = close_result.index[trade_position - 1]
                    trade_previous_allocations = extract_weight_allocations(close_result.loc[prev_trade_idx], prefix="weight_")
                    prev_trade_code = normalize_code(confirmed_holding.loc[prev_trade_idx])
                    prev_trade_exposure = float(confirmed_exposure_series.loc[prev_trade_idx]) if pd.notna(confirmed_exposure_series.loc[prev_trade_idx]) else None
                    if not trade_previous_allocations and prev_trade_exposure not in (None, 0.0) and prev_trade_code is not None:
                        trade_previous_allocations = [(prev_trade_code, float(prev_trade_exposure))]
                trade_code = normalize_code(confirmed_holding.loc[trade_idx])
                trade_exposure = float(confirmed_exposure_series.loc[trade_idx]) if pd.notna(confirmed_exposure_series.loc[trade_idx]) else None
                if not trade_current_allocations and trade_exposure not in (None, 0.0) and trade_code is not None:
                    trade_current_allocations = [(trade_code, float(trade_exposure))]

    confirmed_trade_parts: list[str] = []
    action_map = {"BUY": "买入", "SELL": "卖出", "ADD": "加仓", "REDUCE": "减仓"}
    for row in trade_rows.itertuples(index=False):
        action_text = action_map.get(str(row.action), str(row.action))
        from_exposure = getattr(row, "from_exposure", None)
        to_exposure = getattr(row, "to_exposure", None)
        exposure_text = ""
        if from_exposure is not None and to_exposure is not None and pd.notna(from_exposure) and pd.notna(to_exposure):
            from monitor_render import format_exposure_transition_for_display
            display_transition = format_exposure_transition_for_display(float(from_exposure), float(to_exposure))
            if display_transition is None:
                continue
            exposure_text = f" {display_transition}"
        confirmed_trade_parts.append(f"{action_text} {row.theme} / {row.name} ({row.code}){exposure_text}")
    confirmed_trade_details = "；".join(confirmed_trade_parts) if confirmed_trade_parts else "无"

    return {
        "current_allocations": current_allocations,
        "previous_allocations": previous_allocations,
        "desired_allocations": desired_allocations,
        "current_holding_date": current_holding_date,
        "previous_holding_date": previous_holding_date,
        "current_exposure": current_exposure,
        "previous_exposure": previous_exposure,
        "previous_portfolio": previous_portfolio,
        "current_portfolio": current_portfolio,
        "desired_portfolio": desired_portfolio,
        "desired_code_norm": desired_code_norm,
        "current_price": current_price,
        "previous_close_price": previous_close_price,
        "intraday_price_return": intraday_price_return,
        "base_exposure": base_exposure,
        "top2_close_cap_triggered": top2_close_cap_triggered,
        "top2_close_gap": top2_close_gap,
        "top2_close_risk_cap": top2_close_risk_cap,
        "extra_cap_triggered": extra_cap_triggered,
        "confirmed_trade_date": confirmed_trade_date,
        "trade_previous_allocations": trade_previous_allocations,
        "trade_current_allocations": trade_current_allocations,
        "confirmed_trade_details": confirmed_trade_details,
    }



def build_signal_snapshot(
    snapshot_cls,
    *,
    prices: pd.DataFrame,
    strategy_config: dict[str, object],
    result: pd.DataFrame,
    close_result: pd.DataFrame,
    close_trades: pd.DataFrame,
    base_target_exposure: pd.Series,
    market_proxy_context: dict[str, float | str | None],
    spot_prices: dict[str, float],
    raw_closes: dict[str, float],
    now: datetime,
):
    selected_pool = strategy_config["selected_pool"]
    signal = result["signal"]
    confirmed_holding = close_result["holding"] if "holding" in close_result.columns else close_result["signal"]
    confirmed_exposure_series = close_result["exposure"] if "exposure" in close_result.columns else close_result["target_exposure"]
    desired_target_exposure = result["target_exposure"]

    latest_idx = signal.index[-1]
    prev_idx = signal.index[-2]
    current_signal_code = signal.loc[latest_idx]
    previous_signal_code = signal.loc[prev_idx]
    confirmed_latest_idx = confirmed_holding.index[-1]
    confirmed_prev_idx = confirmed_holding.index[-2]
    current_code = confirmed_holding.loc[confirmed_latest_idx]
    previous_code = confirmed_holding.loc[confirmed_prev_idx]

    name_map = {item["code"]: item["name"] for item in selected_pool}
    theme_map = {item["code"]: item["theme"] for item in selected_pool}

    leader_context = build_leader_context(
        latest_idx=latest_idx,
        prices=prices,
        strategy_id=str(strategy_config['strategy_id']),
        theme_map=theme_map,
        name_map=name_map,
    )
    reference_and_risk = build_reference_and_risk_context(
        latest_idx=latest_idx,
        result=result,
        close_result=close_result,
        backtest_nav_file=Path(strategy_config["backtest_nav_file"]),
        prices=prices,
        spot_prices=spot_prices,
        raw_closes=raw_closes,
        now=now,
    )
    current_mom = reference_and_risk["current_mom"]
    effective_mom = reference_and_risk["effective_mom"]
    max_drawdown = reference_and_risk["max_drawdown"]
    current_drawdown = reference_and_risk["current_drawdown"]
    current_nav = reference_and_risk["current_nav"]
    nav_date = reference_and_risk["nav_date"]
    peak_nav_date = reference_and_risk["peak_nav_date"]
    peak_nav_value = reference_and_risk["peak_nav_value"]
    market_session_label = reference_and_risk["market_session_label"]
    live_nav = reference_and_risk["live_nav"]
    live_nav_date = reference_and_risk["live_nav_date"]
    live_nav_coverage = reference_and_risk["live_nav_coverage"]
    risk_context = reference_and_risk["risk_context"]
    momentum_lookback_days = reference_and_risk["momentum_lookback_days"]
    momentum_start_date = reference_and_risk["momentum_start_date"]
    momentum_end_date = reference_and_risk["momentum_end_date"]

    current_signal_code_norm = normalize_code(current_signal_code)
    previous_signal_code_norm = normalize_code(previous_signal_code)
    current_code_norm = normalize_code(current_code)
    previous_code_norm = normalize_code(previous_code)
    desired_exposure = float(desired_target_exposure.loc[latest_idx]) if pd.notna(desired_target_exposure.loc[latest_idx]) else 0.0

    position_and_trade = build_position_and_trade_context(
        latest_idx=latest_idx,
        confirmed_latest_idx=confirmed_latest_idx,
        confirmed_prev_idx=confirmed_prev_idx,
        result=result,
        close_result=close_result,
        close_trades=close_trades,
        confirmed_holding=confirmed_holding,
        confirmed_exposure_series=confirmed_exposure_series,
        current_signal_code_norm=current_signal_code_norm,
        current_code_norm=current_code_norm,
        previous_code_norm=previous_code_norm,
        desired_exposure=desired_exposure,
        theme_map=theme_map,
        name_map=name_map,
        prices=prices,
        spot_prices=spot_prices,
        raw_closes=raw_closes,
        market_session_label=market_session_label,
        base_target_exposure=base_target_exposure,
    )
    current_allocations = position_and_trade["current_allocations"]
    previous_allocations = position_and_trade["previous_allocations"]
    desired_allocations = position_and_trade["desired_allocations"]
    current_holding_date = position_and_trade["current_holding_date"]
    previous_holding_date = position_and_trade["previous_holding_date"]
    current_exposure = position_and_trade["current_exposure"]
    previous_exposure = position_and_trade["previous_exposure"]
    previous_portfolio = position_and_trade["previous_portfolio"]
    current_portfolio = position_and_trade["current_portfolio"]
    desired_portfolio = position_and_trade["desired_portfolio"]
    desired_code_norm = position_and_trade["desired_code_norm"]
    current_price = position_and_trade["current_price"]
    previous_close_price = position_and_trade["previous_close_price"]
    intraday_price_return = position_and_trade["intraday_price_return"]
    base_exposure = position_and_trade["base_exposure"]
    top2_close_cap_triggered = position_and_trade["top2_close_cap_triggered"]
    top2_close_gap = position_and_trade["top2_close_gap"]
    top2_close_risk_cap = position_and_trade["top2_close_risk_cap"]
    extra_cap_triggered = position_and_trade["extra_cap_triggered"]
    confirmed_trade_date = position_and_trade["confirmed_trade_date"]
    trade_previous_allocations = position_and_trade["trade_previous_allocations"]
    trade_current_allocations = position_and_trade["trade_current_allocations"]
    confirmed_trade_details = position_and_trade["confirmed_trade_details"]
    confirmed_trade_reason = build_confirmed_trade_reason(
        trade_previous_allocations=trade_previous_allocations,
        trade_current_allocations=trade_current_allocations,
        strategy_id=str(strategy_config["strategy_id"]),
        current_momentum=current_mom,
        effective_momentum=effective_mom,
        current_drawdown=current_drawdown,
        base_exposure=base_exposure,
        desired_exposure=desired_exposure,
        extra_cap_triggered=extra_cap_triggered,
        top2_close_cap_triggered=top2_close_cap_triggered,
        top2_close_gap=top2_close_gap,
        top2_close_risk_cap=top2_close_risk_cap,
        theme_map=theme_map,
        name_map=name_map,
        risk_leader=leader_context['risk_leader'],
        risk_leader_name=leader_context['risk_leader_name'],
        risk_leader_theme=leader_context['risk_leader_theme'],
        risk_leader_momentum=leader_context['risk_leader_momentum'],
        defensive_leader=leader_context['defensive_leader'],
        defensive_leader_name=leader_context['defensive_leader_name'],
        defensive_leader_theme=leader_context['defensive_leader_theme'],
        defensive_leader_momentum=leader_context['defensive_leader_momentum'],
    )

    trade_details, changed = build_trade_details_from_allocations(
        current_allocations=current_allocations,
        desired_allocations=desired_allocations,
        theme_map=theme_map,
        name_map=name_map,
    )
    pending_trade_reason = build_confirmed_trade_reason(
        trade_previous_allocations=current_allocations,
        trade_current_allocations=desired_allocations,
        strategy_id=str(strategy_config["strategy_id"]),
        current_momentum=current_mom,
        effective_momentum=effective_mom,
        current_drawdown=current_drawdown,
        base_exposure=base_exposure,
        desired_exposure=desired_exposure,
        extra_cap_triggered=extra_cap_triggered,
        top2_close_cap_triggered=top2_close_cap_triggered,
        top2_close_gap=top2_close_gap,
        top2_close_risk_cap=top2_close_risk_cap,
        theme_map=theme_map,
        name_map=name_map,
        risk_leader=leader_context['risk_leader'],
        risk_leader_name=leader_context['risk_leader_name'],
        risk_leader_theme=leader_context['risk_leader_theme'],
        risk_leader_momentum=leader_context['risk_leader_momentum'],
        defensive_leader=leader_context['defensive_leader'],
        defensive_leader_name=leader_context['defensive_leader_name'],
        defensive_leader_theme=leader_context['defensive_leader_theme'],
        defensive_leader_momentum=leader_context['defensive_leader_momentum'],
    )

    return snapshot_cls(
        strategy_id=str(strategy_config["strategy_id"]),
        strategy_label=str(strategy_config["strategy_label"]),
        trade_date=str(latest_idx.date()),
        previous_signal=previous_signal_code_norm,
        previous_signal_name=name_map.get(previous_signal_code_norm) if previous_signal_code_norm else None,
        previous_signal_theme=theme_map.get(previous_signal_code_norm) if previous_signal_code_norm else None,
        current_signal=current_signal_code_norm,
        current_signal_name=name_map.get(current_signal_code_norm) if current_signal_code_norm else None,
        current_signal_theme=theme_map.get(current_signal_code_norm) if current_signal_code_norm else None,
        previous_holding_date=previous_holding_date,
        current_holding_date=current_holding_date,
        previous_holding=previous_code_norm,
        previous_name=name_map.get(previous_code_norm) if previous_code_norm else None,
        previous_theme=theme_map.get(previous_code_norm) if previous_code_norm else None,
        previous_exposure=previous_exposure,
        previous_portfolio=previous_portfolio,
        current_holding=current_code_norm,
        current_name=name_map.get(current_code_norm) if current_code_norm else None,
        current_theme=theme_map.get(current_code_norm) if current_code_norm else None,
        current_portfolio=current_portfolio,
        desired_holding=desired_code_norm,
        desired_name=name_map.get(desired_code_norm) if desired_code_norm else None,
        desired_theme=theme_map.get(desired_code_norm) if desired_code_norm else None,
        desired_exposure=desired_exposure,
        desired_portfolio=desired_portfolio,
        risk_leader=leader_context['risk_leader'],
        risk_leader_name=leader_context['risk_leader_name'],
        risk_leader_theme=leader_context['risk_leader_theme'],
        risk_leader_momentum=leader_context['risk_leader_momentum'],
        defensive_leader=leader_context['defensive_leader'],
        defensive_leader_name=leader_context['defensive_leader_name'],
        defensive_leader_theme=leader_context['defensive_leader_theme'],
        defensive_leader_momentum=leader_context['defensive_leader_momentum'],
        momentum_lookback_days=momentum_lookback_days,
        momentum_start_date=momentum_start_date,
        momentum_end_date=momentum_end_date,
        current_momentum=current_mom,
        effective_momentum=effective_mom,
        current_exposure=current_exposure,
        current_price=current_price,
        previous_close_price=previous_close_price,
        intraday_price_return=intraday_price_return,
        market_session_label=market_session_label,
        live_nav_coverage=live_nav_coverage,
        market_volume_mode=str(market_proxy_context.get("mode")) if market_proxy_context.get("mode") is not None else None,
        market_volume_progress=float(market_proxy_context["progress"]) if market_proxy_context.get("progress") is not None else None,
        market_amount_ratio_20_60=(float(market_proxy_context["market_amount_ratio_20_60"]) if market_proxy_context.get("market_amount_ratio_20_60") is not None else None),
        market_amount_ratio_5_20=(float(market_proxy_context["market_amount_ratio_5_20"]) if market_proxy_context.get("market_amount_ratio_5_20") is not None else None),
        market_breadth_proxy=(float(market_proxy_context["market_breadth_proxy"]) if market_proxy_context.get("market_breadth_proxy") is not None else None),
        max_drawdown=max_drawdown,
        nav_date=nav_date,
        current_nav=current_nav,
        live_nav_date=live_nav_date,
        live_nav=live_nav,
        current_drawdown=current_drawdown,
        peak_nav_date=peak_nav_date,
        peak_nav_value=peak_nav_value,
        entry_risk_score=risk_context["entry_risk_score"],
        entry_risk_level=risk_context["entry_risk_level"],
        historical_regime_label=risk_context["historical_regime_label"],
        historical_regime_count=risk_context["historical_regime_count"],
        historical_win_rate_60=risk_context["historical_win_rate_60"],
        historical_win_rate_60_percentile=risk_context["historical_win_rate_60_percentile"],
        historical_avg_ret_60=risk_context["historical_avg_ret_60"],
        historical_avg_mdd_60=risk_context["historical_avg_mdd_60"],
        historical_avg_ret_60_percentile=risk_context["historical_avg_ret_60_percentile"],
        momentum_percentile=risk_context["momentum_percentile"],
        drawdown_buffer_ratio=risk_context["drawdown_buffer_ratio"],
        entry_advice=risk_context["entry_advice"],
        extra_cap_triggered=extra_cap_triggered,
        extra_cap_label=str(strategy_config["extra_cap_label"]),
        top2_close_cap_triggered=top2_close_cap_triggered,
        top2_close_gap=top2_close_gap,
        top2_close_risk_cap=top2_close_risk_cap,
        base_exposure=base_exposure,
        confirmed_trade_date=confirmed_trade_date,
        confirmed_trade_details=confirmed_trade_details,
        confirmed_trade_reason=confirmed_trade_reason,
        trade_details=trade_details,
        pending_trade_reason=pending_trade_reason,
        changed=changed,
    )
