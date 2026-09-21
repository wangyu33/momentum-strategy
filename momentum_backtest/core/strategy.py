"""正式策略核心主链。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from .backtest import (
    build_trades_from_weight_frame,
    build_weight_frame_from_holding_exposure,
    compute_signal_asset_momentum,
    finalize_position_columns,
    recompute_return_chain,
    run_threshold_dual_strategy,
    run_target_weights_strategy,
)
from .config import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_ASSET_PERCENTILE_CAP_CODES,
    DEFAULT_ASSET_PERCENTILE_CAP_EXTREME_CAP,
    DEFAULT_ASSET_PERCENTILE_CAP_HIGH,
    DEFAULT_ASSET_PERCENTILE_CAP_HIGH_CAP,
    DEFAULT_ASSET_PERCENTILE_CAP_MID,
    DEFAULT_ASSET_PERCENTILE_CAP_MID_CAP,
    DEFAULT_ASSET_PERCENTILE_CAP_MIN_PERIODS,
    DEFAULT_ASSET_PERCENTILE_CAP_START,
    DEFAULT_ASSET_PERCENTILE_CAP_STATE_LOOKBACK,
    DEFAULT_BASELINE_DROP_CODES,
    DEFAULT_BOLL_EXTREME_HOT_BANDWIDTH_PCT,
    DEFAULT_BOLL_EXTREME_HOT_CAP,
    DEFAULT_BOLL_HOT_BANDWIDTH_PCT,
    DEFAULT_BOLL_HOT_CAP,
    DEFAULT_CLOSE_TOP2_GAP,
    DEFAULT_CLOSE_TOP2_RISK_CAP,
    DEFAULT_DUAL_CASH_EXIT_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_LOOKBACK,
    DEFAULT_MA_TREND_CUT,
    DEFAULT_MA_TREND_WINDOW,
    DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT,
    DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT,
    DEFAULT_REGIME_MIX_MOMENTUM_CUT,
    DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
    DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
    DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
    DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_CUT,
    DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_EXPOSURE,
    DEFAULT_REGIME_MIX_PRE_OVERHEAT_START_CUT,
    DEFAULT_REGIME_MIX_PROXY_KIND,
    DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT,
    DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
    DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING,
    DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT,
    DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT,
    DEFAULT_RISK_POOL_INCLUDE_TREASURY,
    DEFAULT_SIGNAL_LEADER_MARGIN,
    DEFAULT_SIGNAL_QUALITY_METHOD,
    DEFAULT_SIGNAL_SLOPE_PENALTY,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_STRESS_BOND_BREADTH_CUT,
    DEFAULT_STRESS_BOND_CODE,
    DEFAULT_STRESS_BOND_ENTER_DAYS,
    DEFAULT_STRESS_BOND_EXIT_DAYS,
    DEFAULT_STRESS_BOND_FILL_RESIDUAL_CASH,
    DEFAULT_STRESS_BOND_RATIO_CUT,
    DEFAULT_STRESS_BOND_RISK_CAP,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    DEFAULT_YEARS,
    DEFENSIVE_CODES,
    RISK_CODES,
)
from .io import normalize_code
from .proxy import (
    build_dynamic_core_target_weights,
    build_parametrized_hs300_trend_filter,
    build_proxy_catalog,
    load_market_volume_proxy,
)
from .signals import (
    build_asset_own_momentum_percentile,
    build_asset_relative_volatility_percentile,
    build_quality_momentum_single_asset_signal,
    build_signal_quality_score,
    build_signal_stability_score,
    build_top2_close_risk_flag,
    choose_signal_winner_with_margin,
    filter_score_row_by_confirmation,
    resolve_strategy_universe,
)


def run_simple_quality_momentum_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = build_default_strategy_params()
    candidate_codes = [str(code) for code in selected["code"] if str(code) in prices.columns]
    signal, target_exposure, current_momentum, max_momentum = build_quality_momentum_single_asset_signal(
        prices,
        lookback=DEFAULT_LOOKBACK,
        signal_quality_method=str(params.get("signal_quality_method", DEFAULT_SIGNAL_QUALITY_METHOD)),
        slope_penalty=float(params.get("signal_slope_penalty", DEFAULT_SIGNAL_SLOPE_PENALTY)),
        volatility_penalty=float(params.get("signal_volatility_penalty", 0.0)),
        downside_volatility_penalty=float(params.get("signal_downside_volatility_penalty", 0.0)),
        r2_penalty=float(params.get("signal_r2_penalty", 0.0)),
        volatility_state_lookback=int(params.get("signal_volatility_state_lookback", 252)),
        volatility_percentile_penalty=float(params.get("signal_volatility_percentile_penalty", 0.0)),
        downside_volatility_percentile_penalty=float(params.get("signal_downside_volatility_percentile_penalty", 0.0)),
        volatility_percentile_divisor=float(params.get("signal_volatility_percentile_divisor", 0.0)),
        leader_margin=float(params.get("signal_leader_margin", 0.0)),
        candidate_codes=candidate_codes,
    )

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype="float64")
    for dt_idx in prices.index:
        code = normalize_code(signal.loc[dt_idx])
        if code is None or code not in target_weights.columns:
            continue
        exposure = float(target_exposure.loc[dt_idx]) if pd.notna(target_exposure.loc[dt_idx]) else 0.0
        if exposure > 0:
            target_weights.loc[dt_idx, code] = exposure

    result, _ = run_target_weights_strategy(prices, selected, target_weights, current_momentum, fee_rate, slippage_rate)
    return_weight_cols = [col for col in result.columns if col.startswith("weight_")]
    return_weights = result[return_weight_cols].copy()
    return_weights.columns = [col.removeprefix("weight_") for col in return_weight_cols]
    result = finalize_position_columns(
        result,
        selected,
        confirmed_weights=target_weights,
        return_weights=return_weights,
        return_holding=result["holding"] if "holding" in result.columns else None,
        return_exposure=result["exposure"] if "exposure" in result.columns else None,
    )
    result = recompute_return_chain(result, prices, fee_rate=fee_rate, slippage_rate=slippage_rate)
    trades = build_trades_from_weight_frame(target_weights, result["nav"], selected)

    selected_momentum_percentile = pd.Series(index=prices.index, dtype="float64", name="selected_momentum_percentile")
    signal_momentum_percentile = build_asset_own_momentum_percentile(
        prices[candidate_codes],
        lookback=DEFAULT_LOOKBACK,
        state_lookback=756,
        min_periods=120,
    )
    for dt_idx in prices.index:
        code = normalize_code(signal.loc[dt_idx])
        if code is None or code not in signal_momentum_percentile.columns:
            continue
        pct_value = signal_momentum_percentile.loc[dt_idx, code]
        if pd.notna(pct_value):
            selected_momentum_percentile.loc[dt_idx] = float(pct_value)

    signal_asset_momentum = compute_signal_asset_momentum(prices, result["signal"], DEFAULT_LOOKBACK)
    result["current_momentum"] = signal_asset_momentum
    result["max_momentum"] = max_momentum.reindex(result.index)
    result["effective_momentum"] = signal_asset_momentum
    result["signal_asset_momentum"] = signal_asset_momentum
    result["base_target_exposure"] = target_weights.sum(axis=1).rename("base_target_exposure")
    result["target_exposure"] = target_weights.sum(axis=1).rename("target_exposure")
    result["selected_momentum_percentile"] = selected_momentum_percentile.reindex(result.index)
    result["selected_momentum_pct_cap_triggered"] = False
    result["defensive_signal_momentum_pct_cap_triggered"] = False
    result["weak_market_trigger"] = False
    result["top2_close_risk_cap_triggered"] = False
    result["extra_cap_triggered"] = False
    result["extra_cap_reason"] = pd.Series(pd.NA, index=result.index, dtype="object")
    result["stress_bond_trigger"] = False
    result = pd.concat([result, target_weights.add_prefix("target_weight_")], axis=1)
    return result, trades


def apply_selected_signal_momentum_pct_cap_to_target_weights(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    cap_start: float,
    cap_end: float,
    cap_floor: float,
    lookback: int = DEFAULT_LOOKBACK,
    state_lookback: int = 756,
    min_periods: int = 120,
    scope: str = "all",
    risk_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if cap_floor <= 0 or cap_floor > 1.0:
        raise ValueError("cap_floor must be within (0, 1]")
    if cap_end < cap_start:
        raise ValueError("cap_end must be >= cap_start")

    adjusted = target_weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0).fillna(0.0).copy()
    signal = adjusted.idxmax(axis=1).where(adjusted.max(axis=1) > 1e-12, pd.NA)
    selected_pct = pd.Series(index=prices.index, dtype="float64", name="selected_momentum_percentile")
    triggered = pd.Series(False, index=prices.index, dtype=bool, name="selected_momentum_pct_cap_triggered")
    signal_pct = build_asset_own_momentum_percentile(
        prices,
        lookback=lookback,
        state_lookback=state_lookback,
        min_periods=min_periods,
    )
    risk_set = set(risk_codes or [])
    all_set = set(prices.columns)
    if scope == "all":
        eligible_codes = all_set
    elif scope == "risk":
        eligible_codes = risk_set
    elif scope == "defensive":
        eligible_codes = all_set - risk_set
    else:
        raise ValueError(f"unsupported momentum pct cap scope: {scope}")

    for dt_idx in adjusted.index:
        signal_code = normalize_code(signal.loc[dt_idx])
        if signal_code is None or signal_code not in eligible_codes or signal_code not in signal_pct.columns:
            continue
        pct_value = signal_pct.loc[dt_idx, signal_code]
        if pd.isna(pct_value):
            continue
        selected_pct.loc[dt_idx] = float(pct_value)
        if float(pct_value) < cap_start:
            continue
        if cap_end > cap_start:
            progress = min(max((float(pct_value) - cap_start) / (cap_end - cap_start), 0.0), 1.0)
            cap_value = 1.0 + (cap_floor - 1.0) * progress
        else:
            cap_value = cap_floor
        total_weight = float(adjusted.loc[dt_idx].sum())
        if total_weight > cap_value + 1e-12:
            adjusted.loc[dt_idx, :] = adjusted.loc[dt_idx, :] * (cap_value / total_weight)
            triggered.loc[dt_idx] = True

    return adjusted, selected_pct, triggered


def apply_boll_hot_cap_to_target_weights(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    cap: float = DEFAULT_BOLL_HOT_CAP,
    bandwidth_pct_cut: float = DEFAULT_BOLL_HOT_BANDWIDTH_PCT,
    extreme_cap: float = DEFAULT_BOLL_EXTREME_HOT_CAP,
    extreme_bandwidth_pct_cut: float = DEFAULT_BOLL_EXTREME_HOT_BANDWIDTH_PCT,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    adjusted = target_weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0).fillna(0.0).copy()
    signal = adjusted.idxmax(axis=1).where(adjusted.max(axis=1) > 1e-12, pd.NA)

    ma20 = prices.rolling(20, min_periods=20).mean()
    std20 = prices.rolling(20, min_periods=20).std(ddof=0)
    upper20 = ma20 + 2.0 * std20
    lower20 = ma20 - 2.0 * std20
    bandwidth = ((upper20 - lower20) / ma20.replace(0.0, pd.NA)).replace([float("inf"), -float("inf")], pd.NA)
    bandwidth_pct120 = bandwidth.rolling(120, min_periods=30).apply(
        lambda arr: (
            (pd.Series(arr).dropna() < pd.Series(arr).dropna().iloc[-1]).sum()
            + 0.5 * (pd.Series(arr).dropna() == pd.Series(arr).dropna().iloc[-1]).sum()
        )
        / max(len(pd.Series(arr).dropna()), 1)
        if len(pd.Series(arr).dropna()) > 0
        else float("nan"),
        raw=False,
    )

    triggered = pd.Series(False, index=adjusted.index, dtype=bool, name="boll_hot_cap_triggered")
    extreme_triggered = pd.Series(False, index=adjusted.index, dtype=bool, name="boll_extreme_hot_cap_triggered")
    for dt_idx in adjusted.index:
        signal_code = normalize_code(signal.loc[dt_idx])
        if signal_code is None or signal_code not in adjusted.columns:
            continue

        original_total_weight = float(adjusted.loc[dt_idx].sum())
        if original_total_weight <= cap + 1e-12:
            continue

        price_now = prices.loc[dt_idx, signal_code]
        upper_now = upper20.loc[dt_idx, signal_code]
        bandwidth_pct = bandwidth_pct120.loc[dt_idx, signal_code]
        if pd.isna(price_now) or pd.isna(upper_now) or pd.isna(bandwidth_pct):
            continue
        if float(price_now) < float(upper_now) or float(bandwidth_pct) < bandwidth_pct_cut:
            continue

        triggered.loc[dt_idx] = True
        target_cap = cap
        if float(bandwidth_pct) >= extreme_bandwidth_pct_cut:
            target_cap = min(target_cap, extreme_cap)
            extreme_triggered.loc[dt_idx] = True
        adjusted.loc[dt_idx, :] = adjusted.loc[dt_idx, :] * (target_cap / original_total_weight)

    return adjusted, triggered, extreme_triggered


def apply_asset_momentum_percentile_cap_to_target_weights(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    asset_codes: list[str] | None = None,
    cap_start: float = DEFAULT_ASSET_PERCENTILE_CAP_START,
    cap_mid: float = DEFAULT_ASSET_PERCENTILE_CAP_MID,
    mid_cap: float = DEFAULT_ASSET_PERCENTILE_CAP_MID_CAP,
    cap_high: float = DEFAULT_ASSET_PERCENTILE_CAP_HIGH,
    high_cap: float = DEFAULT_ASSET_PERCENTILE_CAP_HIGH_CAP,
    extreme_cap: float = DEFAULT_ASSET_PERCENTILE_CAP_EXTREME_CAP,
    state_lookback: int = DEFAULT_ASSET_PERCENTILE_CAP_STATE_LOOKBACK,
    min_periods: int = DEFAULT_ASSET_PERCENTILE_CAP_MIN_PERIODS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    adjusted = target_weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0).fillna(0.0).copy()
    selected_pct = pd.Series(index=adjusted.index, dtype="float64", name="asset_momentum_percentile")
    cap_series = pd.Series(1.0, index=adjusted.index, dtype="float64", name="asset_momentum_percentile_cap")
    triggered = pd.Series(False, index=adjusted.index, dtype=bool, name="asset_momentum_percentile_cap_triggered")
    triggered_asset = pd.Series(pd.NA, index=adjusted.index, dtype="object", name="asset_momentum_percentile_cap_code")

    effective_asset_codes = [
        code for code in (DEFAULT_ASSET_PERCENTILE_CAP_CODES if asset_codes is None else asset_codes)
        if code in adjusted.columns and code in prices.columns
    ]
    if not effective_asset_codes:
        return adjusted, selected_pct, cap_series, triggered, triggered_asset

    signal_pct = build_asset_own_momentum_percentile(
        prices[effective_asset_codes],
        lookback=DEFAULT_LOOKBACK,
        state_lookback=state_lookback,
        min_periods=min_periods,
    )

    for asset_code in effective_asset_codes:
        for dt_idx in adjusted.index:
            asset_weight = float(adjusted.loc[dt_idx, asset_code]) if pd.notna(adjusted.loc[dt_idx, asset_code]) else 0.0
            if asset_weight <= 1e-12:
                continue

            pct_value = signal_pct.loc[dt_idx, asset_code]
            if pd.isna(pct_value):
                continue

            pct_value = float(pct_value)
            selected_pct.loc[dt_idx] = pct_value
            triggered_asset.loc[dt_idx] = asset_code
            cap_value = None
            if cap_start <= pct_value < cap_mid:
                progress = min(max((pct_value - cap_start) / (cap_mid - cap_start), 0.0), 1.0)
                cap_value = 1.0 + (mid_cap - 1.0) * progress
            elif cap_mid <= pct_value < cap_high:
                if cap_high > cap_mid:
                    progress = min(max((pct_value - cap_mid) / (cap_high - cap_mid), 0.0), 1.0)
                    cap_value = mid_cap + (high_cap - mid_cap) * progress
                else:
                    cap_value = high_cap
            elif pct_value >= cap_high:
                cap_value = extreme_cap

            if cap_value is None:
                continue

            total_weight = float(adjusted.loc[dt_idx].sum())
            cap_series.loc[dt_idx] = cap_value
            if total_weight > cap_value + 1e-12:
                adjusted.loc[dt_idx, :] = adjusted.loc[dt_idx, :] * (cap_value / total_weight)
                triggered.loc[dt_idx] = True

    return adjusted, selected_pct, cap_series, triggered, triggered_asset


def apply_defensive_signal_momentum_pct_cap_to_target_weights(
    target_weights: pd.DataFrame,
    *,
    selected_percentile: pd.Series,
    cap_start: float,
    cap_floor: float,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if cap_floor <= 0 or cap_floor > 1.0:
        raise ValueError("cap_floor must be within (0, 1]")
    if cap_start < 0 or cap_start > 1.0:
        raise ValueError("cap_start must be within [0, 1]")

    adjusted = target_weights.copy()
    triggered = pd.Series(False, index=adjusted.index, dtype=bool, name="defensive_signal_momentum_pct_cap_triggered")
    defensive_set = set(defensive_codes or [])
    if not defensive_set or adjusted.empty:
        return adjusted, triggered

    signal = adjusted.idxmax(axis=1).where(adjusted.max(axis=1) > 1e-12, pd.NA)
    aligned_pct = selected_percentile.reindex(adjusted.index)
    for dt_idx in adjusted.index:
        signal_code = normalize_code(signal.loc[dt_idx])
        if signal_code is None or signal_code not in defensive_set:
            continue
        pct_value = aligned_pct.loc[dt_idx]
        if pd.isna(pct_value) or float(pct_value) < cap_start:
            continue
        total_weight = float(adjusted.loc[dt_idx].sum())
        if total_weight > cap_floor + 1e-12:
            adjusted.loc[dt_idx, :] = adjusted.loc[dt_idx, :] * (cap_floor / total_weight)
            triggered.loc[dt_idx] = True
    return adjusted, triggered


def apply_min_rebalance_threshold_to_target_weights(
    target_weights: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.Series]:
    adjusted = target_weights.copy()
    blocked = pd.Series(False, index=adjusted.index, dtype=bool, name="rebalance_threshold_blocked")
    if threshold <= 0 or adjusted.empty:
        return adjusted, blocked

    previous = adjusted.iloc[0].copy()
    for dt_idx in adjusted.index[1:]:
        desired = adjusted.loc[dt_idx]
        turnover = float((desired - previous).abs().sum())
        if turnover < threshold:
            adjusted.loc[dt_idx] = previous
            blocked.loc[dt_idx] = True
        else:
            previous = desired.copy()
    return adjusted, blocked


def build_absolute_momentum_threshold_series(
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str] | object],
) -> pd.Series:
    index = prices.index
    proxy = proxy.reindex(index).ffill()

    base_threshold = float(params.get("absolute_momentum_threshold", DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD))
    threshold = pd.Series(base_threshold, index=index, dtype="float64", name="absolute_momentum_threshold")
    mode = str(params.get("dynamic_threshold_mode", "fixed"))

    if mode == "fixed":
        return threshold

    amount_20_60 = proxy["market_amount_ratio_20_60"]
    amount_5_20 = proxy["market_amount_ratio_5_20"]
    breadth = proxy["market_breadth_proxy"]

    if mode == "weak_proxy_step":
        weak_mask = (
            (amount_20_60 < float(params.get("dynamic_threshold_amount_20_60_cut", 1.0)))
            & (amount_5_20 < float(params.get("dynamic_threshold_amount_5_20_cut", 1.0)))
            & (breadth < float(params.get("dynamic_threshold_breadth_cut", 0.0)))
        ).fillna(False)
        threshold.loc[weak_mask] = float(params.get("dynamic_threshold_weak_value", base_threshold))
        return threshold

    if mode == "weak_proxy_tiered":
        tier_step = float(params.get("dynamic_threshold_tier_step", 0.005))
        max_threshold = float(params.get("dynamic_threshold_max", base_threshold))
        weak_score = pd.Series(0.0, index=index, dtype="float64")
        weak_score = weak_score.add((amount_20_60 < float(params.get("dynamic_threshold_amount_20_60_cut", 1.0))).astype(float), fill_value=0.0)
        weak_score = weak_score.add((amount_5_20 < float(params.get("dynamic_threshold_amount_5_20_cut", 1.0))).astype(float), fill_value=0.0)
        weak_score = weak_score.add((breadth < float(params.get("dynamic_threshold_breadth_cut", 0.0))).astype(float), fill_value=0.0)
        threshold = (base_threshold + weak_score * tier_step).clip(upper=max_threshold)
        threshold.name = "absolute_momentum_threshold"
        return threshold

    if mode == "asymmetric_band":
        weak_mask = (
            (amount_20_60 < float(params.get("dynamic_threshold_amount_20_60_cut", 1.0)))
            & (breadth < float(params.get("dynamic_threshold_breadth_cut", 0.0)))
        ).fillna(False)
        strong_mask = (
            (amount_20_60 > float(params.get("dynamic_threshold_strong_amount_20_60_cut", 1.05)))
            & (breadth > float(params.get("dynamic_threshold_strong_breadth_cut", 0.02)))
        ).fillna(False)
        threshold.loc[weak_mask] = float(params.get("dynamic_threshold_weak_value", base_threshold))
        threshold.loc[strong_mask] = float(params.get("dynamic_threshold_strong_value", base_threshold))
        return threshold.clip(lower=0.0)

    raise ValueError(f"unsupported dynamic_threshold_mode: {mode}")


def build_dynamic_core_target_weights_with_transition(
    prices: pd.DataFrame,
    lookback: int,
    absolute_threshold: float | pd.Series,
    weak_trend_defensive_weight: float,
    core_weight: float,
    hs300_mom60_cut: float,
    hs300_mom120_cut: float,
    hs300_ma_window: int,
    signal_quality_method: str = "raw",
    slope_penalty: float = 0.0,
    volatility_penalty: float = 0.0,
    downside_volatility_penalty: float = 0.0,
    r2_penalty: float = 0.0,
    volatility_state_lookback: int = 252,
    volatility_percentile_penalty: float = 0.0,
    downside_volatility_percentile_penalty: float = 0.0,
    volatility_percentile_divisor: float = 0.0,
    confirmation_lookback: int = 0,
    confirmation_top_n: int = 0,
    secondary_stability_method: str = "none",
    secondary_stability_gap: float = 0.0,
    leader_margin: float = 0.0,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
    transition_mode: str = "step",
    transition_start_cut: float = 0.0,
    transition_end_cut: float | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    threshold_is_series = isinstance(absolute_threshold, pd.Series)
    absolute_threshold_series = None
    if threshold_is_series:
        absolute_threshold_series = absolute_threshold.reindex(prices.index).astype("float64")

    if transition_mode != "continuous" and not threshold_is_series:
        return build_dynamic_core_target_weights(
            prices,
            lookback=lookback,
            absolute_threshold=float(absolute_threshold),
            weak_trend_defensive_weight=weak_trend_defensive_weight,
            core_weight=core_weight,
            hs300_mom60_cut=hs300_mom60_cut,
            hs300_mom120_cut=hs300_mom120_cut,
            hs300_ma_window=hs300_ma_window,
            signal_quality_method=signal_quality_method,
            slope_penalty=slope_penalty,
            volatility_penalty=volatility_penalty,
            downside_volatility_penalty=downside_volatility_penalty,
            r2_penalty=r2_penalty,
            volatility_state_lookback=volatility_state_lookback,
            volatility_percentile_penalty=volatility_percentile_penalty,
            downside_volatility_percentile_penalty=downside_volatility_percentile_penalty,
            volatility_percentile_divisor=volatility_percentile_divisor,
            confirmation_lookback=confirmation_lookback,
            confirmation_top_n=confirmation_top_n,
            secondary_stability_method=secondary_stability_method,
            secondary_stability_gap=secondary_stability_gap,
            leader_margin=leader_margin,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
        )

    if transition_end_cut is None:
        transition_end_cut = absolute_threshold
    if not threshold_is_series and transition_end_cut <= transition_start_cut:
        raise ValueError("transition_end_cut must be greater than transition_start_cut")

    momentum, score = build_signal_quality_score(
        prices,
        lookback=lookback,
        method=signal_quality_method,
        slope_penalty=slope_penalty,
        volatility_penalty=volatility_penalty,
        downside_volatility_penalty=downside_volatility_penalty,
        r2_penalty=r2_penalty,
        volatility_state_lookback=volatility_state_lookback,
        volatility_percentile_penalty=volatility_percentile_penalty,
        downside_volatility_percentile_penalty=downside_volatility_percentile_penalty,
        volatility_percentile_divisor=volatility_percentile_divisor,
    )
    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    risk_score = score[active_risk_codes]
    defensive_score = score[active_defensive_codes]
    risk_confirmation = None
    if confirmation_lookback > 0 and confirmation_top_n > 0:
        risk_confirmation = prices[active_risk_codes] / prices[active_risk_codes].shift(confirmation_lookback) - 1
    secondary_score = build_signal_stability_score(prices, lookback=lookback, method=secondary_stability_method)
    risk_secondary_score = secondary_score[active_risk_codes] if secondary_score is not None else None
    defensive_secondary_score = secondary_score[active_defensive_codes] if secondary_score is not None else None

    satellite_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype="float64")
    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    prev_signal: str | None = None

    for dt_idx in prices.index:
        prev_risk = prev_signal if prev_signal in active_risk_codes else None
        prev_def = prev_signal if prev_signal in active_defensive_codes else None
        risk_score_row = risk_score.loc[dt_idx]
        if risk_confirmation is not None:
            risk_score_row = filter_score_row_by_confirmation(risk_score_row, risk_confirmation.loc[dt_idx], confirmation_top_n)
        r_asset = choose_signal_winner_with_margin(
            risk_score_row,
            prev_risk,
            leader_margin,
            None if risk_secondary_score is None else risk_secondary_score.loc[dt_idx],
            secondary_stability_gap,
        )
        d_asset = choose_signal_winner_with_margin(
            defensive_score.loc[dt_idx],
            prev_def,
            leader_margin,
            None if defensive_secondary_score is None else defensive_secondary_score.loc[dt_idx],
            secondary_stability_gap,
        )
        r_score = float(momentum.loc[dt_idx, r_asset]) if r_asset and pd.notna(momentum.loc[dt_idx, r_asset]) else float("nan")
        d_score = float(momentum.loc[dt_idx, d_asset]) if d_asset and pd.notna(momentum.loc[dt_idx, d_asset]) else float("nan")

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
            prev_signal = None
            continue

        threshold_value = float(absolute_threshold_series.loc[dt_idx]) if absolute_threshold_series is not None else float(absolute_threshold)
        transition_end_value = threshold_value if absolute_threshold_series is not None else float(transition_end_cut)
        if transition_mode == "continuous" and transition_end_value <= transition_start_cut:
            raise ValueError("transition_end_cut must be greater than transition_start_cut")

        if pd.notna(r_score) and r_score > threshold_value:
            satellite_weights.loc[dt_idx, str(r_asset)] = 1.0
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = float(r_score)
            prev_signal = str(r_asset) if r_asset else None
            continue

        if pd.notna(r_score) and r_score > transition_start_cut:
            if transition_mode == "continuous":
                blend_ratio = (float(r_score) - transition_start_cut) / (transition_end_value - transition_start_cut)
                blend_ratio = min(max(blend_ratio, 0.0), 1.0)
                risk_weight = blend_ratio
                defensive_weight = 0.0
                if pd.notna(d_asset):
                    defensive_weight = (1.0 - blend_ratio) * weak_trend_defensive_weight
                    satellite_weights.loc[dt_idx, str(d_asset)] = defensive_weight
                if pd.notna(r_asset):
                    satellite_weights.loc[dt_idx, str(r_asset)] = risk_weight
                target_exposure.loc[dt_idx] = risk_weight + defensive_weight
                current_momentum.loc[dt_idx] = float(r_score)
                if risk_weight >= defensive_weight and pd.notna(r_asset):
                    signal.loc[dt_idx] = r_asset
                    prev_signal = str(r_asset)
                elif pd.notna(d_asset):
                    signal.loc[dt_idx] = d_asset
                    prev_signal = str(d_asset)
                else:
                    signal.loc[dt_idx] = r_asset
                    prev_signal = str(r_asset) if pd.notna(r_asset) else None
                continue

            if pd.notna(d_asset):
                signal.loc[dt_idx] = d_asset
                target_exposure.loc[dt_idx] = weak_trend_defensive_weight
                satellite_weights.loc[dt_idx, str(d_asset)] = weak_trend_defensive_weight
                current_momentum.loc[dt_idx] = float(r_score)
                prev_signal = str(d_asset)
                continue

        if pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            satellite_weights.loc[dt_idx, str(d_asset)] = target_exposure.loc[dt_idx]
            current_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
            prev_signal = str(d_asset)
            continue

        signal.loc[dt_idx] = pd.NA
        target_exposure.loc[dt_idx] = 0.0
        current_momentum.loc[dt_idx] = float("nan")
        prev_signal = None

    trend_filter = build_parametrized_hs300_trend_filter(
        prices,
        mom60_cut=hs300_mom60_cut,
        mom120_cut=hs300_mom120_cut,
        ma_window=hs300_ma_window,
    )
    hs300_code = "510300"
    hs300_mom60 = prices[hs300_code] / prices[hs300_code].shift(60) - 1
    dynamic_core_weight = trend_filter.astype(float) * core_weight
    satellite_scale = 1.0 - dynamic_core_weight

    target_weights = satellite_weights.mul(satellite_scale, axis=0)
    target_weights[hs300_code] = target_weights.get(hs300_code, 0.0) + dynamic_core_weight
    blended_momentum = (satellite_scale * current_momentum + dynamic_core_weight * hs300_mom60).rename("current_momentum")
    return target_weights, blended_momentum, signal, target_exposure, trend_filter


def build_official_target_weights(
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str] | object],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        defensive_codes=[str(code) for code in params.get("defensive_codes", DEFENSIVE_CODES)],
    )
    active_signal_codes = list(dict.fromkeys([*active_risk_codes, *active_defensive_codes]))
    transition_mode = str(params.get("regime_transition_mode", "step"))
    transition_start_cut = float(params.get("regime_transition_start_cut", 0.0))
    transition_end_cut = float(params.get("regime_transition_end_cut", DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD))
    absolute_threshold_series = build_absolute_momentum_threshold_series(prices, proxy, params)

    aggressive_weights, aggressive_momentum, _, _, aggressive_trend = build_dynamic_core_target_weights_with_transition(
        prices,
        lookback=DEFAULT_LOOKBACK,
        absolute_threshold=absolute_threshold_series,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        core_weight=float(params["aggressive_core_weight"]),
        hs300_mom60_cut=0.05,
        hs300_mom120_cut=0.10,
        hs300_ma_window=90,
        signal_quality_method=str(params.get("signal_quality_method", "raw")),
        slope_penalty=float(params.get("signal_slope_penalty", 0.0)),
        volatility_penalty=float(params.get("signal_volatility_penalty", 0.0)),
        downside_volatility_penalty=float(params.get("signal_downside_volatility_penalty", 0.0)),
        r2_penalty=float(params.get("signal_r2_penalty", 0.0)),
        volatility_state_lookback=int(params.get("signal_volatility_state_lookback", 252)),
        volatility_percentile_penalty=float(params.get("signal_volatility_percentile_penalty", 0.0)),
        downside_volatility_percentile_penalty=float(params.get("signal_downside_volatility_percentile_penalty", 0.0)),
        volatility_percentile_divisor=float(params.get("signal_volatility_percentile_divisor", 0.0)),
        confirmation_lookback=int(params.get("signal_confirmation_lookback", 0)),
        confirmation_top_n=int(params.get("signal_confirmation_top_n", 0)),
        secondary_stability_method=str(params.get("signal_secondary_stability_method", "none")),
        secondary_stability_gap=float(params.get("signal_secondary_stability_gap", 0.0)),
        leader_margin=float(params.get("signal_leader_margin", 0.0)),
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
        transition_mode=transition_mode,
        transition_start_cut=transition_start_cut,
        transition_end_cut=transition_end_cut,
    )
    conservative_weights, conservative_momentum, _, _, _ = build_dynamic_core_target_weights_with_transition(
        prices,
        lookback=DEFAULT_LOOKBACK,
        absolute_threshold=absolute_threshold_series,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        core_weight=float(params["conservative_core_weight"]),
        hs300_mom60_cut=0.04,
        hs300_mom120_cut=0.10,
        hs300_ma_window=120,
        signal_quality_method=str(params.get("signal_quality_method", "raw")),
        slope_penalty=float(params.get("signal_slope_penalty", 0.0)),
        volatility_penalty=float(params.get("signal_volatility_penalty", 0.0)),
        downside_volatility_penalty=float(params.get("signal_downside_volatility_penalty", 0.0)),
        r2_penalty=float(params.get("signal_r2_penalty", 0.0)),
        volatility_state_lookback=int(params.get("signal_volatility_state_lookback", 252)),
        volatility_percentile_penalty=float(params.get("signal_volatility_percentile_penalty", 0.0)),
        downside_volatility_percentile_penalty=float(params.get("signal_downside_volatility_percentile_penalty", 0.0)),
        volatility_percentile_divisor=float(params.get("signal_volatility_percentile_divisor", 0.0)),
        confirmation_lookback=int(params.get("signal_confirmation_lookback", 0)),
        confirmation_top_n=int(params.get("signal_confirmation_top_n", 0)),
        secondary_stability_method=str(params.get("signal_secondary_stability_method", "none")),
        secondary_stability_gap=float(params.get("signal_secondary_stability_gap", 0.0)),
        leader_margin=float(params.get("signal_leader_margin", 0.0)),
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
        transition_mode=transition_mode,
        transition_start_cut=transition_start_cut,
        transition_end_cut=transition_end_cut,
    )
    aggressive_mask = aggressive_trend & (aggressive_momentum >= float(params["regime_momentum_cut"]))
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    mixed_momentum = conservative_momentum.copy()
    mixed_momentum.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    mixed_momentum = mixed_momentum.rename("current_momentum")

    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = mixed_target_weights[active_risk_codes].sum(axis=1)
    volume_weak_mask = (
        (proxy["market_amount_ratio_20_60"] < float(params["volume_ratio_cut"]))
        & (proxy["market_amount_ratio_5_20"] < float(params["volume_short_ratio_cut"]))
        & (proxy["market_breadth_proxy"] < float(params["volume_breadth_cut"]))
        & (mixed_momentum <= float(params["volume_guard_momentum_ceiling"]))
    ).fillna(False)

    capped_target_weights = mixed_target_weights.copy()
    weak_cap = float(params["volume_guard_cap"])
    weak_scale_mask = volume_weak_mask & (row_risk_weight > weak_cap)
    if weak_scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[weak_scale_mask] = weak_cap / row_risk_weight.loc[weak_scale_mask]
        capped_target_weights.loc[weak_scale_mask, active_risk_codes] = capped_target_weights.loc[
            weak_scale_mask, active_risk_codes
        ].mul(scale.loc[weak_scale_mask], axis=0)

    risk_weight_after_weak_guard = capped_target_weights[active_risk_codes].sum(axis=1)
    overheat_stability_method = str(params.get("overheat_stability_method", "none"))
    overheat_stability_min_rank = float(params.get("overheat_stability_min_rank", 0.0))
    overheat_stability_cap = float(params.get("overheat_stability_cap", 1.0))
    close_gap = float(params.get("close_top2_gap", 0.0))
    close_risk_cap = float(params.get("close_top2_risk_cap", 1.0))
    top2_close_mask = build_top2_close_risk_flag(
        prices,
        lookback=DEFAULT_LOOKBACK,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        signal_quality_method=str(params.get("signal_quality_method", "raw")),
        slope_penalty=float(params.get("signal_slope_penalty", 0.0)),
        volatility_penalty=float(params.get("signal_volatility_penalty", 0.0)),
        downside_volatility_penalty=float(params.get("signal_downside_volatility_penalty", 0.0)),
        r2_penalty=float(params.get("signal_r2_penalty", 0.0)),
        volatility_state_lookback=int(params.get("signal_volatility_state_lookback", 252)),
        volatility_percentile_penalty=float(params.get("signal_volatility_percentile_penalty", 0.0)),
        downside_volatility_percentile_penalty=float(params.get("signal_downside_volatility_percentile_penalty", 0.0)),
        volatility_percentile_divisor=float(params.get("signal_volatility_percentile_divisor", 0.0)),
        confirmation_lookback=int(params.get("signal_confirmation_lookback", 0)),
        confirmation_top_n=int(params.get("signal_confirmation_top_n", 0)),
        risk_codes=active_risk_codes,
        close_gap=close_gap,
    )
    close_scale_mask = top2_close_mask & (risk_weight_after_weak_guard > close_risk_cap)
    if close_scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[close_scale_mask] = close_risk_cap / risk_weight_after_weak_guard.loc[close_scale_mask]
        capped_target_weights.loc[close_scale_mask, active_risk_codes] = capped_target_weights.loc[
            close_scale_mask, active_risk_codes
        ].mul(scale.loc[close_scale_mask], axis=0)

    selected_volatility_rank = pd.Series(index=prices.index, dtype="float64", name="selected_volatility_rank")
    selected_volatility_cap_triggered = pd.Series(False, index=prices.index, dtype=bool, name="selected_volatility_cap_triggered")
    selected_momentum_percentile = pd.Series(index=prices.index, dtype="float64", name="selected_momentum_percentile")
    selected_momentum_pct_cap_triggered = pd.Series(False, index=prices.index, dtype=bool, name="selected_momentum_pct_cap_triggered")
    preliminary_result = None
    preliminary_signal = None
    vol_cap_start = float(params.get("signal_selected_volatility_cap_start", 1.0))
    vol_cap_end = float(params.get("signal_selected_volatility_cap_end", 1.0))
    vol_cap_floor = float(params.get("signal_selected_volatility_cap_floor", 1.0))
    if vol_cap_floor <= 0 or vol_cap_floor > 1.0:
        raise ValueError("signal_selected_volatility_cap_floor must be within (0, 1]")
    if vol_cap_start < 1.0 or vol_cap_end < 1.0:
        if vol_cap_end < vol_cap_start:
            raise ValueError("signal_selected_volatility_cap_end must be >= signal_selected_volatility_cap_start")

        preliminary_result, _ = run_target_weights_strategy(
            prices,
            pd.DataFrame({"code": list(prices.columns), "theme": list(prices.columns), "name": list(prices.columns)}),
            capped_target_weights,
            mixed_momentum,
            DEFAULT_FEE_RATE,
            DEFAULT_SLIPPAGE_RATE,
        )
        preliminary_signal = preliminary_result["signal"].astype("object")
        signal_volatility_rank = build_asset_relative_volatility_percentile(
            prices[active_risk_codes],
            vol_window=20,
            state_lookback=int(params.get("signal_volatility_state_lookback", 252)),
        )
        for dt_idx in prices.index:
            signal_code = preliminary_signal.loc[dt_idx]
            if pd.isna(signal_code) or signal_code not in signal_volatility_rank.columns:
                continue
            selected_volatility_rank.loc[dt_idx] = float(signal_volatility_rank.loc[dt_idx, str(signal_code)])

        risk_weight_before_vol_cap = capped_target_weights[active_risk_codes].sum(axis=1)
        high_vol_mask = selected_volatility_rank >= vol_cap_start
        if high_vol_mask.any():
            cap_series = pd.Series(1.0, index=prices.index, dtype="float64")
            if vol_cap_end > vol_cap_start:
                progress = ((selected_volatility_rank.loc[high_vol_mask] - vol_cap_start) / (vol_cap_end - vol_cap_start)).clip(lower=0.0, upper=1.0)
                cap_series.loc[high_vol_mask] = 1.0 + (vol_cap_floor - 1.0) * progress
            else:
                cap_series.loc[high_vol_mask] = vol_cap_floor

            scale_mask = high_vol_mask & (risk_weight_before_vol_cap > cap_series)
            selected_volatility_cap_triggered.loc[scale_mask] = True
            if scale_mask.any():
                scale = pd.Series(1.0, index=prices.index, dtype="float64")
                scale.loc[scale_mask] = cap_series.loc[scale_mask] / risk_weight_before_vol_cap.loc[scale_mask]
                capped_target_weights.loc[scale_mask, active_risk_codes] = capped_target_weights.loc[
                    scale_mask, active_risk_codes
                ].mul(scale.loc[scale_mask], axis=0)

    momentum_pct_cap_stage = str(params.get("signal_selected_momentum_pct_cap_stage", "core"))
    momentum_pct_cap_start = float(params.get("signal_selected_momentum_pct_cap_start", 1.0))
    momentum_pct_cap_end = float(params.get("signal_selected_momentum_pct_cap_end", 1.0))
    momentum_pct_cap_floor = float(params.get("signal_selected_momentum_pct_cap_floor", 1.0))
    if momentum_pct_cap_floor <= 0 or momentum_pct_cap_floor > 1.0:
        raise ValueError("signal_selected_momentum_pct_cap_floor must be within (0, 1]")
    if momentum_pct_cap_stage == "core" and (momentum_pct_cap_start < 1.0 or momentum_pct_cap_end < 1.0):
        if momentum_pct_cap_end < momentum_pct_cap_start:
            raise ValueError("signal_selected_momentum_pct_cap_end must be >= signal_selected_momentum_pct_cap_start")

        if preliminary_result is None:
            preliminary_result, _ = run_target_weights_strategy(
                prices,
                pd.DataFrame({"code": list(prices.columns), "theme": list(prices.columns), "name": list(prices.columns)}),
                capped_target_weights,
                mixed_momentum,
                DEFAULT_FEE_RATE,
                DEFAULT_SLIPPAGE_RATE,
            )
            preliminary_signal = preliminary_result["signal"].astype("object")

        signal_momentum_percentile = build_asset_own_momentum_percentile(
            prices[active_signal_codes],
            lookback=DEFAULT_LOOKBACK,
            state_lookback=int(params.get("signal_selected_momentum_pct_lookback", 756)),
            min_periods=int(params.get("signal_selected_momentum_pct_min_periods", 120)),
        )
        for dt_idx in prices.index:
            signal_code = preliminary_signal.loc[dt_idx]
            if pd.isna(signal_code) or signal_code not in signal_momentum_percentile.columns:
                continue
            selected_momentum_percentile.loc[dt_idx] = float(signal_momentum_percentile.loc[dt_idx, str(signal_code)])

        total_weight_before_momentum_pct_cap = capped_target_weights.sum(axis=1)
        high_momentum_pct_mask = selected_momentum_percentile >= momentum_pct_cap_start
        if high_momentum_pct_mask.any():
            cap_series = pd.Series(1.0, index=prices.index, dtype="float64")
            if momentum_pct_cap_end > momentum_pct_cap_start:
                progress = (
                    (selected_momentum_percentile.loc[high_momentum_pct_mask] - momentum_pct_cap_start)
                    / (momentum_pct_cap_end - momentum_pct_cap_start)
                ).clip(lower=0.0, upper=1.0)
                cap_series.loc[high_momentum_pct_mask] = 1.0 + (momentum_pct_cap_floor - 1.0) * progress
            else:
                cap_series.loc[high_momentum_pct_mask] = momentum_pct_cap_floor

            scale_mask = high_momentum_pct_mask & (total_weight_before_momentum_pct_cap > cap_series)
            selected_momentum_pct_cap_triggered.loc[scale_mask] = True
            if scale_mask.any():
                scale = pd.Series(1.0, index=prices.index, dtype="float64")
                scale.loc[scale_mask] = cap_series.loc[scale_mask] / total_weight_before_momentum_pct_cap.loc[scale_mask]
                capped_target_weights.loc[scale_mask, :] = capped_target_weights.loc[scale_mask, :].mul(
                    scale.loc[scale_mask],
                    axis=0,
                )

    base_result, _ = run_target_weights_strategy(
        prices,
        pd.DataFrame({"code": list(prices.columns), "theme": list(prices.columns), "name": list(prices.columns)}),
        capped_target_weights,
        mixed_momentum,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
    )
    base_result["weak_market_trigger"] = weak_scale_mask.reindex(base_result.index).fillna(False)
    base_result["absolute_momentum_threshold"] = absolute_threshold_series.reindex(base_result.index).ffill()
    base_result["top2_close_risk_cap_triggered"] = close_scale_mask.reindex(base_result.index).fillna(False)
    base_result["top2_close_gap"] = close_gap
    base_result["top2_close_risk_cap"] = close_risk_cap
    base_result["selected_volatility_rank"] = selected_volatility_rank.reindex(base_result.index)
    base_result["selected_volatility_cap_triggered"] = selected_volatility_cap_triggered.reindex(base_result.index).fillna(False)
    base_result["selected_momentum_percentile"] = selected_momentum_percentile.reindex(base_result.index)
    base_result["selected_momentum_pct_cap_triggered"] = selected_momentum_pct_cap_triggered.reindex(base_result.index).fillna(False)
    selected_stability_rank = None
    if overheat_stability_method != "none" and overheat_stability_min_rank > 0:
        stability_score = build_signal_stability_score(prices, lookback=DEFAULT_LOOKBACK, method=overheat_stability_method)
        if stability_score is not None:
            active_stability_score = stability_score[active_risk_codes]
            stability_rank = active_stability_score.rank(axis=1, pct=True, method="average")
            selected_signal = base_result["signal"].astype("object")
            selected_stability_rank = pd.Series(index=prices.index, dtype="float64", name="selected_stability_rank")
            for dt_idx in prices.index:
                signal_code = selected_signal.loc[dt_idx]
                if pd.isna(signal_code) or signal_code not in stability_rank.columns:
                    continue
                selected_stability_rank.loc[dt_idx] = float(stability_rank.loc[dt_idx, str(signal_code)])
    risk_weight_after_weak_guard = capped_target_weights[active_risk_codes].sum(axis=1)

    extra_cap_triggered = pd.Series(False, index=prices.index, dtype=bool, name="extra_cap_triggered")
    extra_cap_reason = pd.Series(pd.NA, index=prices.index, dtype="object", name="extra_cap_reason")
    overheat_cap_triggered = pd.Series(False, index=prices.index, dtype=bool, name="overheat_cap_triggered")
    overheat_high_cap_triggered = pd.Series(False, index=prices.index, dtype=bool, name="overheat_high_cap_triggered")
    overheat_stability_cap_triggered = pd.Series(False, index=prices.index, dtype=bool, name="overheat_stability_cap_triggered")

    overheat_mode = str(params.get("overheat_cap_mode", "step"))
    if overheat_mode == "continuous":
        pre_start_cut = float(params.get("pre_overheat_start_cut", params["overheat_momentum_cut"]))
        pre_end_cut = float(params.get("pre_overheat_end_cut", params["overheat_momentum_cut"]))
        pre_end_exposure = float(params.get("pre_overheat_end_exposure", 1.0))
        overheat_cap = float(params["overheat_max_exposure"])
        overheat_momentum_cut = float(params["overheat_momentum_cut"])
        overheat_high_cap = float(params["overheat_high_max_exposure"])
        overheat_high_momentum_cut = float(params["overheat_high_momentum_cut"])
        drawdown_cut = float(params["overheat_drawdown_cut"])

        if pre_end_cut < pre_start_cut:
            raise ValueError("pre_overheat_end_cut must be >= pre_overheat_start_cut")
        if overheat_momentum_cut < pre_end_cut:
            raise ValueError("overheat_momentum_cut must be >= pre_overheat_end_cut in continuous mode")
        if overheat_high_momentum_cut <= overheat_momentum_cut:
            raise ValueError("overheat_high_momentum_cut must be greater than overheat_momentum_cut in continuous mode")

        cap_series = pd.Series(float("inf"), index=prices.index, dtype="float64")
        active_mask = (base_result["drawdown"] >= drawdown_cut) & (risk_weight_after_weak_guard > 0)

        pre_mask = active_mask & (mixed_momentum >= pre_start_cut) & (mixed_momentum < pre_end_cut)
        if pre_mask.any() and pre_end_cut > pre_start_cut:
            pre_progress = ((mixed_momentum.loc[pre_mask] - pre_start_cut) / (pre_end_cut - pre_start_cut)).clip(lower=0.0, upper=1.0)
            cap_series.loc[pre_mask] = 1.0 + (pre_end_exposure - 1.0) * pre_progress
        elif pre_mask.any():
            cap_series.loc[pre_mask] = pre_end_exposure

        mid_mask = active_mask & (mixed_momentum >= pre_end_cut) & (mixed_momentum < overheat_momentum_cut)
        if mid_mask.any() and overheat_momentum_cut > pre_end_cut:
            mid_progress = ((mixed_momentum.loc[mid_mask] - pre_end_cut) / (overheat_momentum_cut - pre_end_cut)).clip(lower=0.0, upper=1.0)
            cap_series.loc[mid_mask] = pre_end_exposure + (overheat_cap - pre_end_exposure) * mid_progress
        elif mid_mask.any():
            cap_series.loc[mid_mask] = overheat_cap

        high_mask = active_mask & (mixed_momentum >= overheat_momentum_cut)
        if high_mask.any():
            momentum_progress = ((mixed_momentum.loc[high_mask] - overheat_momentum_cut) / (overheat_high_momentum_cut - overheat_momentum_cut)).clip(lower=0.0, upper=1.0)
            continuous_cap = overheat_cap + (overheat_high_cap - overheat_cap) * momentum_progress
            cap_series.loc[high_mask] = continuous_cap

        stability_clip_mask = pd.Series(False, index=prices.index, dtype=bool)
        if selected_stability_rank is not None and overheat_stability_cap < 1.0:
            unstable_mask = active_mask & (selected_stability_rank <= overheat_stability_min_rank)
            prior_cap_series = cap_series.copy()
            cap_series.loc[unstable_mask] = cap_series.loc[unstable_mask].clip(upper=overheat_stability_cap)
            stability_clip_mask = unstable_mask & (cap_series < prior_cap_series - 1e-12)

        continuous_mask = cap_series.replace(float("inf"), pd.NA).notna() & (risk_weight_after_weak_guard > cap_series)
        if continuous_mask.any():
            scale = pd.Series(1.0, index=prices.index, dtype="float64")
            scale.loc[continuous_mask] = cap_series.loc[continuous_mask] / risk_weight_after_weak_guard.loc[continuous_mask]
            capped_target_weights.loc[continuous_mask, active_risk_codes] = capped_target_weights.loc[
                continuous_mask, active_risk_codes
            ].mul(scale.loc[continuous_mask], axis=0)
            extra_cap_triggered.loc[continuous_mask] = True
            extra_cap_reason.loc[pre_mask & continuous_mask] = "pre_overheat_cap"
            extra_cap_reason.loc[mid_mask & continuous_mask] = "mid_overheat_cap"
            extra_cap_reason.loc[high_mask & continuous_mask] = "overheat_cap"
            overheat_cap_triggered.loc[(mid_mask | high_mask) & continuous_mask] = True
            overheat_high_cap_triggered.loc[high_mask & continuous_mask] = True
            if stability_clip_mask.any():
                extra_cap_reason.loc[stability_clip_mask & continuous_mask] = "stability_cap"
                overheat_stability_cap_triggered.loc[stability_clip_mask & continuous_mask] = True
    else:
        overheat_mask = (
            (risk_weight_after_weak_guard > float(params["overheat_max_exposure"]))
            & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
            & (mixed_momentum >= float(params["overheat_momentum_cut"]))
        )
        if overheat_mask.any():
            scale = pd.Series(1.0, index=prices.index, dtype="float64")
            scale.loc[overheat_mask] = float(params["overheat_max_exposure"]) / risk_weight_after_weak_guard.loc[overheat_mask]
            capped_target_weights.loc[overheat_mask, active_risk_codes] = capped_target_weights.loc[
                overheat_mask, active_risk_codes
            ].mul(scale.loc[overheat_mask], axis=0)
            extra_cap_triggered.loc[overheat_mask] = True
            extra_cap_reason.loc[overheat_mask] = "overheat_cap"
            overheat_cap_triggered.loc[overheat_mask] = True

        if selected_stability_rank is not None and overheat_stability_cap < 1.0:
            stability_mask = (
                (selected_stability_rank <= overheat_stability_min_rank)
                & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
                & (mixed_momentum >= float(params["overheat_momentum_cut"]))
            )
            row_risk_after_overheat = capped_target_weights[active_risk_codes].sum(axis=1)
            extra_cap_mask = stability_mask & (row_risk_after_overheat > overheat_stability_cap)
            if extra_cap_mask.any():
                scale = pd.Series(1.0, index=prices.index, dtype="float64")
                scale.loc[extra_cap_mask] = overheat_stability_cap / row_risk_after_overheat.loc[extra_cap_mask]
                capped_target_weights.loc[extra_cap_mask, active_risk_codes] = capped_target_weights.loc[
                    extra_cap_mask, active_risk_codes
                ].mul(scale.loc[extra_cap_mask], axis=0)
                extra_cap_triggered.loc[extra_cap_mask] = True
                extra_cap_reason.loc[extra_cap_mask] = "stability_cap"
                overheat_stability_cap_triggered.loc[extra_cap_mask] = True

        overheat_high_mask = (
            (capped_target_weights[active_risk_codes].sum(axis=1) > float(params["overheat_high_max_exposure"]))
            & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
            & (mixed_momentum >= float(params["overheat_high_momentum_cut"]))
        )
        if overheat_high_mask.any():
            scale = pd.Series(1.0, index=prices.index, dtype="float64")
            scale.loc[overheat_high_mask] = float(params["overheat_high_max_exposure"]) / capped_target_weights[active_risk_codes].sum(axis=1).loc[overheat_high_mask]
            capped_target_weights.loc[overheat_high_mask, active_risk_codes] = capped_target_weights.loc[
                overheat_high_mask, active_risk_codes
            ].mul(scale.loc[overheat_high_mask], axis=0)
            extra_cap_triggered.loc[overheat_high_mask] = True
            extra_cap_reason.loc[overheat_high_mask] = "overheat_high_cap"
            overheat_high_cap_triggered.loc[overheat_high_mask] = True

    base_result["extra_cap_triggered"] = extra_cap_triggered.reindex(base_result.index).fillna(False)
    base_result["extra_cap_reason"] = extra_cap_reason.reindex(base_result.index)
    base_result["overheat_cap_triggered"] = overheat_cap_triggered.reindex(base_result.index).fillna(False)
    base_result["overheat_high_cap_triggered"] = overheat_high_cap_triggered.reindex(base_result.index).fillna(False)
    base_result["overheat_stability_cap_triggered"] = overheat_stability_cap_triggered.reindex(base_result.index).fillna(False)
    return capped_target_weights, mixed_momentum, base_result


def build_default_strategy_params(
    drop_codes: list[str] | None = None,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> dict[str, object]:
    base_risk_codes = list(RISK_CODES if risk_codes is None else risk_codes)
    if DEFAULT_RISK_POOL_INCLUDE_TREASURY and DEFAULT_STRESS_BOND_CODE not in base_risk_codes:
        base_risk_codes.append(DEFAULT_STRESS_BOND_CODE)
    return {
        "drop_codes": list(DEFAULT_BASELINE_DROP_CODES if drop_codes is None else drop_codes),
        "risk_codes": base_risk_codes,
        "defensive_codes": list(DEFENSIVE_CODES if defensive_codes is None else defensive_codes),
        "proxy_kind": DEFAULT_REGIME_MIX_PROXY_KIND,
        "signal_quality_method": DEFAULT_SIGNAL_QUALITY_METHOD,
        "signal_slope_penalty": DEFAULT_SIGNAL_SLOPE_PENALTY,
        "signal_volatility_penalty": 0.0,
        "signal_downside_volatility_penalty": 0.0,
        "signal_r2_penalty": 0.0,
        "signal_volatility_state_lookback": 252,
        "signal_volatility_percentile_penalty": 0.0,
        "signal_downside_volatility_percentile_penalty": 0.0,
        "signal_volatility_percentile_divisor": 0.0,
        "signal_confirmation_lookback": 60,
        "signal_confirmation_top_n": 2,
        "signal_secondary_stability_method": "none",
        "signal_secondary_stability_gap": 0.0,
        "signal_leader_margin": DEFAULT_SIGNAL_LEADER_MARGIN,
        "close_top2_gap": DEFAULT_CLOSE_TOP2_GAP,
        "close_top2_risk_cap": DEFAULT_CLOSE_TOP2_RISK_CAP,
        "aggressive_core_weight": DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT,
        "conservative_core_weight": DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT,
        "regime_momentum_cut": DEFAULT_REGIME_MIX_MOMENTUM_CUT,
        "regime_transition_mode": "step",
        "regime_transition_start_cut": 0.0,
        "regime_transition_end_cut": DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        "absolute_momentum_threshold": DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        "dynamic_threshold_mode": "fixed",
        "dynamic_threshold_amount_20_60_cut": 1.0,
        "dynamic_threshold_amount_5_20_cut": 1.0,
        "dynamic_threshold_breadth_cut": 0.0,
        "dynamic_threshold_weak_value": 0.06,
        "dynamic_threshold_tier_step": 0.005,
        "dynamic_threshold_max": 0.065,
        "dynamic_threshold_strong_amount_20_60_cut": 1.05,
        "dynamic_threshold_strong_breadth_cut": 0.02,
        "dynamic_threshold_strong_value": 0.045,
        "volume_ratio_cut": DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT,
        "volume_short_ratio_cut": DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT,
        "volume_breadth_cut": DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT,
        "volume_guard_cap": DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
        "volume_guard_momentum_ceiling": DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING,
        "overheat_drawdown_cut": DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
        "pre_overheat_start_cut": DEFAULT_REGIME_MIX_PRE_OVERHEAT_START_CUT,
        "pre_overheat_end_cut": DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_CUT,
        "pre_overheat_end_exposure": DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_EXPOSURE,
        "overheat_momentum_cut": DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
        "overheat_max_exposure": DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
        "overheat_high_momentum_cut": DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
        "overheat_high_max_exposure": DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
        "overheat_cap_mode": "step",
        "overheat_stability_method": "none",
        "overheat_stability_min_rank": 0.0,
        "overheat_stability_cap": 1.0,
        "signal_selected_volatility_cap_start": 1.0,
        "signal_selected_volatility_cap_end": 1.0,
        "signal_selected_volatility_cap_floor": 1.0,
        "signal_selected_momentum_pct_cap_stage": "post_overlay",
        "signal_selected_momentum_pct_cap_start": 1.0,
        "signal_selected_momentum_pct_cap_end": 1.0,
        "signal_selected_momentum_pct_cap_floor": 1.0,
        "signal_selected_momentum_pct_lookback": 756,
        "signal_selected_momentum_pct_min_periods": 120,
        "signal_selected_momentum_pct_cap_scope": "all",
        "defensive_signal_selected_momentum_pct_cap_start": 1.0,
        "defensive_signal_selected_momentum_pct_cap_floor": 1.0,
        "target_min_rebalance_threshold": 0.0,
        "ma_trend_window": DEFAULT_MA_TREND_WINDOW,
        "ma_trend_cut": DEFAULT_MA_TREND_CUT,
    }


def build_persistent_trigger_mask(raw_trigger: pd.Series, enter_days: int, exit_days: int) -> pd.Series:
    active = False
    enter_streak = 0
    exit_streak = 0
    result: list[bool] = []

    for is_triggered in raw_trigger.fillna(False).astype(bool).tolist():
        if is_triggered:
            enter_streak += 1
            exit_streak = 0
        else:
            exit_streak += 1
            enter_streak = 0

        if not active and enter_streak >= enter_days:
            active = True
        elif active and exit_streak >= exit_days:
            active = False

        result.append(active)

    return pd.Series(result, index=raw_trigger.index, dtype=bool)


def apply_persistent_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str] | object],
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    enter_days: int,
    exit_days: int,
    fee_rate: float,
    slippage_rate: float,
    fill_residual_cash_to_treasury: bool = False,
    return_target_weights: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_weights, mixed_momentum, _ = build_official_target_weights(prices, proxy, params)
    configured_risk_codes = params.get("risk_codes")
    configured_defensive_codes = params.get("defensive_codes")
    active_risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in configured_risk_codes] if configured_risk_codes is not None else None,
        defensive_codes=[str(code) for code in configured_defensive_codes] if configured_defensive_codes is not None else None,
    )
    risk_budget_codes = list(active_risk_codes)
    if "510300" in prices.columns and "510300" not in risk_budget_codes:
        risk_budget_codes.append("510300")
    overlaid_weights = base_weights.copy()
    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_budget_codes].sum(axis=1)

    raw_trigger = (
        (proxy["market_amount_ratio_20_60"] < ratio_cut)
        & (proxy["market_breadth_proxy"] < breadth_cut)
    ).fillna(False)
    trigger_mask = build_persistent_trigger_mask(raw_trigger, enter_days=enter_days, exit_days=exit_days)

    scale_mask = trigger_mask & (row_risk_weight > risk_cap)
    if scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[scale_mask] = risk_cap / row_risk_weight.loc[scale_mask]
        overlaid_weights.loc[scale_mask, risk_budget_codes] = overlaid_weights.loc[scale_mask, risk_budget_codes].mul(
            scale.loc[scale_mask], axis=0
        )
        moved_weight = row_risk_weight.loc[scale_mask] - risk_cap
        overlaid_weights.loc[scale_mask, treasury_code] = overlaid_weights.loc[scale_mask, treasury_code].add(
            moved_weight,
            fill_value=0.0,
        )

    if fill_residual_cash_to_treasury:
        residual_cash = (1.0 - overlaid_weights.sum(axis=1)).clip(lower=0.0)
        residual_mask = trigger_mask & (residual_cash > 1e-12)
        if residual_mask.any():
            overlaid_weights.loc[residual_mask, treasury_code] = overlaid_weights.loc[residual_mask, treasury_code].add(
                residual_cash.loc[residual_mask],
                fill_value=0.0,
            )

    result, trades = run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)
    if return_target_weights:
        return result, trades, overlaid_weights
    return result, trades


def apply_ma_trend_cut_to_target_weights(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    window: int = DEFAULT_MA_TREND_WINDOW,
    cut: float = DEFAULT_MA_TREND_CUT,
) -> tuple[pd.DataFrame, pd.Series]:
    """持仓标的收盘价跌破自身 window 日均线时，按 cut 比例缩减当日目标权重。

    返回 (调整后权重, 触发掩码)。window<=0 或 cut>=1 时原样返回，相当于关闭。
    """
    triggered = pd.Series(False, index=target_weights.index, name="ma_trend_triggered")
    if window <= 0 or cut >= 1.0:
        return target_weights, triggered
    moving_avg = prices.rolling(window).mean()
    holding = target_weights.idxmax(axis=1)
    current_price = pd.Series(
        [
            float(prices.loc[dt, code])
            if code in prices.columns and pd.notna(prices.loc[dt, code])
            else np.nan
            for dt, code in zip(target_weights.index, holding)
        ],
        index=target_weights.index,
    )
    ma_price = pd.Series(
        [
            float(moving_avg.loc[dt, code])
            if code in moving_avg.columns and pd.notna(moving_avg.loc[dt, code])
            else np.nan
            for dt, code in zip(target_weights.index, holding)
        ],
        index=target_weights.index,
    )
    invested = target_weights.sum(axis=1) > 1e-9
    triggered = ((current_price < ma_price).fillna(False)) & invested
    triggered = triggered.rename("ma_trend_triggered")
    adjusted = target_weights.mul(np.where(triggered.to_numpy(), cut, 1.0), axis=0)
    return adjusted, triggered


def run_default_strategy_with_params(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    params: dict[str, object],
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    market_proxy: pd.DataFrame | None = None,
    treasury_code: str = DEFAULT_STRESS_BOND_CODE,
    risk_cap: float = DEFAULT_STRESS_BOND_RISK_CAP,
    ratio_cut: float = DEFAULT_STRESS_BOND_RATIO_CUT,
    breadth_cut: float = DEFAULT_STRESS_BOND_BREADTH_CUT,
    enter_days: int = DEFAULT_STRESS_BOND_ENTER_DAYS,
    exit_days: int = DEFAULT_STRESS_BOND_EXIT_DAYS,
    fill_residual_cash_to_treasury: bool = DEFAULT_STRESS_BOND_FILL_RESIDUAL_CASH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = max(int(math.ceil((prices.index.max() - prices.index.min()).days / 365.25)) + 1, DEFAULT_YEARS)
    if market_proxy is None:
        market_proxy = load_market_volume_proxy(years=years, refresh=False)
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(
            market_proxy,
            prices,
            risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        )
    }
    proxy_kind = str(params.get("proxy_kind", DEFAULT_REGIME_MIX_PROXY_KIND))
    if proxy_kind not in proxy_catalog:
        raise ValueError(f"unsupported proxy_kind: {proxy_kind}")
    effective_proxy = proxy_catalog[proxy_kind]

    base_weights, _, base_result = build_official_target_weights(prices, effective_proxy, params)
    risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        defensive_codes=[str(code) for code in params.get("defensive_codes", DEFENSIVE_CODES)],
    )
    proxy = effective_proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_codes].sum(axis=1)
    raw_trigger = (
        (proxy["market_amount_ratio_20_60"] < ratio_cut)
        & (proxy["market_breadth_proxy"] < breadth_cut)
    ).fillna(False)
    stress_mask = build_persistent_trigger_mask(raw_trigger, enter_days=enter_days, exit_days=exit_days)
    result, trades, target_weights = apply_persistent_overlay(
        prices,
        selected,
        effective_proxy,
        params,
        treasury_code=treasury_code,
        risk_cap=risk_cap,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
        enter_days=enter_days,
        exit_days=exit_days,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        fill_residual_cash_to_treasury=fill_residual_cash_to_treasury,
        return_target_weights=True,
    )
    internal_momentum = result["current_momentum"].copy().rename("current_momentum")
    post_overlay_selected_momentum_percentile = None
    post_overlay_selected_momentum_trigger = None
    post_overlay_defensive_momentum_trigger = None
    post_overlay_threshold_blocked = None

    momentum_pct_stage = str(params.get("signal_selected_momentum_pct_cap_stage", "core"))
    momentum_pct_cap_start = float(params.get("signal_selected_momentum_pct_cap_start", 1.0))
    momentum_pct_cap_end = float(params.get("signal_selected_momentum_pct_cap_end", 1.0))
    momentum_pct_cap_floor = float(params.get("signal_selected_momentum_pct_cap_floor", 1.0))
    if momentum_pct_stage == "post_overlay" and (momentum_pct_cap_start < 1.0 or momentum_pct_cap_end < 1.0):
        target_weights, post_overlay_selected_momentum_percentile, post_overlay_selected_momentum_trigger = (
            apply_selected_signal_momentum_pct_cap_to_target_weights(
                prices,
                target_weights,
                cap_start=momentum_pct_cap_start,
                cap_end=momentum_pct_cap_end,
                cap_floor=momentum_pct_cap_floor,
                lookback=DEFAULT_LOOKBACK,
                state_lookback=int(params.get("signal_selected_momentum_pct_lookback", 756)),
                min_periods=int(params.get("signal_selected_momentum_pct_min_periods", 120)),
                scope=str(params.get("signal_selected_momentum_pct_cap_scope", "all")),
                risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
            )
        )
    defensive_momentum_pct_cap_start = float(params.get("defensive_signal_selected_momentum_pct_cap_start", 1.0))
    defensive_momentum_pct_cap_floor = float(params.get("defensive_signal_selected_momentum_pct_cap_floor", 1.0))
    selected_pct_for_defensive_cap = (
        post_overlay_selected_momentum_percentile
        if post_overlay_selected_momentum_percentile is not None
        else base_result.get("selected_momentum_percentile")
    )
    if (
        selected_pct_for_defensive_cap is not None
        and defensive_momentum_pct_cap_start < 1.0
        and defensive_momentum_pct_cap_floor < 1.0
    ):
        target_weights, post_overlay_defensive_momentum_trigger = apply_defensive_signal_momentum_pct_cap_to_target_weights(
            target_weights,
            selected_percentile=selected_pct_for_defensive_cap,
            cap_start=defensive_momentum_pct_cap_start,
            cap_floor=defensive_momentum_pct_cap_floor,
            defensive_codes=[str(code) for code in params.get("defensive_codes", DEFENSIVE_CODES)],
        )

    rebalance_threshold = float(params.get("target_min_rebalance_threshold", 0.0))
    if rebalance_threshold > 0:
        target_weights, post_overlay_threshold_blocked = apply_min_rebalance_threshold_to_target_weights(
            target_weights,
            rebalance_threshold,
        )

    ma_trend_triggered = None
    ma_trend_window = int(params.get("ma_trend_window", DEFAULT_MA_TREND_WINDOW))
    ma_trend_cut = float(params.get("ma_trend_cut", DEFAULT_MA_TREND_CUT))
    if ma_trend_window > 0 and ma_trend_cut < 1.0:
        target_weights, ma_trend_triggered = apply_ma_trend_cut_to_target_weights(
            prices,
            target_weights,
            window=ma_trend_window,
            cut=ma_trend_cut,
        )

    if (
        post_overlay_selected_momentum_percentile is not None
        or post_overlay_threshold_blocked is not None
        or ma_trend_triggered is not None
    ):
        result, trades = run_target_weights_strategy(
            prices,
            selected,
            target_weights,
            internal_momentum,
            fee_rate,
            slippage_rate,
        )
    return_weight_cols = [col for col in result.columns if col.startswith("weight_")]
    if return_weight_cols:
        return_weights = result[return_weight_cols].copy()
        return_weights.columns = [col.removeprefix("weight_") for col in return_weight_cols]
    else:
        return_weights = build_weight_frame_from_holding_exposure(
            result.index,
            [str(code) for code in selected["code"]],
            result["holding"],
            result["exposure"],
        )
    result = finalize_position_columns(
        result,
        selected,
        confirmed_weights=target_weights,
        return_weights=return_weights,
        return_holding=result["holding"] if "holding" in result.columns else None,
        return_exposure=result["exposure"] if "exposure" in result.columns else None,
    )
    result = recompute_return_chain(result, prices, fee_rate=fee_rate, slippage_rate=slippage_rate)
    trades = build_trades_from_weight_frame(target_weights, result["nav"], selected)

    effective_momentum = result["current_momentum"].copy().rename("effective_momentum")
    signal_momentum = compute_signal_asset_momentum(prices, result["signal"], DEFAULT_LOOKBACK)
    result["effective_momentum"] = effective_momentum
    result["signal_asset_momentum"] = signal_momentum
    result["current_momentum"] = signal_momentum
    result["base_target_exposure"] = base_result["exposure"].reindex(result.index).fillna(0.0)
    if "weak_market_trigger" in base_result.columns:
        result["weak_market_trigger"] = base_result["weak_market_trigger"].reindex(result.index).fillna(False)
    if "top2_close_risk_cap_triggered" in base_result.columns:
        result["top2_close_risk_cap_triggered"] = base_result["top2_close_risk_cap_triggered"].reindex(result.index).fillna(False)
        result["top2_close_gap"] = base_result["top2_close_gap"].reindex(result.index).ffill()
        result["top2_close_risk_cap"] = base_result["top2_close_risk_cap"].reindex(result.index).ffill()
    if "selected_volatility_rank" in base_result.columns:
        result["selected_volatility_rank"] = base_result["selected_volatility_rank"].reindex(result.index)
    if "absolute_momentum_threshold" in base_result.columns:
        result["absolute_momentum_threshold"] = base_result["absolute_momentum_threshold"].reindex(result.index).ffill()
    if "selected_volatility_cap_triggered" in base_result.columns:
        result["selected_volatility_cap_triggered"] = base_result["selected_volatility_cap_triggered"].reindex(result.index).fillna(False)
    if "selected_momentum_percentile" in base_result.columns:
        result["selected_momentum_percentile"] = base_result["selected_momentum_percentile"].reindex(result.index)
    if "selected_momentum_pct_cap_triggered" in base_result.columns:
        result["selected_momentum_pct_cap_triggered"] = base_result["selected_momentum_pct_cap_triggered"].reindex(result.index).fillna(False)
    if post_overlay_selected_momentum_percentile is not None:
        result["selected_momentum_percentile"] = post_overlay_selected_momentum_percentile.reindex(result.index)
    if post_overlay_selected_momentum_trigger is not None:
        result["selected_momentum_pct_cap_triggered"] = post_overlay_selected_momentum_trigger.reindex(result.index).fillna(False)
    if post_overlay_defensive_momentum_trigger is not None:
        combined_selected_pct_trigger = result["selected_momentum_pct_cap_triggered"].reindex(result.index).fillna(False)
        result["selected_momentum_pct_cap_triggered"] = (
            combined_selected_pct_trigger | post_overlay_defensive_momentum_trigger.reindex(result.index).fillna(False)
        )
        result["defensive_signal_momentum_pct_cap_triggered"] = post_overlay_defensive_momentum_trigger.reindex(result.index).fillna(False)
    if "extra_cap_triggered" in base_result.columns:
        result["extra_cap_triggered"] = base_result["extra_cap_triggered"].reindex(result.index).fillna(False)
    if "extra_cap_reason" in base_result.columns:
        result["extra_cap_reason"] = base_result["extra_cap_reason"].reindex(result.index)
    if "overheat_cap_triggered" in base_result.columns:
        result["overheat_cap_triggered"] = base_result["overheat_cap_triggered"].reindex(result.index).fillna(False)
    if "overheat_high_cap_triggered" in base_result.columns:
        result["overheat_high_cap_triggered"] = base_result["overheat_high_cap_triggered"].reindex(result.index).fillna(False)
    if "overheat_stability_cap_triggered" in base_result.columns:
        result["overheat_stability_cap_triggered"] = base_result["overheat_stability_cap_triggered"].reindex(result.index).fillna(False)
    result["target_exposure"] = target_weights.sum(axis=1).rename("target_exposure")
    result["stress_bond_trigger"] = stress_mask
    if ma_trend_triggered is not None:
        result["ma_trend_triggered"] = ma_trend_triggered.reindex(result.index).fillna(False)
    if post_overlay_threshold_blocked is not None:
        result["rebalance_threshold_blocked"] = post_overlay_threshold_blocked.reindex(result.index).fillna(False)
    result = pd.concat([result, target_weights.add_prefix("target_weight_")], axis=1)
    return result, trades


def run_default_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    market_proxy: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_result, _ = run_threshold_dual_strategy(
        prices,
        selected,
        lookback=DEFAULT_LOOKBACK,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        cash_exit_threshold=DEFAULT_DUAL_CASH_EXIT_THRESHOLD,
        risk_codes=list(RISK_CODES),
        defensive_codes=list(DEFENSIVE_CODES),
    )
    base_target_weights = base_result[[col for col in base_result.columns if col.startswith("weight_")]].copy()
    base_target_weights.columns = [col.removeprefix("weight_") for col in base_target_weights.columns]
    target_weights, asset_pct_rank, asset_pct_cap, asset_pct_cap_triggered, asset_pct_cap_code = apply_asset_momentum_percentile_cap_to_target_weights(
        prices,
        base_target_weights,
    )
    target_weights, boll_hot_triggered, boll_extreme_hot_triggered = apply_boll_hot_cap_to_target_weights(
        prices,
        target_weights,
        cap=DEFAULT_BOLL_HOT_CAP,
        bandwidth_pct_cut=DEFAULT_BOLL_HOT_BANDWIDTH_PCT,
        extreme_cap=DEFAULT_BOLL_EXTREME_HOT_CAP,
        extreme_bandwidth_pct_cut=DEFAULT_BOLL_EXTREME_HOT_BANDWIDTH_PCT,
    )
    ma_trend_triggered = None
    if DEFAULT_MA_TREND_WINDOW > 0 and DEFAULT_MA_TREND_CUT < 1.0:
        target_weights, ma_trend_triggered = apply_ma_trend_cut_to_target_weights(
            prices,
            target_weights,
            window=DEFAULT_MA_TREND_WINDOW,
            cut=DEFAULT_MA_TREND_CUT,
        )
    result, _ = run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        base_result["current_momentum"],
        fee_rate,
        slippage_rate,
    )
    return_weight_cols = [col for col in result.columns if col.startswith("weight_")]
    return_weights = result[return_weight_cols].copy()
    return_weights.columns = [col.removeprefix("weight_") for col in return_weight_cols]
    result = finalize_position_columns(
        result,
        selected,
        confirmed_weights=target_weights,
        return_weights=return_weights,
        return_holding=result["holding"] if "holding" in result.columns else None,
        return_exposure=result["exposure"] if "exposure" in result.columns else None,
    )
    result = recompute_return_chain(result, prices, fee_rate=fee_rate, slippage_rate=slippage_rate)
    trades = build_trades_from_weight_frame(target_weights, result["nav"], selected)
    signal_momentum = compute_signal_asset_momentum(prices, result["signal"], DEFAULT_LOOKBACK)
    result["current_momentum"] = signal_momentum
    result["effective_momentum"] = signal_momentum
    result["signal_asset_momentum"] = signal_momentum
    result["base_target_exposure"] = base_target_weights.sum(axis=1).rename("base_target_exposure")
    result["target_exposure"] = target_weights.sum(axis=1).rename("target_exposure")
    result["asset_momentum_percentile"] = asset_pct_rank.reindex(result.index)
    result["asset_momentum_percentile_cap"] = asset_pct_cap.reindex(result.index).fillna(1.0)
    result["asset_momentum_percentile_cap_triggered"] = asset_pct_cap_triggered.reindex(result.index).fillna(False)
    result["asset_momentum_percentile_cap_code"] = asset_pct_cap_code.reindex(result.index)
    result["boll_hot_cap_triggered"] = boll_hot_triggered.reindex(result.index).fillna(False)
    result["boll_extreme_hot_cap_triggered"] = boll_extreme_hot_triggered.reindex(result.index).fillna(False)
    result["extra_cap_triggered"] = (
        asset_pct_cap_triggered.reindex(result.index).fillna(False)
        | boll_hot_triggered.reindex(result.index).fillna(False)
    )
    result["extra_cap_reason"] = pd.Series(pd.NA, index=result.index, dtype="object")
    result.loc[result["asset_momentum_percentile_cap_triggered"], "extra_cap_reason"] = "asset_momentum_percentile_cap"
    result.loc[result["boll_hot_cap_triggered"], "extra_cap_reason"] = "boll_hot_cap"
    result.loc[result["boll_extreme_hot_cap_triggered"], "extra_cap_reason"] = "boll_extreme_hot_cap"
    if ma_trend_triggered is not None:
        result["ma_trend_triggered"] = ma_trend_triggered.reindex(result.index).fillna(False)
    result = pd.concat([result, target_weights.add_prefix("target_weight_")], axis=1)
    return result, trades
