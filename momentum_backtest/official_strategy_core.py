#!/usr/bin/env python3
"""正式动量策略主链构建器。

目标：把当前正式基线的 target weight 主链收口到唯一实现，
减少研究脚本复制正式逻辑后微调所带来的口径漂移。
"""

from __future__ import annotations

import pandas as pd


def build_absolute_momentum_threshold_series(
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
) -> pd.Series:
    """按市场状态生成逐日动量阈值序列。"""
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

try:
    from .compare_goal_optimizations import build_dynamic_core_target_weights, build_parametrized_hs300_trend_filter
    from .compare_hs300_regime_fixes import run_target_weights_strategy
    from .run_backtest import (
        DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
        DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        DEFENSIVE_CODES,
        RISK_CODES,
        build_asset_relative_volatility_percentile,
        build_signal_stability_score,
        build_top2_close_risk_flag,
        build_signal_quality_score,
        choose_signal_winner_with_margin,
        filter_score_row_by_confirmation,
        resolve_strategy_universe,
    )
except ImportError:
    from compare_goal_optimizations import build_dynamic_core_target_weights, build_parametrized_hs300_trend_filter
    from compare_hs300_regime_fixes import run_target_weights_strategy
    from run_backtest import (
        DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
        DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        DEFENSIVE_CODES,
        RISK_CODES,
        build_asset_relative_volatility_percentile,
        build_signal_stability_score,
        build_top2_close_risk_flag,
        build_signal_quality_score,
        choose_signal_winner_with_margin,
        filter_score_row_by_confirmation,
        resolve_strategy_universe,
    )


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
    secondary_score = build_signal_stability_score(
        prices,
        lookback=lookback,
        method=secondary_stability_method,
    )
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
    params: dict[str, float | list[str]],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """构建当前正式基线的目标权重主链。

    该函数统一封装：
    1. core-satellite / regime mix 主体
    2. weak market guard
    3. top2 close risk cap
    4. continuous overheat cap

    stress bond overlay 仍在上层 runner 中处理，因为它依赖独立的持续触发逻辑。
    """
    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        defensive_codes=[str(code) for code in params.get("defensive_codes", DEFENSIVE_CODES)],
    )
    transition_mode = str(params.get("regime_transition_mode", "step"))
    transition_start_cut = float(params.get("regime_transition_start_cut", 0.0))
    transition_end_cut = float(params.get("regime_transition_end_cut", DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD))
    absolute_threshold_series = build_absolute_momentum_threshold_series(prices, proxy, params)

    aggressive_weights, aggressive_momentum, _, _, aggressive_trend = build_dynamic_core_target_weights_with_transition(
        prices,
        lookback=25,
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
        lookback=25,
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
        lookback=25,
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
        signal_volatility_rank = build_asset_relative_volatility_percentile(
            prices[active_risk_codes],
            vol_window=20,
            state_lookback=int(params.get("signal_volatility_state_lookback", 252)),
        )
        preliminary_signal = preliminary_result["signal"].astype("object")
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
    selected_stability_rank = None
    if overheat_stability_method != "none" and overheat_stability_min_rank > 0:
        stability_score = build_signal_stability_score(prices, lookback=25, method=overheat_stability_method)
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
            momentum_progress = (
                (mixed_momentum.loc[high_mask] - overheat_momentum_cut) / (overheat_high_momentum_cut - overheat_momentum_cut)
            ).clip(lower=0.0, upper=1.0)
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
            scale.loc[overheat_high_mask] = (
                float(params["overheat_high_max_exposure"]) / capped_target_weights[active_risk_codes].sum(axis=1).loc[overheat_high_mask]
            )
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
