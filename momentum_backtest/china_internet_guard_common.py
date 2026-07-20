#!/usr/bin/env python3
"""中概互联保护规则的共享实现。"""

from __future__ import annotations

import pandas as pd

try:
    from .candidate_pool_common import BASE_DEFENSIVE_CODES, BASE_RISK_CODES
    from .compare_strategy_refinements import run_signal_strategy
except ImportError:
    from candidate_pool_common import BASE_DEFENSIVE_CODES, BASE_RISK_CODES
    from compare_strategy_refinements import run_signal_strategy
from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
    DEFAULT_OVERHEAT_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    build_strategy_summary,
)


CHINA_INTERNET_CODE = "513050"
FLOAT_TOL = 1e-12


def summarize_china_internet_guard_result(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | str]:
    weight_col = f"weight_{CHINA_INTERNET_CODE}"
    if weight_col in result.columns:
        ci_weights = result[weight_col].fillna(0.0).astype(float)
        ci_holding_share = float((ci_weights.abs() > FLOAT_TOL).mean())
        ci_exposure_share = float(ci_weights.mean())
    else:
        holding_codes = result["holding"].map(lambda value: str(value).replace(".0", "") if pd.notna(value) else "")
        exposure = result["exposure"].fillna(0.0).astype(float)
        ci_mask = holding_codes == CHINA_INTERNET_CODE
        ci_holding_share = float(ci_mask.mean())
        ci_exposure_share = float(exposure.where(ci_mask, 0.0).mean())
    return {
        "start_date": result.index[0].date().isoformat(),
        "end_date": result.index[-1].date().isoformat(),
        **build_strategy_summary(result, trades),
        "china_internet_holding_share": ci_holding_share,
        "china_internet_exposure_share": ci_exposure_share,
    }


def build_china_internet_guard_components(prices: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [code for code in BASE_RISK_CODES + [CHINA_INTERNET_CODE] if code in prices.columns]
    defensive_codes = [code for code in BASE_DEFENSIVE_CODES if code in prices.columns]
    return momentum, momentum[risk_codes], momentum[defensive_codes]


def choose_china_internet_guard_signal(
    risk_scores: pd.Series,
    defensive_scores: pd.Series,
    china_abs_threshold: float | None = None,
    china_margin_threshold: float | None = None,
) -> tuple[object, float, float]:
    valid_risk = risk_scores.dropna().sort_values(ascending=False)
    valid_def = defensive_scores.dropna().sort_values(ascending=False)
    r_asset = valid_risk.index[0] if not valid_risk.empty else pd.NA
    r_score = float(valid_risk.iloc[0]) if not valid_risk.empty else float("nan")
    d_asset = valid_def.index[0] if not valid_def.empty else pd.NA
    d_score = float(valid_def.iloc[0]) if not valid_def.empty else float("nan")

    candidate_asset = r_asset
    candidate_score = r_score
    if pd.notna(r_asset) and str(r_asset) == CHINA_INTERNET_CODE:
        second_score = float(valid_risk.iloc[1]) if len(valid_risk) >= 2 else float("nan")
        second_asset = valid_risk.index[1] if len(valid_risk) >= 2 else pd.NA
        if china_abs_threshold is not None and pd.notna(r_score) and r_score <= china_abs_threshold:
            candidate_asset = second_asset
            candidate_score = second_score
        elif china_margin_threshold is not None and pd.notna(second_score) and (r_score - second_score) < china_margin_threshold:
            candidate_asset = second_asset
            candidate_score = second_score

    if pd.notna(candidate_score) and candidate_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
        return candidate_asset, 1.0, float(candidate_score)
    if pd.notna(candidate_score) and candidate_score > 0 and pd.notna(d_asset):
        return d_asset, DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT, float(candidate_score)
    if pd.notna(d_asset):
        return d_asset, (1.0 if pd.notna(d_score) and d_score > 0 else 0.0), d_score
    return pd.NA, 0.0, float("nan")


def run_china_internet_guard_variant(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    china_abs_threshold: float | None = None,
    china_margin_threshold: float | None = None,
    china_max_exposure: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, risk_mom, defensive_mom = build_china_internet_guard_components(prices, lookback)
    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        asset, exposure, momentum = choose_china_internet_guard_signal(
            risk_scores=risk_mom.loc[dt_idx],
            defensive_scores=defensive_mom.loc[dt_idx],
            china_abs_threshold=china_abs_threshold,
            china_margin_threshold=china_margin_threshold,
        )
        signal.loc[dt_idx] = asset
        target_exposure.loc[dt_idx] = exposure
        current_momentum.loc[dt_idx] = momentum

    base_result, _ = run_signal_strategy(prices, selected, signal, target_exposure, current_momentum, fee_rate, slippage_rate)
    adjusted_exposure = target_exposure.copy()
    if china_max_exposure is not None:
        china_mask = signal == CHINA_INTERNET_CODE
        adjusted_exposure.loc[china_mask] = adjusted_exposure.loc[china_mask].clip(upper=china_max_exposure)

    risk_codes = [code for code in risk_mom.columns if code in prices.columns]
    overheat_mask = (
        signal.isin(risk_codes)
        & (adjusted_exposure > DEFAULT_OVERHEAT_MAX_EXPOSURE)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_MOMENTUM_CUT)
    )
    adjusted_exposure.loc[overheat_mask] = DEFAULT_OVERHEAT_MAX_EXPOSURE
    extreme_mask = (
        signal.isin(risk_codes)
        & (adjusted_exposure > DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT)
    )
    adjusted_exposure.loc[extreme_mask] = DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE

    return run_signal_strategy(prices, selected, signal, adjusted_exposure, current_momentum, fee_rate, slippage_rate)
