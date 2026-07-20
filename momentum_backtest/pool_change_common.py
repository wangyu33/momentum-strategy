#!/usr/bin/env python3
"""候选池加减替换实验共用的池改写与评估 helper。"""

from __future__ import annotations

import pandas as pd

try:
    from .official_baseline import apply_official_baseline_nav_anchor
    from .run_backtest import (
        DEFENSIVE_CODES,
        RISK_CODES,
        build_default_strategy_params,
        run_default_strategy_with_params,
    )
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor
    from run_backtest import (
        DEFENSIVE_CODES,
        RISK_CODES,
        build_default_strategy_params,
        run_default_strategy_with_params,
    )


def evaluate_pool(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    market_proxy: pd.DataFrame,
    drop_codes: list[str],
    risk_codes: list[str],
    defensive_codes: list[str],
    fee_rate: float,
    slippage_rate: float,
    use_official_baseline_anchor: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按给定候选池与风控分组，回放默认正式策略。"""
    result, trades = run_default_strategy_with_params(
        prices=prices,
        selected=selected,
        params=build_default_strategy_params(
            drop_codes=drop_codes,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
        ),
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        market_proxy=market_proxy,
    )
    if use_official_baseline_anchor:
        result = apply_official_baseline_nav_anchor(result)
    return result, trades


def apply_pool_change(
    base_selected: pd.DataFrame,
    base_drop_codes: list[str],
    change: dict[str, object],
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """按候选池变更描述生成新的标的池与风险/防守分组。"""
    selected = base_selected.copy()
    risk_codes = [code for code in RISK_CODES if code not in base_drop_codes]
    defensive_codes = [code for code in DEFENSIVE_CODES if code not in base_drop_codes]
    effective_drop_codes = list(base_drop_codes)

    drop_codes = [str(code) for code in change.get("drop_codes", [])]
    if drop_codes:
        selected = selected[~selected["code"].astype(str).isin(drop_codes)].reset_index(drop=True)
        risk_codes = [code for code in risk_codes if code not in drop_codes]
        defensive_codes = [code for code in defensive_codes if code not in drop_codes]

    candidate = change.get("candidate")
    candidate_kind = change.get("candidate_kind")
    if candidate is not None:
        selected = pd.concat([selected, pd.DataFrame([candidate])], ignore_index=True)
        candidate_code = str(candidate["code"])
        if candidate_code in effective_drop_codes:
            effective_drop_codes = [code for code in effective_drop_codes if code != candidate_code]
        if candidate_kind == "risk":
            risk_codes.append(candidate_code)
        elif candidate_kind == "defensive":
            defensive_codes.append(candidate_code)

    candidates = change.get("candidates", [])
    for item in candidates:
        selected = pd.concat([selected, pd.DataFrame([item])], ignore_index=True)
        candidate_code = str(item["code"])
        if candidate_code in effective_drop_codes:
            effective_drop_codes = [code for code in effective_drop_codes if code != candidate_code]
        if candidate_kind == "risk":
            risk_codes.append(candidate_code)
        elif candidate_kind == "defensive":
            defensive_codes.append(candidate_code)

    selected = selected.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    risk_codes = list(dict.fromkeys(risk_codes))
    defensive_codes = list(dict.fromkeys(defensive_codes))
    return selected, risk_codes, defensive_codes, effective_drop_codes
