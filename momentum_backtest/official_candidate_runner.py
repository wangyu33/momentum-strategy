#!/usr/bin/env python3
"""正式基线上候选研究的共享辅助函数。"""

from __future__ import annotations

import pandas as pd

try:
    from .compare_market_proxy_variants import build_proxy_catalog
    from .compare_hs300_regime_fixes import run_target_weights_strategy
    from .run_backtest import (
        build_default_strategy_params,
        build_strategy_summary,
        load_core_selected_and_prices,
        run_default_strategy_with_params,
    )
except ImportError:
    from compare_market_proxy_variants import build_proxy_catalog
    from compare_hs300_regime_fixes import run_target_weights_strategy
    from run_backtest import (
        build_default_strategy_params,
        build_strategy_summary,
        load_core_selected_and_prices,
        run_default_strategy_with_params,
    )


def load_official_research_context() -> dict[str, object]:
    selected, prices = load_core_selected_and_prices(today=pd.Timestamp("2026-06-05").date(), allow_same_day_close=False)
    market_proxy = pd.read_csv(
        "momentum_backtest/output/research/goal_optimizations/market_volume_proxy.csv", parse_dates=["date"]
    ).set_index("date")
    params = build_default_strategy_params()
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(market_proxy, prices, risk_codes=[str(code) for code in params["risk_codes"]])
    }
    effective_proxy = proxy_catalog[str(params["proxy_kind"])]
    official_result, official_trades = run_default_strategy_with_params(
        prices=prices,
        selected=selected,
        params=params,
        market_proxy=effective_proxy,
    )
    return {
        "selected": selected,
        "prices": prices,
        "market_proxy": market_proxy,
        "effective_proxy": effective_proxy,
        "params": params,
        "official_result": official_result,
        "official_trades": official_trades,
    }


def extract_target_weights(result: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in result.columns if c.startswith("target_weight_")]
    if not cols:
        raise RuntimeError("missing target_weight_ columns")
    target = result[cols].copy()
    target.columns = [c.removeprefix("target_weight_") for c in cols]
    return target


def run_candidate_from_weights(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    weights: pd.DataFrame,
    official_result: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effective_momentum = (
        official_result["effective_momentum"] if "effective_momentum" in official_result.columns else official_result["current_momentum"]
    )
    result, trades = run_target_weights_strategy(
        prices,
        selected,
        weights,
        effective_momentum.rename("current_momentum"),
        fee_rate=0.0003,
        slippage_rate=0.0002,
    )
    for col in [
        "signal",
        "current_momentum",
        "effective_momentum",
        "base_target_exposure",
        "target_exposure",
        "top2_close_risk_cap_triggered",
        "top2_close_gap",
        "top2_close_risk_cap",
        "stress_bond_trigger",
        "weak_market_trigger",
    ]:
        if col in official_result.columns:
            result[col] = official_result[col]
    if "target_exposure" not in result.columns:
        result["target_exposure"] = weights.sum(axis=1)
    result = pd.concat([result, weights.add_prefix("target_weight_")], axis=1)
    return result, trades


def summarize_candidate(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    variant: str,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)
    summary["variant"] = variant
    if extra_fields:
        summary.update(extra_fields)
    return summary
