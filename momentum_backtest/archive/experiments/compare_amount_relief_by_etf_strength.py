#!/usr/bin/env python3
"""测试当 A 股量能转弱但 ETF 横截面仍强时，是否放松弱量能闸门。"""

from __future__ import annotations

import argparse

try:
    from ..runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from compare_goal_optimizations import load_market_volume_proxy
from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
from compare_market_proxy_variants import build_proxy_catalog, build_risk_proxy_features
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RISK_CODES,
    build_default_strategy_params,
    ensure_output_dirs,
    load_default_strategy_backtest_pool,
    resolve_strategy_universe,
    write_dataframe_csv_atomic,
)
from compare_goal_optimizations import build_dynamic_core_target_weights


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_amount_relief_by_etf_strength")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试当 A 股量能转弱但 ETF 横截面仍强时，是否放松弱量能闸门。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def build_relief_strategy_result(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    market_proxy: pd.DataFrame,
    params: dict[str, object],
    *,
    breadth_blend_floor: float | None = None,
    above_ma20_floor: float | None = None,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(
            market_proxy,
            prices,
            risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        )
    }
    proxy_kind = str(params["proxy_kind"])
    proxy = proxy_catalog[proxy_kind].reindex(prices.index).ffill()
    features = build_risk_proxy_features(
        prices,
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
    )

    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        defensive_codes=[str(code) for code in params.get("defensive_codes", [])],
    )

    aggressive_weights, aggressive_momentum, _, _, aggressive_trend = build_dynamic_core_target_weights(
        prices,
        lookback=25,
        absolute_threshold=0.05,
        weak_trend_defensive_weight=0.8,
        core_weight=float(params["aggressive_core_weight"]),
        hs300_mom60_cut=0.05,
        hs300_mom120_cut=0.10,
        hs300_ma_window=90,
        signal_quality_method=str(params.get("signal_quality_method", "raw")),
        slope_penalty=float(params.get("signal_slope_penalty", 0.0)),
        leader_margin=float(params.get("signal_leader_margin", 0.0)),
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    conservative_weights, conservative_momentum, _, _, _ = build_dynamic_core_target_weights(
        prices,
        lookback=25,
        absolute_threshold=0.05,
        weak_trend_defensive_weight=0.8,
        core_weight=float(params["conservative_core_weight"]),
        hs300_mom60_cut=0.04,
        hs300_mom120_cut=0.10,
        hs300_ma_window=120,
        signal_quality_method=str(params.get("signal_quality_method", "raw")),
        slope_penalty=float(params.get("signal_slope_penalty", 0.0)),
        leader_margin=float(params.get("signal_leader_margin", 0.0)),
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )

    aggressive_mask = aggressive_trend & (aggressive_momentum >= float(params["regime_momentum_cut"]))
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    mixed_momentum = conservative_momentum.copy()
    mixed_momentum.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    mixed_momentum = mixed_momentum.rename("current_momentum")

    row_risk_weight = mixed_target_weights[active_risk_codes].sum(axis=1)
    volume_weak_mask = (
        (proxy["market_amount_ratio_20_60"] < float(params["volume_ratio_cut"]))
        & (proxy["market_amount_ratio_5_20"] < float(params["volume_short_ratio_cut"]))
        & (proxy["market_breadth_proxy"] < float(params["volume_breadth_cut"]))
        & (mixed_momentum <= float(params["volume_guard_momentum_ceiling"]))
    ).fillna(False)

    relief_mask = pd.Series(False, index=prices.index, dtype=bool)
    if breadth_blend_floor is not None:
        relief_mask = relief_mask | (features["breadth_blend"] >= breadth_blend_floor).fillna(False)
    if above_ma20_floor is not None:
        relief_mask = relief_mask | (features["above_ma20_frac"] >= above_ma20_floor).fillna(False)

    guarded_weights = mixed_target_weights.copy()
    weak_cap = float(params["volume_guard_cap"])
    effective_guard_mask = volume_weak_mask & (~relief_mask) & (row_risk_weight > weak_cap)
    if effective_guard_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[effective_guard_mask] = weak_cap / row_risk_weight.loc[effective_guard_mask]
        guarded_weights.loc[effective_guard_mask, active_risk_codes] = guarded_weights.loc[
            effective_guard_mask, active_risk_codes
        ].mul(scale.loc[effective_guard_mask], axis=0)

    result, trades = run_target_weights_strategy(
        prices,
        selected,
        guarded_weights,
        mixed_momentum,
        fee_rate,
        slippage_rate,
    )
    return result, trades


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = load_default_strategy_backtest_pool()
    _, cached_prices = load_cached_data()
    start_ts = cached_prices.index.max() - pd.DateOffset(years=args.years)
    prices = cached_prices.loc[cached_prices.index >= start_ts].copy()
    selected_codes = selected["code"].astype(str).tolist()
    prices = prices[[code for code in selected_codes if code in prices.columns]].copy()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)

    base_params = build_default_strategy_params()
    rows: list[dict[str, object]] = []
    baseline: dict[str, float] | None = None

    variants = [
        ("baseline", None, None),
        ("relief_breadth_blend_ge_m2", -0.02, None),
        ("relief_breadth_blend_ge_m1", -0.01, None),
        ("relief_breadth_blend_ge_0", 0.0, None),
        ("relief_above_ma20_ge_55", None, 0.55),
        ("relief_above_ma20_ge_60", None, 0.60),
        ("relief_above_ma20_ge_65", None, 0.65),
        ("relief_combo_m1_or_ma60", -0.01, 0.60),
    ]

    for name, breadth_floor, ma20_floor in variants:
        result, trades = build_relief_strategy_result(
            prices=prices,
            selected=selected,
            market_proxy=market_proxy,
            params=base_params,
            breadth_blend_floor=breadth_floor,
            above_ma20_floor=ma20_floor,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        summary = summarize(result, trades, selected)
        row = {
            "name": name,
            "annualized_return": float(summary["annualized_return"]),
            "sharpe_rf0": float(summary["sharpe_rf0"]),
            "max_drawdown": float(summary["max_drawdown"]),
            "max_drawdown_integral": float(summary["max_drawdown_integral"]),
            "trade_action_count": int(summary["trade_action_count"]),
            "breadth_blend_floor": breadth_floor,
            "above_ma20_floor": ma20_floor,
        }
        if baseline is None:
            baseline = row.copy()
            row["annualized_diff"] = 0.0
            row["sharpe_diff"] = 0.0
            row["max_drawdown_diff"] = 0.0
            row["max_drawdown_integral_diff"] = 0.0
        else:
            row["annualized_diff"] = row["annualized_return"] - baseline["annualized_return"]
            row["sharpe_diff"] = row["sharpe_rf0"] - baseline["sharpe_rf0"]
            row["max_drawdown_diff"] = row["max_drawdown"] - baseline["max_drawdown"]
            row["max_drawdown_integral_diff"] = row["max_drawdown_integral"] - baseline["max_drawdown_integral"]
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    write_dataframe_csv_atomic(summary_df, SUMMARY_PATH, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nsummary saved to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
