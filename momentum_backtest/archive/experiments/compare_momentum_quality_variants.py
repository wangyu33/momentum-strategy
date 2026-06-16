#!/usr/bin/env python3
"""对比不同“质量化动量”口径在当前正式基线下的表现。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    from ..runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from compare_defensive_persistence import build_persistent_mask
from compare_goal_optimizations import build_parametrized_hs300_trend_filter, load_market_volume_proxy
from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
from compare_market_proxy_variants import build_proxy_catalog
from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_LOOKBACK,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_STRESS_BOND_BREADTH_CUT,
    DEFAULT_STRESS_BOND_CODE,
    DEFAULT_STRESS_BOND_ENTER_DAYS,
    DEFAULT_STRESS_BOND_EXIT_DAYS,
    DEFAULT_STRESS_BOND_RISK_CAP,
    DEFAULT_STRESS_BOND_RATIO_CUT,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    RISK_CODES,
    build_default_strategy_params,
    ensure_output_dirs,
    fetch_histories,
    load_default_strategy_backtest_pool,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_momentum_quality_variants")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比不同质量化动量口径在当前正式基线下的表现。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取价格，而不是复用 core 缓存。")
    parser.add_argument("--slope-grid-only", action="store_true", help="只跑 slope_penalty 小范围搜参。")
    parser.add_argument("--slope-start", type=float, default=0.6, help="slope_penalty 搜参起点。")
    parser.add_argument("--slope-stop", type=float, default=1.0, help="slope_penalty 搜参终点。")
    parser.add_argument("--slope-step", type=float, default=0.05, help="slope_penalty 搜参步长。")
    return parser.parse_args()


def build_quality_score(
    prices: pd.DataFrame,
    method: str,
    lookback: int = DEFAULT_LOOKBACK,
    downvol_scale: float = 0.0,
    drawdown_penalty: float = 0.0,
    slope_penalty: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_momentum = prices / prices.shift(lookback) - 1
    ret5 = prices / prices.shift(5) - 1
    drawdown20 = prices / prices.rolling(20).max() - 1
    returns = prices.pct_change()
    downside_vol20 = returns.where(returns < 0, 0.0).rolling(20).std(ddof=0) * math.sqrt(252)
    excess_slope = (ret5 - raw_momentum / max(lookback / 5, 1)).clip(lower=0.0)

    if method == "raw":
        score = raw_momentum.copy()
    elif method == "downvol":
        score = raw_momentum / (1.0 + downvol_scale * downside_vol20)
    elif method == "drawdown":
        score = raw_momentum - drawdown_penalty * drawdown20.abs()
    elif method == "slope":
        score = raw_momentum - slope_penalty * excess_slope
    elif method == "combo":
        score = raw_momentum / (1.0 + downvol_scale * downside_vol20)
        score = score - drawdown_penalty * drawdown20.abs() - slope_penalty * excess_slope
    else:
        raise ValueError(f"unsupported method: {method}")
    return raw_momentum, score


def choose_winner_with_margin(
    score_row: pd.Series,
    prev_asset: str | None,
    leader_margin: float,
) -> str | None:
    valid = score_row.dropna().sort_values(ascending=False)
    if valid.empty:
        return None
    top_asset = str(valid.index[0])
    top_score = float(valid.iloc[0])
    if leader_margin > 0 and prev_asset and prev_asset in valid.index:
        prev_score = float(valid.loc[prev_asset])
        if top_score - prev_score < leader_margin:
            return str(prev_asset)
    return top_asset


def build_quality_signal(
    prices: pd.DataFrame,
    method: str,
    lookback: int = DEFAULT_LOOKBACK,
    absolute_threshold: float = DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    weak_trend_defensive_weight: float = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    downvol_scale: float = 0.0,
    drawdown_penalty: float = 0.0,
    slope_penalty: float = 0.0,
    leader_margin: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw_momentum, score = build_quality_score(
        prices,
        method=method,
        lookback=lookback,
        downvol_scale=downvol_scale,
        drawdown_penalty=drawdown_penalty,
        slope_penalty=slope_penalty,
    )
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    defensive_codes = [code for code in ["511580", "518880", "512890"] if code in prices.columns]
    risk_score = score[risk_codes]
    defensive_score = score[defensive_codes]

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    prev_signal: str | None = None
    for dt_idx in prices.index:
        prev_risk = prev_signal if prev_signal in risk_codes else None
        prev_def = prev_signal if prev_signal in defensive_codes else None
        r_asset = choose_winner_with_margin(risk_score.loc[dt_idx], prev_risk, leader_margin)
        d_asset = choose_winner_with_margin(defensive_score.loc[dt_idx], prev_def, leader_margin)
        r_score = float(raw_momentum.loc[dt_idx, r_asset]) if r_asset and pd.notna(raw_momentum.loc[dt_idx, r_asset]) else float("nan")
        d_score = float(raw_momentum.loc[dt_idx, d_asset]) if d_asset and pd.notna(raw_momentum.loc[dt_idx, d_asset]) else float("nan")

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
            prev_signal = None
        elif pd.notna(r_score) and r_score > absolute_threshold:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = float(r_score)
            prev_signal = str(r_asset) if r_asset else None
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = weak_trend_defensive_weight
            current_momentum.loc[dt_idx] = float(r_score)
            prev_signal = str(d_asset) if d_asset else None
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            current_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
            prev_signal = str(d_asset) if d_asset else None
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
            prev_signal = None

    return signal, target_exposure, current_momentum


def build_dynamic_core_target_weights_quality(
    prices: pd.DataFrame,
    method: str,
    core_weight: float,
    hs300_mom60_cut: float,
    hs300_mom120_cut: float,
    hs300_ma_window: int,
    downvol_scale: float = 0.0,
    drawdown_penalty: float = 0.0,
    slope_penalty: float = 0.0,
    leader_margin: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    signal, target_exposure, current_momentum = build_quality_signal(
        prices,
        method=method,
        lookback=DEFAULT_LOOKBACK,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        downvol_scale=downvol_scale,
        drawdown_penalty=drawdown_penalty,
        slope_penalty=slope_penalty,
        leader_margin=leader_margin,
    )
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

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_weights[hs300_code] += dynamic_core_weight
    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_scale.loc[code_mask] * target_exposure.loc[code_mask]

    blended_momentum = (satellite_scale * current_momentum + dynamic_core_weight * hs300_mom60).rename("current_momentum")
    return target_weights, blended_momentum, trend_filter


def build_base_target_weights_quality(
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    method: str,
    downvol_scale: float = 0.0,
    drawdown_penalty: float = 0.0,
    slope_penalty: float = 0.0,
    leader_margin: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    aggressive_weights, aggressive_momentum, aggressive_trend = build_dynamic_core_target_weights_quality(
        prices,
        method=method,
        core_weight=float(params["aggressive_core_weight"]),
        hs300_mom60_cut=0.05,
        hs300_mom120_cut=0.10,
        hs300_ma_window=90,
        downvol_scale=downvol_scale,
        drawdown_penalty=drawdown_penalty,
        slope_penalty=slope_penalty,
        leader_margin=leader_margin,
    )
    conservative_weights, conservative_momentum, _ = build_dynamic_core_target_weights_quality(
        prices,
        method=method,
        core_weight=float(params["conservative_core_weight"]),
        hs300_mom60_cut=0.04,
        hs300_mom120_cut=0.10,
        hs300_ma_window=120,
        downvol_scale=downvol_scale,
        drawdown_penalty=drawdown_penalty,
        slope_penalty=slope_penalty,
        leader_margin=leader_margin,
    )

    aggressive_mask = aggressive_trend & (aggressive_momentum >= float(params["regime_momentum_cut"]))
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    mixed_momentum = conservative_momentum.copy()
    mixed_momentum.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    mixed_momentum = mixed_momentum.rename("current_momentum")

    proxy = proxy.reindex(prices.index).ffill()
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    row_risk_weight = mixed_target_weights[risk_codes].sum(axis=1)
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
        capped_target_weights.loc[weak_scale_mask, risk_codes] = capped_target_weights.loc[weak_scale_mask, risk_codes].mul(
            scale.loc[weak_scale_mask], axis=0
        )

    base_result, _ = run_target_weights_strategy(
        prices,
        pd.DataFrame({"code": list(prices.columns), "theme": list(prices.columns), "name": list(prices.columns)}),
        capped_target_weights,
        mixed_momentum,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
    )
    risk_weight_after_weak_guard = capped_target_weights[risk_codes].sum(axis=1)
    overheat_mask = (
        (risk_weight_after_weak_guard > float(params["overheat_max_exposure"]))
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (mixed_momentum >= float(params["overheat_momentum_cut"]))
    )
    if overheat_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[overheat_mask] = float(params["overheat_max_exposure"]) / risk_weight_after_weak_guard.loc[overheat_mask]
        capped_target_weights.loc[overheat_mask, risk_codes] = capped_target_weights.loc[overheat_mask, risk_codes].mul(
            scale.loc[overheat_mask], axis=0
        )

    overheat_high_mask = (
        (capped_target_weights[risk_codes].sum(axis=1) > float(params["overheat_high_max_exposure"]))
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (mixed_momentum >= float(params["overheat_high_momentum_cut"]))
    )
    if overheat_high_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[overheat_high_mask] = (
            float(params["overheat_high_max_exposure"]) / capped_target_weights[risk_codes].sum(axis=1).loc[overheat_high_mask]
        )
        capped_target_weights.loc[overheat_high_mask, risk_codes] = capped_target_weights.loc[
            overheat_high_mask, risk_codes
        ].mul(scale.loc[overheat_high_mask], axis=0)

    return capped_target_weights, mixed_momentum, base_result


def run_quality_variant(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    fee_rate: float,
    slippage_rate: float,
    method: str,
    downvol_scale: float = 0.0,
    drawdown_penalty: float = 0.0,
    slope_penalty: float = 0.0,
    leader_margin: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_weights, mixed_momentum, _ = build_base_target_weights_quality(
        prices=prices,
        proxy=proxy,
        params=params,
        method=method,
        downvol_scale=downvol_scale,
        drawdown_penalty=drawdown_penalty,
        slope_penalty=slope_penalty,
        leader_margin=leader_margin,
    )
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    overlaid_weights = base_weights.copy()
    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_codes].sum(axis=1)

    raw_trigger = (
        (proxy["market_amount_ratio_20_60"] < DEFAULT_STRESS_BOND_RATIO_CUT)
        & (proxy["market_breadth_proxy"] < DEFAULT_STRESS_BOND_BREADTH_CUT)
    ).fillna(False)
    trigger_mask = build_persistent_mask(
        raw_trigger,
        enter_days=DEFAULT_STRESS_BOND_ENTER_DAYS,
        exit_days=DEFAULT_STRESS_BOND_EXIT_DAYS,
    )
    scale_mask = trigger_mask & (row_risk_weight > DEFAULT_STRESS_BOND_RISK_CAP)
    if scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[scale_mask] = DEFAULT_STRESS_BOND_RISK_CAP / row_risk_weight.loc[scale_mask]
        overlaid_weights.loc[scale_mask, risk_codes] = overlaid_weights.loc[scale_mask, risk_codes].mul(
            scale.loc[scale_mask], axis=0
        )
        moved_weight = row_risk_weight.loc[scale_mask] - DEFAULT_STRESS_BOND_RISK_CAP
        overlaid_weights.loc[scale_mask, DEFAULT_STRESS_BOND_CODE] = overlaid_weights.loc[
            scale_mask, DEFAULT_STRESS_BOND_CODE
        ].add(moved_weight, fill_value=0.0)

    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = load_default_strategy_backtest_pool()
    if args.refresh:
        prices = fetch_histories(selected, years=args.years)
    else:
        _, cached_prices = load_cached_data()
        start_ts = cached_prices.index.max() - pd.DateOffset(years=args.years)
        keep_codes = [code for code in selected["code"].astype(str) if code in cached_prices.columns]
        prices = cached_prices.loc[cached_prices.index >= start_ts, keep_codes].copy()

    params = build_default_strategy_params()
    base_market_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)
    proxy_catalog = {item["name"]: item["proxy"] for item in build_proxy_catalog(base_market_proxy, prices)}
    proxy = proxy_catalog["hybrid_breadth_blend"]

    variants: list[dict[str, object]] = [{"strategy": "baseline_raw", "method": "raw"}]
    if args.slope_grid_only:
        current = args.slope_start
        while current <= args.slope_stop + 1e-12:
            strategy_name = f"slope_{int(round(current * 100)):03d}"
            variants.append({"strategy": strategy_name, "method": "slope", "slope_penalty": round(current, 4)})
            current += args.slope_step
    else:
        variants.extend(
            [
                {"strategy": "downvol_150", "method": "downvol", "downvol_scale": 1.5},
                {"strategy": "downvol_200", "method": "downvol", "downvol_scale": 2.0},
                {"strategy": "drawdown_025", "method": "drawdown", "drawdown_penalty": 0.25},
                {"strategy": "drawdown_035", "method": "drawdown", "drawdown_penalty": 0.35},
                {"strategy": "slope_050", "method": "slope", "slope_penalty": 0.5},
                {"strategy": "slope_080", "method": "slope", "slope_penalty": 0.8},
                {
                    "strategy": "combo_light",
                    "method": "combo",
                    "downvol_scale": 1.0,
                    "drawdown_penalty": 0.15,
                    "slope_penalty": 0.35,
                },
                {
                    "strategy": "combo_margin",
                    "method": "combo",
                    "downvol_scale": 1.0,
                    "drawdown_penalty": 0.15,
                    "slope_penalty": 0.35,
                    "leader_margin": 0.015,
                },
                {
                    "strategy": "raw_margin",
                    "method": "raw",
                    "leader_margin": 0.015,
                },
            ]
        )

    rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame(index=prices.index)
    baseline_summary: dict[str, object] | None = None

    for spec in variants:
        result, trades = run_quality_variant(
            prices=prices,
            selected=selected,
            proxy=proxy,
            params=params,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            method=str(spec["method"]),
            downvol_scale=float(spec.get("downvol_scale", 0.0)),
            drawdown_penalty=float(spec.get("drawdown_penalty", 0.0)),
            slope_penalty=float(spec.get("slope_penalty", 0.0)),
            leader_margin=float(spec.get("leader_margin", 0.0)),
        )
        summary = summarize(result, trades, selected)
        summary["strategy"] = str(spec["strategy"])
        summary["method"] = str(spec["method"])
        summary["downvol_scale"] = float(spec.get("downvol_scale", 0.0))
        summary["drawdown_penalty"] = float(spec.get("drawdown_penalty", 0.0))
        summary["slope_penalty"] = float(spec.get("slope_penalty", 0.0))
        summary["leader_margin"] = float(spec.get("leader_margin", 0.0))
        rows.append(summary)
        nav_compare[str(spec["strategy"])] = result["nav"]
        if str(spec["strategy"]) == "baseline_raw":
            baseline_summary = summary

    if baseline_summary is None:
        raise RuntimeError("missing baseline summary")

    summary_df = pd.DataFrame(rows)
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(baseline_summary["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(baseline_summary["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = (
        summary_df["max_drawdown_integral"] - float(baseline_summary["max_drawdown_integral"])
    )
    summary_df["max_drawdown_diff"] = summary_df["max_drawdown"] - float(baseline_summary["max_drawdown"])
    summary_df["is_soft_improvement"] = (
        (summary_df["annualized_return"] >= float(baseline_summary["annualized_return"]) - 0.01)
        & (summary_df["max_drawdown_integral"] < float(baseline_summary["max_drawdown_integral"]) - 1e-12)
    )
    summary_df = summary_df.sort_values(
        ["is_soft_improvement", "max_drawdown_integral", "max_drawdown", "annualized_return"],
        ascending=[False, True, False, False],
    )
    write_dataframe_csv_atomic(summary_df, SUMMARY_PATH, index=False)
    write_dataframe_csv_atomic(nav_compare, COMPARE_PATH)

    print(
        summary_df[
            [
                "strategy",
                "annualized_return",
                "sharpe_rf0",
                "max_drawdown_integral",
                "max_drawdown",
                "annualized_diff",
                "max_drawdown_integral_diff",
                "max_drawdown_diff",
                "trade_count",
                "is_soft_improvement",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
