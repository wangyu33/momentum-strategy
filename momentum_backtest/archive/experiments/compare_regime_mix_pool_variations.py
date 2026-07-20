#!/usr/bin/env python3
"""Scan pool additions/removals/replacements on top of the current notified regime-mix strategy.

This script intentionally keeps its own historical regime-mix baseline and must not be
anchored to the official 28.2691 formal baseline chain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

try:
    from ...runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from archive_data_loaders import build_archive_flat_output_dir, load_workspace_selected_prices
from candidate_pool_common import (
    ETF_510230,
    ETF_510500,
    ETF_510880,
    ETF_510900,
    ETF_511380,
    ETF_512100,
    ETF_513030,
    ETF_513050,
    ETF_588000,
)
from hs300_regime_common import load_cached_data, run_target_weights_strategy, summarize
from overlay_candidate_catalog import TREASURY_10Y, TREASURY_30Y
from overlay_strategy_helpers import apply_risk_cap
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
)
from archive_strategy_common import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    fetch_histories,
    load_fixed_etf_pool,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("compare_regime_mix_pool_variations")
INTENTIONALLY_UNANCHORED_BASELINE = True
BASE_RISK_CODES = ["510300", "159949", "159954", "159941", "513650", "513880"]
BASE_DEFENSIVE_CODES = ["511580", "518880", "512890"]
NOTIFIED_OVERHEAT_CAP = 0.30
NOTIFIED_OVERHEAT_HIGH_CAP = 0.20
NOTIFIED_OVERHEAT_HIGH_MOMENTUM = 0.32
ADDITION_CANDIDATES = [
    ("risk", ETF_588000),
    ("risk", ETF_513030),
    ("risk", ETF_513050),
    ("risk", ETF_510500),
    ("risk", ETF_512100),
    ("risk", ETF_510900),
    ("risk", ETF_510230),
    ("defensive", TREASURY_30Y),
    ("defensive", TREASURY_10Y),
    ("defensive", ETF_511380),
    ("defensive", ETF_510880),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pool variations on top of the notified regime-mix strategy.")
    parser.add_argument("--years", type=int, default=15, help="History years when refresh is enabled.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--start-date", type=str, default="2012-01-01", help="Analysis start date.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Fetch required histories for addition/replacement variants.")
    return parser.parse_args()


def build_custom_threshold_signal(
    prices: pd.DataFrame,
    risk_codes: list[str],
    defensive_codes: list[str],
    lookback: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [code for code in risk_codes if code in prices.columns]
    defensive_codes = [code for code in defensive_codes if code in prices.columns]
    risk_mom = momentum[risk_codes]
    defensive_mom = momentum[defensive_codes]

    risk_winner = risk_mom.idxmax(axis=1, skipna=True)
    risk_best = risk_mom.max(axis=1, skipna=True)
    defensive_winner = defensive_mom.idxmax(axis=1, skipna=True)
    defensive_best = defensive_mom.max(axis=1, skipna=True)

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        r_asset = risk_winner.loc[dt_idx]
        r_score = risk_best.loc[dt_idx]
        d_asset = defensive_winner.loc[dt_idx]
        d_score = defensive_best.loc[dt_idx]
        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
        elif pd.notna(r_score) and r_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT
            current_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            current_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
    return signal, target_exposure, current_momentum


def build_hs300_trend_filter(prices: pd.DataFrame, mom60_cut: float, mom120_cut: float, ma_window: int) -> pd.Series:
    close = prices["510300"]
    mom60 = close / close.shift(60) - 1
    mom120 = close / close.shift(120) - 1
    ma = close.rolling(ma_window).mean()
    return ((mom60 > mom60_cut) & (mom120 > mom120_cut) & (close > ma)).rename("hs300_trend_filter")


def build_custom_dynamic_core_target_weights(
    prices: pd.DataFrame,
    risk_codes: list[str],
    defensive_codes: list[str],
    lookback: int,
    core_weight: float,
    mom60_cut: float,
    mom120_cut: float,
    ma_window: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    signal, target_exposure, current_momentum = build_custom_threshold_signal(
        prices=prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
        lookback=lookback,
    )
    trend_filter = build_hs300_trend_filter(prices, mom60_cut=mom60_cut, mom120_cut=mom120_cut, ma_window=ma_window)
    hs300_mom60 = prices["510300"] / prices["510300"].shift(60) - 1
    dynamic_core_weight = trend_filter.astype(float) * core_weight
    satellite_scale = 1.0 - dynamic_core_weight
    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_weights["510300"] += dynamic_core_weight
    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_scale.loc[code_mask] * target_exposure.loc[code_mask]
    blended_momentum = (satellite_scale * current_momentum + dynamic_core_weight * hs300_mom60).rename("current_momentum")
    return target_weights, blended_momentum, trend_filter


def run_custom_regime_mix_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    risk_codes: list[str],
    defensive_codes: list[str],
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    aggressive_core_weight: float = 0.30,
    conservative_core_weight: float = 0.28,
    regime_momentum_cut: float = 0.06,
    overheat_cap: float = NOTIFIED_OVERHEAT_CAP,
    overheat_high_cap: float = NOTIFIED_OVERHEAT_HIGH_CAP,
    overheat_high_momentum: float = NOTIFIED_OVERHEAT_HIGH_MOMENTUM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggressive_weights, aggressive_momentum, aggressive_trend = build_custom_dynamic_core_target_weights(
        prices=prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
        lookback=lookback,
        core_weight=aggressive_core_weight,
        mom60_cut=0.05,
        mom120_cut=0.10,
        ma_window=90,
    )
    conservative_weights, conservative_momentum, _ = build_custom_dynamic_core_target_weights(
        prices=prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
        lookback=lookback,
        core_weight=conservative_core_weight,
        mom60_cut=0.04,
        mom120_cut=0.10,
        ma_window=120,
    )
    aggressive_mask = aggressive_trend & (aggressive_momentum >= regime_momentum_cut)
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    mixed_momentum = conservative_momentum.copy()
    mixed_momentum.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    mixed_momentum = mixed_momentum.rename("current_momentum")

    risk_codes = [code for code in risk_codes if code in prices.columns]
    base_result, _ = run_target_weights_strategy(
        prices,
        selected,
        mixed_target_weights,
        mixed_momentum,
        fee_rate,
        slippage_rate,
    )
    row_risk_weight = mixed_target_weights[risk_codes].sum(axis=1)
    reduce_mask = (
        (row_risk_weight > overheat_cap)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (mixed_momentum >= DEFAULT_OVERHEAT_MOMENTUM_CUT)
    )
    capped_target_weights = apply_risk_cap(
        mixed_target_weights,
        risk_budget_codes=risk_codes,
        trigger_mask=reduce_mask,
        risk_cap=overheat_cap,
    )
    high_reduce_mask = (
        (capped_target_weights[risk_codes].sum(axis=1) > overheat_high_cap)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (mixed_momentum >= overheat_high_momentum)
    )
    capped_target_weights = apply_risk_cap(
        capped_target_weights,
        risk_budget_codes=risk_codes,
        trigger_mask=high_reduce_mask,
        risk_cap=overheat_high_cap,
    )
    return run_target_weights_strategy(
        prices,
        selected,
        capped_target_weights,
        mixed_momentum,
        fee_rate,
        slippage_rate,
    )


def build_variants(refresh: bool) -> list[dict[str, object]]:
    base_pool = load_fixed_etf_pool()
    variants: list[dict[str, object]] = [
        {
            "name": "base_regime_mix",
            "selected": base_pool,
            "risk_codes": BASE_RISK_CODES,
            "defensive_codes": BASE_DEFENSIVE_CODES,
        }
    ]
    for code in BASE_RISK_CODES + BASE_DEFENSIVE_CODES:
        selected = base_pool[base_pool["code"] != code].copy()
        variants.append(
            {
                "name": f"remove_{code}",
                "selected": selected,
                "risk_codes": [item for item in BASE_RISK_CODES if item != code],
                "defensive_codes": [item for item in BASE_DEFENSIVE_CODES if item != code],
            }
        )

    remove_511580_selected = base_pool[base_pool["code"] != "511580"].copy()
    for overheat_cap, overheat_high_cap, overheat_high_momentum in [
        (0.28, 0.18, 0.30),
        (0.28, 0.20, 0.30),
        (0.29, 0.18, 0.30),
        (0.29, 0.20, 0.30),
        (0.30, 0.18, 0.30),
    ]:
        variants.append(
            {
                "name": (
                    f"remove_511580_cap{int(round(overheat_cap * 100)):02d}_"
                    f"hi{int(round(overheat_high_cap * 100)):02d}_"
                    f"hm{int(round(overheat_high_momentum * 100)):02d}"
                ),
                "selected": remove_511580_selected,
                "risk_codes": [item for item in BASE_RISK_CODES],
                "defensive_codes": ["518880", "512890"],
                "params": {
                    "overheat_cap": overheat_cap,
                    "overheat_high_cap": overheat_high_cap,
                    "overheat_high_momentum": overheat_high_momentum,
                },
            }
        )

    if refresh:
        for side, item in ADDITION_CANDIDATES:
            selected = pd.concat([base_pool, pd.DataFrame([item])], ignore_index=True)
            risk_codes = BASE_RISK_CODES + ([item["code"]] if side == "risk" else [])
            defensive_codes = BASE_DEFENSIVE_CODES + ([item["code"]] if side == "defensive" else [])
            variants.append(
                {
                    "name": f"add_{item['code']}",
                    "selected": selected,
                    "risk_codes": risk_codes,
                    "defensive_codes": defensive_codes,
                }
            )
        selected = pd.concat([base_pool[base_pool["code"] != "512890"], pd.DataFrame([ETF_510880])], ignore_index=True)
        variants.append(
            {
                "name": "replace_512890_with_510880",
                "selected": selected,
                "risk_codes": BASE_RISK_CODES,
                "defensive_codes": ["511580", "518880", "510880"],
            }
        )
    return variants


def load_base_prices(*, years: int, start_date: str, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """优先按本脚本固定池加载价格；formal core 缓存缺列时自动补抓。"""
    base_selected = load_fixed_etf_pool()
    _, core_cached_prices = load_cached_data()
    extra_paths = [Path("momentum_backtest/output/core/prices.csv")] if not core_cached_prices.empty else None
    prices = load_workspace_selected_prices(
        base_selected,
        years=years if refresh else None,
        start_date=pd.Timestamp(start_date),
        refresh=refresh,
        fetch_fn=fetch_histories,
        extra_paths=extra_paths,
        missing_label="required regime-mix histories",
    )
    prices = prices.dropna(how="all")
    return base_selected, prices


def load_refresh_price_panel(*, years: int, start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """refresh 模式下一次性抓全候选池，避免每个变体重复拉历史。"""
    base_selected = load_fixed_etf_pool()
    extra_selected = pd.DataFrame([item for _, item in ADDITION_CANDIDATES])
    combined_selected = pd.concat([base_selected, extra_selected], ignore_index=True)
    combined_selected = combined_selected.drop_duplicates(subset="code", keep="first").reset_index(drop=True)
    prices = fetch_histories(combined_selected, years=years)
    prices = prices.loc[prices.index >= pd.Timestamp(start_date)].copy()
    prices = prices.dropna(how="all")
    return combined_selected, prices


def main() -> int:
    args = parse_args()

    refresh_selected: pd.DataFrame | None = None
    refresh_prices: pd.DataFrame | None = None
    if args.refresh:
        refresh_selected, refresh_prices = load_refresh_price_panel(years=args.years, start_date=args.start_date)
        base_selected = load_fixed_etf_pool()
        base_codes = [code for code in base_selected["code"].astype(str) if code in refresh_prices.columns]
        base_prices = refresh_prices[base_codes].copy()
    else:
        base_selected, base_prices = load_base_prices(
            years=args.years,
            start_date=args.start_date,
            refresh=False,
        )
    variants = build_variants(refresh=args.refresh)

    rows: list[dict[str, object]] = []
    compare_df = build_compare_frame(base_prices)
    variant_count = 0
    base_summary: dict[str, float | int | str] | None = None

    for variant in variants:
        selected = pd.DataFrame(variant["selected"]).copy()
        selected_codes = selected["code"].astype(str).tolist()
        uses_only_base = set(selected_codes).issubset(set(base_selected["code"].astype(str)))
        if args.refresh:
            assert refresh_prices is not None
            prices = refresh_prices[[code for code in selected_codes if code in refresh_prices.columns]].copy()
        elif uses_only_base:
            prices = base_prices[[code for code in selected_codes if code in base_prices.columns]].copy()
        else:
            continue
        if prices.empty or "510300" not in prices.columns:
            continue
        prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)].copy()
        prices = prices.dropna(how="all")
        if prices.empty:
            continue

        result, trades = run_custom_regime_mix_strategy(
            prices=prices,
            selected=selected,
            risk_codes=list(variant["risk_codes"]),
            defensive_codes=list(variant["defensive_codes"]),
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        summary = summarize(result, trades, selected)
        summary = append_variant_result(
            rows,
            compare_df,
            None,
            strategy=str(variant["name"]),
            result=result,
            summary=summary,
            strategy_field="variant",
            nav_column=str(variant["name"]),
            extra_fields={
                "start_date": str(result.index.min().date()),
                "end_date": str(result.index.max().date()),
                "risk_codes": ",".join([code for code in variant["risk_codes"] if code in prices.columns]),
                "defensive_codes": ",".join([code for code in variant["defensive_codes"] if code in prices.columns]),
            },
        )
        variant_count += 1
        if variant["name"] == "base_regime_mix":
            base_summary = summary

    if not rows or base_summary is None:
        raise RuntimeError("no valid pool variants were evaluated")

    summary_df = finalize_baseline_diff_summary(
        rows,
        baseline_field="variant",
        baseline_value="base_regime_mix",
        metric_mappings=(
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "mdd_diff"),
            ("trade_count", "trade_diff"),
        ),
        sort_by=["annualized_return", "sharpe_rf0", "max_drawdown"],
        ascending=[False, False, False],
    )
    focus = ["base_regime_mix"] + summary_df["variant"].head(6).tolist()
    focus = list(dict.fromkeys(focus))
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary_df,
        compare_df,
        plot_filename="top_variants.png",
        title=f"Regime Mix Pool Variations ({variant_count} variants)",
        lines=[
            (name, name, 2.1 if name == "base_regime_mix" else 1.7)
            for name in focus
            if name in compare_df.columns
        ],
        benchmark_label="HS300",
        benchmark_column="hs300_benchmark",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
