#!/usr/bin/env python3
"""Scan pool additions/removals/replacements on top of the current notified regime-mix strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ..runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from compare_candidate_pool_additions import (
    ETF_510230,
    ETF_510500,
    ETF_510880,
    ETF_510900,
    ETF_511090,
    ETF_511260,
    ETF_511380,
    ETF_512100,
    ETF_513030,
    ETF_513050,
    ETF_588000,
)
from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_regime_mix_pool_variations")
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
    ("defensive", ETF_511090),
    ("defensive", ETF_511260),
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
    capped_target_weights = mixed_target_weights.copy()
    if reduce_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[reduce_mask] = overheat_cap / row_risk_weight.loc[reduce_mask]
        capped_target_weights.loc[reduce_mask, risk_codes] = capped_target_weights.loc[reduce_mask, risk_codes].mul(
            scale.loc[reduce_mask], axis=0
        )
    high_reduce_mask = (
        (capped_target_weights[risk_codes].sum(axis=1) > overheat_high_cap)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (mixed_momentum >= overheat_high_momentum)
    )
    if high_reduce_mask.any():
        high_scale = pd.Series(1.0, index=prices.index, dtype="float64")
        high_scale.loc[high_reduce_mask] = (
            overheat_high_cap / capped_target_weights[risk_codes].sum(axis=1).loc[high_reduce_mask]
        )
        capped_target_weights.loc[high_reduce_mask, risk_codes] = capped_target_weights.loc[
            high_reduce_mask, risk_codes
        ].mul(high_scale.loc[high_reduce_mask], axis=0)
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


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_selected, base_prices = load_cached_data()
    base_prices = base_prices.loc[base_prices.index >= pd.Timestamp(args.start_date)].copy()
    variants = build_variants(refresh=args.refresh)

    rows: list[dict[str, object]] = []
    compare_df = pd.DataFrame(index=base_prices.index)
    variant_count = 0
    base_summary: dict[str, float | int | str] | None = None

    for variant in variants:
        selected = pd.DataFrame(variant["selected"]).copy()
        uses_only_base = set(selected["code"]).issubset(set(base_selected["code"]))
        if uses_only_base:
            prices = base_prices[[code for code in selected["code"] if code in base_prices.columns]].copy()
        else:
            if not args.refresh:
                continue
            prices = fetch_histories(selected, years=args.years)
            prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)].copy()
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
        compare_df[variant["name"]] = result["nav"].reindex(compare_df.index)
        summary = summarize(result, trades, selected)
        summary["variant"] = variant["name"]
        summary["start_date"] = str(result.index.min().date())
        summary["end_date"] = str(result.index.max().date())
        summary["risk_codes"] = ",".join([code for code in variant["risk_codes"] if code in prices.columns])
        summary["defensive_codes"] = ",".join([code for code in variant["defensive_codes"] if code in prices.columns])
        rows.append(summary)
        variant_count += 1
        if variant["name"] == "base_regime_mix":
            base_summary = summary

    if not rows or base_summary is None:
        raise RuntimeError("no valid pool variants were evaluated")

    summary_df = pd.DataFrame(rows)
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(base_summary["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(base_summary["sharpe_rf0"])
    summary_df["mdd_diff"] = summary_df["max_drawdown"] - float(base_summary["max_drawdown"])
    summary_df["trade_diff"] = summary_df["trade_count"] - int(base_summary["trade_count"])
    summary_df = summary_df.sort_values(["annualized_return", "sharpe_rf0", "max_drawdown"], ascending=[False, False, False])
    write_dataframe_csv_atomic(summary_df, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    focus = ["base_regime_mix"] + summary_df["variant"].head(6).tolist()
    focus = list(dict.fromkeys(focus))
    for name in focus:
        if name not in compare_df.columns:
            continue
        ax.plot(compare_df.index, compare_df[name], linewidth=2.1 if name == "base_regime_mix" else 1.7, label=name)
    ax.set_title(f"Regime Mix Pool Variations ({variant_count} variants)", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "top_variants.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
