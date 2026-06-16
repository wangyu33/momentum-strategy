#!/usr/bin/env python3
"""Test guard rules for adding Nonferrous Metals ETF into the risk pool."""

from __future__ import annotations

import argparse

try:
    from ..runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from compare_candidate_pool_additions import BASE_DEFENSIVE_CODES, BASE_RISK_CODES
from compare_strategy_refinements import run_signal_strategy
from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
    DEFAULT_OVERHEAT_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    RESEARCH_OUTPUT_DIR,
    annualized_return,
    build_benchmark_nav,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    max_drawdown,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

NONFERROUS_ETF = {"theme": "有色金属", "code": "512400", "name": "有色金属ETF南方", "sina_symbol": "sh512400"}
NONFERROUS_CODE = "512400"
OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_nonferrous_guards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Nonferrous Metals ETF guard variants.")
    parser.add_argument("--years", type=int, default=15, help="History years to request before overlap alignment.")
    parser.add_argument("--start-date", type=str, default="2012-01-01", help="Analysis start date.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    return parser.parse_args()


def summarize(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | str]:
    daily_ret = result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    holding_share = float((result["holding"] == NONFERROUS_CODE).fillna(False).mean())
    return {
        "start_date": result.index[0].date().isoformat(),
        "end_date": result.index[-1].date().isoformat(),
        "total_return": float(result["nav"].iloc[-1] - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown(result["nav"]),
        "trade_count": int(len(trades)),
        "avg_exposure": float(result["exposure"].mean()),
        "nonferrous_holding_share": holding_share,
        "latest_momentum": float(result["current_momentum"].dropna().iloc[-1]),
        "latest_exposure": float(result["exposure"].iloc[-1]),
    }


def build_base_components(prices: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [code for code in BASE_RISK_CODES + [NONFERROUS_CODE] if code in prices.columns]
    defensive_codes = [code for code in BASE_DEFENSIVE_CODES if code in prices.columns]
    return momentum, momentum[risk_codes], momentum[defensive_codes]


def choose_signal(
    risk_scores: pd.Series,
    defensive_scores: pd.Series,
    nonferrous_abs_threshold: float | None = None,
    nonferrous_margin_threshold: float | None = None,
) -> tuple[object, float, float]:
    valid_risk = risk_scores.dropna().sort_values(ascending=False)
    valid_def = defensive_scores.dropna().sort_values(ascending=False)
    r_asset = valid_risk.index[0] if not valid_risk.empty else pd.NA
    r_score = float(valid_risk.iloc[0]) if not valid_risk.empty else float("nan")
    d_asset = valid_def.index[0] if not valid_def.empty else pd.NA
    d_score = float(valid_def.iloc[0]) if not valid_def.empty else float("nan")

    candidate_asset = r_asset
    candidate_score = r_score
    if pd.notna(r_asset) and str(r_asset) == NONFERROUS_CODE:
        second_score = float(valid_risk.iloc[1]) if len(valid_risk) >= 2 else float("nan")
        second_asset = valid_risk.index[1] if len(valid_risk) >= 2 else pd.NA
        if nonferrous_abs_threshold is not None and pd.notna(r_score) and r_score <= nonferrous_abs_threshold:
            candidate_asset = second_asset
            candidate_score = second_score
        elif (
            nonferrous_margin_threshold is not None
            and pd.notna(second_score)
            and (r_score - second_score) < nonferrous_margin_threshold
        ):
            candidate_asset = second_asset
            candidate_score = second_score

    if pd.notna(candidate_score) and candidate_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
        return candidate_asset, 1.0, float(candidate_score)
    if pd.notna(candidate_score) and candidate_score > 0 and pd.notna(d_asset):
        return d_asset, DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT, float(candidate_score)
    if pd.notna(d_asset):
        return d_asset, (1.0 if pd.notna(d_score) and d_score > 0 else 0.0), d_score
    return pd.NA, 0.0, float("nan")


def run_variant(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    nonferrous_abs_threshold: float | None = None,
    nonferrous_margin_threshold: float | None = None,
    nonferrous_max_exposure: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, risk_mom, defensive_mom = build_base_components(prices, lookback)
    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        asset, exposure, momentum = choose_signal(
            risk_scores=risk_mom.loc[dt_idx],
            defensive_scores=defensive_mom.loc[dt_idx],
            nonferrous_abs_threshold=nonferrous_abs_threshold,
            nonferrous_margin_threshold=nonferrous_margin_threshold,
        )
        signal.loc[dt_idx] = asset
        target_exposure.loc[dt_idx] = exposure
        current_momentum.loc[dt_idx] = momentum

    base_result, _ = run_signal_strategy(prices, selected, signal, target_exposure, current_momentum, fee_rate, slippage_rate)
    adjusted_exposure = target_exposure.copy()
    if nonferrous_max_exposure is not None:
        nonferrous_mask = signal == NONFERROUS_CODE
        adjusted_exposure.loc[nonferrous_mask] = adjusted_exposure.loc[nonferrous_mask].clip(upper=nonferrous_max_exposure)

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


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    selected = pd.concat([base_pool, pd.DataFrame([NONFERROUS_ETF])], ignore_index=True)
    prices = fetch_histories(selected, years=args.years).dropna(how="any")
    prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)]

    variants = [
        ("base_pool", None, None, None, False),
        ("nonferrous_risk", None, None, None, True),
        ("nonferrous_cap_70", None, None, 0.70, True),
        ("nonferrous_cap_60", None, None, 0.60, True),
        ("nonferrous_abs_08", 0.08, None, None, True),
        ("nonferrous_abs_10", 0.10, None, None, True),
        ("nonferrous_margin_03", None, 0.03, None, True),
        ("nonferrous_margin_05", None, 0.05, None, True),
        ("nonferrous_margin_03_cap_70", None, 0.03, 0.70, True),
    ]

    rows: list[dict[str, object]] = []
    compare_df = None
    base_metrics: dict[str, float | int] = {}
    base_prices = prices[base_pool["code"].tolist()].copy()

    for name, abs_thr, margin_thr, max_exposure, include_nonferrous in variants:
        current_selected = selected if include_nonferrous else base_pool.copy()
        current_prices = prices if include_nonferrous else base_prices
        result, trades = run_variant(
            current_prices,
            current_selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            nonferrous_abs_threshold=abs_thr if include_nonferrous else None,
            nonferrous_margin_threshold=margin_thr if include_nonferrous else None,
            nonferrous_max_exposure=max_exposure if include_nonferrous else None,
        )
        if compare_df is None:
            compare_df = pd.DataFrame(index=result.index)
            compare_df["hs300_benchmark"] = build_benchmark_nav(current_prices, benchmark_code="510300")
        compare_df[name] = result["nav"].reindex(compare_df.index)
        row = {"variant": name, **summarize(result, trades)}
        rows.append(row)
        if name == "base_pool":
            base_metrics = {
                "total_return": float(row["total_return"]),
                "annualized_return": float(row["annualized_return"]),
                "sharpe_rf0": float(row["sharpe_rf0"]),
                "max_drawdown": float(row["max_drawdown"]),
                "trade_count": int(row["trade_count"]),
            }

    summary = pd.DataFrame(rows)
    summary["return_diff"] = summary["total_return"] - float(base_metrics["total_return"])
    summary["annualized_diff"] = summary["annualized_return"] - float(base_metrics["annualized_return"])
    summary["sharpe_diff"] = summary["sharpe_rf0"] - float(base_metrics["sharpe_rf0"])
    summary["mdd_diff"] = summary["max_drawdown"] - float(base_metrics["max_drawdown"])
    summary["trade_diff"] = summary["trade_count"] - int(base_metrics["trade_count"])

    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    compare_df.to_csv(OUTPUT_DIR / "nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    for column in [
        "base_pool",
        "nonferrous_risk",
        "nonferrous_cap_70",
        "nonferrous_abs_08",
        "nonferrous_margin_03",
        "nonferrous_margin_03_cap_70",
    ]:
        ax.plot(compare_df.index, compare_df[column], linewidth=1.8 if column != "base_pool" else 2.3, label=column)
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.5, linestyle="--", label="HS300 ETF")
    ax.set_title("Nonferrous Metals ETF Guard Variants", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
