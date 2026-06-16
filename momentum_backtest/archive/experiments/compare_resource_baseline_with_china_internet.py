#!/usr/bin/env python3
"""Compare China Internet variants using resource_abs08 as the baseline."""

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

from compare_candidate_pool_additions import BASE_DEFENSIVE_CODES, BASE_RISK_CODES, ETF_513050
from compare_resource_guards import RESOURCE_CODE
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
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

CHINA_INTERNET_CODE = "513050"
OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_resource_baseline_with_china_internet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare China Internet variants on top of resource_abs08 baseline.")
    parser.add_argument("--years", type=int, default=6, help="History years to request before overlap alignment.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    return parser.parse_args()


def summarize(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | str]:
    daily_ret = result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    ci_share = float((result["holding"] == CHINA_INTERNET_CODE).fillna(False).mean())
    resource_share = float((result["holding"] == RESOURCE_CODE).fillna(False).mean())
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
        "china_internet_holding_share": ci_share,
        "resource_holding_share": resource_share,
        "latest_momentum": float(result["current_momentum"].dropna().iloc[-1]),
        "latest_exposure": float(result["exposure"].iloc[-1]),
    }


def choose_signal(
    risk_scores: pd.Series,
    defensive_scores: pd.Series,
    china_mode: str,
) -> tuple[object, float, float]:
    valid_risk = risk_scores.dropna().sort_values(ascending=False)
    valid_def = defensive_scores.dropna().sort_values(ascending=False)
    if valid_risk.empty:
        r_asset = pd.NA
        r_score = float("nan")
    else:
        r_asset = valid_risk.index[0]
        r_score = float(valid_risk.iloc[0])
    d_asset = valid_def.index[0] if not valid_def.empty else pd.NA
    d_score = float(valid_def.iloc[0]) if not valid_def.empty else float("nan")

    candidate_asset = r_asset
    candidate_score = r_score
    if pd.notna(candidate_asset) and str(candidate_asset) == RESOURCE_CODE and pd.notna(candidate_score) and candidate_score <= 0.08:
        if len(valid_risk) >= 2:
            candidate_asset = valid_risk.index[1]
            candidate_score = float(valid_risk.iloc[1])

    if pd.notna(candidate_asset) and str(candidate_asset) == CHINA_INTERNET_CODE:
        if china_mode == "exclude_to_second":
            if len(valid_risk) >= 2:
                candidate_asset = valid_risk.index[1]
                candidate_score = float(valid_risk.iloc[1])
            else:
                candidate_asset = pd.NA
                candidate_score = float("nan")
        elif china_mode == "reverse_to_defensive":
            candidate_asset = d_asset
            candidate_score = float(d_score) if pd.notna(d_score) else float("nan")

    if pd.notna(candidate_score) and candidate_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
        exposure = 1.0
        if pd.notna(candidate_asset) and str(candidate_asset) == CHINA_INTERNET_CODE and china_mode == "cap70":
            exposure = 0.70
        return candidate_asset, exposure, float(candidate_score)
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
    risk_codes: list[str],
    china_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1
    risk_scores = momentum[[code for code in risk_codes if code in prices.columns]]
    defensive_scores = momentum[[code for code in BASE_DEFENSIVE_CODES if code in prices.columns]]

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    for dt_idx in prices.index:
        asset, exposure, score = choose_signal(
            risk_scores=risk_scores.loc[dt_idx],
            defensive_scores=defensive_scores.loc[dt_idx],
            china_mode=china_mode,
        )
        signal.loc[dt_idx] = asset
        target_exposure.loc[dt_idx] = exposure
        current_momentum.loc[dt_idx] = score

    base_result, _ = run_signal_strategy(prices, selected, signal, target_exposure, current_momentum, fee_rate, slippage_rate)
    adjusted_exposure = target_exposure.copy()
    risk_code_set = [code for code in risk_scores.columns if code in prices.columns]
    overheat_mask = (
        signal.isin(risk_code_set)
        & (adjusted_exposure > DEFAULT_OVERHEAT_MAX_EXPOSURE)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_MOMENTUM_CUT)
    )
    adjusted_exposure.loc[overheat_mask] = DEFAULT_OVERHEAT_MAX_EXPOSURE
    extreme_mask = (
        signal.isin(risk_code_set)
        & (adjusted_exposure > DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT)
    )
    adjusted_exposure.loc[extreme_mask] = DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE
    return run_signal_strategy(prices, selected, signal, adjusted_exposure, current_momentum, fee_rate, slippage_rate)


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    selected = pd.concat([base_pool, pd.DataFrame([ETF_513050])], ignore_index=True)
    prices = fetch_histories(selected, years=args.years).dropna(how="any")

    resource_risk_codes = BASE_RISK_CODES + [RESOURCE_CODE]
    resource_ci_risk_codes = BASE_RISK_CODES + [RESOURCE_CODE, CHINA_INTERNET_CODE]
    variants = [
        ("resource_abs08_baseline", resource_risk_codes, "normal"),
        ("resource_abs08_plus_ci", resource_ci_risk_codes, "normal"),
        ("resource_abs08_plus_ci_cap70", resource_ci_risk_codes, "cap70"),
        ("resource_abs08_ci_exclude_to_second", resource_ci_risk_codes, "exclude_to_second"),
        ("resource_abs08_ci_reverse_to_defensive", resource_ci_risk_codes, "reverse_to_defensive"),
    ]

    rows: list[dict[str, object]] = []
    compare_df = None
    base_metrics: dict[str, float | int] = {}
    for name, risk_codes, china_mode in variants:
        result, trades = run_variant(
            prices=prices,
            selected=selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            risk_codes=risk_codes,
            china_mode=china_mode,
        )
        row = {"variant": name, **summarize(result, trades)}
        rows.append(row)
        if compare_df is None:
            compare_df = pd.DataFrame(index=result.index)
            compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")
        compare_df[name] = result["nav"].reindex(compare_df.index)
        if name == "resource_abs08_baseline":
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
    write_dataframe_csv_atomic(summary, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    for column in [
        "resource_abs08_baseline",
        "resource_abs08_plus_ci",
        "resource_abs08_plus_ci_cap70",
        "resource_abs08_ci_exclude_to_second",
        "resource_abs08_ci_reverse_to_defensive",
    ]:
        ax.plot(compare_df.index, compare_df[column], linewidth=2.0 if column == "resource_abs08_baseline" else 1.8, label=column)
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.5, linestyle="--", label="HS300 ETF")
    ax.set_title("Resource Abs08 Baseline vs China Internet Variants", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
