#!/usr/bin/env python3
"""测试将资源 ETF 加入风险池后的保护规则。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

try:
    from .candidate_pool_common import BASE_DEFENSIVE_CODES, BASE_RISK_CODES
    from .official_baseline import apply_official_baseline_nav_anchor
    from .compare_strategy_refinements import run_signal_strategy
except ImportError:
    from candidate_pool_common import BASE_DEFENSIVE_CODES, BASE_RISK_CODES
    from official_baseline import apply_official_baseline_nav_anchor
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
    build_benchmark_nav,
    build_strategy_summary,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

RESOURCE_ETF = {"theme": "资源", "code": "510410", "name": "资源ETF", "sina_symbol": "sh510410"}
RESOURCE_CODE = "510410"
FLOAT_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试将资源 ETF 加入风险池后的保护规则。")
    parser.add_argument("--years", type=int, default=6, help="向前抓取多少年历史数据，再对齐公共区间。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def summarize(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | str]:
    weight_col = f"weight_{RESOURCE_CODE}"
    if weight_col in result.columns:
        resource_weights = result[weight_col].fillna(0.0).astype(float)
        holding_share = float((resource_weights.abs() > FLOAT_TOL).mean())
        exposure_share = float(resource_weights.mean())
    else:
        holding_codes = result["holding"].map(lambda value: str(value).replace(".0", "") if pd.notna(value) else "")
        exposure = result["exposure"].fillna(0.0).astype(float)
        resource_mask = holding_codes == RESOURCE_CODE
        holding_share = float(resource_mask.mean())
        exposure_share = float(exposure.where(resource_mask, 0.0).mean())
    return {
        "start_date": result.index[0].date().isoformat(),
        "end_date": result.index[-1].date().isoformat(),
        **build_strategy_summary(result, trades),
        "resource_holding_share": holding_share,
        "resource_exposure_share": exposure_share,
    }


def build_base_components(prices: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [code for code in BASE_RISK_CODES + [RESOURCE_CODE] if code in prices.columns]
    defensive_codes = [code for code in BASE_DEFENSIVE_CODES if code in prices.columns]
    return momentum, momentum[risk_codes], momentum[defensive_codes]


def choose_signal(
    risk_scores: pd.Series,
    defensive_scores: pd.Series,
    resource_abs_threshold: float | None = None,
    resource_margin_threshold: float | None = None,
) -> tuple[object, float, float]:
    valid_risk = risk_scores.dropna().sort_values(ascending=False)
    valid_def = defensive_scores.dropna().sort_values(ascending=False)
    r_asset = valid_risk.index[0] if not valid_risk.empty else pd.NA
    r_score = float(valid_risk.iloc[0]) if not valid_risk.empty else float("nan")
    d_asset = valid_def.index[0] if not valid_def.empty else pd.NA
    d_score = float(valid_def.iloc[0]) if not valid_def.empty else float("nan")

    candidate_asset = r_asset
    candidate_score = r_score
    if pd.notna(r_asset) and str(r_asset) == RESOURCE_CODE:
        second_score = float(valid_risk.iloc[1]) if len(valid_risk) >= 2 else float("nan")
        second_asset = valid_risk.index[1] if len(valid_risk) >= 2 else pd.NA
        if resource_abs_threshold is not None and pd.notna(r_score) and r_score <= resource_abs_threshold:
            candidate_asset = second_asset
            candidate_score = second_score
        elif (
            resource_margin_threshold is not None
            and pd.notna(second_score)
            and (r_score - second_score) < resource_margin_threshold
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
    resource_abs_threshold: float | None = None,
    resource_margin_threshold: float | None = None,
    resource_max_exposure: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, risk_mom, defensive_mom = build_base_components(prices, lookback)
    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        asset, exposure, momentum = choose_signal(
            risk_scores=risk_mom.loc[dt_idx],
            defensive_scores=defensive_mom.loc[dt_idx],
            resource_abs_threshold=resource_abs_threshold,
            resource_margin_threshold=resource_margin_threshold,
        )
        signal.loc[dt_idx] = asset
        target_exposure.loc[dt_idx] = exposure
        current_momentum.loc[dt_idx] = momentum

    base_result, _ = run_signal_strategy(prices, selected, signal, target_exposure, current_momentum, fee_rate, slippage_rate)
    adjusted_exposure = target_exposure.copy()
    if resource_max_exposure is not None:
        resource_mask = signal == RESOURCE_CODE
        adjusted_exposure.loc[resource_mask] = adjusted_exposure.loc[resource_mask].clip(upper=resource_max_exposure)

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
    selected = pd.concat([base_pool, pd.DataFrame([RESOURCE_ETF])], ignore_index=True)
    prices = fetch_histories(selected, years=args.years).dropna(how="any")

    variants = [
        ("base_pool", None, None, None, False),
        ("resource_risk", None, None, None, True),
        ("resource_cap_70", None, None, 0.70, True),
        ("resource_cap_60", None, None, 0.60, True),
        ("resource_abs_08", 0.08, None, None, True),
        ("resource_abs_10", 0.10, None, None, True),
        ("resource_margin_03", None, 0.03, None, True),
        ("resource_margin_05", None, 0.05, None, True),
        ("resource_margin_03_cap_70", None, 0.03, 0.70, True),
    ]

    rows: list[dict[str, object]] = []
    compare_df = None
    base_metrics: dict[str, float | int] = {}
    base_prices = prices[base_pool["code"].tolist()].copy()

    for name, abs_thr, margin_thr, max_exposure, include_resource in variants:
        current_selected = selected if include_resource else base_pool.copy()
        current_prices = prices if include_resource else base_prices
        result, trades = run_variant(
            current_prices,
            current_selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            resource_abs_threshold=abs_thr if include_resource else None,
            resource_margin_threshold=margin_thr if include_resource else None,
            resource_max_exposure=max_exposure if include_resource else None,
        )
        if name == "base_pool":
            result = apply_official_baseline_nav_anchor(result)
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
    write_dataframe_csv_atomic(summary, RESEARCH_OUTPUT_DIR / "resource_guard_summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, RESEARCH_OUTPUT_DIR / "resource_guard_nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    for column in [
        "base_pool",
        "resource_risk",
        "resource_cap_70",
        "resource_abs_08",
        "resource_margin_03",
        "resource_margin_03_cap_70",
    ]:
        ax.plot(compare_df.index, compare_df[column], linewidth=1.8 if column != "base_pool" else 2.3, label=column)
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.5, linestyle="--", label="HS300 ETF")
    ax.set_title("Resource ETF Guard Variants", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, RESEARCH_OUTPUT_DIR / "resource_guard_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
