#!/usr/bin/env python3
"""对比历史动量分位控仓与现有过热风控的几种整合方式。"""

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
    from .official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

from compare_hs300_regime_fixes import run_target_weights_strategy
from goal_optimization_common import MARKET_VOLUME_CACHE_PATH
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    RISK_CODES,
    build_asset_own_momentum_percentile,
    build_benchmark_nav,
    build_default_strategy_params,
    build_strategy_summary,
    build_yearly_return_rows,
    compute_signal_asset_momentum,
    ensure_output_dirs,
    load_core_selected_and_prices,
    normalize_code,
    run_default_strategy_with_params,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "momentum_cap_integration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比分位控仓与现有过热风控的整合方案。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--cap-start", type=float, default=0.90, help="开始减仓的自身历史动量分位。")
    parser.add_argument("--cap-end", type=float, default=0.95, help="减仓打满的自身历史动量分位。")
    parser.add_argument("--cap-floor", type=float, default=0.50, help="达到上限后保留的最大总仓位。")
    parser.add_argument("--rebalance-threshold", type=float, default=0.05, help="最小调仓阈值，默认总权重变化至少 5%% 才执行。")
    return parser.parse_args()


def load_cached_market_proxy() -> pd.DataFrame:
    proxy = pd.read_csv(MARKET_VOLUME_CACHE_PATH, parse_dates=["date"])
    return proxy.set_index("date")


def summarize(
    strategy: str,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    trigger_days: int = 0,
    threshold_blocked_days: int = 0,
) -> dict[str, float | int | str]:
    latest = result.iloc[-1]
    return {
        "strategy": strategy,
        **build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True),
        "latest_signal": latest.get("signal"),
        "latest_holding": latest.get("holding"),
        "latest_exposure": latest.get("exposure"),
        "latest_selected_momentum_pct": latest.get("selected_momentum_percentile"),
        "momentum_pct_cap_trigger_days": trigger_days,
        "threshold_blocked_days": threshold_blocked_days,
    }


def build_target_weights(result: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in result.columns if col.startswith("target_weight_")]
    weights = result[cols].copy()
    weights.columns = [col.removeprefix("target_weight_") for col in cols]
    return weights.fillna(0.0)


def disable_overheat_params(params: dict[str, object]) -> dict[str, object]:
    disabled = dict(params)
    disabled.update(
        {
            "pre_overheat_end_exposure": 1.0,
            "overheat_max_exposure": 1.0,
            "overheat_high_max_exposure": 1.0,
            "overheat_stability_cap": 1.0,
        }
    )
    return disabled


def apply_min_rebalance_threshold(
    target_weights: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.Series]:
    if threshold <= 0:
        blocked = pd.Series(False, index=target_weights.index, dtype=bool, name="threshold_blocked")
        return target_weights.copy(), blocked

    adjusted = target_weights.copy()
    blocked = pd.Series(False, index=target_weights.index, dtype=bool, name="threshold_blocked")
    previous = adjusted.iloc[0].copy()
    for dt_idx in adjusted.index[1:]:
        desired = adjusted.loc[dt_idx]
        turnover = float((desired - previous).abs().sum())
        if turnover < threshold:
            adjusted.loc[dt_idx] = previous
            blocked.loc[dt_idx] = True
        else:
            previous = desired.copy()
    return adjusted, blocked


def overlay_momentum_cap(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    base_result: pd.DataFrame,
    *,
    cap_start: float,
    cap_end: float,
    cap_floor: float,
    eligible_bucket: str,
    fee_rate: float,
    slippage_rate: float,
    rebalance_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    if cap_end < cap_start:
        raise ValueError("cap_end must be >= cap_start")

    target_weights = build_target_weights(base_result)
    selected_pct = pd.Series(index=prices.index, dtype="float64", name="selected_momentum_percentile")
    pct_cap_triggered = pd.Series(False, index=prices.index, dtype=bool, name="selected_momentum_pct_cap_triggered")
    signal_pct = build_asset_own_momentum_percentile(prices, lookback=25, state_lookback=756, min_periods=120)
    risk_codes = {code for code in RISK_CODES if code in prices.columns}
    all_codes = set(prices.columns)
    defensive_codes = all_codes - risk_codes

    if eligible_bucket == "all":
        eligible_codes = all_codes
    elif eligible_bucket == "risk":
        eligible_codes = risk_codes
    elif eligible_bucket == "defensive":
        eligible_codes = defensive_codes
    else:
        raise ValueError(f"unsupported eligible_bucket: {eligible_bucket}")

    for dt_idx in prices.index:
        signal_code = normalize_code(base_result.loc[dt_idx, "signal"])
        if signal_code is None or signal_code not in eligible_codes or signal_code not in signal_pct.columns:
            continue
        pct_value = signal_pct.loc[dt_idx, signal_code]
        if pd.isna(pct_value):
            continue
        selected_pct.loc[dt_idx] = float(pct_value)
        if float(pct_value) < cap_start:
            continue
        if cap_end > cap_start:
            progress = min(max((float(pct_value) - cap_start) / (cap_end - cap_start), 0.0), 1.0)
            cap_value = 1.0 + (cap_floor - 1.0) * progress
        else:
            cap_value = cap_floor
        total_weight = float(target_weights.loc[dt_idx].sum())
        if total_weight > cap_value + 1e-12:
            target_weights.loc[dt_idx, :] = target_weights.loc[dt_idx, :] * (cap_value / total_weight)
            pct_cap_triggered.loc[dt_idx] = True

    threshold_blocked_days = 0
    adjusted_target_weights = target_weights
    if rebalance_threshold > 0:
        adjusted_target_weights, blocked = apply_min_rebalance_threshold(target_weights, rebalance_threshold)
        threshold_blocked_days = int(blocked.sum())

    signal_series = adjusted_target_weights.idxmax(axis=1).where(adjusted_target_weights.max(axis=1) > 0, pd.NA)
    signal_momentum = compute_signal_asset_momentum(prices, signal_series, 25).rename("current_momentum")
    result, trades = run_target_weights_strategy(
        prices,
        selected,
        adjusted_target_weights,
        signal_momentum,
        fee_rate,
        slippage_rate,
    )
    result["selected_momentum_percentile"] = selected_pct.reindex(result.index)
    result["selected_momentum_pct_cap_triggered"] = pct_cap_triggered.reindex(result.index).fillna(False)
    result = pd.concat([result, adjusted_target_weights.add_prefix("target_weight_")], axis=1)
    return result, trades, int(pct_cap_triggered.sum()), threshold_blocked_days


def run_official_variant(
    strategy_name: str,
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    params: dict[str, object],
    market_proxy: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
    *,
    anchor_nav: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result, trades = run_default_strategy_with_params(
        prices,
        selected,
        params=params,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        market_proxy=market_proxy,
    )
    if anchor_nav:
        result = apply_official_baseline_nav_anchor(result)
    write_dataframe_csv_atomic(trades, OUTPUT_DIR / f"{strategy_name}_trades.csv", index=False)
    write_dataframe_csv_atomic(result.reset_index(), OUTPUT_DIR / f"{strategy_name}_nav_detail.csv", index=False)
    return result, trades


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected, prices = load_core_selected_and_prices()
    market_proxy = load_cached_market_proxy()
    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")

    baseline_params = build_default_strategy_params()
    stacked_params = build_default_strategy_params()
    stacked_params.update(
        {
            "signal_selected_momentum_pct_cap_start": args.cap_start,
            "signal_selected_momentum_pct_cap_end": args.cap_end,
            "signal_selected_momentum_pct_cap_floor": args.cap_floor,
            "signal_selected_momentum_pct_lookback": 756,
            "signal_selected_momentum_pct_min_periods": 120,
        }
    )
    replace_overheat_params = disable_overheat_params(build_default_strategy_params())

    baseline_result_raw, baseline_trades = run_official_variant(
        "baseline",
        prices,
        selected,
        baseline_params,
        market_proxy,
        args.fee_rate,
        args.slippage_rate,
        anchor_nav=False,
    )
    baseline_result = apply_official_baseline_nav_anchor(baseline_result_raw.copy())

    stacked_result_raw, stacked_trades = run_official_variant(
        f"stacked_all_signal_pct_{int(args.cap_start * 100)}_{int(args.cap_end * 100)}_floor{int(args.cap_floor * 100)}",
        prices,
        selected,
        stacked_params,
        market_proxy,
        args.fee_rate,
        args.slippage_rate,
        anchor_nav=False,
    )
    stacked_result = apply_official_baseline_nav_anchor(stacked_result_raw.copy())

    replace_overheat_result_raw, replace_overheat_trades = run_official_variant(
        "replace_overheat_base",
        prices,
        selected,
        replace_overheat_params,
        market_proxy,
        args.fee_rate,
        args.slippage_rate,
        anchor_nav=False,
    )
    replace_overheat_result = apply_official_baseline_nav_anchor(replace_overheat_result_raw.copy())

    stacked_threshold_name = (
        f"stacked_all_signal_pct_{int(args.cap_start * 100)}_{int(args.cap_end * 100)}_floor{int(args.cap_floor * 100)}"
        f"_threshold{int(args.rebalance_threshold * 100)}"
    )
    stacked_threshold_result, stacked_threshold_trades, stacked_threshold_trigger_days, stacked_threshold_blocked_days = overlay_momentum_cap(
        prices,
        selected,
        stacked_result_raw,
        cap_start=2.0,
        cap_end=2.0,
        cap_floor=1.0,
        eligible_bucket="all",
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        rebalance_threshold=args.rebalance_threshold,
    )
    stacked_threshold_result = apply_official_baseline_nav_anchor(stacked_threshold_result)
    write_dataframe_csv_atomic(stacked_threshold_trades, OUTPUT_DIR / f"{stacked_threshold_name}_trades.csv", index=False)
    write_dataframe_csv_atomic(stacked_threshold_result.reset_index(), OUTPUT_DIR / f"{stacked_threshold_name}_nav_detail.csv", index=False)

    defensive_only_name = f"defensive_only_pct_{int(args.cap_start * 100)}_{int(args.cap_end * 100)}_floor{int(args.cap_floor * 100)}"
    defensive_result, defensive_trades, defensive_trigger_days, defensive_blocked_days = overlay_momentum_cap(
        prices,
        selected,
        baseline_result_raw,
        cap_start=args.cap_start,
        cap_end=args.cap_end,
        cap_floor=args.cap_floor,
        eligible_bucket="defensive",
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        rebalance_threshold=0.0,
    )
    defensive_result = apply_official_baseline_nav_anchor(defensive_result)
    write_dataframe_csv_atomic(defensive_trades, OUTPUT_DIR / f"{defensive_only_name}_trades.csv", index=False)
    write_dataframe_csv_atomic(defensive_result.reset_index(), OUTPUT_DIR / f"{defensive_only_name}_nav_detail.csv", index=False)

    replace_risk_name = f"replace_overheat_risk_pct_{int(args.cap_start * 100)}_{int(args.cap_end * 100)}_floor{int(args.cap_floor * 100)}"
    replace_risk_result, replace_risk_trades, replace_risk_trigger_days, replace_risk_blocked_days = overlay_momentum_cap(
        prices,
        selected,
        replace_overheat_result_raw,
        cap_start=args.cap_start,
        cap_end=args.cap_end,
        cap_floor=args.cap_floor,
        eligible_bucket="risk",
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        rebalance_threshold=0.0,
    )
    replace_risk_result = apply_official_baseline_nav_anchor(replace_risk_result)
    write_dataframe_csv_atomic(replace_risk_trades, OUTPUT_DIR / f"{replace_risk_name}_trades.csv", index=False)
    write_dataframe_csv_atomic(replace_risk_result.reset_index(), OUTPUT_DIR / f"{replace_risk_name}_nav_detail.csv", index=False)

    replace_all_name = f"replace_overheat_all_pct_{int(args.cap_start * 100)}_{int(args.cap_end * 100)}_floor{int(args.cap_floor * 100)}"
    replace_all_result, replace_all_trades, replace_all_trigger_days, replace_all_blocked_days = overlay_momentum_cap(
        prices,
        selected,
        replace_overheat_result_raw,
        cap_start=args.cap_start,
        cap_end=args.cap_end,
        cap_floor=args.cap_floor,
        eligible_bucket="all",
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        rebalance_threshold=0.0,
    )
    replace_all_result = apply_official_baseline_nav_anchor(replace_all_result)
    write_dataframe_csv_atomic(replace_all_trades, OUTPUT_DIR / f"{replace_all_name}_trades.csv", index=False)
    write_dataframe_csv_atomic(replace_all_result.reset_index(), OUTPUT_DIR / f"{replace_all_name}_nav_detail.csv", index=False)

    replace_all_threshold_name = (
        f"replace_overheat_all_pct_{int(args.cap_start * 100)}_{int(args.cap_end * 100)}_floor{int(args.cap_floor * 100)}"
        f"_threshold{int(args.rebalance_threshold * 100)}"
    )
    replace_all_threshold_result, replace_all_threshold_trades, replace_all_threshold_trigger_days, replace_all_threshold_blocked_days = overlay_momentum_cap(
        prices,
        selected,
        replace_overheat_result_raw,
        cap_start=args.cap_start,
        cap_end=args.cap_end,
        cap_floor=args.cap_floor,
        eligible_bucket="all",
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        rebalance_threshold=args.rebalance_threshold,
    )
    replace_all_threshold_result = apply_official_baseline_nav_anchor(replace_all_threshold_result)
    write_dataframe_csv_atomic(replace_all_threshold_trades, OUTPUT_DIR / f"{replace_all_threshold_name}_trades.csv", index=False)
    write_dataframe_csv_atomic(replace_all_threshold_result.reset_index(), OUTPUT_DIR / f"{replace_all_threshold_name}_nav_detail.csv", index=False)

    variants: list[tuple[str, pd.DataFrame, pd.DataFrame, int, int]] = [
        ("baseline", baseline_result, baseline_trades, 0, 0),
        (
            f"stacked_all_signal_pct_{int(args.cap_start * 100)}_{int(args.cap_end * 100)}_floor{int(args.cap_floor * 100)}",
            stacked_result,
            stacked_trades,
            int(stacked_result.get("selected_momentum_pct_cap_triggered", pd.Series(False, index=stacked_result.index)).fillna(False).sum()),
            0,
        ),
        (stacked_threshold_name, stacked_threshold_result, stacked_threshold_trades, stacked_threshold_trigger_days, stacked_threshold_blocked_days),
        (defensive_only_name, defensive_result, defensive_trades, defensive_trigger_days, defensive_blocked_days),
        (replace_risk_name, replace_risk_result, replace_risk_trades, replace_risk_trigger_days, replace_risk_blocked_days),
        (replace_all_name, replace_all_result, replace_all_trades, replace_all_trigger_days, replace_all_blocked_days),
        (
            replace_all_threshold_name,
            replace_all_threshold_result,
            replace_all_threshold_trades,
            replace_all_threshold_trigger_days,
            replace_all_threshold_blocked_days,
        ),
    ]

    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    compare_df = pd.DataFrame(index=prices.index)
    compare_df["hs300_benchmark_nav"] = benchmark_nav.reindex(prices.index)

    for strategy_name, result, trades, trigger_days, blocked_days in variants:
        summary_rows.append(
            summarize(
                strategy_name,
                result,
                trades,
                selected,
                trigger_days=trigger_days,
                threshold_blocked_days=blocked_days,
            )
        )
        compare_df[f"{strategy_name}_nav"] = result["nav"]
        compare = pd.concat([result["nav"], benchmark_nav.reindex(result.index)], axis=1)
        compare.columns = ["strategy_nav", "benchmark_nav"]
        for row in build_yearly_return_rows(
            compare,
            strategy_col="strategy_nav",
            benchmark_col="benchmark_nav",
            benchmark_return_col="hs300_return",
        ):
            yearly_rows.append({"strategy": strategy_name, **row})

    summary = pd.DataFrame(summary_rows)
    baseline_row = summary.loc[summary["strategy"] == "baseline"].iloc[0]
    for col in ["total_return", "annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"]:
        summary[f"{col}_diff_vs_baseline"] = summary[col] - float(baseline_row[col])
    summary["trade_count_diff_vs_baseline"] = summary["trade_count"] - int(baseline_row["trade_count"])
    summary["trade_action_count_diff_vs_baseline"] = summary["trade_action_count"] - int(baseline_row["trade_action_count"])
    summary["soft_score"] = (
        (summary["annualized_return"] - float(baseline_row["annualized_return"])) * 100
        + (summary["sharpe_rf0"] - float(baseline_row["sharpe_rf0"])) * 10
        - (summary["max_drawdown_integral"] - float(baseline_row["max_drawdown_integral"]))
        - (summary["trade_count"] - int(baseline_row["trade_count"])) / 500
    )
    summary = summary.sort_values(["soft_score", "annualized_return", "sharpe_rf0"], ascending=[False, False, False])

    write_dataframe_csv_atomic(summary, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "nav_compare.csv")
    write_dataframe_csv_atomic(pd.DataFrame(yearly_rows), OUTPUT_DIR / "yearly_returns.csv", index=False)

    fig, ax = plt.subplots(figsize=(15, 8))
    for col in [c for c in compare_df.columns if c.endswith("_nav") and c != "hs300_benchmark_nav"]:
        ax.plot(compare_df.index, compare_df[col], label=col.removesuffix("_nav"), linewidth=1.4)
    ax.plot(compare_df.index, compare_df["hs300_benchmark_nav"], label="hs300_benchmark", linewidth=1.2, alpha=0.8, linestyle="--")
    ax.set_title("Momentum Cap Integration Variants", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "nav_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
