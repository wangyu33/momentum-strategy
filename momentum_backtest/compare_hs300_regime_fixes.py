#!/usr/bin/env python3
"""对比修复沪深300主导牛市漏判问题的几种方案。"""

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

from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    DEFENSIVE_CODES,
    RESEARCH_OUTPUT_DIR,
    build_benchmark_nav,
    build_strategy_summary,
    build_yearly_return_rows,
    build_threshold_dual_signal,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    load_core_selected_and_prices,
    run_threshold_dual_strategy,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


HS300_CODE = "510300"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比修复沪深300主导牛市漏判问题的几种方案。")
    parser.add_argument("--years", type=int, default=10, help="回测最近多少年。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--absolute-threshold", type=float, default=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD, help="风险资产绝对动量阈值。")
    parser.add_argument("--weak-trend-defensive-weight", type=float, default=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT, help="风险趋势偏弱但仍为正时的防守仓位。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取历史数据，而不是复用本地缓存。")
    return parser.parse_args()


def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_core_selected_and_prices()


def summarize(result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    return build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)


def build_trades_from_weights(result: pd.DataFrame, target_weights: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    weights = target_weights.shift(1).fillna(0.0)
    prev_weights = weights.shift(1).fillna(0.0)
    trades: list[dict[str, object]] = []

    for dt_idx in weights.index:
        current = weights.loc[dt_idx]
        prev = prev_weights.loc[dt_idx]
        for code in weights.columns:
            curr_w = float(current[code])
            prev_w = float(prev[code])
            if abs(curr_w - prev_w) < 1e-12:
                continue
            if curr_w > prev_w:
                trades.append(
                    {
                        "date": dt_idx,
                        "action": "BUY" if prev_w == 0 else "ADD",
                        "code": code,
                        "theme": code_to_theme.get(code, ""),
                        "name": code_to_name.get(code, ""),
                        "from_exposure": prev_w,
                        "to_exposure": curr_w,
                        "nav": float(result.loc[dt_idx, "nav"]),
                    }
                )
            else:
                trades.append(
                    {
                        "date": dt_idx,
                        "action": "SELL" if curr_w == 0 else "REDUCE",
                        "code": code,
                        "theme": code_to_theme.get(code, ""),
                        "name": code_to_name.get(code, ""),
                        "from_exposure": prev_w,
                        "to_exposure": curr_w,
                        "nav": float(result.loc[dt_idx, "nav"]),
                    }
                )

    return pd.DataFrame(trades)


def run_target_weights_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    target_weights: pd.DataFrame,
    signal_momentum: pd.Series,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change().fillna(0.0)
    target_weights = target_weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0)
    weights = target_weights.shift(1).fillna(0.0)
    prev_weights = weights.shift(1).fillna(0.0)
    gross_ret = (weights * returns).sum(axis=1)
    turnover = (weights - prev_weights).abs().sum(axis=1)
    trade_cost_rate = turnover * (fee_rate + slippage_rate)
    strategy_ret = (1 + gross_ret) * (1 - trade_cost_rate) - 1
    nav = (1 + strategy_ret.fillna(0.0)).cumprod()
    nav.iloc[0] = 1.0
    exposure = weights.sum(axis=1).rename("exposure")
    primary_weight = weights.max(axis=1)
    holding = weights.idxmax(axis=1).where(primary_weight > 0, pd.NA).rename("holding")
    signal = target_weights.idxmax(axis=1).where(target_weights.max(axis=1) > 0, pd.NA).rename("signal")
    drawdown = nav / nav.cummax() - 1

    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": signal_momentum,
            "signal": signal,
            "holding": holding,
            "exposure": exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )
    result = pd.concat([result, weights.add_prefix("weight_")], axis=1)
    trades = build_trades_from_weights(result, target_weights, selected)
    return result, trades


def build_hs300_trend_filter(prices: pd.DataFrame) -> pd.Series:
    close = prices[HS300_CODE]
    mom60 = close / close.shift(60) - 1
    mom120 = close / close.shift(120) - 1
    ma120 = close.rolling(120).mean()
    return ((mom60 > 0.05) & (mom120 > 0.10) & (close > ma120)).rename("hs300_trend_filter")


def run_hs300_guard_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
    )
    trend_filter = build_hs300_trend_filter(prices)
    hs300_mom60 = prices[HS300_CODE] / prices[HS300_CODE].shift(60) - 1

    guard_mask = trend_filter & (signal.isna() | signal.isin(DEFENSIVE_CODES))
    adjusted_signal = signal.copy()
    adjusted_target_exposure = target_exposure.copy()
    adjusted_momentum = current_momentum.copy()

    adjusted_signal.loc[guard_mask] = HS300_CODE
    adjusted_target_exposure.loc[guard_mask] = 1.0
    adjusted_momentum.loc[guard_mask] = hs300_mom60.loc[guard_mask]

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for code in prices.columns:
        code_mask = adjusted_signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] = adjusted_target_exposure.loc[code_mask]

    return run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        adjusted_momentum,
        fee_rate,
        slippage_rate,
    )


def run_dynamic_core_satellite_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    core_weight: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
    )
    trend_filter = build_hs300_trend_filter(prices)
    hs300_mom60 = prices[HS300_CODE] / prices[HS300_CODE].shift(60) - 1
    dynamic_core_weight = trend_filter.astype(float) * core_weight
    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_weights[HS300_CODE] += dynamic_core_weight
    satellite_scale = 1.0 - dynamic_core_weight

    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_scale.loc[code_mask] * target_exposure.loc[code_mask]

    blended_momentum = (satellite_scale * current_momentum + dynamic_core_weight * hs300_mom60).rename("current_momentum")
    return run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )


def run_hs300_partial_override_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    override_weight: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
    )
    trend_filter = build_hs300_trend_filter(prices)
    hs300_mom60 = prices[HS300_CODE] / prices[HS300_CODE].shift(60) - 1
    defensive_mask = signal.isin(DEFENSIVE_CODES)
    partial_mask = trend_filter & defensive_mask

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] = target_exposure.loc[code_mask]

    override_weight = min(max(override_weight, 0.0), 1.0)
    if partial_mask.any():
        for dt_idx in prices.index[partial_mask]:
            defensive_code = signal.loc[dt_idx]
            current_override = min(override_weight, float(target_weights.loc[dt_idx, defensive_code]))
            target_weights.loc[dt_idx, HS300_CODE] += current_override
            target_weights.loc[dt_idx, defensive_code] -= current_override

    adjusted_momentum = current_momentum.copy()
    adjusted_momentum.loc[partial_mask] = (
        (1.0 - override_weight) * current_momentum.loc[partial_mask]
        + override_weight * hs300_mom60.loc[partial_mask]
    )
    return run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        adjusted_momentum.rename("current_momentum"),
        fee_rate,
        slippage_rate,
    )


def run_core_satellite_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    core_weight: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
    )

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_weights[HS300_CODE] = core_weight
    satellite_weight = 1.0 - core_weight

    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_weight * target_exposure.loc[code_mask]

    blended_momentum = satellite_weight * current_momentum + core_weight * (prices[HS300_CODE] / prices[HS300_CODE].shift(60) - 1)
    return run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        blended_momentum.rename("current_momentum"),
        fee_rate,
        slippage_rate,
    )


def calculate_yearly_returns(result: pd.DataFrame, benchmark_nav: pd.Series) -> pd.DataFrame:
    compare = pd.concat([result["nav"], benchmark_nav], axis=1)
    compare.columns = ["strategy_nav", "benchmark_nav"]
    rows = build_yearly_return_rows(
        compare,
        strategy_col="strategy_nav",
        benchmark_col="benchmark_nav",
        benchmark_return_col="hs300_return",
    )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.refresh:
        selected = load_fixed_etf_pool()
        prices = fetch_histories(selected, years=args.years)
    else:
        selected, prices = load_cached_data()

    benchmark_nav = build_benchmark_nav(prices, benchmark_code=HS300_CODE)
    baseline_result, baseline_trades = run_threshold_dual_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        absolute_threshold=args.absolute_threshold,
        weak_trend_defensive_weight=args.weak_trend_defensive_weight,
    )
    guard_result, guard_trades = run_hs300_guard_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        absolute_threshold=args.absolute_threshold,
        weak_trend_defensive_weight=args.weak_trend_defensive_weight,
    )
    core_result, core_trades = run_core_satellite_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        absolute_threshold=args.absolute_threshold,
        weak_trend_defensive_weight=args.weak_trend_defensive_weight,
    )
    adaptive_core_result, adaptive_core_trades = run_dynamic_core_satellite_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        absolute_threshold=args.absolute_threshold,
        weak_trend_defensive_weight=args.weak_trend_defensive_weight,
    )
    partial_override_results = []
    for override_weight in [0.30, 0.40, 0.50]:
        result, trades = run_hs300_partial_override_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            absolute_threshold=args.absolute_threshold,
            weak_trend_defensive_weight=args.weak_trend_defensive_weight,
            override_weight=override_weight,
        )
        partial_override_results.append((override_weight, result, trades))

    rows = [
        {"strategy": "baseline_threshold_dual", **summarize(baseline_result, baseline_trades, selected)},
        {"strategy": "hs300_trend_guard", **summarize(guard_result, guard_trades, selected)},
        {"strategy": "core_satellite_70_30", **summarize(core_result, core_trades, selected)},
        {"strategy": "adaptive_core_satellite_80_20", **summarize(adaptive_core_result, adaptive_core_trades, selected)},
    ]
    for override_weight, result, trades in partial_override_results:
        rows.append(
            {
                "strategy": f"hs300_partial_override_{int(round(override_weight * 100)):02d}",
                **summarize(result, trades, selected),
            }
        )
    summary = pd.DataFrame(rows)
    baseline_row = summary[summary["strategy"] == "baseline_threshold_dual"].iloc[0]
    summary["return_diff"] = summary["total_return"] - float(baseline_row["total_return"])
    summary["annualized_diff"] = summary["annualized_return"] - float(baseline_row["annualized_return"])
    summary["sharpe_diff"] = summary["sharpe_rf0"] - float(baseline_row["sharpe_rf0"])
    summary["mdd_diff"] = summary["max_drawdown"] - float(baseline_row["max_drawdown"])
    summary["trade_diff"] = summary["trade_count"] - int(baseline_row["trade_count"])

    compare_df = pd.DataFrame(index=prices.index)
    compare_df["baseline_threshold_dual_nav"] = baseline_result["nav"]
    compare_df["hs300_trend_guard_nav"] = guard_result["nav"]
    compare_df["core_satellite_70_30_nav"] = core_result["nav"]
    compare_df["adaptive_core_satellite_80_20_nav"] = adaptive_core_result["nav"]
    for override_weight, result, _ in partial_override_results:
        compare_df[f"hs300_partial_override_{int(round(override_weight * 100)):02d}_nav"] = result["nav"]
    compare_df["hs300_benchmark"] = benchmark_nav

    yearly_frames = []
    for name, result in [
        ("baseline_threshold_dual", baseline_result),
        ("hs300_trend_guard", guard_result),
        ("core_satellite_70_30", core_result),
        ("adaptive_core_satellite_80_20", adaptive_core_result),
    ]:
        yearly = calculate_yearly_returns(result, benchmark_nav)
        yearly.insert(0, "strategy", name)
        yearly_frames.append(yearly)
    for override_weight, result, _ in partial_override_results:
        name = f"hs300_partial_override_{int(round(override_weight * 100)):02d}"
        yearly = calculate_yearly_returns(result, benchmark_nav)
        yearly.insert(0, "strategy", name)
        yearly_frames.append(yearly)
    yearly_summary = pd.concat(yearly_frames, ignore_index=True)

    ensure_output_dirs()
    write_dataframe_csv_atomic(summary, RESEARCH_OUTPUT_DIR / "hs300_regime_fix_summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, RESEARCH_OUTPUT_DIR / "hs300_regime_fix_nav_compare.csv")
    write_dataframe_csv_atomic(yearly_summary, RESEARCH_OUTPUT_DIR / "hs300_regime_fix_yearly_returns.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["baseline_threshold_dual_nav"], linewidth=2.2, label="Baseline")
    ax.plot(compare_df.index, compare_df["adaptive_core_satellite_80_20_nav"], linewidth=1.8, label="Adaptive Core 80/20")
    ax.plot(compare_df.index, compare_df["hs300_partial_override_30_nav"], linewidth=1.8, label="Partial Override 30%")
    ax.plot(compare_df.index, compare_df["hs300_partial_override_40_nav"], linewidth=1.8, label="Partial Override 40%")
    ax.plot(compare_df.index, compare_df["hs300_partial_override_50_nav"], linewidth=1.8, label="Partial Override 50%")
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.5, linestyle="--", label="HS300")
    ax.set_title("HS300 Regime Fix Comparison", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, RESEARCH_OUTPUT_DIR / "hs300_regime_fix_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    print(yearly_summary[yearly_summary["year"].isin([2017, 2018])].to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
