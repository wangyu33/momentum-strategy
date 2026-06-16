#!/usr/bin/env python3
"""对比 threshold dual momentum 的若干实用改法。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports, write_text_atomic
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports, write_text_atomic

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    RESEARCH_OUTPUT_DIR,
    RISK_CODES,
    DEFENSIVE_CODES,
    build_benchmark_nav,
    build_strategy_summary,
    ensure_output_dirs,
    fetch_histories,
    load_core_selected_and_prices,
    load_fixed_etf_pool,
    normalize_code,
    run_threshold_dual_strategy,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 threshold dual momentum 的若干实用改法。")
    parser.add_argument("--years", type=int, default=10, help="回测最近多少年。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取历史数据，而不是复用本地缓存。")
    return parser.parse_args()


def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_core_selected_and_prices()


def summarize(result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    exposure_series = result["exposure"] if "exposure" in result.columns else result["holding"].notna().astype(float)
    return build_strategy_summary(result, trades, selected=selected, exposure_series=exposure_series)


def build_trades(result: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    holding = result["holding"]
    exposure = result["exposure"].fillna(0.0)
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)
    trades = []
    for dt_idx in result.index:
        asset = normalize_code(holding.loc[dt_idx])
        prev_asset = normalize_code(prev_holding.loc[dt_idx])
        weight = float(exposure.loc[dt_idx]) if pd.notna(exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0
        if prev_asset == asset and abs(weight - prev_weight) < 1e-12:
            continue
        if prev_asset and (prev_asset != asset or prev_weight > weight):
            trades.append(
                {
                    "date": dt_idx,
                    "action": "SELL" if prev_asset != asset else "REDUCE",
                    "code": prev_asset,
                    "theme": code_to_theme.get(prev_asset, ""),
                    "name": code_to_name.get(prev_asset, ""),
                    "from_exposure": prev_weight,
                    "to_exposure": weight if prev_asset == asset else 0.0,
                    "nav": float(result.loc[dt_idx, "nav"]),
                }
            )
        if asset and (prev_asset != asset or weight > prev_weight):
            trades.append(
                {
                    "date": dt_idx,
                    "action": "BUY" if prev_asset != asset else "ADD",
                    "code": asset,
                    "theme": code_to_theme.get(asset, ""),
                    "name": code_to_name.get(asset, ""),
                    "from_exposure": prev_weight if prev_asset == asset else 0.0,
                    "to_exposure": weight,
                    "nav": float(result.loc[dt_idx, "nav"]),
                }
            )
    return pd.DataFrame(trades)


def run_signal_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    signal: pd.Series,
    target_exposure: pd.Series,
    signal_momentum: pd.Series,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change()
    holding = signal.shift(1)
    exposure = target_exposure.shift(1).fillna(0.0).rename("exposure")
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)
    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")
    per_side_cost = fee_rate + slippage_rate

    for dt_idx in prices.index:
        asset = holding.loc[dt_idx]
        prev_asset = prev_holding.loc[dt_idx]
        weight = float(exposure.loc[dt_idx]) if pd.notna(exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0
        gross_ret = 0.0
        if pd.notna(asset) and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = weight * float(returns.loc[dt_idx, asset])

        if pd.isna(asset) and pd.isna(prev_asset):
            day_turnover = abs(weight - prev_weight)
        elif pd.notna(asset) and pd.notna(prev_asset) and asset == prev_asset:
            day_turnover = abs(weight - prev_weight)
        else:
            day_turnover = prev_weight + weight

        turnover.loc[dt_idx] = day_turnover
        cost_rate = day_turnover * per_side_cost
        trade_cost_rate.loc[dt_idx] = cost_rate
        strategy_ret.loc[dt_idx] = (1 + gross_ret) * (1 - cost_rate) - 1

    nav = (1 + strategy_ret.fillna(0.0)).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1
    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": signal_momentum,
            "signal": signal,
            "holding": holding,
            "exposure": exposure,
            "target_exposure": target_exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )
    return result, build_trades(result, selected)


def build_base_components(prices: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    defensive_codes = [code for code in DEFENSIVE_CODES if code in prices.columns]
    risk_mom = momentum[risk_codes]
    defensive_mom = momentum[defensive_codes]
    risk_best = risk_mom.max(axis=1, skipna=True)
    risk_winner = risk_mom.idxmax(axis=1, skipna=True)
    return momentum, risk_mom, risk_best, risk_winner


def run_hysteresis_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    entry_threshold: float = 0.06,
    exit_threshold: float = 0.03,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum, _, risk_best, risk_winner = build_base_components(prices, lookback)
    defensive_codes = [code for code in DEFENSIVE_CODES if code in prices.columns]
    defensive_mom = momentum[defensive_codes]
    defensive_best = defensive_mom.max(axis=1, skipna=True)
    defensive_winner = defensive_mom.idxmax(axis=1, skipna=True)

    regime = "defensive"
    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    signal_mom = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        r_score = risk_best.loc[dt_idx]
        r_asset = risk_winner.loc[dt_idx]
        d_score = defensive_best.loc[dt_idx]
        d_asset = defensive_winner.loc[dt_idx]

        if pd.notna(r_score) and r_score >= entry_threshold:
            regime = "risk"
        elif pd.notna(r_score) and r_score <= exit_threshold:
            regime = "defensive"

        if regime == "risk" and pd.notna(r_asset):
            signal.loc[dt_idx] = r_asset
            exposure.loc[dt_idx] = 1.0
            signal_mom.loc[dt_idx] = float(r_score)
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            if pd.notna(r_score) and r_score > 0:
                exposure.loc[dt_idx] = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT
                signal_mom.loc[dt_idx] = float(r_score)
            else:
                exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
                signal_mom.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
        else:
            signal.loc[dt_idx] = pd.NA
            exposure.loc[dt_idx] = 0.0
            signal_mom.loc[dt_idx] = float("nan")

    return run_signal_strategy(prices, selected, signal, exposure, signal_mom, fee_rate, slippage_rate)


def run_defensive_multihorizon_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum, _, risk_best, risk_winner = build_base_components(prices, lookback)
    def25 = prices / prices.shift(25) - 1
    def60 = prices / prices.shift(60) - 1
    defensive_codes = [code for code in DEFENSIVE_CODES if code in prices.columns]
    def_score = 0.6 * def25[defensive_codes] + 0.4 * def60[defensive_codes]
    defensive_best = def_score.max(axis=1, skipna=True)
    defensive_winner = def_score.idxmax(axis=1, skipna=True)

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    signal_mom = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        r_score = risk_best.loc[dt_idx]
        r_asset = risk_winner.loc[dt_idx]
        d_score = defensive_best.loc[dt_idx]
        d_asset = defensive_winner.loc[dt_idx]

        if pd.notna(r_score) and r_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
            signal.loc[dt_idx] = r_asset
            exposure.loc[dt_idx] = 1.0
            signal_mom.loc[dt_idx] = float(r_score)
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT
            signal_mom.loc[dt_idx] = float(r_score)
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            signal_mom.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
        else:
            signal.loc[dt_idx] = pd.NA
            exposure.loc[dt_idx] = 0.0
            signal_mom.loc[dt_idx] = float("nan")

    return run_signal_strategy(prices, selected, signal, exposure, signal_mom, fee_rate, slippage_rate)


def run_lead_margin_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    lead_margin: float = 0.015,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    defensive_codes = [code for code in DEFENSIVE_CODES if code in prices.columns]
    risk_mom = momentum[risk_codes]
    defensive_mom = momentum[defensive_codes]
    risk_best = risk_mom.max(axis=1, skipna=True)
    risk_sorted = risk_mom.apply(lambda row: row.dropna().sort_values(ascending=False).tolist(), axis=1)
    risk_winner = risk_mom.idxmax(axis=1, skipna=True)
    defensive_best = defensive_mom.max(axis=1, skipna=True)
    defensive_winner = defensive_mom.idxmax(axis=1, skipna=True)

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    signal_mom = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    prev_signal = None
    prev_exposure = 0.0

    for dt_idx in prices.index:
        r_score = risk_best.loc[dt_idx]
        r_asset = risk_winner.loc[dt_idx]
        d_score = defensive_best.loc[dt_idx]
        d_asset = defensive_winner.loc[dt_idx]
        top_scores = risk_sorted.loc[dt_idx]
        score_gap = None
        if len(top_scores) >= 2:
            score_gap = top_scores[0] - top_scores[1]

        if pd.notna(r_score) and r_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
            candidate_signal = r_asset
            candidate_exposure = 1.0
            candidate_momentum = float(r_score)
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            candidate_signal = d_asset
            candidate_exposure = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT
            candidate_momentum = float(r_score)
        elif pd.notna(d_asset):
            candidate_signal = d_asset
            candidate_exposure = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            candidate_momentum = float(d_score) if pd.notna(d_score) else float("nan")
        else:
            candidate_signal = pd.NA
            candidate_exposure = 0.0
            candidate_momentum = float("nan")

        if (
            prev_signal is not None
            and normalize_code(candidate_signal) != normalize_code(prev_signal)
            and score_gap is not None
            and score_gap < lead_margin
        ):
            signal.loc[dt_idx] = prev_signal
            exposure.loc[dt_idx] = prev_exposure
            signal_mom.loc[dt_idx] = candidate_momentum
        else:
            signal.loc[dt_idx] = candidate_signal
            exposure.loc[dt_idx] = candidate_exposure
            signal_mom.loc[dt_idx] = candidate_momentum

        prev_signal = signal.loc[dt_idx]
        prev_exposure = float(exposure.loc[dt_idx])

    return run_signal_strategy(prices, selected, signal, exposure, signal_mom, fee_rate, slippage_rate)


def main() -> int:
    args = parse_args()
    if args.refresh:
        selected = load_fixed_etf_pool()
        prices = fetch_histories(selected, years=args.years)
    else:
        selected, prices = load_cached_data()

    base_result, base_trades = run_threshold_dual_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    hysteresis_result, hysteresis_trades = run_hysteresis_strategy(
        prices, selected, args.lookback, args.fee_rate, args.slippage_rate
    )
    defmh_result, defmh_trades = run_defensive_multihorizon_strategy(
        prices, selected, args.lookback, args.fee_rate, args.slippage_rate
    )
    margin_result, margin_trades = run_lead_margin_strategy(
        prices, selected, args.lookback, args.fee_rate, args.slippage_rate
    )

    rows = [
        {"strategy": "current_threshold_dual", **summarize(base_result, base_trades, selected)},
        {"strategy": "hysteresis_thresholds", **summarize(hysteresis_result, hysteresis_trades, selected)},
        {"strategy": "defensive_multihorizon", **summarize(defmh_result, defmh_trades, selected)},
        {"strategy": "lead_margin_filter", **summarize(margin_result, margin_trades, selected)},
    ]
    summary = pd.DataFrame(rows)
    base = summary.iloc[0]
    summary["return_diff"] = summary["total_return"] - base["total_return"]
    summary["annualized_diff"] = summary["annualized_return"] - base["annualized_return"]
    summary["sharpe_diff"] = summary["sharpe_rf0"] - base["sharpe_rf0"]
    summary["mdd_diff"] = summary["max_drawdown"] - base["max_drawdown"]
    summary["trade_diff"] = summary["trade_count"] - base["trade_count"]

    compare_df = pd.DataFrame(index=prices.index)
    compare_df["current_threshold_dual_nav"] = base_result["nav"]
    compare_df["hysteresis_thresholds_nav"] = hysteresis_result["nav"]
    compare_df["defensive_multihorizon_nav"] = defmh_result["nav"]
    compare_df["lead_margin_filter_nav"] = margin_result["nav"]
    compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")

    ensure_output_dirs()
    write_dataframe_csv_atomic(summary, RESEARCH_OUTPUT_DIR / "strategy_refinements_summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, RESEARCH_OUTPUT_DIR / "strategy_refinements_nav_compare.csv")
    write_dataframe_csv_atomic(hysteresis_result, RESEARCH_OUTPUT_DIR / "hysteresis_thresholds_nav.csv")
    write_dataframe_csv_atomic(hysteresis_trades, RESEARCH_OUTPUT_DIR / "hysteresis_thresholds_trades.csv", index=False)
    write_dataframe_csv_atomic(defmh_result, RESEARCH_OUTPUT_DIR / "defensive_multihorizon_nav.csv")
    write_dataframe_csv_atomic(defmh_trades, RESEARCH_OUTPUT_DIR / "defensive_multihorizon_trades.csv", index=False)
    write_dataframe_csv_atomic(margin_result, RESEARCH_OUTPUT_DIR / "lead_margin_filter_nav.csv")
    write_dataframe_csv_atomic(margin_trades, RESEARCH_OUTPUT_DIR / "lead_margin_filter_trades.csv", index=False)

    write_text_atomic(
        RESEARCH_OUTPUT_DIR / "strategy_refinements_rules.txt",
        (
            "hysteresis_thresholds: enter risk > 6%, exit risk < 3%\n"
            "defensive_multihorizon: defensive ranking = 0.6 * 25d + 0.4 * 60d\n"
            "lead_margin_filter: do not switch when risk winner leads runner-up by < 1.5%\n"
        ),
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["current_threshold_dual_nav"], linewidth=2.2, label="Current Threshold Dual")
    ax.plot(compare_df.index, compare_df["hysteresis_thresholds_nav"], linewidth=1.8, label="Hysteresis")
    ax.plot(compare_df.index, compare_df["defensive_multihorizon_nav"], linewidth=1.8, label="Defensive MultiHorizon")
    ax.plot(compare_df.index, compare_df["lead_margin_filter_nav"], linewidth=1.8, label="Lead Margin")
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.6, linestyle="--", label="HS300")
    ax.set_title("Strategy Refinements Comparison", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, RESEARCH_OUTPUT_DIR / "strategy_refinements_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
