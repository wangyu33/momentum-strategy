#!/usr/bin/env python3
"""对比基于信号标的自身历史动量分位的两类减仓方案。"""

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
    ensure_output_dirs,
    load_core_selected_and_prices,
    run_default_strategy_with_params,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "momentum_percentile_caps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比基于信号标的自身历史动量分位的两类减仓方案。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--cap-start", type=float, default=0.90, help="开始减仓的自身历史动量分位。")
    parser.add_argument("--cap-end", type=float, default=0.95, help="减仓打满的自身历史动量分位。")
    parser.add_argument("--cap-floor", type=float, default=0.70, help="达到上限后保留的最大总仓位。")
    return parser.parse_args()


def load_cached_market_proxy() -> pd.DataFrame:
    proxy = pd.read_csv(MARKET_VOLUME_CACHE_PATH, parse_dates=["date"])
    return proxy.set_index("date")


def summarize(name: str, result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    latest = result.iloc[-1]
    return {
        "strategy": name,
        **build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True),
        "latest_signal": latest.get("signal"),
        "latest_holding": latest.get("holding"),
        "latest_exposure": latest.get("exposure"),
        "latest_selected_momentum_pct": latest.get("selected_momentum_percentile"),
        "momentum_pct_cap_trigger_days": int(result.get("selected_momentum_pct_cap_triggered", pd.Series(False, index=result.index)).fillna(False).sum()),
    }


def build_baseline_target_weights(result: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in result.columns if col.startswith("target_weight_")]
    weights = result[cols].copy()
    weights.columns = [col.removeprefix("target_weight_") for col in cols]
    return weights.fillna(0.0)


def apply_risk_only_momentum_pct_cap(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    baseline_result: pd.DataFrame,
    *,
    cap_start: float,
    cap_end: float,
    cap_floor: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cap_end < cap_start:
        raise ValueError("cap_end must be >= cap_start")

    target_weights = build_baseline_target_weights(baseline_result)
    signal_momentum = baseline_result["current_momentum"].copy().rename("current_momentum")
    selected_pct = pd.Series(index=prices.index, dtype="float64", name="selected_momentum_percentile")
    pct_cap_triggered = pd.Series(False, index=prices.index, dtype=bool, name="selected_momentum_pct_cap_triggered")
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    signal_pct = build_asset_own_momentum_percentile(
        prices,
        lookback=25,
        state_lookback=756,
        min_periods=120,
    )

    for dt_idx in prices.index:
        signal_code = baseline_result.loc[dt_idx, "signal"]
        if pd.isna(signal_code):
            continue
        signal_code = str(int(signal_code)) if isinstance(signal_code, float) and signal_code.is_integer() else str(signal_code)
        if signal_code not in signal_pct.columns:
            continue
        pct_value = signal_pct.loc[dt_idx, signal_code]
        if pd.isna(pct_value):
            continue
        selected_pct.loc[dt_idx] = float(pct_value)
        if signal_code not in risk_codes or float(pct_value) < cap_start:
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

    result, trades = run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        signal_momentum,
        fee_rate,
        slippage_rate,
    )
    result = apply_official_baseline_nav_anchor(result)
    result["selected_momentum_percentile"] = selected_pct.reindex(result.index)
    result["selected_momentum_pct_cap_triggered"] = pct_cap_triggered.reindex(result.index).fillna(False)
    result = pd.concat([result, target_weights.add_prefix("target_weight_")], axis=1)
    return result, trades


def build_variants(cap_start: float, cap_end: float, cap_floor: float) -> list[tuple[str, dict[str, object]]]:
    baseline = build_default_strategy_params()

    all_signal_cap = build_default_strategy_params()
    all_signal_cap.update(
        {
            "signal_selected_momentum_pct_cap_start": cap_start,
            "signal_selected_momentum_pct_cap_end": cap_end,
            "signal_selected_momentum_pct_cap_floor": cap_floor,
            "signal_selected_momentum_pct_lookback": 756,
            "signal_selected_momentum_pct_min_periods": 120,
        }
    )

    return [
        ("baseline", baseline),
        (f"all_signal_momentum_pct_cap_{int(cap_start*100)}_{int(cap_end*100)}_floor{int(cap_floor*100)}", all_signal_cap),
    ]


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected, prices = load_core_selected_and_prices()
    market_proxy = load_cached_market_proxy()
    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")

    variants = build_variants(args.cap_start, args.cap_end, args.cap_floor)
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    compare_df = pd.DataFrame(index=prices.index)
    compare_df["hs300_benchmark_nav"] = benchmark_nav.reindex(prices.index)
    baseline_result_for_overlay: pd.DataFrame | None = None

    for strategy_name, params in variants:
        result, trades = run_default_strategy_with_params(
            prices,
            selected,
            params=params,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            market_proxy=market_proxy,
        )
        result = apply_official_baseline_nav_anchor(result)
        if strategy_name == "baseline":
            baseline_result_for_overlay = result.copy()
        summary_rows.append(summarize(strategy_name, result, trades, selected))
        compare_df[f"{strategy_name}_nav"] = result["nav"]

        compare = pd.concat([result["nav"], benchmark_nav.reindex(result.index)], axis=1)
        compare.columns = ["strategy_nav", "benchmark_nav"]
        for row in build_yearly_return_rows(compare, strategy_col="strategy_nav", benchmark_col="benchmark_nav", benchmark_return_col="hs300_return"):
            yearly_rows.append({"strategy": strategy_name, **row})

        write_dataframe_csv_atomic(trades, OUTPUT_DIR / f"{strategy_name}_trades.csv", index=False)
        write_dataframe_csv_atomic(result.reset_index(), OUTPUT_DIR / f"{strategy_name}_nav_detail.csv", index=False)

    if baseline_result_for_overlay is None:
        raise RuntimeError("baseline result missing")

    risk_only_name = f"risk_only_momentum_pct_cap_{int(args.cap_start*100)}_{int(args.cap_end*100)}_floor{int(args.cap_floor*100)}"
    risk_only_result, risk_only_trades = apply_risk_only_momentum_pct_cap(
        prices,
        selected,
        baseline_result_for_overlay,
        cap_start=args.cap_start,
        cap_end=args.cap_end,
        cap_floor=args.cap_floor,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    summary_rows.append(summarize(risk_only_name, risk_only_result, risk_only_trades, selected))
    compare_df[f"{risk_only_name}_nav"] = risk_only_result["nav"]
    compare = pd.concat([risk_only_result["nav"], benchmark_nav.reindex(risk_only_result.index)], axis=1)
    compare.columns = ["strategy_nav", "benchmark_nav"]
    for row in build_yearly_return_rows(compare, strategy_col="strategy_nav", benchmark_col="benchmark_nav", benchmark_return_col="hs300_return"):
        yearly_rows.append({"strategy": risk_only_name, **row})
    write_dataframe_csv_atomic(risk_only_trades, OUTPUT_DIR / f"{risk_only_name}_trades.csv", index=False)
    write_dataframe_csv_atomic(risk_only_result.reset_index(), OUTPUT_DIR / f"{risk_only_name}_nav_detail.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    baseline_row = summary.loc[summary["strategy"] == "baseline"].iloc[0]
    for col in ["total_return", "annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"]:
        summary[f"{col}_diff_vs_baseline"] = summary[col] - float(baseline_row[col])
    summary["trade_count_diff_vs_baseline"] = summary["trade_count"] - int(baseline_row["trade_count"])
    summary["trade_action_count_diff_vs_baseline"] = summary["trade_action_count"] - int(baseline_row["trade_action_count"])

    write_dataframe_csv_atomic(summary, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "nav_compare.csv")
    write_dataframe_csv_atomic(pd.DataFrame(yearly_rows), OUTPUT_DIR / "yearly_returns.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    for col in [c for c in compare_df.columns if c.endswith("_nav") and c != "hs300_benchmark_nav"]:
        ax.plot(compare_df.index, compare_df[col], label=col.replace("_nav", ""), linewidth=1.6)
    ax.plot(compare_df.index, compare_df["hs300_benchmark_nav"], label="hs300_benchmark", linewidth=1.3, alpha=0.8, linestyle="--")
    ax.set_title("Momentum Percentile Cap Variants", loc="left", fontsize=16, fontweight="bold")
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
