#!/usr/bin/env python3
"""对比当前正式基线与下一轮低副作用优化候选。"""

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
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
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


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "next_optimization_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比当前正式基线与下一轮低副作用优化候选。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def summarize(name: str, result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    return {"strategy": name, **build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)}


def build_variants() -> list[tuple[str, dict[str, object]]]:
    baseline = build_default_strategy_params()
    vg60 = build_default_strategy_params()
    vg60.update({"volume_guard_cap": 0.60})

    vg50 = build_default_strategy_params()
    vg50.update({"volume_guard_cap": 0.50})

    top2 = build_default_strategy_params()
    top2.update({"close_top2_gap": 0.005, "close_top2_risk_cap": 0.70})

    vg50_top2 = build_default_strategy_params()
    vg50_top2.update({"volume_guard_cap": 0.50, "close_top2_gap": 0.005, "close_top2_risk_cap": 0.70})

    leader = build_default_strategy_params()
    leader.update({"signal_leader_margin": 0.005})

    combo = build_default_strategy_params()
    combo.update(
        {
            "volume_guard_cap": 0.60,
            "close_top2_gap": 0.005,
            "close_top2_risk_cap": 0.70,
            "signal_leader_margin": 0.005,
        }
    )

    return [
        ("baseline", baseline),
        ("vg60", vg60),
        ("vg50", vg50),
        ("top2_gap05_cap70", top2),
        ("vg50_top2_gap05_cap70", vg50_top2),
        ("leader_margin05", leader),
        ("combo_vg60_top2_leader", combo),
    ]


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected, prices = load_core_selected_and_prices()
    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")

    variants = build_variants()
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    compare_df = pd.DataFrame(index=prices.index)
    compare_df["hs300_benchmark_nav"] = benchmark_nav.reindex(prices.index)

    baseline_result: pd.DataFrame | None = None
    baseline_trades: pd.DataFrame | None = None

    for strategy_name, params in variants:
        result, trades = run_default_strategy_with_params(
            prices,
            selected,
            params=params,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        if strategy_name == "baseline":
            baseline_result = result
            baseline_trades = trades

        summary_rows.append(summarize(strategy_name, result, trades, selected))
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

        write_dataframe_csv_atomic(trades, OUTPUT_DIR / f"{strategy_name}_trades.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    if baseline_result is None or baseline_trades is None:
        raise RuntimeError("missing baseline result")

    baseline_row = summary[summary["strategy"] == "baseline"].iloc[0]
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
    ax.set_title("Next Optimization Candidates", loc="left", fontsize=16, fontweight="bold")
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
