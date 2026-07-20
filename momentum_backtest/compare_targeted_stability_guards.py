#!/usr/bin/env python3
"""对比稳定性作为次级排序和过热阶段额外风控的 targeted 方案。"""

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

from goal_optimization_common import MARKET_VOLUME_CACHE_PATH
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


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "targeted_stability_guards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比稳定性作为次级排序和过热阶段额外风控的 targeted 方案。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def summarize(name: str, result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    return {"strategy": name, **build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)}


def load_cached_market_proxy() -> pd.DataFrame:
    proxy = pd.read_csv(MARKET_VOLUME_CACHE_PATH, parse_dates=["date"])
    return proxy.set_index("date")


def build_variants() -> list[tuple[str, dict[str, object]]]:
    baseline = build_default_strategy_params()

    tie_r2_003 = build_default_strategy_params()
    tie_r2_003.update({"signal_secondary_stability_method": "r2", "signal_secondary_stability_gap": 0.003})

    tie_r2_005 = build_default_strategy_params()
    tie_r2_005.update({"signal_secondary_stability_method": "r2", "signal_secondary_stability_gap": 0.005})

    tie_downvol_003 = build_default_strategy_params()
    tie_downvol_003.update({"signal_secondary_stability_method": "downside_vol", "signal_secondary_stability_gap": 0.003})

    tie_downvol_005 = build_default_strategy_params()
    tie_downvol_005.update({"signal_secondary_stability_method": "downside_vol", "signal_secondary_stability_gap": 0.005})

    overheat_r2_020 = build_default_strategy_params()
    overheat_r2_020.update(
        {
            "overheat_stability_method": "r2",
            "overheat_stability_min_rank": 0.35,
            "overheat_stability_cap": 0.20,
        }
    )

    overheat_downvol_020 = build_default_strategy_params()
    overheat_downvol_020.update(
        {
            "overheat_stability_method": "downside_vol",
            "overheat_stability_min_rank": 0.35,
            "overheat_stability_cap": 0.20,
        }
    )

    combo_tie_r2_overheat_downvol = build_default_strategy_params()
    combo_tie_r2_overheat_downvol.update(
        {
            "signal_secondary_stability_method": "r2",
            "signal_secondary_stability_gap": 0.005,
            "overheat_stability_method": "downside_vol",
            "overheat_stability_min_rank": 0.35,
            "overheat_stability_cap": 0.20,
        }
    )

    combo_tie_downvol_overheat_r2 = build_default_strategy_params()
    combo_tie_downvol_overheat_r2.update(
        {
            "signal_secondary_stability_method": "downside_vol",
            "signal_secondary_stability_gap": 0.005,
            "overheat_stability_method": "r2",
            "overheat_stability_min_rank": 0.35,
            "overheat_stability_cap": 0.20,
        }
    )

    return [
        ("baseline", baseline),
        ("tie_r2_gap003", tie_r2_003),
        ("tie_r2_gap005", tie_r2_005),
        ("tie_downvol_gap003", tie_downvol_003),
        ("tie_downvol_gap005", tie_downvol_005),
        ("overheat_r2_rank35_cap20", overheat_r2_020),
        ("overheat_downvol_rank35_cap20", overheat_downvol_020),
        ("combo_tie_r2_overheat_downvol", combo_tie_r2_overheat_downvol),
        ("combo_tie_downvol_overheat_r2", combo_tie_downvol_overheat_r2),
    ]


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected, prices = load_core_selected_and_prices()
    market_proxy = load_cached_market_proxy()
    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")

    variants = build_variants()
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    compare_df = pd.DataFrame(index=prices.index)
    compare_df["hs300_benchmark_nav"] = benchmark_nav.reindex(prices.index)

    for strategy_name, params in variants:
        result, trades = run_default_strategy_with_params(
            prices,
            selected,
            params=params,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            market_proxy=market_proxy,
        )
        if strategy_name == "baseline":
            result = apply_official_baseline_nav_anchor(result)
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
    baseline_row = summary.loc[summary["strategy"] == "baseline"].iloc[0]
    for col in ["total_return", "annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"]:
        summary[f"{col}_diff_vs_baseline"] = summary[col] - float(baseline_row[col])
    summary["trade_count_diff_vs_baseline"] = summary["trade_count"] - int(baseline_row["trade_count"])
    summary["trade_action_count_diff_vs_baseline"] = summary["trade_action_count"] - int(baseline_row["trade_action_count"])
    summary["is_soft_improvement"] = (
        (summary["annualized_return"] >= float(baseline_row["annualized_return"]))
        & (summary["max_drawdown_integral"] <= float(baseline_row["max_drawdown_integral"]))
    )
    summary = summary.sort_values(
        ["is_soft_improvement", "annualized_return", "max_drawdown_integral", "max_drawdown"],
        ascending=[False, False, True, True],
    )

    write_dataframe_csv_atomic(summary, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "nav_compare.csv")
    write_dataframe_csv_atomic(pd.DataFrame(yearly_rows), OUTPUT_DIR / "yearly_returns.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    for col in [c for c in compare_df.columns if c.endswith("_nav") and c != "hs300_benchmark_nav"]:
        ax.plot(compare_df.index, compare_df[col], label=col.replace("_nav", ""), linewidth=1.6)
    ax.plot(compare_df.index, compare_df["hs300_benchmark_nav"], label="hs300_benchmark", linewidth=1.3, alpha=0.8, linestyle="--")
    ax.set_title("Targeted Stability Guards", loc="left", fontsize=16, fontweight="bold")
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
