#!/usr/bin/env python3
"""测试将中概互联加入风险池后的保护规则。"""

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
    from .candidate_pool_common import BASE_DEFENSIVE_CODES, BASE_RISK_CODES, ETF_513050
    from .china_internet_guard_common import (
        CHINA_INTERNET_CODE,
        run_china_internet_guard_variant,
        summarize_china_internet_guard_result,
    )
    from .official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from candidate_pool_common import BASE_DEFENSIVE_CODES, BASE_RISK_CODES, ETF_513050
    from china_internet_guard_common import (
        CHINA_INTERNET_CODE,
        run_china_internet_guard_variant,
        summarize_china_internet_guard_result,
    )
    from official_baseline import apply_official_baseline_nav_anchor

from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    build_benchmark_nav,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


# Backward-compatible exports for monitor/validation callers that still import from this module.
run_variant = run_china_internet_guard_variant
summarize = summarize_china_internet_guard_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试将中概互联加入风险池后的保护规则。")
    parser.add_argument("--years", type=int, default=6, help="向前抓取多少年历史数据，再对齐公共区间。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    selected = pd.concat([base_pool, pd.DataFrame([ETF_513050])], ignore_index=True)
    prices = fetch_histories(selected, years=args.years)
    prices = prices.dropna(how="any")

    variants = [
        ("base_pool", None, None, None, False),
        ("china_internet_risk", None, None, None, True),
        ("ci_cap_70", None, None, 0.70, True),
        ("ci_cap_60", None, None, 0.60, True),
        ("ci_abs_08", 0.08, None, None, True),
        ("ci_abs_10", 0.10, None, None, True),
        ("ci_margin_03", None, 0.03, None, True),
        ("ci_margin_05", None, 0.05, None, True),
        ("ci_margin_03_cap_70", None, 0.03, 0.70, True),
    ]

    rows: list[dict[str, object]] = []
    compare_df = None
    base_metrics: dict[str, float | int] = {}
    base_prices = prices[base_pool["code"].tolist()].copy()

    for name, abs_thr, margin_thr, max_exposure, include_ci in variants:
        current_selected = selected if include_ci else base_pool.copy()
        current_prices = prices if include_ci else base_prices
        result, trades = run_china_internet_guard_variant(
            current_prices,
            current_selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            china_abs_threshold=abs_thr if include_ci else None,
            china_margin_threshold=margin_thr if include_ci else None,
            china_max_exposure=max_exposure if include_ci else None,
        )
        if name == "base_pool":
            result = apply_official_baseline_nav_anchor(result)
        if compare_df is None:
            compare_df = pd.DataFrame(index=result.index)
            compare_df["hs300_benchmark"] = build_benchmark_nav(current_prices, benchmark_code="510300")
        compare_df[name] = result["nav"].reindex(compare_df.index)
        row = {"variant": name, **summarize_china_internet_guard_result(result, trades)}
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
    write_dataframe_csv_atomic(summary, RESEARCH_OUTPUT_DIR / "china_internet_guard_summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, RESEARCH_OUTPUT_DIR / "china_internet_guard_nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    for column in ["base_pool", "china_internet_risk", "ci_cap_70", "ci_abs_08", "ci_margin_03", "ci_margin_03_cap_70"]:
        ax.plot(compare_df.index, compare_df[column], linewidth=1.8 if column != "base_pool" else 2.3, label=column)
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.5, linestyle="--", label="HS300 ETF")
    ax.set_title("China Internet Guard Variants", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, RESEARCH_OUTPUT_DIR / "china_internet_guard_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
