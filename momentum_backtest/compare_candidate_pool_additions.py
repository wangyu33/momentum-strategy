#!/usr/bin/env python3
"""对比若干实用 ETF 候选标的加入当前池子的效果。"""

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

from candidate_pool_common import (
    BASE_DEFENSIVE_CODES,
    BASE_RISK_CODES,
    ETF_164824,
    ETF_510050,
    ETF_510230,
    ETF_510500,
    ETF_510880,
    ETF_510900,
    ETF_511090,
    ETF_511260,
    ETF_511380,
    ETF_512100,
    ETF_512480,
    ETF_513030,
    ETF_513050,
    ETF_513080,
    ETF_513180,
    ETF_513400,
    ETF_515790,
    ETF_588000,
    run_custom_threshold_dual_with_overheat,
    summarize,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比若干实用 ETF 候选标的加入当前池子的效果。")
    parser.add_argument("--years", type=int, default=15, help="向前抓取多少年历史数据，再对齐公共区间。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--start-date", type=str, default="2012-01-01", help="候选池对比的分析起始日期。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    base_without_lowvol = base_pool[base_pool["code"] != "512890"].copy()
    candidates = [
        {
            "name": "base_pool",
            "selected": base_pool.copy(),
            "risk_codes": BASE_RISK_CODES,
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_kc50",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_588000])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["588000"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_hstech",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_513180])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["513180"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_10y_bond",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_511260])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES,
            "defensive_codes": BASE_DEFENSIVE_CODES + ["511260"],
        },
        {
            "name": "plus_germany",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_513030])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["513030"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_china_internet",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_513050])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["513050"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_30y_bond",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_511090])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES,
            "defensive_codes": BASE_DEFENSIVE_CODES + ["511090"],
        },
        {
            "name": "plus_semiconductor",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_512480])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["512480"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_solar",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_515790])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["515790"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_convertible_bond",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_511380])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES,
            "defensive_codes": BASE_DEFENSIVE_CODES + ["511380"],
        },
        {
            "name": "plus_kc50_convertible_bond",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_588000, ETF_511380])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["588000"],
            "defensive_codes": BASE_DEFENSIVE_CODES + ["511380"],
        },
        {
            "name": "plus_csi500",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_510500])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["510500"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_csi500_convertible_bond",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_510500, ETF_511380])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["510500"],
            "defensive_codes": BASE_DEFENSIVE_CODES + ["511380"],
        },
        {
            "name": "plus_csi1000",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_512100])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["512100"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_csi1000_convertible_bond",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_512100, ETF_511380])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["512100"],
            "defensive_codes": BASE_DEFENSIVE_CODES + ["511380"],
        },
        {
            "name": "plus_hshares",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_510900])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["510900"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_sse50",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_510050])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["510050"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_financial",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_510230])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["510230"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_india_lof",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_164824])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["164824"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "plus_dow",
            "selected": pd.concat([base_pool, pd.DataFrame([ETF_513400])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES + ["513400"],
            "defensive_codes": BASE_DEFENSIVE_CODES,
        },
        {
            "name": "replace_div_lowvol_with_dividend",
            "selected": pd.concat([base_without_lowvol, pd.DataFrame([ETF_510880])], ignore_index=True),
            "risk_codes": BASE_RISK_CODES,
            "defensive_codes": ["511580", "518880", "510880"],
        },
    ]

    rows: list[dict[str, object]] = []
    compare_df = None
    base_total_return = None
    base_annualized = None
    base_sharpe = None
    base_mdd = None
    base_trade_count = None

    union_selected = pd.concat([pd.DataFrame(candidate["selected"]).copy() for candidate in candidates], ignore_index=True)
    union_selected = union_selected.drop_duplicates(subset=["code"], keep="last")
    union_prices = fetch_histories(union_selected, years=args.years)

    for candidate in candidates:
        name = str(candidate["name"])
        selected = pd.DataFrame(candidate["selected"]).copy()
        risk_codes = list(candidate["risk_codes"])
        defensive_codes = list(candidate["defensive_codes"])
        selected_codes = selected["code"].astype(str).tolist()
        prices = union_prices.reindex(columns=selected_codes).copy()
        prices = prices.dropna(how="any")
        prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)]
        result, trades = run_custom_threshold_dual_with_overheat(
            prices=prices,
            selected=selected,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        if name == "base_pool":
            result = apply_official_baseline_nav_anchor(result)
        if compare_df is None:
            compare_df = pd.DataFrame(index=result.index)
            compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")
        compare_df[name] = result["nav"].reindex(compare_df.index)
        summary = {
            "pool": name,
            "risk_codes": ",".join([code for code in risk_codes if code in prices.columns]),
            "defensive_codes": ",".join([code for code in defensive_codes if code in prices.columns]),
            **summarize(result, trades),
        }
        rows.append(summary)
        if name == "base_pool":
            base_total_return = float(summary["total_return"])
            base_annualized = float(summary["annualized_return"])
            base_sharpe = float(summary["sharpe_rf0"])
            base_mdd = float(summary["max_drawdown"])
            base_trade_count = int(summary["trade_count"])

    summary_df = pd.DataFrame(rows)
    summary_df["return_diff"] = summary_df["total_return"] - base_total_return
    summary_df["annualized_diff"] = summary_df["annualized_return"] - base_annualized
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - base_sharpe
    summary_df["mdd_diff"] = summary_df["max_drawdown"] - base_mdd
    summary_df["trade_diff"] = summary_df["trade_count"] - base_trade_count

    ensure_output_dirs()
    write_dataframe_csv_atomic(summary_df, RESEARCH_OUTPUT_DIR / "candidate_pool_additions_summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, RESEARCH_OUTPUT_DIR / "candidate_pool_additions_nav_compare.csv")
    focus_pools = [
        "base_pool",
        "plus_hshares",
        "plus_sse50",
        "plus_financial",
        "replace_div_lowvol_with_dividend",
    ]
    write_dataframe_csv_atomic(
        summary_df[summary_df["pool"].isin(focus_pools)],
        RESEARCH_OUTPUT_DIR / "recommended_candidate_additions_summary.csv",
        index=False,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["base_pool"], linewidth=2.2, label="Base Pool")
    ax.plot(compare_df.index, compare_df["plus_kc50"], linewidth=1.8, label="+ KC50")
    ax.plot(compare_df.index, compare_df["plus_hstech"], linewidth=1.8, label="+ HS Tech")
    ax.plot(compare_df.index, compare_df["plus_10y_bond"], linewidth=1.8, label="+ 10Y Bond")
    ax.plot(compare_df.index, compare_df["plus_germany"], linewidth=1.8, label="+ Germany")
    ax.plot(compare_df.index, compare_df["plus_china_internet"], linewidth=1.8, label="+ China Internet")
    ax.plot(compare_df.index, compare_df["plus_30y_bond"], linewidth=1.8, label="+ 30Y Bond")
    ax.plot(compare_df.index, compare_df["plus_semiconductor"], linewidth=1.8, label="+ Semiconductor")
    ax.plot(compare_df.index, compare_df["plus_solar"], linewidth=1.8, label="+ Solar")
    ax.plot(compare_df.index, compare_df["plus_hshares"], linewidth=1.8, label="+ H Shares")
    ax.plot(compare_df.index, compare_df["plus_sse50"], linewidth=1.8, label="+ SSE50")
    ax.plot(compare_df.index, compare_df["plus_financial"], linewidth=1.8, label="+ Financial")
    ax.plot(compare_df.index, compare_df["plus_india_lof"], linewidth=1.8, label="+ India LOF")
    ax.plot(compare_df.index, compare_df["plus_dow"], linewidth=1.8, label="+ Dow")
    ax.plot(compare_df.index, compare_df["replace_div_lowvol_with_dividend"], linewidth=1.8, label="Replace Div LowVol")
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.6, linestyle="--", label="HS300 ETF")
    ax.set_title("Candidate Pool Additions Comparison", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, RESEARCH_OUTPUT_DIR / "candidate_pool_additions_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    focus_labels = {
        "base_pool": "Base Pool",
        "plus_hshares": "+ H Shares",
        "plus_sse50": "+ SSE50",
        "plus_financial": "+ Financial",
        "replace_div_lowvol_with_dividend": "Replace Div LowVol",
        "hs300_benchmark": "HS300 ETF",
    }
    fig, ax = plt.subplots(figsize=(14, 7))
    for pool in focus_pools:
        ax.plot(compare_df.index, compare_df[pool], linewidth=2.0 if pool == "base_pool" else 1.7, label=focus_labels[pool])
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.6, linestyle="--", label=focus_labels["hs300_benchmark"])
    ax.set_title("Recommended Candidate Additions Comparison", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, RESEARCH_OUTPUT_DIR / "recommended_candidate_additions_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
