#!/usr/bin/env python3
"""对比与当前池子重合度更低的一批额外 ETF 候选。"""

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

from compare_candidate_pool_additions import BASE_DEFENSIVE_CODES, BASE_RISK_CODES, run_custom_threshold_dual_with_overheat
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    annualized_return,
    build_benchmark_nav,
    count_trade_days,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    max_drawdown,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ETF_510880 = {"theme": "上证红利", "code": "510880", "name": "上证红利ETF", "sina_symbol": "sh510880"}
ETF_159985 = {"theme": "豆粕", "code": "159985", "name": "豆粕ETF", "sina_symbol": "sz159985"}
ETF_515220 = {"theme": "煤炭", "code": "515220", "name": "煤炭ETF", "sina_symbol": "sh515220"}
ETF_508000 = {"theme": "REITs", "code": "508000", "name": "REITsETF", "sina_symbol": "sh508000"}
OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_more_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比与当前池子重合度更低的一批额外 ETF 候选。")
    parser.add_argument("--years", type=int, default=6, help="向前抓取多少年历史数据，再对齐公共区间。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def summarize(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | str]:
    daily_ret = result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    return {
        "start_date": result.index[0].date().isoformat(),
        "end_date": result.index[-1].date().isoformat(),
        "total_return": float(result["nav"].iloc[-1] - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown(result["nav"]),
        "trade_count": count_trade_days(trades),
        "trade_action_count": int(len(trades)),
        "avg_exposure": float(result["exposure"].mean()),
        "latest_momentum": float(result["current_momentum"].dropna().iloc[-1]),
        "latest_exposure": float(result["exposure"].iloc[-1]),
    }


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    candidates = [
        ("base_pool", [], BASE_RISK_CODES, BASE_DEFENSIVE_CODES),
        ("plus_sh_dividend", [ETF_510880], BASE_RISK_CODES, BASE_DEFENSIVE_CODES + ["510880"]),
        ("plus_soymeal", [ETF_159985], BASE_RISK_CODES + ["159985"], BASE_DEFENSIVE_CODES),
        ("plus_coal", [ETF_515220], BASE_RISK_CODES + ["515220"], BASE_DEFENSIVE_CODES),
        ("plus_reits", [ETF_508000], BASE_RISK_CODES, BASE_DEFENSIVE_CODES + ["508000"]),
    ]

    rows: list[dict[str, object]] = []
    compare_df = None
    base_metrics: dict[str, float | int] = {}

    for name, additions, risk_codes, defensive_codes in candidates:
        selected = pd.concat([base_pool, pd.DataFrame(additions)], ignore_index=True) if additions else base_pool.copy()
        prices = fetch_histories(selected, years=args.years)
        prices = prices.dropna(how="any")
        result, trades = run_custom_threshold_dual_with_overheat(
            prices=prices,
            selected=selected,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        row = {
            "pool": name,
            "risk_codes": ",".join([code for code in risk_codes if code in prices.columns]),
            "defensive_codes": ",".join([code for code in defensive_codes if code in prices.columns]),
            **summarize(result, trades),
        }
        rows.append(row)
        if compare_df is None:
            compare_df = pd.DataFrame(index=result.index)
            compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")
        compare_df[name] = result["nav"].reindex(compare_df.index)
        if name == "base_pool":
            base_metrics = {
                "total_return": float(row["total_return"]),
                "annualized_return": float(row["annualized_return"]),
                "sharpe_rf0": float(row["sharpe_rf0"]),
                "max_drawdown": float(row["max_drawdown"]),
                "trade_count": int(row["trade_count"]),
            }

    summary_df = pd.DataFrame(rows)
    summary_df["return_diff"] = summary_df["total_return"] - float(base_metrics["total_return"])
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(base_metrics["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(base_metrics["sharpe_rf0"])
    summary_df["mdd_diff"] = summary_df["max_drawdown"] - float(base_metrics["max_drawdown"])
    summary_df["trade_diff"] = summary_df["trade_count"] - int(base_metrics["trade_count"])

    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(summary_df, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["base_pool"], linewidth=2.2, label="Base Pool")
    ax.plot(compare_df.index, compare_df["plus_sh_dividend"], linewidth=1.8, label="+ SH Dividend")
    ax.plot(compare_df.index, compare_df["plus_soymeal"], linewidth=1.8, label="+ Soymeal")
    ax.plot(compare_df.index, compare_df["plus_coal"], linewidth=1.8, label="+ Coal")
    ax.plot(compare_df.index, compare_df["plus_reits"], linewidth=1.8, label="+ REITs")
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.6, linestyle="--", label="HS300 ETF")
    ax.set_title("Additional ETF Candidate Comparison", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
