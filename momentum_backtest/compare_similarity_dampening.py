#!/usr/bin/env python3
"""测试通过 leader_margin 抑制高相似资产之间的抢信号。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__, include_module_dir=False)

import pandas as pd

try:
    from .compare_hs300_regime_fixes import summarize
    from .compare_goal_optimizations import load_market_volume_proxy
    from .official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from compare_hs300_regime_fixes import summarize
    from compare_goal_optimizations import load_market_volume_proxy
    from official_baseline import apply_official_baseline_nav_anchor

from run_backtest import (
    DEFAULT_BASELINE_DROP_CODES,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    build_default_strategy_params,
    fetch_histories,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "similarity_dampening"
LEADER_MARGINS = [0.0, 0.005, 0.01, 0.015, 0.02]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试通过 leader_margin 抑制高相似资产之间的抢信号。")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--start-date", type=str, default="2012-01-01")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = load_default_strategy_backtest_pool().copy()
    prices = fetch_histories(selected, years=args.years)
    prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)].copy()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)
    rows: list[dict[str, object]] = []

    for margin in LEADER_MARGINS:
        params = build_default_strategy_params(drop_codes=list(DEFAULT_BASELINE_DROP_CODES))
        params["signal_leader_margin"] = margin
        result, trades = run_default_strategy_with_params(
            prices=prices,
            selected=selected,
            params=params,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            market_proxy=market_proxy,
        )
        if abs(float(margin)) < 1e-12:
            result = apply_official_baseline_nav_anchor(result)
        row = summarize(result, trades, selected)
        row["leader_margin"] = margin
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    base_row = summary_df.loc[summary_df["leader_margin"] == 0.0].iloc[0]
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(base_row["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(base_row["sharpe_rf0"])
    summary_df["mdd_diff"] = summary_df["max_drawdown"] - float(base_row["max_drawdown"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(base_row["max_drawdown_integral"])
    summary_df["trade_diff"] = summary_df["trade_count"] - int(base_row["trade_count"])
    write_dataframe_csv_atomic(summary_df, OUTPUT_DIR / "summary.csv", index=False)
    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
