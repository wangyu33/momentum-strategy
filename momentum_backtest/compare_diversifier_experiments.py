#!/usr/bin/env python3
"""聚焦测试低相关候选加入/替换后的效果。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import prepare_local_imports
    from .official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from runtime_env import prepare_local_imports
    from official_baseline import apply_official_baseline_nav_anchor

prepare_local_imports(__file__, include_module_dir=False)

import pandas as pd

try:
    from .candidate_pool_common import (
        ETF_159930,
        ETF_159985,
        ETF_510410,
        ETF_515220,
    )
    from .hs300_regime_common import summarize
    from .goal_optimization_common import load_market_volume_proxy
    from .pool_change_common import apply_pool_change, evaluate_pool
except ImportError:
    from candidate_pool_common import (
        ETF_159930,
        ETF_159985,
        ETF_510410,
        ETF_515220,
    )
    from hs300_regime_common import summarize
    from goal_optimization_common import load_market_volume_proxy
    from pool_change_common import apply_pool_change, evaluate_pool

from run_backtest import (
    DEFAULT_BASELINE_DROP_CODES,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    fetch_histories,
    load_default_strategy_backtest_pool,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "diversifier_experiments"

EXPERIMENTS = [
    {"pool": "base_pool", "kind": "base"},
    {"pool": "plus_soymeal_159985", "kind": "add", "candidate_kind": "risk", "candidate": ETF_159985},
    {"pool": "plus_energy_159930", "kind": "add", "candidate_kind": "risk", "candidate": ETF_159930},
    {"pool": "plus_coal_515220", "kind": "add", "candidate_kind": "risk", "candidate": ETF_515220},
    {"pool": "plus_resource_510410", "kind": "add", "candidate_kind": "risk", "candidate": ETF_510410},
    {"pool": "replace_cyb50_soymeal", "kind": "replace", "drop_codes": ["159949"], "candidate_kind": "risk", "candidate": ETF_159985},
    {"pool": "replace_cyb50_energy", "kind": "replace", "drop_codes": ["159949"], "candidate_kind": "risk", "candidate": ETF_159930},
    {"pool": "replace_cyb50_coal", "kind": "replace", "drop_codes": ["159949"], "candidate_kind": "risk", "candidate": ETF_515220},
    {"pool": "replace_cyb50_resource", "kind": "replace", "drop_codes": ["159949"], "candidate_kind": "risk", "candidate": ETF_510410},
    {"pool": "replace_nikkei_soymeal", "kind": "replace", "drop_codes": ["513880"], "candidate_kind": "risk", "candidate": ETF_159985},
    {"pool": "replace_nikkei_energy", "kind": "replace", "drop_codes": ["513880"], "candidate_kind": "risk", "candidate": ETF_159930},
    {"pool": "replace_nikkei_coal", "kind": "replace", "drop_codes": ["513880"], "candidate_kind": "risk", "candidate": ETF_515220},
    {"pool": "replace_nikkei_resource", "kind": "replace", "drop_codes": ["513880"], "candidate_kind": "risk", "candidate": ETF_510410},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聚焦测试低相关候选加入/替换后的效果。")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--start-date", type=str, default="2012-01-01")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_drop_codes = list(DEFAULT_BASELINE_DROP_CODES)
    base_selected = load_default_strategy_backtest_pool().copy()
    selected_union = pd.concat(
        [
            base_selected,
            pd.DataFrame([ETF_159985, ETF_159930, ETF_515220, ETF_510410]),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    all_prices = fetch_histories(selected_union, years=args.years)
    all_prices = all_prices.loc[all_prices.index >= pd.Timestamp(args.start_date)].copy()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)

    rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame(index=all_prices.index)

    for change in EXPERIMENTS:
        selected, risk_codes, defensive_codes, effective_drop_codes = apply_pool_change(base_selected, base_drop_codes, change)
        price_codes = [str(code) for code in selected["code"].tolist() if str(code) in all_prices.columns]
        prices = all_prices[price_codes].copy()
        result, trades = evaluate_pool(
            selected=selected,
            prices=prices,
            market_proxy=market_proxy,
            drop_codes=effective_drop_codes,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            use_official_baseline_anchor=(str(change["pool"]) == "base_pool"),
        )
        if str(change["pool"]) == "base_pool":
            result = apply_official_baseline_nav_anchor(result)
        row = summarize(result, trades, selected)
        row["pool"] = str(change["pool"])
        row["change_kind"] = str(change["kind"])
        row["risk_codes"] = ",".join(code for code in risk_codes if code in prices.columns)
        row["defensive_codes"] = ",".join(code for code in defensive_codes if code in prices.columns)
        rows.append(row)
        nav_compare[row["pool"]] = result["nav"].reindex(nav_compare.index)

    summary_df = pd.DataFrame(rows)
    base_row = summary_df.loc[summary_df["pool"] == "base_pool"].iloc[0]
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(base_row["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(base_row["sharpe_rf0"])
    summary_df["mdd_diff"] = summary_df["max_drawdown"] - float(base_row["max_drawdown"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(base_row["max_drawdown_integral"])
    summary_df["trade_diff"] = summary_df["trade_count"] - int(base_row["trade_count"])

    write_dataframe_csv_atomic(summary_df, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(nav_compare, OUTPUT_DIR / "nav_compare.csv")
    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
