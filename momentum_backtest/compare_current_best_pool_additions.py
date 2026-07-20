#!/usr/bin/env python3
"""评估当前正式框架下的候选池增删与替换方案。"""

from __future__ import annotations

import argparse
try:
    from .runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__, include_module_dir=False)

import pandas as pd

try:
    from .candidate_pool_common import (
        ETF_159930,
        ETF_159985,
        ETF_508000,
        ETF_510500,
        ETF_510880,
        ETF_510900,
        ETF_510410,
        ETF_511090,
        ETF_511260,
        ETF_511380,
        ETF_512480,
        ETF_513030,
        ETF_513080,
        ETF_513300,
        ETF_513660,
        ETF_515790,
        ETF_515220,
        ETF_588000,
    )
    from .goal_optimization_common import load_market_volume_proxy
    from .hs300_regime_common import summarize
    from .pool_change_common import apply_pool_change, evaluate_pool
except ImportError:
    from candidate_pool_common import (
        ETF_159930,
        ETF_159985,
        ETF_508000,
        ETF_510500,
        ETF_510880,
        ETF_510900,
        ETF_510410,
        ETF_511090,
        ETF_511260,
        ETF_511380,
        ETF_512480,
        ETF_513030,
        ETF_513080,
        ETF_513300,
        ETF_513660,
        ETF_515790,
        ETF_515220,
        ETF_588000,
    )
    from goal_optimization_common import load_market_volume_proxy
    from hs300_regime_common import summarize
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


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "current_best_pool_additions"

POOL_CHANGES = [
    {"pool": "base_pool", "kind": "base"},
    {"pool": "restore_511580", "kind": "add", "candidate_kind": "defensive", "candidate": {"theme": "政金ETF", "code": "511580", "name": "国债政金债ETF招商", "sina_symbol": "sh511580"}},
    {"pool": "restore_513650", "kind": "add", "candidate_kind": "risk", "candidate": {"theme": "标普500", "code": "513650", "name": "标普500ETF南方", "sina_symbol": "sh513650"}},
    {"pool": "plus_resource_510410", "kind": "add", "candidate_kind": "risk", "candidate": ETF_510410},
    {"pool": "plus_energy_159930", "kind": "add", "candidate_kind": "risk", "candidate": ETF_159930},
    {"pool": "plus_energy_159930_defensive", "kind": "add", "candidate_kind": "defensive", "candidate": ETF_159930},
    {"pool": "plus_hshares_510900", "kind": "add", "candidate_kind": "risk", "candidate": ETF_510900},
    {"pool": "plus_coal_515220", "kind": "add", "candidate_kind": "risk", "candidate": ETF_515220},
    {"pool": "plus_soymeal_159985", "kind": "add", "candidate_kind": "risk", "candidate": ETF_159985},
    {"pool": "plus_soymeal_159985_defensive", "kind": "add", "candidate_kind": "defensive", "candidate": ETF_159985},
    {"pool": "plus_germany_513030", "kind": "add", "candidate_kind": "risk", "candidate": ETF_513030},
    {"pool": "plus_france_513080", "kind": "add", "candidate_kind": "risk", "candidate": ETF_513080},
    {"pool": "plus_france_germany", "kind": "add_pair", "candidate_kind": "risk", "candidates": [ETF_513080, ETF_513030]},
    {"pool": "plus_kc50_588000", "kind": "add", "candidate_kind": "risk", "candidate": ETF_588000},
    {"pool": "plus_semiconductor_512480", "kind": "add", "candidate_kind": "risk", "candidate": ETF_512480},
    {"pool": "plus_solar_515790", "kind": "add", "candidate_kind": "risk", "candidate": ETF_515790},
    {"pool": "plus_csi500_510500", "kind": "add", "candidate_kind": "risk", "candidate": ETF_510500},
    {"pool": "plus_global_dividend_513300", "kind": "add", "candidate_kind": "defensive", "candidate": ETF_513300},
    {"pool": "plus_hk_div_lowvol_513660", "kind": "add", "candidate_kind": "defensive", "candidate": ETF_513660},
    {"pool": "plus_reits_508000", "kind": "add", "candidate_kind": "defensive", "candidate": ETF_508000},
    {"pool": "plus_dividend_510880", "kind": "add", "candidate_kind": "defensive", "candidate": ETF_510880},
    {"pool": "plus_treasury10_511260", "kind": "add", "candidate_kind": "defensive", "candidate": ETF_511260},
    {"pool": "plus_treasury30_511090", "kind": "add", "candidate_kind": "defensive", "candidate": ETF_511090},
    {"pool": "plus_convertible_511380", "kind": "add", "candidate_kind": "defensive", "candidate": ETF_511380},
    {"pool": "drop_hshares_159954", "kind": "drop", "drop_codes": ["159954"]},
    {"pool": "drop_nikkei_513880", "kind": "drop", "drop_codes": ["513880"]},
    {"pool": "drop_cyb50_159949", "kind": "drop", "drop_codes": ["159949"]},
    {"pool": "drop_divlow_512890", "kind": "drop", "drop_codes": ["512890"]},
    {"pool": "replace_513650_resource", "kind": "replace", "drop_codes": ["513650"], "candidate_kind": "risk", "candidate": ETF_510410},
    {"pool": "replace_513650_energy", "kind": "replace", "drop_codes": ["513650"], "candidate_kind": "risk", "candidate": ETF_159930},
    {"pool": "replace_513650_hshares", "kind": "replace", "drop_codes": ["513650"], "candidate_kind": "risk", "candidate": ETF_510900},
    {"pool": "replace_513880_germany", "kind": "replace", "drop_codes": ["513880"], "candidate_kind": "risk", "candidate": ETF_513030},
    {"pool": "replace_159954_germany", "kind": "replace", "drop_codes": ["159954"], "candidate_kind": "risk", "candidate": ETF_513030},
    {"pool": "replace_513880_hshares_alt", "kind": "replace", "drop_codes": ["513880"], "candidate_kind": "risk", "candidate": ETF_510900},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估当前正式框架下的候选池增删与替换方案。")
    parser.add_argument("--years", type=int, default=15, help="向前抓取多少年历史数据。")
    parser.add_argument("--start-date", type=str, default="2012-01-01", help="分析起始日期。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--pools", nargs="*", default=[], help="只评估指定 pool 名称。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_drop_codes = list(DEFAULT_BASELINE_DROP_CODES)
    base_selected = load_default_strategy_backtest_pool().copy()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)
    pool_filters = set(args.pools)
    changes_to_run = [change for change in POOL_CHANGES if not pool_filters or str(change["pool"]) in pool_filters]

    prepared_changes: list[dict[str, object]] = []
    union_selected_frames: list[pd.DataFrame] = []
    for change in changes_to_run:
        selected, risk_codes, defensive_codes, effective_drop_codes = apply_pool_change(base_selected, base_drop_codes, change)
        prepared_changes.append(
            {
                "change": change,
                "selected": selected,
                "risk_codes": risk_codes,
                "defensive_codes": defensive_codes,
                "effective_drop_codes": effective_drop_codes,
            }
        )
        union_selected_frames.append(selected)

    union_selected = pd.concat(union_selected_frames, ignore_index=True).drop_duplicates(subset=["code"], keep="last")
    union_prices = fetch_histories(union_selected, years=args.years)

    rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame()

    for prepared in prepared_changes:
        change = dict(prepared["change"])
        pool_name = str(change["pool"])
        selected = pd.DataFrame(prepared["selected"]).copy()
        risk_codes = list(prepared["risk_codes"])
        defensive_codes = list(prepared["defensive_codes"])
        effective_drop_codes = list(prepared["effective_drop_codes"])
        selected_codes = selected["code"].astype(str).tolist()
        prices = union_prices.reindex(columns=selected_codes).copy()
        prices = prices.dropna(how="any")
        prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)].copy()
        result, trades = evaluate_pool(
            selected=selected,
            prices=prices,
            market_proxy=market_proxy,
            drop_codes=effective_drop_codes,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            use_official_baseline_anchor=(pool_name == "base_pool"),
        )
        row = summarize(result, trades, selected)
        row["pool"] = pool_name
        row["change_kind"] = str(change["kind"])
        row["start_date"] = result.index.min().date().isoformat()
        row["risk_codes"] = ",".join(code for code in risk_codes if code in prices.columns)
        row["defensive_codes"] = ",".join(code for code in defensive_codes if code in prices.columns)
        rows.append(row)
        nav_compare[pool_name] = result["nav"]

    summary_df = pd.DataFrame(rows)
    base_row = summary_df[summary_df["pool"] == "base_pool"].iloc[0]
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(base_row["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(base_row["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(base_row["max_drawdown_integral"])
    summary_df["mdd_diff"] = summary_df["max_drawdown"] - float(base_row["max_drawdown"])
    summary_df["trade_diff"] = summary_df["trade_count"] - int(base_row["trade_count"])
    summary_df["is_valid_change"] = (
        (summary_df["pool"] != "base_pool")
        & (summary_df["start_date"] <= "2012-12-31")
        & (summary_df["annualized_return"] >= float(base_row["annualized_return"]) - 1e-12)
        & (summary_df["sharpe_rf0"] >= float(base_row["sharpe_rf0"]) - 1e-12)
        & (summary_df["max_drawdown_integral"] <= float(base_row["max_drawdown_integral"]) + 1e-12)
        & (
            (summary_df["annualized_return"] > float(base_row["annualized_return"]) + 1e-12)
            | (summary_df["sharpe_rf0"] > float(base_row["sharpe_rf0"]) + 1e-12)
            | (summary_df["max_drawdown_integral"] < float(base_row["max_drawdown_integral"]) - 1e-12)
        )
    )

    valid_df = summary_df[summary_df["is_valid_change"]].copy()
    valid_additions_df = valid_df[valid_df["change_kind"] == "add"].copy()

    write_dataframe_csv_atomic(summary_df, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(valid_df, OUTPUT_DIR / "valid_pool_changes.csv", index=False)
    write_dataframe_csv_atomic(valid_additions_df, OUTPUT_DIR / "valid_additions.csv", index=False)
    write_dataframe_csv_atomic(nav_compare, OUTPUT_DIR / "nav_compare.csv")

    print(summary_df.to_csv(index=False))
    if valid_df.empty:
        print("No valid pool changes under current framework.")
    else:
        print(valid_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
