#!/usr/bin/env python3
"""Inventory current factor coverage and carry / term-structure feasibility."""

from __future__ import annotations

import pandas as pd

from common import (
    FRONTIER_OUTPUT_DIR,
    code_to_theme_map,
    ensure_frontier_output_dir,
    load_selected_prices,
    render_markdown_table,
    save_markdown,
    write_dataframe_csv_atomic,
)


def build_factor_inventory() -> pd.DataFrame:
    rows = [
        {
            "factor_family": "price_trend",
            "frontier_factor": "single-window momentum",
            "repo_status": "existing",
            "current_repo_evidence": "official raw / quality momentum signal",
            "best_fit_data_source": "existing output/core/prices.csv",
            "implementation_direction": "keep as baseline reference layer",
        },
        {
            "factor_family": "price_trend",
            "frontier_factor": "multi-horizon trend consistency",
            "repo_status": "partial",
            "current_repo_evidence": "archived multihorizon experiments exist, not in formal main chain",
            "best_fit_data_source": "existing output/core/prices.csv",
            "implementation_direction": "promote 10/20/40/60/120 horizon blend to formal research track",
        },
        {
            "factor_family": "trend_quality",
            "frontier_factor": "downside vol / r2 / slope / drawdown penalties",
            "repo_status": "existing",
            "current_repo_evidence": "signal quality score, stability score, drawdown overlays",
            "best_fit_data_source": "existing output/core/prices.csv",
            "implementation_direction": "already usable for ML features or scoring inputs",
        },
        {
            "factor_family": "state",
            "frontier_factor": "breadth / amount / regime proxies",
            "repo_status": "existing",
            "current_repo_evidence": "market_amount_ratio_20_60, market_amount_ratio_5_20, market_breadth_proxy",
            "best_fit_data_source": "existing market proxy cache",
            "implementation_direction": "keep as state conditioning layer",
        },
        {
            "factor_family": "risk_control",
            "frontier_factor": "momentum percentile / overheat / signal confirmation",
            "repo_status": "existing",
            "current_repo_evidence": "selected momentum percentile cap, leader margin, confirmation lookback",
            "best_fit_data_source": "existing output/core/prices.csv",
            "implementation_direction": "suitable for dynamic exposure mapping",
        },
        {
            "factor_family": "carry",
            "frontier_factor": "carry / roll yield / term-structure slope",
            "repo_status": "missing",
            "current_repo_evidence": "no futures curve, yield, basis, dividend yield, or roll data in repo",
            "best_fit_data_source": "external futures curve, bond yield curve, ETF distribution yield, or index-level carry feeds",
            "implementation_direction": "highest-priority missing layer for cross-asset expansion",
        },
        {
            "factor_family": "relative_value",
            "frontier_factor": "value / relative valuation",
            "repo_status": "missing",
            "current_repo_evidence": "no valuation or macro spread inputs in repo",
            "best_fit_data_source": "index valuation series, bond real yield, commodity basis proxies",
            "implementation_direction": "secondary priority after carry",
        },
        {
            "factor_family": "crowding",
            "frontier_factor": "flow / positioning / correlation crowding",
            "repo_status": "partial",
            "current_repo_evidence": "price-only correlation and concentration proxies possible; no true flow data",
            "best_fit_data_source": "ETF flows, OI / COT, financing, issuance, holdings overlap",
            "implementation_direction": "start with price-based proxy, then upgrade with external flow data",
        },
    ]
    return pd.DataFrame(rows)


def build_carry_feasibility(selected: pd.DataFrame) -> pd.DataFrame:
    relevant = {
        "518880": {
            "asset_theme": "黄金ETF",
            "carry_need": "gold lease / real-rate / futures curve carry proxy",
            "current_local_data": "price only",
            "recommended_source": "AU/COMEX futures curve or real-rate proxy",
            "local_ready": "no",
            "next_step": "add gold carry proxy from curve or real yield",
        },
        "513880": {
            "asset_theme": "日经ETF",
            "carry_need": "equity index carry = dividend yield - financing",
            "current_local_data": "price only",
            "recommended_source": "Nikkei dividend yield + short-rate / futures basis",
            "local_ready": "no",
            "next_step": "add equity index implied carry proxy",
        },
        "159941": {
            "asset_theme": "纳指ETF",
            "carry_need": "equity index carry = dividend yield - financing",
            "current_local_data": "price only",
            "recommended_source": "Nasdaq dividend yield + SOFR / futures basis",
            "local_ready": "no",
            "next_step": "add US equity carry proxy",
        },
        "511260": {
            "asset_theme": "十年国债ETF",
            "carry_need": "yield roll-down / term-structure carry",
            "current_local_data": "price only",
            "recommended_source": "China 10Y yield curve slope and bond ETF yield-to-maturity",
            "local_ready": "no",
            "next_step": "add bond roll-down / carry proxy first",
        },
        "159985": {
            "asset_theme": "豆粕ETF",
            "carry_need": "commodity carry / convenience yield / curve slope",
            "current_local_data": "price only",
            "recommended_source": "m1-m3 soymeal futures curve or index roll yield",
            "local_ready": "no",
            "next_step": "add commodity term-structure slope proxy",
        },
    }
    code_theme = code_to_theme_map(selected)
    rows = []
    for code in selected["code"].astype(str):
        if code not in relevant:
            continue
        row = {"code": code, "selected_theme": code_theme.get(code, ""), **relevant[code]}
        rows.append(row)
    return pd.DataFrame(rows)


def build_markdown(inventory: pd.DataFrame, carry: pd.DataFrame) -> str:
    lines = [
        "# Frontier Factor Inventory",
        "",
        "This note records which factor families are already covered by the current strategy stack,",
        "which ones are only partially covered, and which ones require new external data.",
        "",
        "## Factor Inventory",
        "",
        render_markdown_table(inventory),
        "",
        "## Carry / Term-Structure Feasibility For Current Universe",
        "",
        render_markdown_table(carry),
        "",
        "## Immediate Takeaways",
        "",
        "- Carry / term-structure is the clearest missing factor family for the current cross-asset ETF universe.",
        "- The repo is already strong on price trend, trend quality, regime, and risk-control features.",
        "- True carry needs new upstream data; price-only substitutes should stay exploratory and should not be treated as real carry.",
    ]
    return "\n".join(lines)


def main() -> int:
    ensure_frontier_output_dir()
    selected, _ = load_selected_prices()
    inventory = build_factor_inventory()
    carry = build_carry_feasibility(selected)

    write_dataframe_csv_atomic(inventory, FRONTIER_OUTPUT_DIR / "factor_inventory.csv", index=False)
    write_dataframe_csv_atomic(carry, FRONTIER_OUTPUT_DIR / "carry_term_structure_feasibility.csv", index=False)
    save_markdown(
        FRONTIER_OUTPUT_DIR / "factor_inventory.md",
        build_markdown(inventory, carry),
    )
    print(inventory.to_string(index=False))
    print()
    print(carry.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
