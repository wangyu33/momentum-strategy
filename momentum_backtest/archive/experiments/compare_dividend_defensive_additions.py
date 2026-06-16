#!/usr/bin/env python3
"""Compare adding dividend-style defensive ETFs to the current threshold dual momentum pool."""

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

from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    RESEARCH_OUTPUT_DIR,
    RISK_CODES,
    annualized_return,
    build_benchmark_nav,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    max_drawdown,
    normalize_code,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_dividend_defensive_additions"


ETF_510880 = {"theme": "红利ETF", "code": "510880", "name": "红利ETF华泰柏瑞", "sina_symbol": "sh510880"}
ETF_520550 = {
    "theme": "港股红利低波ETF",
    "code": "520550",
    "name": "招商恒生港股通高股息低波动ETF",
    "sina_symbol": "sh520550",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare adding 510880 / 520550 into the defensive bucket.")
    parser.add_argument("--years", type=int, default=10, help="Backtest years.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    return parser.parse_args()


def run_custom_threshold_dual(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    defensive_codes: list[str],
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change()
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    defensive_codes = [code for code in defensive_codes if code in prices.columns]

    risk_mom = momentum[risk_codes]
    defensive_mom = momentum[defensive_codes]
    risk_winner = risk_mom.idxmax(axis=1, skipna=True)
    risk_best = risk_mom.max(axis=1, skipna=True)
    defensive_winner = defensive_mom.idxmax(axis=1, skipna=True)
    defensive_best = defensive_mom.max(axis=1, skipna=True)

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        r_asset = risk_winner.loc[dt_idx]
        r_score = risk_best.loc[dt_idx]
        d_asset = defensive_winner.loc[dt_idx]
        d_score = defensive_best.loc[dt_idx]

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
        elif pd.notna(r_score) and r_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT
            current_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            current_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")

    holding = signal.shift(1)
    exposure = target_exposure.shift(1).fillna(0.0).rename("exposure")
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)
    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")
    per_side_cost = fee_rate + slippage_rate

    for dt_idx in prices.index:
        asset = holding.loc[dt_idx]
        prev_asset = prev_holding.loc[dt_idx]
        weight = float(exposure.loc[dt_idx]) if pd.notna(exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0

        gross_ret = 0.0
        if pd.notna(asset) and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = weight * float(returns.loc[dt_idx, asset])

        if pd.isna(asset) and pd.isna(prev_asset):
            day_turnover = abs(weight - prev_weight)
        elif pd.notna(asset) and pd.notna(prev_asset) and asset == prev_asset:
            day_turnover = abs(weight - prev_weight)
        else:
            day_turnover = prev_weight + weight

        turnover.loc[dt_idx] = day_turnover
        cost_rate = day_turnover * per_side_cost
        trade_cost_rate.loc[dt_idx] = cost_rate
        strategy_ret.loc[dt_idx] = (1 + gross_ret) * (1 - cost_rate) - 1

    nav = (1 + strategy_ret.fillna(0.0)).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1
    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": current_momentum,
            "signal": signal,
            "holding": holding,
            "exposure": exposure,
            "target_exposure": target_exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )

    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    trades = []
    for dt_idx in prices.index:
        asset = normalize_code(holding.loc[dt_idx])
        prev_asset = normalize_code(prev_holding.loc[dt_idx])
        weight = float(exposure.loc[dt_idx]) if pd.notna(exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0
        if prev_asset == asset and abs(weight - prev_weight) < 1e-12:
            continue
        if prev_asset and (prev_asset != asset or prev_weight > weight):
            trades.append(
                {
                    "date": dt_idx,
                    "action": "SELL" if prev_asset != asset else "REDUCE",
                    "code": prev_asset,
                    "theme": code_to_theme.get(prev_asset, ""),
                    "name": code_to_name.get(prev_asset, ""),
                    "from_exposure": prev_weight,
                    "to_exposure": weight if prev_asset == asset else 0.0,
                    "nav": float(result.loc[dt_idx, "nav"]),
                }
            )
        if asset and (prev_asset != asset or weight > prev_weight):
            trades.append(
                {
                    "date": dt_idx,
                    "action": "BUY" if prev_asset != asset else "ADD",
                    "code": asset,
                    "theme": code_to_theme.get(asset, ""),
                    "name": code_to_name.get(asset, ""),
                    "from_exposure": prev_weight if prev_asset == asset else 0.0,
                    "to_exposure": weight,
                    "nav": float(result.loc[dt_idx, "nav"]),
                }
            )

    return result, pd.DataFrame(trades)


def summarize(result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    daily_ret = result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    latest_holding = normalize_code(result["holding"].dropna().iloc[-1]) if not result["holding"].dropna().empty else ""
    latest_theme = ""
    latest_name = ""
    if latest_holding:
        row = selected[selected["code"] == latest_holding].iloc[0]
        latest_theme = row["theme"]
        latest_name = row["name"]
    return {
        "total_return": float(result["nav"].iloc[-1] - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown(result["nav"]),
        "trade_count": int(len(trades)),
        "avg_exposure": float(result["exposure"].mean()),
        "latest_holding_code": latest_holding,
        "latest_holding_theme": latest_theme,
        "latest_holding_name": latest_name,
        "latest_momentum": float(result["current_momentum"].dropna().iloc[-1]),
        "latest_exposure": float(result["exposure"].iloc[-1]),
    }


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    base_defensive = ["511580", "518880", "512890"]

    candidates = [
        ("base_pool", [], base_defensive),
        ("plus_510880", [ETF_510880], base_defensive + ["510880"]),
        ("plus_520550", [ETF_520550], base_defensive + ["520550"]),
        ("plus_both", [ETF_510880, ETF_520550], base_defensive + ["510880", "520550"]),
    ]

    rows: list[dict[str, float | int | str]] = []
    compare_df = None
    for name, additions, defensive_codes in candidates:
        selected = pd.concat([base_pool, pd.DataFrame(additions)], ignore_index=True) if additions else base_pool.copy()
        prices = fetch_histories(selected, years=args.years)
        result, trades = run_custom_threshold_dual(
            prices,
            selected,
            defensive_codes=defensive_codes,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        rows.append(
            {
                "pool": name,
                "defensive_codes": ",".join(defensive_codes),
                **summarize(result, trades, selected),
            }
        )
        if compare_df is None:
            compare_df = pd.DataFrame(index=result.index)
            compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")
        compare_df[f"{name}_nav"] = result["nav"].reindex(compare_df.index)
        write_dataframe_csv_atomic(result, OUTPUT_DIR / f"{name}_nav.csv")
        write_dataframe_csv_atomic(trades, OUTPUT_DIR / f"{name}_trades.csv", index=False)

    summary = pd.DataFrame(rows)
    base = summary.iloc[0]
    summary["return_diff"] = summary["total_return"] - base["total_return"]
    summary["annualized_diff"] = summary["annualized_return"] - base["annualized_return"]
    summary["sharpe_diff"] = summary["sharpe_rf0"] - base["sharpe_rf0"]
    summary["mdd_diff"] = summary["max_drawdown"] - base["max_drawdown"]
    summary["trade_diff"] = summary["trade_count"] - base["trade_count"]

    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(summary, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["base_pool_nav"], linewidth=2.2, label="Base Pool")
    ax.plot(compare_df.index, compare_df["plus_510880_nav"], linewidth=1.8, label="+510880")
    ax.plot(compare_df.index, compare_df["plus_520550_nav"], linewidth=1.8, label="+520550")
    ax.plot(compare_df.index, compare_df["plus_both_nav"], linewidth=1.8, label="+Both")
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.6, linestyle="--", label="HS300")
    ax.set_title("Dividend Defensive Additions Comparison", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
