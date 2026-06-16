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

from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
    DEFAULT_OVERHEAT_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    RESEARCH_OUTPUT_DIR,
    build_benchmark_nav,
    build_strategy_summary,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    normalize_code,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ETF_588000 = {"theme": "科创50", "code": "588000", "name": "科创50ETF华夏", "sina_symbol": "sh588000"}
ETF_513180 = {"theme": "恒生科技", "code": "513180", "name": "恒生科技指数ETF", "sina_symbol": "sh513180"}
ETF_511260 = {"theme": "10年国债", "code": "511260", "name": "十年国债ETF", "sina_symbol": "sh511260"}
ETF_513030 = {"theme": "德国ETF", "code": "513030", "name": "德国ETF", "sina_symbol": "sh513030"}
ETF_513050 = {"theme": "中概互联", "code": "513050", "name": "中概互联网ETF", "sina_symbol": "sh513050"}
ETF_511090 = {"theme": "30年国债", "code": "511090", "name": "30年国债ETF", "sina_symbol": "sh511090"}
ETF_512480 = {"theme": "半导体", "code": "512480", "name": "半导体ETF", "sina_symbol": "sh512480"}
ETF_515790 = {"theme": "光伏", "code": "515790", "name": "光伏ETF", "sina_symbol": "sh515790"}
ETF_511380 = {"theme": "可转债", "code": "511380", "name": "可转债ETF", "sina_symbol": "sh511380"}
ETF_510500 = {"theme": "中证500", "code": "510500", "name": "中证500ETF", "sina_symbol": "sh510500"}
ETF_512100 = {"theme": "中证1000", "code": "512100", "name": "中证1000ETF", "sina_symbol": "sh512100"}
ETF_510900 = {"theme": "H股", "code": "510900", "name": "H股ETF", "sina_symbol": "sh510900"}
ETF_510050 = {"theme": "上证50", "code": "510050", "name": "上证50ETF", "sina_symbol": "sh510050"}
ETF_510230 = {"theme": "金融", "code": "510230", "name": "金融ETF", "sina_symbol": "sh510230"}
ETF_510880 = {"theme": "红利ETF", "code": "510880", "name": "红利ETF", "sina_symbol": "sh510880"}

BASE_RISK_CODES = ["510300", "159949", "159941", "513650", "513880"]
BASE_DEFENSIVE_CODES = ["511580", "518880", "512890"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比若干实用 ETF 候选标的加入当前池子的效果。")
    parser.add_argument("--years", type=int, default=15, help="向前抓取多少年历史数据，再对齐公共区间。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--start-date", type=str, default="2012-01-01", help="候选池对比的分析起始日期。")
    return parser.parse_args()


def run_custom_threshold_dual_with_overheat(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    risk_codes: list[str],
    defensive_codes: list[str],
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1
    returns = prices.pct_change()
    risk_codes = [code for code in risk_codes if code in prices.columns]
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

    base_holding = signal.shift(1)
    base_exposure = target_exposure.shift(1).fillna(0.0)
    base_strategy_ret = pd.Series(0.0, index=prices.index, name="base_strategy_return")
    per_side_cost = fee_rate + slippage_rate

    prev_holding = base_holding.shift(1)
    prev_exposure = base_exposure.shift(1).fillna(0.0)
    for dt_idx in prices.index:
        asset = normalize_code(base_holding.loc[dt_idx])
        prev_asset = normalize_code(prev_holding.loc[dt_idx])
        weight = float(base_exposure.loc[dt_idx]) if pd.notna(base_exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0
        gross_ret = 0.0
        if asset and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = weight * float(returns.loc[dt_idx, asset])
        if not asset and not prev_asset:
            turnover = abs(weight - prev_weight)
        elif asset and prev_asset and asset == prev_asset:
            turnover = abs(weight - prev_weight)
        else:
            turnover = prev_weight + weight
        base_strategy_ret.loc[dt_idx] = (1 + gross_ret) * (1 - turnover * per_side_cost) - 1

    base_nav = (1 + base_strategy_ret.fillna(0.0)).cumprod()
    base_nav.iloc[0] = 1.0
    base_drawdown = base_nav / base_nav.cummax() - 1

    reduce_mask = (
        signal.isin(risk_codes)
        & (target_exposure > DEFAULT_OVERHEAT_MAX_EXPOSURE)
        & (base_drawdown >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_MOMENTUM_CUT)
    )
    capped_exposure = target_exposure.copy()
    capped_exposure.loc[reduce_mask] = DEFAULT_OVERHEAT_MAX_EXPOSURE
    high_reduce_mask = (
        signal.isin(risk_codes)
        & (capped_exposure > DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE)
        & (base_drawdown >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT)
    )
    capped_exposure.loc[high_reduce_mask] = DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE

    holding = signal.shift(1)
    exposure = capped_exposure.shift(1).fillna(0.0).rename("exposure")
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)
    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")

    for dt_idx in prices.index:
        asset = normalize_code(holding.loc[dt_idx])
        prev_asset = normalize_code(prev_holding.loc[dt_idx])
        weight = float(exposure.loc[dt_idx]) if pd.notna(exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0
        gross_ret = 0.0
        if asset and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = weight * float(returns.loc[dt_idx, asset])
        if not asset and not prev_asset:
            day_turnover = abs(weight - prev_weight)
        elif asset and prev_asset and asset == prev_asset:
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
            "target_exposure": capped_exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )

    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    trades: list[dict[str, object]] = []
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


def summarize(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        "start_date": result.index[0].date().isoformat(),
        "end_date": result.index[-1].date().isoformat(),
        **build_strategy_summary(result, trades),
    }


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

    for candidate in candidates:
        name = str(candidate["name"])
        selected = pd.DataFrame(candidate["selected"]).copy()
        risk_codes = list(candidate["risk_codes"])
        defensive_codes = list(candidate["defensive_codes"])
        prices = fetch_histories(selected, years=args.years)
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
