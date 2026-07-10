#!/usr/bin/env python3
"""A 股行业 ETF 动量策略回测入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from momentum_backtest.runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from momentum_backtest.run_backtest import (
    build_strategy_summary,
    configure_matplotlib,
    fetch_histories,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("a_share_industry_etf_momentum_backtest/output")
CORE_OUTPUT_DIR = OUTPUT_DIR / "core"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"

DEFAULT_YEARS = 10
WARMUP_DAYS = 180
DEFAULT_COST_RATE = 0.0005

INDUSTRY_ETFS = [
    {"role": "industry", "theme": "半导体", "code": "512480", "name": "半导体ETF国联安", "sina_symbol": "sh512480"},
    {"role": "industry", "theme": "军工", "code": "512660", "name": "军工ETF国泰", "sina_symbol": "sh512660"},
    {"role": "industry", "theme": "医药", "code": "512010", "name": "医药ETF易方达", "sina_symbol": "sh512010"},
    {"role": "industry", "theme": "医疗", "code": "512170", "name": "医疗ETF华宝", "sina_symbol": "sh512170"},
    {"role": "industry", "theme": "白酒", "code": "512690", "name": "酒ETF鹏华", "sina_symbol": "sh512690"},
    {"role": "industry", "theme": "消费", "code": "159928", "name": "消费ETF汇添富", "sina_symbol": "sz159928"},
    {"role": "industry", "theme": "家电", "code": "159996", "name": "家电ETF国泰", "sina_symbol": "sz159996"},
    {"role": "industry", "theme": "银行", "code": "512800", "name": "银行ETF华宝", "sina_symbol": "sh512800"},
    {"role": "industry", "theme": "证券", "code": "512880", "name": "证券ETF国泰", "sina_symbol": "sh512880"},
    {"role": "industry", "theme": "煤炭", "code": "515220", "name": "煤炭ETF国泰", "sina_symbol": "sh515220"},
    {"role": "industry", "theme": "有色", "code": "512400", "name": "有色金属ETF南方", "sina_symbol": "sh512400"},
    {"role": "industry", "theme": "稀土", "code": "516150", "name": "稀土ETF嘉实", "sina_symbol": "sh516150"},
    {"role": "industry", "theme": "化工", "code": "159870", "name": "化工ETF鹏华", "sina_symbol": "sz159870"},
    {"role": "industry", "theme": "地产", "code": "512200", "name": "房地产ETF南方", "sina_symbol": "sh512200"},
    {"role": "industry", "theme": "通信", "code": "515880", "name": "通信ETF国泰", "sina_symbol": "sh515880"},
    {"role": "industry", "theme": "电子", "code": "159997", "name": "电子ETF天弘", "sina_symbol": "sz159997"},
    {"role": "industry", "theme": "软件", "code": "159852", "name": "软件ETF嘉实", "sina_symbol": "sz159852"},
    {"role": "industry", "theme": "传媒游戏", "code": "159869", "name": "游戏ETF华夏", "sina_symbol": "sz159869"},
    {"role": "industry", "theme": "传媒", "code": "512980", "name": "传媒ETF广发", "sina_symbol": "sh512980"},
    {"role": "industry", "theme": "新能源", "code": "516160", "name": "新能源ETF南方", "sina_symbol": "sh516160"},
    {"role": "industry", "theme": "光伏", "code": "515790", "name": "光伏ETF华泰柏瑞", "sina_symbol": "sh515790"},
    {"role": "industry", "theme": "新能源车", "code": "515030", "name": "新能源车ETF华夏", "sina_symbol": "sh515030"},
    {"role": "industry", "theme": "农业", "code": "516810", "name": "农业ETF华夏", "sina_symbol": "sh516810"},
]

DEFENSE_ASSETS = [
    {"role": "defense", "theme": "十年国债", "code": "511260", "name": "十年国债ETF", "sina_symbol": "sh511260"},
    {"role": "defense", "theme": "现金替代", "code": "CASH", "name": "现金替代", "sina_symbol": ""},
]

BENCHMARK_ASSET = {"role": "benchmark", "theme": "沪深300", "code": "510300", "name": "沪深300ETF华泰柏瑞", "sina_symbol": "sh510300"}

STRATEGY_VARIANTS = {
    "top2_bond": {
        "label": "行业动量 Top2 严格阈值 + 国债防守",
        "max_positions": 2,
        "absolute_score": 0.15,
        "trend_filter": False,
        "fallback": "511260",
    },
    "top3_bond": {
        "label": "行业动量 Top3 严格阈值 + 国债防守",
        "max_positions": 3,
        "absolute_score": 0.15,
        "trend_filter": False,
        "fallback": "511260",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 A 股行业 ETF 动量策略回测。")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="分析最近多少年。")
    parser.add_argument("--cost-rate", type=float, default=DEFAULT_COST_RATE, help="按换手收取的一次性总成本率。")
    parser.add_argument(
        "--strategy",
        choices=["all", *STRATEGY_VARIANTS.keys()],
        default="all",
        help="运行单个策略或全部策略。",
    )
    return parser.parse_args()


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, CORE_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"{path.stem}_",
            dir=str(path.parent),
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def build_selected() -> pd.DataFrame:
    return pd.DataFrame([*INDUSTRY_ETFS, *DEFENSE_ASSETS, BENCHMARK_ASSET])


def load_prices(years: int) -> pd.DataFrame:
    selected = build_selected()
    real_selected = selected[selected["code"] != "CASH"].copy()
    prices = fetch_histories(real_selected, years=max(years + 2, 8))
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.ffill(limit=3)
    prices["CASH"] = 1.0
    return prices[[str(code) for code in selected["code"] if str(code) in prices.columns]].copy()


def compute_analysis_start(prices: pd.DataFrame, years: int) -> pd.Timestamp:
    end_date = prices.index.max()
    requested_start = end_date - pd.DateOffset(years=years)
    min_available_start = min(series.index.min() for _, series in prices.items() if not series.dropna().empty)
    warmup_start = pd.Timestamp(min_available_start) + pd.Timedelta(days=WARMUP_DAYS)
    return max(requested_start, warmup_start)


def compute_composite_score(prices: pd.DataFrame, industry_codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ret20 = prices[industry_codes] / prices[industry_codes].shift(20) - 1.0
    ret60 = prices[industry_codes] / prices[industry_codes].shift(60) - 1.0
    ret120 = prices[industry_codes] / prices[industry_codes].shift(120) - 1.0
    sma120 = prices[industry_codes].rolling(120).mean()
    score = 0.5 * ret20 + 0.3 * ret60 + 0.2 * ret120
    trend_ok = prices[industry_codes] > sma120
    return score, trend_ok, sma120


def annualized_return_from_nav(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    days = max((nav.index[-1] - nav.index[0]).days, 1)
    years = days / 365.25
    return float(nav.iloc[-1] ** (1.0 / years) - 1.0)


def build_dual_benchmark_yearly_returns(result: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    years = sorted(int(year) for year in result.index.year.unique())
    for year in years:
        year_mask = result.index.year == year
        if not year_mask.any():
            continue
        period = result.loc[year_mask]
        start_date = period.index[0]
        end_date = period.index[-1]

        strategy_start = float(result["nav"].shift(1).loc[start_date]) if pd.notna(result["nav"].shift(1).loc[start_date]) else 1.0
        industry_start = float(result["industry_equal_benchmark_nav"].shift(1).loc[start_date]) if pd.notna(result["industry_equal_benchmark_nav"].shift(1).loc[start_date]) else 1.0
        hs300_start = float(result["hs300_benchmark_nav"].shift(1).loc[start_date]) if pd.notna(result["hs300_benchmark_nav"].shift(1).loc[start_date]) else 1.0

        rows.append(
            {
                "year": year,
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "strategy_return": float(result.loc[end_date, "nav"] / strategy_start - 1.0),
                "industry_equal_benchmark_return": float(result.loc[end_date, "industry_equal_benchmark_nav"] / industry_start - 1.0),
                "hs300_return": float(result.loc[end_date, "hs300_benchmark_nav"] / hs300_start - 1.0),
            }
        )
    return pd.DataFrame(rows)


def build_drawdown_episodes(nav: pd.Series) -> pd.DataFrame:
    drawdown = nav / nav.cummax() - 1.0
    episodes: list[dict[str, object]] = []
    in_drawdown = False
    start_idx = nav.index[0]
    trough_idx = nav.index[0]
    trough_value = 0.0

    for dt_idx, dd in drawdown.items():
        dd = float(dd)
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            start_idx = dt_idx
            trough_idx = dt_idx
            trough_value = dd
        elif dd < 0 and in_drawdown and dd < trough_value:
            trough_idx = dt_idx
            trough_value = dd
        elif dd >= 0 and in_drawdown:
            episodes.append(
                {
                    "start_date": start_idx.date().isoformat(),
                    "trough_date": trough_idx.date().isoformat(),
                    "recovery_date": dt_idx.date().isoformat(),
                    "max_drawdown": trough_value,
                    "duration_days": int((dt_idx - start_idx).days),
                    "trough_days": int((trough_idx - start_idx).days),
                }
            )
            in_drawdown = False

    if in_drawdown:
        end_idx = nav.index[-1]
        episodes.append(
            {
                "start_date": start_idx.date().isoformat(),
                "trough_date": trough_idx.date().isoformat(),
                "recovery_date": "",
                "max_drawdown": trough_value,
                "duration_days": int((end_idx - start_idx).days),
                "trough_days": int((trough_idx - start_idx).days),
            }
        )

    if not episodes:
        return pd.DataFrame(columns=["start_date", "trough_date", "recovery_date", "max_drawdown", "duration_days", "trough_days"])
    return pd.DataFrame(episodes).sort_values(["max_drawdown", "duration_days"]).reset_index(drop=True)


def build_monthly_targets(
    prices: pd.DataFrame,
    industry_codes: list[str],
    *,
    max_positions: int,
    absolute_score: float,
    trend_filter: bool,
    fallback: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score, trend_ok, sma120 = compute_composite_score(prices, industry_codes)
    month_end_index = prices.groupby(prices.index.to_period("M")).tail(1).index

    target_weights = pd.DataFrame(0.0, index=month_end_index, columns=prices.columns, dtype="float64")
    snapshot_rows: list[dict[str, object]] = []

    for dt_idx in month_end_index:
        valid_scores = score.loc[dt_idx].dropna().sort_values(ascending=False)
        chosen_codes: list[str] = []
        unit_weight = 1.0 / max_positions

        for code, value in valid_scores.items():
            if float(value) <= absolute_score:
                continue
            if trend_filter and not bool(trend_ok.loc[dt_idx, code]):
                continue
            chosen_codes.append(str(code))
            if len(chosen_codes) >= max_positions:
                break

        if chosen_codes:
            for code in chosen_codes:
                target_weights.loc[dt_idx, code] = unit_weight
        fallback_weight = 1.0 - len(chosen_codes) * unit_weight
        if fallback_weight > 0:
            target_weights.loc[dt_idx, fallback] = fallback_weight

        top_code = chosen_codes[0] if chosen_codes else fallback
        top_score = float(score.loc[dt_idx, top_code]) if top_code in industry_codes and pd.notna(score.loc[dt_idx, top_code]) else float("nan")
        snapshot_rows.append(
            {
                "date": dt_idx,
                "winner_code": top_code,
                "winner_count": len(chosen_codes),
                "winner_score": top_score,
                "chosen_codes": ",".join(chosen_codes) if chosen_codes else fallback,
                "fallback_used": int(len(chosen_codes) == 0),
                **{f"score_{code}": float(score.loc[dt_idx, code]) if pd.notna(score.loc[dt_idx, code]) else float("nan") for code in industry_codes},
                **{f"trend_ok_{code}": int(bool(trend_ok.loc[dt_idx, code])) if pd.notna(trend_ok.loc[dt_idx, code]) else 0 for code in industry_codes},
                **{f"sma120_{code}": float(sma120.loc[dt_idx, code]) if pd.notna(sma120.loc[dt_idx, code]) else float("nan") for code in industry_codes},
            }
        )

    snapshot = pd.DataFrame(snapshot_rows).set_index("date")
    return target_weights, snapshot


def build_daily_target_weights(prices: pd.DataFrame, monthly_targets: pd.DataFrame, analysis_start: pd.Timestamp, fallback: str) -> pd.DataFrame:
    analysis_index = prices.loc[prices.index >= analysis_start].index
    daily_targets = pd.DataFrame(float("nan"), index=analysis_index, columns=prices.columns, dtype="float64")
    first_day_by_month = analysis_index.to_series().groupby(analysis_index.to_period("M")).min()

    previous_month_targets = monthly_targets.loc[monthly_targets.index < analysis_index.min()]
    if not previous_month_targets.empty:
        daily_targets.iloc[0] = previous_month_targets.iloc[-1]
    else:
        daily_targets.iloc[0, daily_targets.columns.get_loc(fallback)] = 1.0

    for rebalance_dt, weights in monthly_targets.iterrows():
        next_period = rebalance_dt.to_period("M") + 1
        if next_period not in first_day_by_month.index:
            continue
        effective_dt = pd.Timestamp(first_day_by_month.loc[next_period])
        if effective_dt not in daily_targets.index:
            continue
        daily_targets.loc[effective_dt] = weights

    return daily_targets.ffill().fillna(0.0)


def build_trades(target_weights: pd.DataFrame, nav: pd.Series) -> pd.DataFrame:
    prev_weights = target_weights.shift(1).fillna(target_weights.iloc[0])
    rows: list[dict[str, object]] = []
    for dt_idx in target_weights.index:
        current = target_weights.loc[dt_idx]
        previous = prev_weights.loc[dt_idx]
        for code in target_weights.columns:
            curr_w = float(current[code])
            prev_w = float(previous[code])
            if abs(curr_w - prev_w) < 1e-12:
                continue
            rows.append(
                {
                    "date": dt_idx,
                    "code": code,
                    "action": "BUY" if curr_w > prev_w else "SELL",
                    "from_weight": prev_w,
                    "to_weight": curr_w,
                    "nav": float(nav.loc[dt_idx]),
                }
            )
    return pd.DataFrame(rows)


def build_holding_series(target_weights: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    code_to_theme = dict(zip(selected["code"].astype(str), selected["theme"].astype(str)))
    holdings: list[str | None] = []
    labels: list[str] = []
    for _, row in target_weights.iterrows():
        active = row[row > 1e-12].sort_values(ascending=False)
        if active.empty:
            holdings.append(pd.NA)
            labels.append("")
            continue
        active_codes = [str(code) for code in active.index]
        holdings.append(active_codes[0])
        labels.append(" / ".join(f"{code_to_theme.get(code, code)} {active.loc[code]:.0%}" for code in active_codes))
    return pd.Series(holdings, index=target_weights.index, name="holding"), pd.Series(labels, index=target_weights.index, name="portfolio")


def run_strategy_variant(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    years: int,
    cost_rate: float,
    name: str,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    industry_codes = [item["code"] for item in INDUSTRY_ETFS]
    analysis_start = compute_analysis_start(prices, years)
    analysis_prices = prices.loc[prices.index >= analysis_start].copy()

    monthly_targets, monthly_snapshot = build_monthly_targets(
        prices,
        industry_codes,
        max_positions=int(config["max_positions"]),
        absolute_score=float(config["absolute_score"]),
        trend_filter=bool(config["trend_filter"]),
        fallback=str(config["fallback"]),
    )
    daily_targets = build_daily_target_weights(prices, monthly_targets, analysis_start, fallback=str(config["fallback"]))
    daily_targets = daily_targets.reindex(analysis_prices.index).ffill().fillna(0.0)

    raw_returns = analysis_prices.pct_change()
    returns = raw_returns.fillna(0.0)
    industry_return = raw_returns[industry_codes].mean(axis=1, skipna=True).fillna(0.0)
    hs300_return = raw_returns[BENCHMARK_ASSET["code"]].fillna(0.0)
    industry_benchmark_nav = (1.0 + industry_return).cumprod()
    industry_benchmark_nav.iloc[0] = 1.0
    hs300_nav = (1.0 + hs300_return).cumprod()
    hs300_nav.iloc[0] = 1.0

    target_turnover = daily_targets.sub(daily_targets.shift(1).fillna(daily_targets.iloc[0])).abs().sum(axis=1)
    trade_cost_rate = target_turnover * cost_rate
    return_weights = daily_targets.shift(1)
    return_weights.iloc[0] = daily_targets.iloc[0]
    gross_return = (return_weights * returns).sum(axis=1)
    strategy_return = (1.0 + gross_return) * (1.0 - trade_cost_rate) - 1.0
    nav = (1.0 + strategy_return).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1.0

    holding, portfolio = build_holding_series(daily_targets, selected)
    latest_winner_codes = monthly_snapshot["chosen_codes"].reindex(analysis_prices.index).ffill().fillna("")
    latest_winner_score = monthly_snapshot["winner_score"].reindex(analysis_prices.index).ffill()

    result = pd.DataFrame(index=analysis_prices.index)
    result["nav"] = nav
    result["strategy_return"] = strategy_return
    result["drawdown"] = drawdown
    result["turnover"] = target_turnover
    result["trade_cost_rate"] = trade_cost_rate
    result["holding"] = holding
    result["portfolio"] = portfolio
    result["signal"] = latest_winner_codes
    result["current_momentum"] = latest_winner_score
    result["max_momentum"] = latest_winner_score
    result["exposure"] = daily_targets[industry_codes].sum(axis=1)
    result["industry_equal_benchmark_nav"] = industry_benchmark_nav.reindex(result.index).ffill()
    result["hs300_benchmark_nav"] = hs300_nav.reindex(result.index).ffill()
    result["industry_equal_benchmark_return"] = industry_return.reindex(result.index).fillna(0.0)
    result["hs300_benchmark_return"] = hs300_return.reindex(result.index).fillna(0.0)
    for code in daily_targets.columns:
        result[f"weight_{code}"] = daily_targets[code]

    trades = build_trades(daily_targets, nav)
    summary = build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)
    summary.update(
        {
            "strategy_key": name,
            "strategy_name": str(config["label"]),
            "analysis_start": analysis_prices.index[0].date().isoformat(),
            "analysis_end": analysis_prices.index[-1].date().isoformat(),
            "industry_benchmark_total_return": float(industry_benchmark_nav.iloc[-1] - 1.0),
            "industry_benchmark_annualized_return": annualized_return_from_nav(industry_benchmark_nav),
            "hs300_total_return": float(hs300_nav.iloc[-1] - 1.0),
            "hs300_annualized_return": annualized_return_from_nav(hs300_nav),
            "latest_signal_codes": str(latest_winner_codes.iloc[-1]),
            "latest_portfolio": str(portfolio.iloc[-1]),
        }
    )

    yearly = build_dual_benchmark_yearly_returns(result)
    drawdown_episodes = build_drawdown_episodes(result["nav"])
    return result, trades, monthly_snapshot, summary, yearly, drawdown_episodes


def plot_nav(result_map: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    first_result = next(iter(result_map.values()))
    ax.plot(first_result.index, first_result["industry_equal_benchmark_nav"], label="行业ETF等权基准", linewidth=1.6, color="#4a5568")
    ax.plot(first_result.index, first_result["hs300_benchmark_nav"], label="沪深300ETF", linewidth=1.4, color="#7c3aed")
    for name, result in result_map.items():
        ax.plot(result.index, result["nav"], label=name, linewidth=2.0)
    ax.set_title("A股行业ETF动量策略净值对比", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, ANALYSIS_OUTPUT_DIR / "nav_vs_benchmarks.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(result_map: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, result in result_map.items():
        ax.plot(result.index, result["drawdown"], label=name, linewidth=1.8)
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("A股行业ETF动量策略回撤", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, ANALYSIS_OUTPUT_DIR / "drawdown_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    ensure_output_dirs()

    selected = build_selected()
    prices = load_prices(args.years)
    selected = selected.copy()
    selected["history_start"] = selected["code"].map(
        lambda code: prices[code].dropna().index.min().date().isoformat() if code in prices.columns and not prices[code].dropna().empty else ""
    )
    price_output = prices.copy().reset_index()
    write_dataframe_csv_atomic(selected, CORE_OUTPUT_DIR / "selected_etfs.csv", index=False)
    write_dataframe_csv_atomic(price_output, CORE_OUTPUT_DIR / "prices.csv", index=False)

    strategy_names = list(STRATEGY_VARIANTS) if args.strategy == "all" else [args.strategy]
    result_map: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    for name in strategy_names:
        result, trades, monthly_snapshot, summary, yearly, drawdown_episodes = run_strategy_variant(
            prices,
            selected,
            years=args.years,
            cost_rate=args.cost_rate,
            name=name,
            config=STRATEGY_VARIANTS[name],
        )
        result_map[name] = result
        summary_rows.append(summary)

        write_dataframe_csv_atomic(result.reset_index(), CORE_OUTPUT_DIR / f"backtest_nav_{name}.csv", index=False)
        write_dataframe_csv_atomic(trades, CORE_OUTPUT_DIR / f"trades_{name}.csv", index=False)
        write_dataframe_csv_atomic(monthly_snapshot.reset_index(), CORE_OUTPUT_DIR / f"monthly_signals_{name}.csv", index=False)
        write_dataframe_csv_atomic(yearly, CORE_OUTPUT_DIR / f"yearly_returns_{name}.csv", index=False)
        write_dataframe_csv_atomic(drawdown_episodes, CORE_OUTPUT_DIR / f"drawdown_episodes_{name}.csv", index=False)
        write_json_atomic(CORE_OUTPUT_DIR / f"summary_{name}.json", summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(["sharpe_rf0", "annualized_return"], ascending=False)
    write_dataframe_csv_atomic(summary_df, CORE_OUTPUT_DIR / "summary.csv", index=False)
    write_json_atomic(CORE_OUTPUT_DIR / "summary.json", summary_df.to_dict("records"))

    plot_nav(result_map)
    plot_drawdown(result_map)


if __name__ == "__main__":
    main()
