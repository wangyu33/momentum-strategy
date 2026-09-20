"""正式策略输出与汇总。"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
from matplotlib import font_manager
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .backtest import (
    build_benchmark_nav,
    build_strategy_summary,
    build_yearly_return_rows,
    count_trade_days,
)
from .config import ANALYSIS_OUTPUT_DIR, CORE_OUTPUT_DIR, DEFAULT_STRATEGY_NAME
from .io import normalize_code, save_figure_atomic, write_dataframe_csv_atomic
from .signals import build_current_etf_momentum_percentile_table


PREFERRED_CJK_FONT_FILES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
]


def configure_matplotlib() -> None:
    for font_path in PREFERRED_CJK_FONT_FILES:
        if not Path(font_path).exists():
            continue
        try:
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = [font_name]
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def count_rebalance_days(
    trades: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> int:
    if trades.empty or "date" not in trades.columns:
        return 0
    trade_dates = pd.to_datetime(trades["date"], errors="coerce").dropna()
    mask = (trade_dates >= start_date) & (trade_dates <= end_date)
    return int(trade_dates.loc[mask].dt.normalize().nunique())


def is_recovered(drawdown_value: float, tol: float = 1e-10) -> bool:
    return abs(float(drawdown_value)) <= tol


def get_latest_portfolio_text(selected: pd.DataFrame, result: pd.DataFrame) -> str:
    if result.empty:
        return ""
    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    latest_row = result.iloc[-1]
    weight_cols = [col for col in result.columns if col.startswith("weight_")]
    allocations: list[tuple[str, float]] = []
    if weight_cols:
        for col in weight_cols:
            code = col.removeprefix("weight_")
            raw_value = latest_row.get(col, 0.0)
            weight = float(raw_value) if pd.notna(raw_value) else 0.0
            if abs(weight) > 1e-12:
                allocations.append((code, weight))
    if not allocations:
        holding = normalize_code(latest_row.get("holding"))
        exposure_raw = latest_row.get("exposure", 1.0)
        exposure = float(exposure_raw) if pd.notna(exposure_raw) else 0.0
        if holding and abs(exposure) > 1e-12:
            allocations.append((holding, exposure))
    if not allocations:
        return "空仓"
    allocations.sort(key=lambda item: item[1], reverse=True)
    parts = []
    for code, weight in allocations:
        theme = code_to_theme.get(code, "")
        name = code_to_name.get(code, "")
        label = f"{theme} / {name} ({code})" if theme else (f"{name} ({code})" if name else code)
        parts.append(f"{label} {weight:.0%}")
    return "；".join(parts)


def build_contribution_summary(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    result: pd.DataFrame,
) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).reindex(result.index)
    prev_nav = result["nav"].shift(1).fillna(1.0)
    total_profit = float(result["nav"].iloc[-1] - 1.0)
    codes = [str(code) for code in selected["code"]]

    weight_cols = [col for col in result.columns if col.startswith("weight_")]
    confirmed_weights = result[weight_cols].copy()
    confirmed_weights.columns = [col.removeprefix("weight_") for col in weight_cols]

    return_weight_cols = [col for col in result.columns if col.startswith("return_weight_")]
    if return_weight_cols:
        return_weights = result[return_weight_cols].copy()
        return_weights.columns = [col.removeprefix("return_weight_") for col in return_weight_cols]
    else:
        return_weights = confirmed_weights.shift(1).fillna(0.0)

    confirmed_weights = confirmed_weights.reindex(columns=codes, fill_value=0.0).fillna(0.0)
    return_weights = return_weights.reindex(columns=codes, fill_value=0.0).fillna(0.0)
    prev_confirmed_weights = confirmed_weights.shift(1).fillna(0.0)
    gross_profit_by_code = (return_weights * returns.reindex(columns=return_weights.columns).fillna(0.0)).mul(prev_nav, axis=0)
    gross_profit_total = gross_profit_by_code.sum(axis=1)
    daily_net_profit_total = prev_nav * result["strategy_return"].fillna(0.0)
    daily_cost_drag_total = daily_net_profit_total - gross_profit_total
    turnover = result["turnover"].fillna(0.0)
    turnover_by_code = (confirmed_weights - prev_confirmed_weights).abs()
    turnover_nonzero = turnover.where(turnover != 0.0, other=float("nan"))
    turnover_share = turnover_by_code.div(turnover_nonzero, axis=0).fillna(0.0)
    cost_drag_by_code = turnover_share.mul(daily_cost_drag_total, axis=0)
    net_profit_by_code = gross_profit_by_code + cost_drag_by_code
    holding_mask = confirmed_weights.abs() > 1e-12
    total_holding_days = int(holding_mask.any(axis=1).sum())

    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    rows: list[dict[str, object]] = []

    for code in selected["code"]:
        code = str(code)
        holding_days = int(holding_mask[code].sum())
        gross_profit = float(gross_profit_by_code[code].sum())
        cost_drag = float(cost_drag_by_code[code].sum())
        net_profit = float(net_profit_by_code[code].sum())
        rows.append(
            {
                "code": code,
                "theme": code_to_theme.get(code, ""),
                "name": code_to_name.get(code, ""),
                "holding_days": holding_days,
                "holding_day_share": holding_days / total_holding_days if total_holding_days else 0.0,
                "gross_profit_contribution": gross_profit,
                "cost_drag_contribution": cost_drag,
                "net_profit_contribution": net_profit,
                "gross_contribution_rate": gross_profit / total_profit if total_profit else 0.0,
                "net_contribution_rate": net_profit / total_profit if total_profit else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("net_profit_contribution", ascending=False).reset_index(drop=True)


def build_contribution_overview(result: pd.DataFrame, contribution_df: pd.DataFrame) -> pd.DataFrame:
    weight_cols = [col for col in result.columns if col.startswith("weight_")]
    active_days = int((result[weight_cols].fillna(0.0).abs().sum(axis=1) > 1e-12).sum())
    return pd.DataFrame(
        [
            {
                "total_profit": float(result["nav"].iloc[-1] - 1.0),
                "total_gross_profit": float(contribution_df["gross_profit_contribution"].sum()),
                "total_cost_profit": float(contribution_df["cost_drag_contribution"].sum()),
                "total_net_profit": float(contribution_df["net_profit_contribution"].sum()),
                "holding_days": active_days,
            }
        ]
    )


def save_contribution_chart(contribution_df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
    chart_df = contribution_df.sort_values("net_profit_contribution", ascending=True)
    labels = [f"{row.theme}({row.code})" for row in chart_df.itertuples(index=False)]
    colors = ["#1f9d55" if value >= 0 else "#d64545" for value in chart_df["net_contribution_rate"]]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(labels, chart_df["net_contribution_rate"], color=colors)
    ax.axvline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("Pool Contribution Rate to Total Return", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Contribution Rate")
    fig.tight_layout()
    save_figure_atomic(fig, output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_episode_holdings(
    nav: pd.DataFrame,
    theme_map: dict[str, str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    tol: float = 1e-10,
) -> str:
    weight_cols = [col for col in nav.columns if col.startswith("weight_")]
    weight_frame = nav.loc[start_date:end_date, weight_cols].copy()
    weight_frame.columns = [col.removeprefix("weight_") for col in weight_cols]
    avg_weights = weight_frame.fillna(0.0).mean().sort_values(ascending=False)
    avg_weights = avg_weights[avg_weights > tol].head(3)
    if not avg_weights.empty:
        return "; ".join(f"{theme_map.get(code, code)}({code}) {share:.1%}" for code, share in avg_weights.items())
    return ""


def build_drawdown_episode_report(
    nav: pd.DataFrame,
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    selected: pd.DataFrame,
    compare: pd.DataFrame,
) -> pd.DataFrame:
    theme_map = dict(zip(selected["code"], selected["theme"]))
    nav_frame = nav.copy()
    nav_frame["holding_code"] = nav_frame["holding"].map(normalize_code)
    drawdown = nav_frame["nav"] / nav_frame["nav"].cummax() - 1
    underwater = drawdown < 0

    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_segment = False
    start: pd.Timestamp | None = None
    for dt_idx, is_underwater in underwater.items():
        if is_underwater and not in_segment:
            in_segment = True
            start = dt_idx
        elif not is_underwater and in_segment and start is not None:
            segments.append((start, dt_idx))
            in_segment = False
            start = None
    if in_segment and start is not None:
        segments.append((start, nav_frame.index[-1]))

    rows: list[dict[str, object]] = []
    for start_date, end_date in segments:
        segment_dd = drawdown.loc[start_date:end_date]
        trough_date = segment_dd.idxmin()
        peak_date = nav_frame.loc[:start_date, "nav"].idxmax()
        peak_nav = float(nav_frame.loc[peak_date, "nav"])
        trough_nav = float(nav_frame.loc[trough_date, "nav"])
        recovered = is_recovered(float(drawdown.loc[end_date]))

        top_holdings_text = summarize_episode_holdings(nav_frame, theme_map, peak_date, end_date)
        asset_rets = (prices.loc[trough_date] / prices.loc[peak_date] - 1).dropna().sort_values()
        worst_assets_text = "; ".join(f"{theme_map.get(code, code)}({code}) {ret:.1%}" for code, ret in asset_rets.head(3).items())
        trade_action_count = int(
            (
                (pd.to_datetime(trades["date"], errors="coerce") >= peak_date)
                & (pd.to_datetime(trades["date"], errors="coerce") <= end_date)
            ).sum()
        ) if not trades.empty else 0
        switches = count_rebalance_days(trades, peak_date, end_date)
        peak_holding = normalize_code(nav_frame.loc[peak_date, "holding"])
        trough_holding = normalize_code(nav_frame.loc[trough_date, "holding"])
        recovery_holding = normalize_code(nav_frame.loc[end_date, "holding"])

        rows.append(
            {
                "peak_date": peak_date.date().isoformat(),
                "start_date": start_date.date().isoformat(),
                "trough_date": trough_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "peak_to_trough_days": int(nav_frame.loc[peak_date:trough_date].shape[0] - 1),
                "recovery_days": int(nav_frame.loc[trough_date:end_date].shape[0] - 1) if recovered else "",
                "total_days": int(nav_frame.loc[peak_date:end_date].shape[0] - 1),
                "max_drawdown": float(segment_dd.min()),
                "strategy_peak_to_trough_return": float(trough_nav / peak_nav - 1),
                "hs300_peak_to_trough_return": float(compare.loc[trough_date, "benchmark_nav"] / compare.loc[peak_date, "benchmark_nav"] - 1),
                "peak_holding": f"{theme_map.get(peak_holding, peak_holding)}({peak_holding})" if peak_holding else "",
                "trough_holding": f"{theme_map.get(trough_holding, trough_holding)}({trough_holding})" if trough_holding else "",
                "recovery_holding": f"{theme_map.get(recovery_holding, recovery_holding)}({recovery_holding})" if recovery_holding else "",
                "switches_in_episode": switches,
                "trade_actions_in_episode": trade_action_count,
                "top_holdings": top_holdings_text,
                "worst_assets_peak_to_trough": worst_assets_text,
                "recovered": recovered,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("max_drawdown").reset_index(drop=True)


def save_analysis_drawdown_charts(result: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
    drawdown = result["nav"] / result["nav"].cummax() - 1

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(drawdown.index, drawdown, color="#d64545", linewidth=1.8)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("历史回撤相对净值高点", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    save_figure_atomic(fig, ANALYSIS_OUTPUT_DIR / "historical_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(result.index, result["nav"], color="#2b6cb0", linewidth=2.0)
    ax1.set_title("策略净值与回撤", loc="left", fontsize=16, fontweight="bold")
    ax1.set_ylabel("净值")
    ax2.plot(drawdown.index, drawdown, color="#d64545", linewidth=1.8)
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax2.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax2.set_xlabel("日期")
    ax2.set_ylabel("回撤")
    ax2.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    save_figure_atomic(fig, ANALYSIS_OUTPUT_DIR / "nav_and_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_analysis_outputs(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    compare_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
) -> None:
    contribution_overview = build_contribution_overview(result, contribution_df)
    write_dataframe_csv_atomic(contribution_df, ANALYSIS_OUTPUT_DIR / "contribution_summary.csv", index=False)
    write_dataframe_csv_atomic(contribution_overview, ANALYSIS_OUTPUT_DIR / "contribution_overview.csv", index=False)
    save_contribution_chart(contribution_df, ANALYSIS_OUTPUT_DIR / "contribution_rate.png")

    drawdown_report = build_drawdown_episode_report(result, prices, trades, selected, compare_df)
    write_dataframe_csv_atomic(drawdown_report, ANALYSIS_OUTPUT_DIR / "drawdown_episodes.csv", index=False)
    save_analysis_drawdown_charts(result)


def save_outputs(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    lookback: int,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
) -> None:
    prices_to_save = prices.copy()
    prices_to_save.index.name = "date"
    result_to_save = result.copy()
    result_to_save.index.name = "date"
    write_dataframe_csv_atomic(selected, CORE_OUTPUT_DIR / "selected_etfs.csv", index=False)
    write_dataframe_csv_atomic(prices_to_save, CORE_OUTPUT_DIR / "prices.csv")
    write_dataframe_csv_atomic(result_to_save, CORE_OUTPUT_DIR / "backtest_nav.csv")
    write_dataframe_csv_atomic(trades, CORE_OUTPUT_DIR / "trades.csv", index=False)
    write_dataframe_csv_atomic(result_to_save[["nav"]].rename(columns={"nav": "historical_nav"}), CORE_OUTPUT_DIR / "historical_nav.csv")
    percentile_snapshot = build_current_etf_momentum_percentile_table(selected, prices, lookback=lookback)
    write_dataframe_csv_atomic(percentile_snapshot, CORE_OUTPUT_DIR / "etf_momentum_percentiles.csv", index=False)

    buy_trades = trades[trades["action"] == "BUY"]
    sell_trades = trades[trades["action"] == "SELL"]
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result.index, result["nav"], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    if not buy_trades.empty:
        ax.scatter(buy_trades["date"], buy_trades["nav"], color="#1f9d55", marker="^", s=55, label="Buy", zorder=3)
    if not sell_trades.empty:
        ax.scatter(sell_trades["date"], sell_trades["nav"], color="#d64545", marker="v", s=55, label="Sell", zorder=3)
    ax.set_title(f"ETF Strategy NAV ({strategy_name}, lookback={lookback})", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / "nav_with_trades.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result.index, result["current_momentum"], color="#dd6b20", linewidth=2.0)
    ax.axhline(0.0, color="#4a5568", linestyle="--", linewidth=1.0)
    ax.set_title(f"Signal Momentum (trailing {lookback} trading days)", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Momentum")
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / "momentum_series.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    drawdown = result["drawdown"]
    max_dd_date = drawdown.idxmin()
    max_dd_value = float(drawdown.loc[max_dd_date])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(drawdown.index, drawdown, color="#d64545", linewidth=1.8)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax.scatter([max_dd_date], [max_dd_value], color="#9b2c2c", s=70, zorder=3)
    ax.annotate(f"Max DD {max_dd_value:.2%}\n{max_dd_date.date()}", xy=(max_dd_date, max_dd_value), xytext=(12, -8), textcoords="offset points", fontsize=10, color="#742a2a", ha="left", va="top")
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("Historical Drawdown vs Running Peak", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / "max_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")
    compare_df = pd.concat([result["nav"], benchmark_nav], axis=1)
    write_dataframe_csv_atomic(compare_df, CORE_OUTPUT_DIR / "strategy_vs_hs300.csv")
    rows = build_yearly_return_rows(compare_df, strategy_col="nav", benchmark_col="benchmark_nav", benchmark_return_col="hs300_etf_return")
    write_dataframe_csv_atomic(pd.DataFrame(rows), CORE_OUTPUT_DIR / "yearly_returns.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["nav"], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    ax.plot(compare_df.index, compare_df["benchmark_nav"], color="#d64545", linewidth=2.0, label="HS300 Benchmark")
    ax.set_title("Strategy vs HS300 Benchmark", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / "strategy_vs_hs300.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    contribution_df = build_contribution_summary(prices, selected, result)
    write_dataframe_csv_atomic(contribution_df, CORE_OUTPUT_DIR / "contribution_summary.csv", index=False)
    save_contribution_chart(contribution_df, CORE_OUTPUT_DIR / "contribution_rate.png")
    save_contribution_chart(contribution_df, CORE_OUTPUT_DIR / "contribution_rate_cn.png")
    save_analysis_outputs(selected, prices, result, trades, compare_df, contribution_df)


def print_summary(
    selected: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    lookback: int,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
) -> None:
    summary = {
        "strategy": strategy_name,
        "lookback_days": lookback,
        **build_strategy_summary(
            result,
            trades,
            selected=selected,
            include_max_drawdown_integral=True,
            get_latest_portfolio_text=get_latest_portfolio_text,
        ),
        "avg_trade_cost_rate": float(result["trade_cost_rate"][result["trade_cost_rate"] > 0].mean() if (result["trade_cost_rate"] > 0).any() else 0.0),
    }

    print("Fixed ETF pool:")
    print(selected.to_string(index=False))
    print("\nBacktest summary:")
    for key, value in summary.items():
        if isinstance(value, float) and math.isfinite(value):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
