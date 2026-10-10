"""正式策略输出与汇总。"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.lines import Line2D
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
    *,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).reindex(result.index)
    prev_nav = result["nav"].shift(1).fillna(1.0)
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
    if window_start is not None or window_end is not None:
        mask = pd.Series(True, index=result.index)
        if window_start is not None:
            mask &= result.index >= window_start
        if window_end is not None:
            mask &= result.index <= window_end
        gross_profit_by_code = gross_profit_by_code.loc[mask]
        cost_drag_by_code = cost_drag_by_code.loc[mask]
        net_profit_by_code = net_profit_by_code.loc[mask]
        daily_net_profit_total = daily_net_profit_total.loc[mask]
        confirmed_weights = confirmed_weights.loc[mask]
    total_profit = float(daily_net_profit_total.sum())
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


def build_holding_style_maps(selected: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    palette = [
        "#2b6cb0",
        "#d64545",
        "#1f9d55",
        "#805ad5",
        "#dd6b20",
        "#319795",
        "#b7791f",
        "#718096",
        "#c05621",
        "#2f855a",
    ]
    codes = [str(code) for code in selected["code"]]
    color_map = {code: palette[idx % len(palette)] for idx, code in enumerate(codes)}
    color_map["CASH"] = "#a0aec0"
    label_map = {str(row.code): str(row.theme) for row in selected.itertuples(index=False)}
    label_map["CASH"] = "空仓"
    return color_map, label_map


def plot_nav_with_exposure_panels(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    title: str,
    date_label: str,
    output_path: Path,
) -> None:
    color_map, label_map = build_holding_style_maps(selected)
    working = frame.copy()
    exposure_series = pd.to_numeric(working.get("exposure", 0.0), errors="coerce").fillna(0.0)
    holding_series = working.get("holding", pd.Series(index=working.index, dtype="object")).map(normalize_code)
    holding_state = holding_series.where(exposure_series.abs() > 1e-12, "CASH").fillna("CASH")

    fig, (ax_nav, ax_exp) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )
    states = holding_state.tolist()
    shown_codes: set[str] = set()
    segment_start = 0
    for idx in range(1, len(working) + 1):
        is_break = idx == len(working) or states[idx] != states[segment_start]
        if not is_break:
            continue
        code = states[segment_start]
        start_idx = max(segment_start - 1, 0)
        segment = working.iloc[start_idx:idx]
        ax_nav.plot(
            segment.index,
            segment["nav"],
            color=color_map.get(code, "#4a5568"),
            linewidth=2.3,
            solid_capstyle="round",
        )
        shown_codes.add(code)
        segment_start = idx
    ax_nav.set_title(title, loc="left", fontsize=16, fontweight="bold")
    ax_nav.set_ylabel("Net Value")

    weight_cols = [col for col in working.columns if col.startswith("weight_")]
    exposure_frame = pd.DataFrame(index=working.index)
    for col in weight_cols:
        code = col.removeprefix("weight_")
        exposure_frame[code] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)
    if exposure_frame.empty:
        exposure_frame["CASH"] = exposure_series.where(exposure_series <= 1e-12, 0.0)
    else:
        exposure_frame = exposure_frame.reindex(columns=[str(code) for code in selected["code"]], fill_value=0.0)
        cash_series = (1.0 - exposure_frame.sum(axis=1)).clip(lower=0.0)
        exposure_frame["CASH"] = cash_series
    nonzero_cols = [col for col in exposure_frame.columns if float(exposure_frame[col].abs().sum()) > 1e-10]
    if not nonzero_cols:
        nonzero_cols = ["CASH"]
        exposure_frame = pd.DataFrame({"CASH": pd.Series(1.0, index=working.index)})
    plot_cols = nonzero_cols
    for code in plot_cols:
        ax_exp.step(
            working.index,
            exposure_frame[code] * 100.0,
            where="post",
            color=color_map.get(code, "#4a5568"),
            linewidth=1.8,
            label=label_map.get(code, code),
        )
    ax_exp.set_ylabel("仓位%")
    ax_exp.set_xlabel(date_label)
    ax_exp.set_ylim(0, 105)
    ax_exp.axhline(0.0, color="#4a5568", linewidth=1.0)
    legend_handles = [
        Line2D([0], [0], color=color_map.get(code, "#4a5568"), lw=2.2, label=label_map.get(code, code))
        for code in plot_cols
    ]
    ax_nav.legend(handles=legend_handles, ncol=min(4, len(plot_cols)), fontsize=8, loc="upper left")

    fig.tight_layout()
    save_figure_atomic(fig, output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_recent_contribution_outputs(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    result: pd.DataFrame,
    *,
    months: int = 6,
) -> None:
    recent_result = build_recent_window_frame(result, months=months)
    if recent_result.empty:
        return
    recent_start = pd.Timestamp(recent_result.index.min()).normalize()
    recent_end = pd.Timestamp(recent_result.index.max()).normalize()
    contribution_df = build_contribution_summary(
        prices,
        selected,
        result,
        window_start=recent_start,
        window_end=recent_end,
    )
    if contribution_df.empty:
        return
    write_dataframe_csv_atomic(
        contribution_df,
        CORE_OUTPUT_DIR / f"recent_{months}m_contribution_summary.csv",
        index=False,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
    chart_df = contribution_df.sort_values("net_profit_contribution", ascending=True)
    labels = [f"{row.theme}({row.code})" for row in chart_df.itertuples(index=False)]
    colors = ["#1f9d55" if value >= 0 else "#d64545" for value in chart_df["net_contribution_rate"]]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(labels, chart_df["net_contribution_rate"], color=colors)
    ax.axvline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title(f"Pool Contribution Rate to Recent {months}M Return", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Contribution Rate")
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / f"recent_{months}m_contribution_rate.png", dpi=180, bbox_inches="tight")
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


def build_recent_window_frame(frame: pd.DataFrame, *, months: int = 3) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    latest_date = pd.Timestamp(frame.index.max()).normalize()
    start_date = (latest_date - pd.DateOffset(months=months)).normalize()
    recent = frame.loc[frame.index >= start_date].copy()
    return recent if not recent.empty else frame.copy()


def save_recent_core_charts(
    selected: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    lookback: int,
    strategy_name: str,
    months: int = 6,
) -> None:
    recent_result = build_recent_window_frame(result, months=months)
    if recent_result.empty:
        return

    recent_start = pd.Timestamp(recent_result.index.min()).date().isoformat()
    recent_end = pd.Timestamp(recent_result.index.max()).date().isoformat()
    recent_trades = trades.copy()
    if not recent_trades.empty and "date" in recent_trades.columns:
        trade_dates = pd.to_datetime(recent_trades["date"], errors="coerce")
        recent_trades = recent_trades.loc[trade_dates >= recent_result.index.min()].copy()

    buy_trades = recent_trades[recent_trades["action"] == "BUY"] if not recent_trades.empty else pd.DataFrame()
    sell_trades = recent_trades[recent_trades["action"] == "SELL"] if not recent_trades.empty else pd.DataFrame()

    plot_nav_with_exposure_panels(
        recent_result,
        selected,
        title=(
        f"Recent {months}M NAV With Trades ({strategy_name}, lookback={lookback})",
        ),
        date_label=f"Date ({recent_start} -> {recent_end})",
        output_path=CORE_OUTPUT_DIR / f"recent_{months}m_nav_with_trades.png",
    )

    drawdown = recent_result["nav"] / recent_result["nav"].cummax() - 1
    max_dd_date = drawdown.idxmin()
    max_dd_value = float(drawdown.loc[max_dd_date])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(drawdown.index, drawdown, color="#d64545", linewidth=1.8)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax.scatter([max_dd_date], [max_dd_value], color="#9b2c2c", s=70, zorder=3)
    ax.annotate(
        f"Max DD {max_dd_value:.2%}\n{max_dd_date.date()}",
        xy=(max_dd_date, max_dd_value),
        xytext=(12, -8),
        textcoords="offset points",
        fontsize=10,
        color="#742a2a",
        ha="left",
        va="top",
    )
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title(f"Recent {months}M Drawdown vs Running Peak", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"Date ({recent_start} -> {recent_end})")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / f"recent_{months}m_max_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    compare_codes = ["510300", "159941"]
    compare_labels = {
        "510300": "HS300 Benchmark",
        "159941": "Nasdaq ETF 159941",
    }
    compare_colors = {
        "510300": "#d64545",
        "159941": "#805ad5",
    }
    recent_prices = prices.reindex(recent_result.index)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(recent_result.index, recent_result["nav"] / recent_result["nav"].iloc[0], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    for code in compare_codes:
        if code not in recent_prices.columns:
            continue
        series = pd.to_numeric(recent_prices[code], errors="coerce").dropna()
        if series.empty:
            continue
        normalized = series / float(series.iloc[0])
        ax.plot(normalized.index, normalized.values, color=compare_colors[code], linewidth=2.0, label=compare_labels[code])
    ax.set_title(
        f"Recent {months}M Strategy vs HS300 vs Nasdaq",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel(f"Date ({recent_start} -> {recent_end})")
    ax.set_ylabel("Normalized NAV")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / f"recent_{months}m_strategy_vs_hs300_nasdaq.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


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

    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()

    plot_nav_with_exposure_panels(
        result,
        selected,
        title=f"ETF Strategy NAV ({strategy_name}, lookback={lookback})",
        date_label="Date",
        output_path=CORE_OUTPUT_DIR / "nav_with_trades.png",
    )

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
    save_recent_core_charts(selected, result, trades, prices, lookback=lookback, strategy_name=strategy_name, months=6)
    save_recent_contribution_outputs(prices, selected, result, months=6)
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
