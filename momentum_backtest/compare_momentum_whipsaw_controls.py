#!/usr/bin/env python3
"""对比 25 日动量边界噪音的三种防抖方案。"""

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

from compare_hs300_regime_fixes import run_target_weights_strategy
from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    DEFENSIVE_CODES,
    RESEARCH_OUTPUT_DIR,
    RISK_CODES,
    build_benchmark_nav,
    build_strategy_summary,
    choose_signal_winner_with_margin,
    ensure_output_dirs,
    fetch_histories,
    load_core_selected_and_prices,
    load_fixed_etf_pool,
    normalize_code,
    resolve_strategy_universe,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "momentum_whipsaw_controls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 25 日动量边界噪音的三种防抖方案。")
    parser.add_argument("--years", type=int, default=15, help="刷新数据时拉取最近多少年。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--leader-margin", type=float, default=0.005, help="领先阈值，默认 0.5%。")
    parser.add_argument("--confirmation-days", type=int, default=2, help="连续确认天数。")
    parser.add_argument("--anchor-window", type=int, default=3, help="平滑锚点窗口，默认取 25 日前后 3 日均价。")
    parser.add_argument("--transition-start", type=float, default=0.0, help="连续映射起始动量，默认 0%。")
    parser.add_argument("--transition-end", type=float, default=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD, help="连续映射结束动量，默认等于绝对动量阈值。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--absolute-threshold", type=float, default=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD, help="风险资产绝对动量阈值。")
    parser.add_argument("--weak-trend-defensive-weight", type=float, default=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT, help="风险趋势偏弱但仍为正时的防守仓位。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取历史数据，而不是复用 output/core 缓存。")
    return parser.parse_args()


def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_core_selected_and_prices()


def build_raw_momentum(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return prices / prices.shift(lookback) - 1


def build_smoothed_anchor_momentum(prices: pd.DataFrame, lookback: int, anchor_window: int) -> pd.DataFrame:
    if anchor_window < 1:
        raise ValueError("anchor_window must be at least 1")
    half = anchor_window // 2
    anchor_components = [prices.shift(lookback + offset) for offset in range(-half, half + 1)]
    anchor = pd.concat(anchor_components, axis=0).groupby(level=0).mean()
    return prices / anchor - 1


def build_target_weights_from_signal(
    prices: pd.DataFrame,
    signal: pd.Series,
    exposure: pd.Series,
) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype="float64")
    normalized_signal = signal.map(normalize_code)
    exposure = exposure.reindex(prices.index).fillna(0.0)
    for code in prices.columns:
        mask = normalized_signal == code
        if mask.any():
            weights.loc[mask, code] = exposure.loc[mask].astype(float)
    return weights


def build_signal_asset_momentum(momentum: pd.DataFrame, signal: pd.Series) -> pd.Series:
    signal_momentum = pd.Series(index=momentum.index, dtype="float64", name="current_momentum")
    normalized_signal = signal.map(normalize_code)
    for dt_idx in momentum.index:
        code = normalized_signal.loc[dt_idx]
        if code is None or code not in momentum.columns:
            signal_momentum.loc[dt_idx] = float("nan")
            continue
        value = momentum.loc[dt_idx, code]
        signal_momentum.loc[dt_idx] = float(value) if pd.notna(value) else float("nan")
    return signal_momentum


def build_dual_momentum_signal(
    prices: pd.DataFrame,
    momentum: pd.DataFrame,
    *,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    leader_margin: float = 0.0,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    risk_score = momentum[active_risk_codes]
    defensive_score = momentum[active_defensive_codes]

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    prev_signal: str | None = None

    for dt_idx in prices.index:
        prev_risk = prev_signal if prev_signal in active_risk_codes else None
        prev_def = prev_signal if prev_signal in active_defensive_codes else None
        r_asset = choose_signal_winner_with_margin(risk_score.loc[dt_idx], prev_risk, leader_margin)
        d_asset = choose_signal_winner_with_margin(defensive_score.loc[dt_idx], prev_def, leader_margin)
        r_score = float(momentum.loc[dt_idx, r_asset]) if r_asset and pd.notna(momentum.loc[dt_idx, r_asset]) else float("nan")
        d_score = float(momentum.loc[dt_idx, d_asset]) if d_asset and pd.notna(momentum.loc[dt_idx, d_asset]) else float("nan")

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            prev_signal = None
        elif pd.notna(r_score) and r_score > absolute_threshold:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            prev_signal = str(r_asset) if r_asset else None
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = weak_trend_defensive_weight
            prev_signal = str(d_asset) if d_asset else None
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            prev_signal = str(d_asset) if d_asset else None
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            prev_signal = None

    signal_momentum = build_signal_asset_momentum(momentum, signal)
    return signal, target_exposure, signal_momentum


def apply_switch_confirmation(
    signal: pd.Series,
    target_exposure: pd.Series,
    momentum: pd.DataFrame,
    confirmation_days: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if confirmation_days <= 1:
        confirmed_signal = signal.copy()
        confirmed_exposure = target_exposure.copy()
        confirmed_momentum = build_signal_asset_momentum(momentum, confirmed_signal)
        return confirmed_signal, confirmed_exposure, confirmed_momentum

    confirmed_signal = pd.Series(index=signal.index, dtype="object", name="signal")
    confirmed_exposure = pd.Series(index=signal.index, dtype="float64", name="target_exposure")
    pending_signal: str | None = None
    pending_exposure: float | None = None
    pending_streak = 0

    current_signal: str | None = None
    current_exposure = 0.0

    for dt_idx in signal.index:
        desired_signal = normalize_code(signal.loc[dt_idx])
        desired_exposure = float(target_exposure.loc[dt_idx]) if pd.notna(target_exposure.loc[dt_idx]) else 0.0

        if current_signal is None:
            current_signal = desired_signal
            current_exposure = desired_exposure
            pending_signal = None
            pending_exposure = None
            pending_streak = 0
        elif desired_signal == current_signal:
            current_exposure = desired_exposure
            pending_signal = None
            pending_exposure = None
            pending_streak = 0
        else:
            if desired_signal == pending_signal and pending_exposure is not None and abs(desired_exposure - pending_exposure) < 1e-12:
                pending_streak += 1
            else:
                pending_signal = desired_signal
                pending_exposure = desired_exposure
                pending_streak = 1
            if pending_streak >= confirmation_days:
                current_signal = pending_signal
                current_exposure = float(pending_exposure) if pending_exposure is not None else 0.0
                pending_signal = None
                pending_exposure = None
                pending_streak = 0

        confirmed_signal.loc[dt_idx] = current_signal if current_signal is not None else pd.NA
        confirmed_exposure.loc[dt_idx] = current_exposure

    confirmed_momentum = build_signal_asset_momentum(momentum, confirmed_signal)
    return confirmed_signal, confirmed_exposure, confirmed_momentum


def run_variant(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    momentum: pd.DataFrame,
    signal: pd.Series,
    target_exposure: pd.Series,
    signal_momentum: pd.Series,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_weights = build_target_weights_from_signal(prices, signal, target_exposure)
    return run_target_weights_strategy(
        prices=prices,
        selected=selected,
        target_weights=target_weights,
        signal_momentum=signal_momentum,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )


def build_continuous_blend_variant(
    prices: pd.DataFrame,
    momentum: pd.DataFrame,
    *,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    transition_start: float,
    transition_end: float,
    leader_margin: float = 0.0,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if transition_end <= transition_start:
        raise ValueError("transition_end must be greater than transition_start")

    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    risk_score = momentum[active_risk_codes]
    defensive_score = momentum[active_defensive_codes]
    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype="float64")
    signal_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    prev_signal: str | None = None

    for dt_idx in prices.index:
        prev_risk = prev_signal if prev_signal in active_risk_codes else None
        prev_def = prev_signal if prev_signal in active_defensive_codes else None
        r_asset = choose_signal_winner_with_margin(risk_score.loc[dt_idx], prev_risk, leader_margin)
        d_asset = choose_signal_winner_with_margin(defensive_score.loc[dt_idx], prev_def, leader_margin)
        r_score = float(momentum.loc[dt_idx, r_asset]) if r_asset and pd.notna(momentum.loc[dt_idx, r_asset]) else float("nan")
        d_score = float(momentum.loc[dt_idx, d_asset]) if d_asset and pd.notna(momentum.loc[dt_idx, d_asset]) else float("nan")

        if pd.isna(r_score) and pd.isna(d_score):
            signal_momentum.loc[dt_idx] = float("nan")
            prev_signal = None
            continue

        if pd.notna(r_score) and r_score > absolute_threshold and r_asset is not None:
            target_weights.loc[dt_idx, str(r_asset)] = 1.0
            signal_momentum.loc[dt_idx] = float(r_score)
            prev_signal = str(r_asset)
            continue

        if pd.notna(r_score) and r_score > transition_start and r_asset is not None:
            blend_ratio = (float(r_score) - transition_start) / (transition_end - transition_start)
            blend_ratio = min(max(blend_ratio, 0.0), 1.0)
            risk_weight = blend_ratio
            defensive_weight = 0.0
            if pd.notna(d_asset):
                defensive_weight = (1.0 - blend_ratio) * weak_trend_defensive_weight
                target_weights.loc[dt_idx, str(d_asset)] = defensive_weight
            target_weights.loc[dt_idx, str(r_asset)] = risk_weight
            total_weight = risk_weight + defensive_weight
            if total_weight > 1e-12:
                weighted_momentum = risk_weight * float(r_score)
                if pd.notna(d_score):
                    weighted_momentum += defensive_weight * float(d_score)
                signal_momentum.loc[dt_idx] = weighted_momentum / total_weight
            else:
                signal_momentum.loc[dt_idx] = float("nan")
            prev_signal = str(r_asset) if risk_weight >= defensive_weight else (str(d_asset) if pd.notna(d_asset) else str(r_asset))
            continue

        if pd.notna(d_asset):
            target_weights.loc[dt_idx, str(d_asset)] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            signal_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
            prev_signal = str(d_asset)
            continue

        signal_momentum.loc[dt_idx] = float("nan")
        prev_signal = None

    return target_weights, signal_momentum


def summarize_strategy(
    strategy: str,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, object]:
    row = build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)
    signal = result["signal"].map(normalize_code)
    signal_switch_days = int(signal.fillna("").ne(signal.shift(1).fillna("")).sum() - 1)
    row.update(
        {
            "strategy": strategy,
            "signal_switch_days": signal_switch_days,
            "total_turnover": float(result["turnover"].fillna(0.0).sum()),
            "avg_daily_turnover": float(result["turnover"].fillna(0.0).mean()),
        }
    )
    return row


def main() -> int:
    args = parse_args()
    if args.refresh:
        selected = load_fixed_etf_pool()
        prices = fetch_histories(selected, years=args.years)
    else:
        selected, prices = load_cached_data()

    selected = selected[selected["code"].astype(str).isin(prices.columns)].copy()
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    defensive_codes = [code for code in DEFENSIVE_CODES if code in prices.columns]
    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")

    raw_momentum = build_raw_momentum(prices, args.lookback)
    base_signal, base_exposure, base_signal_momentum = build_dual_momentum_signal(
        prices,
        raw_momentum,
        absolute_threshold=args.absolute_threshold,
        weak_trend_defensive_weight=args.weak_trend_defensive_weight,
        leader_margin=0.0,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    margin_signal, margin_exposure, margin_signal_momentum = build_dual_momentum_signal(
        prices,
        raw_momentum,
        absolute_threshold=args.absolute_threshold,
        weak_trend_defensive_weight=args.weak_trend_defensive_weight,
        leader_margin=args.leader_margin,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    confirm_signal, confirm_exposure, confirm_signal_momentum = apply_switch_confirmation(
        base_signal,
        base_exposure,
        raw_momentum,
        confirmation_days=args.confirmation_days,
    )
    smooth_momentum = build_smoothed_anchor_momentum(prices, args.lookback, args.anchor_window)
    smooth_signal, smooth_exposure, smooth_signal_momentum = build_dual_momentum_signal(
        prices,
        smooth_momentum,
        absolute_threshold=args.absolute_threshold,
        weak_trend_defensive_weight=args.weak_trend_defensive_weight,
        leader_margin=0.0,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    continuous_weights, continuous_signal_momentum = build_continuous_blend_variant(
        prices,
        raw_momentum,
        absolute_threshold=args.absolute_threshold,
        weak_trend_defensive_weight=args.weak_trend_defensive_weight,
        transition_start=args.transition_start,
        transition_end=args.transition_end,
        leader_margin=0.0,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )

    strategy_runs = [
        (
            "baseline_raw25",
            "signal",
            raw_momentum,
            base_signal,
            base_exposure,
            base_signal_momentum,
        ),
        (
            f"leader_margin_{int(round(args.leader_margin * 10000)):02d}bp",
            "signal",
            raw_momentum,
            margin_signal,
            margin_exposure,
            margin_signal_momentum,
        ),
        (
            f"confirm_{args.confirmation_days}d",
            "signal",
            raw_momentum,
            confirm_signal,
            confirm_exposure,
            confirm_signal_momentum,
        ),
        (
            f"smooth_anchor_avg{args.anchor_window}",
            "signal",
            smooth_momentum,
            smooth_signal,
            smooth_exposure,
            smooth_signal_momentum,
        ),
        (
            f"continuous_blend_{int(round(args.transition_start * 100)):02d}_{int(round(args.transition_end * 100)):02d}",
            "weights",
            raw_momentum,
            continuous_weights,
            None,
            continuous_signal_momentum,
        ),
    ]

    summary_rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame(index=prices.index)
    latest_rows: list[dict[str, object]] = []

    for strategy_name, run_kind, momentum, signal_or_weights, exposure, signal_momentum in strategy_runs:
        if run_kind == "signal":
            result, trades = run_variant(
                prices,
                selected,
                momentum=momentum,
                signal=signal_or_weights,
                target_exposure=exposure,
                signal_momentum=signal_momentum,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
            )
        else:
            result, trades = run_target_weights_strategy(
                prices=prices,
                selected=selected,
                target_weights=signal_or_weights,
                signal_momentum=signal_momentum,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
            )
        summary_rows.append(summarize_strategy(strategy_name, result, trades, selected))
        nav_compare[f"{strategy_name}_nav"] = result["nav"]
        latest_rows.append(
            {
                "strategy": strategy_name,
                "date": str(result.index[-1].date()),
                "signal": normalize_code(result["signal"].iloc[-1]) or "",
                "holding": normalize_code(result["holding"].iloc[-1]) or "",
                "exposure": float(result["exposure"].iloc[-1]),
                "current_momentum": float(result["current_momentum"].iloc[-1]) if pd.notna(result["current_momentum"].iloc[-1]) else float("nan"),
                "nav": float(result["nav"].iloc[-1]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    baseline_row = summary[summary["strategy"] == "baseline_raw25"].iloc[0]
    for column in [
        "total_return",
        "annualized_return",
        "sharpe_rf0",
        "max_drawdown",
        "trade_count",
        "signal_switch_days",
        "total_turnover",
    ]:
        summary[f"{column}_diff"] = summary[column] - baseline_row[column]
    nav_compare["hs300_benchmark"] = benchmark_nav
    latest_df = pd.DataFrame(latest_rows)

    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(summary, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(nav_compare, OUTPUT_DIR / "nav_compare.csv")
    write_dataframe_csv_atomic(latest_df, OUTPUT_DIR / "latest_snapshot.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(nav_compare.index, nav_compare["baseline_raw25_nav"], linewidth=2.2, label="Baseline Raw25")
    ax.plot(nav_compare.index, nav_compare[f"leader_margin_{int(round(args.leader_margin * 10000)):02d}bp_nav"], linewidth=1.8, label=f"Leader Margin {args.leader_margin:.1%}")
    ax.plot(nav_compare.index, nav_compare[f"confirm_{args.confirmation_days}d_nav"], linewidth=1.8, label=f"Confirm {args.confirmation_days}D")
    ax.plot(nav_compare.index, nav_compare[f"smooth_anchor_avg{args.anchor_window}_nav"], linewidth=1.8, label=f"Smooth Anchor Avg{args.anchor_window}")
    ax.plot(nav_compare.index, nav_compare[f"continuous_blend_{int(round(args.transition_start * 100)):02d}_{int(round(args.transition_end * 100)):02d}_nav"], linewidth=1.8, label="Continuous Blend")
    ax.plot(nav_compare.index, nav_compare["hs300_benchmark"], linewidth=1.4, linestyle="--", label="HS300")
    ax.set_title("Momentum Whipsaw Control Comparison", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "nav_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    print(latest_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
