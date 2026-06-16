#!/usr/bin/env python3
"""Compare defensive-to-risk probe reentry variants against the current default baseline."""

from __future__ import annotations

import argparse

try:
    from ..runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from run_backtest import (
    CORE_OUTPUT_DIR,
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
    DEFAULT_OVERHEAT_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    DEFENSIVE_CODES,
    RESEARCH_OUTPUT_DIR,
    RISK_CODES,
    annualized_return,
    ensure_output_dirs,
    max_drawdown,
    normalize_code,
    run_threshold_dual_with_overheat_cap_strategy,
)


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_defensive_reentry_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare defensive-to-risk probe reentry variants.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--probe-exposures", default="0.5,0.6,0.7", help="Comma-separated exposures for first probe days.")
    parser.add_argument("--probe-days", default="1,2,3", help="Comma-separated day counts to keep probe exposure.")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(CORE_OUTPUT_DIR / "selected_etfs.csv", dtype={"code": str})
    prices = pd.read_csv(CORE_OUTPUT_DIR / "prices.csv", parse_dates=["date"]).set_index("date")
    prices.columns = [normalize_code(col) or str(col) for col in prices.columns]
    return selected, prices


def build_trades(result: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    holding = result["holding"]
    exposure = result["exposure"].fillna(0.0)
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)
    trades: list[dict[str, object]] = []

    for dt_idx in result.index:
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

    return pd.DataFrame(trades)


def run_signal_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    signal: pd.Series,
    target_exposure: pd.Series,
    current_momentum: pd.Series,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change()
    signal = signal.map(normalize_code)
    holding = signal.shift(1)
    exposure = target_exposure.shift(1).fillna(0.0).rename("exposure")
    prev_holding = holding.shift(1)
    prev_exposure = exposure.shift(1).fillna(0.0)
    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")
    per_side_cost = fee_rate + slippage_rate

    for dt_idx in prices.index:
        asset = normalize_code(holding.loc[dt_idx])
        prev_asset = normalize_code(prev_holding.loc[dt_idx])
        weight = float(exposure.loc[dt_idx]) if pd.notna(exposure.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure.loc[dt_idx]) if pd.notna(prev_exposure.loc[dt_idx]) else 0.0
        gross_ret = 0.0
        if asset and asset in returns.columns and pd.notna(returns.loc[dt_idx, asset]):
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
            "target_exposure": target_exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )
    return result, build_trades(result, selected)


def run_defensive_reentry_probe_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    probe_exposure: float,
    probe_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    base_result, _ = run_threshold_dual_with_overheat_cap_strategy(
        prices,
        selected,
        lookback=lookback,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        overheat_drawdown_cut=DEFAULT_OVERHEAT_DRAWDOWN_CUT,
        overheat_momentum_cut=DEFAULT_OVERHEAT_MOMENTUM_CUT,
        overheat_max_exposure=DEFAULT_OVERHEAT_MAX_EXPOSURE,
        overheat_high_momentum_cut=DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
        overheat_high_max_exposure=DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    )
    signal = base_result["signal"].map(normalize_code)
    target_exposure = base_result["target_exposure"].copy()
    current_momentum = base_result["current_momentum"].copy()
    risk_codes = {code for code in RISK_CODES if code in prices.columns}
    defensive_codes = {code for code in DEFENSIVE_CODES if code in prices.columns}

    probe_days_remaining = 0
    probe_asset: str | None = None
    reentry_count = 0
    for dt_idx in prices.index:
        asset = normalize_code(signal.loc[dt_idx])
        prev_asset = normalize_code(signal.shift(1).loc[dt_idx])

        if asset in risk_codes and prev_asset in defensive_codes:
            probe_days_remaining = probe_days
            probe_asset = asset
            reentry_count += 1
        elif asset != probe_asset:
            probe_days_remaining = 0
            probe_asset = None

        if asset in risk_codes and probe_days_remaining > 0 and asset == probe_asset:
            target_exposure.loc[dt_idx] = min(float(target_exposure.loc[dt_idx]), probe_exposure)
            probe_days_remaining -= 1
            if probe_days_remaining == 0:
                probe_asset = None

    result, trades = run_signal_strategy(
        prices,
        selected,
        signal,
        target_exposure,
        current_momentum,
        fee_rate,
        slippage_rate,
    )
    return result, trades, reentry_count


def window_return(result: pd.DataFrame, start: str, end: str) -> float:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return float(result.loc[end_ts, "nav"] / result.loc[start_ts, "nav"] - 1)


def summarize(result: pd.DataFrame, trades: pd.DataFrame, reentry_count: int) -> dict[str, float | int]:
    daily_ret = result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    return {
        "total_return": float(result["nav"].iloc[-1] - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown(result["nav"]),
        "trade_count": int(len(trades)),
        "reentry_count": reentry_count,
        "episode_2018_full": window_return(result, "2017-09-06", "2018-11-28"),
        "episode_2018_nov": window_return(result, "2018-11-01", "2018-11-28"),
        "episode_2024_oct": window_return(result, "2024-10-08", "2024-10-17"),
    }


def main() -> int:
    args = parse_args()
    selected, prices = load_cached_data()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_result, baseline_trades = run_threshold_dual_with_overheat_cap_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        overheat_drawdown_cut=DEFAULT_OVERHEAT_DRAWDOWN_CUT,
        overheat_momentum_cut=DEFAULT_OVERHEAT_MOMENTUM_CUT,
        overheat_max_exposure=DEFAULT_OVERHEAT_MAX_EXPOSURE,
        overheat_high_momentum_cut=DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
        overheat_high_max_exposure=DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    )
    rows: list[dict[str, object]] = [
        {"strategy": "current_default", **summarize(baseline_result, baseline_trades, reentry_count=0)}
    ]
    compare_df = pd.DataFrame(index=prices.index)
    compare_df["current_default_nav"] = baseline_result["nav"]

    for probe_exposure in parse_float_list(args.probe_exposures):
        for probe_days in parse_int_list(args.probe_days):
            result, trades, reentry_count = run_defensive_reentry_probe_strategy(
                prices,
                selected,
                lookback=args.lookback,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
                probe_exposure=probe_exposure,
                probe_days=probe_days,
            )
            name = f"probe_{int(round(probe_exposure * 100)):02d}_d{probe_days}"
            compare_df[f"{name}_nav"] = result["nav"]
            rows.append({"strategy": name, **summarize(result, trades, reentry_count)})

    summary = pd.DataFrame(rows)
    baseline = summary[summary["strategy"] == "current_default"].iloc[0]
    for metric in ["annualized_return", "max_drawdown", "episode_2018_full", "episode_2018_nov", "episode_2024_oct", "sharpe_rf0"]:
        summary[f"{metric}_diff"] = summary[metric] - float(baseline[metric])
    summary["trade_count_diff"] = summary["trade_count"] - int(baseline["trade_count"])

    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    compare_df.to_csv(OUTPUT_DIR / "nav_compare.csv")
    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
