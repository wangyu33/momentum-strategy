#!/usr/bin/env python3
"""对比当前正式基线下，前二信号接近时降低总风险仓位的候选。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import pandas as pd

from run_backtest import (
    build_default_strategy_params,
    build_strategy_summary,
    load_core_selected_and_prices,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)
from compare_market_proxy_variants import build_proxy_catalog

OUTPUT_DIR = Path("momentum_backtest/output/research/top2_riskcap_candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比前二信号接近时降低总风险仓位的候选。")
    parser.add_argument("--gap", type=float, default=0.005, help="前二信号质量差小于等于该值时触发。默认 0.5%。")
    parser.add_argument(
        "--risk-caps",
        type=str,
        default="0.8,0.7,0.6",
        help="触发时总风险仓位上限列表，例如 0.8,0.7,0.6。",
    )
    return parser.parse_args()


def parse_caps(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("empty risk caps")
    for v in values:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"unsupported risk cap {v}")
    return values


def load_official_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    selected, prices = load_core_selected_and_prices(today=pd.Timestamp("2026-06-05").date(), allow_same_day_close=False)
    market_proxy = pd.read_csv(
        "momentum_backtest/output/research/goal_optimizations/market_volume_proxy.csv", parse_dates=["date"]
    ).set_index("date")
    params = build_default_strategy_params()
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(market_proxy, prices, risk_codes=[str(code) for code in params["risk_codes"]])
    }
    effective_proxy = proxy_catalog[str(params["proxy_kind"])]
    return selected, prices, effective_proxy, params


def summarize_variant(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    variant: str,
    close_gap: float,
    risk_cap_when_close: float,
) -> dict[str, object]:
    summary = build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)
    summary["variant"] = variant
    summary["close_gap"] = close_gap
    summary["risk_cap_when_close"] = risk_cap_when_close
    trigger_col = "top2_close_risk_cap_triggered"
    summary["close_days"] = int(result[trigger_col].fillna(False).sum()) if trigger_col in result.columns else 0
    return summary


def main() -> int:
    args = parse_args()
    risk_caps = parse_caps(args.risk_caps)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected, prices, effective_proxy, base_params = load_official_inputs()

    rows: list[dict[str, object]] = []

    baseline_result, baseline_trades = run_default_strategy_with_params(
        prices=prices,
        selected=selected,
        params=base_params,
        market_proxy=effective_proxy,
    )
    baseline_summary = summarize_variant(
        baseline_result,
        baseline_trades,
        selected,
        variant="baseline",
        close_gap=float(base_params["close_top2_gap"]),
        risk_cap_when_close=float(base_params["close_top2_risk_cap"]),
    )
    rows.append(baseline_summary)

    for risk_cap in risk_caps:
        if abs(float(risk_cap) - float(base_params["close_top2_risk_cap"])) < 1e-12 and abs(float(args.gap) - float(base_params["close_top2_gap"])) < 1e-12:
            continue
        params = dict(base_params)
        params["close_top2_gap"] = float(args.gap)
        params["close_top2_risk_cap"] = float(risk_cap)
        result, trades = run_default_strategy_with_params(
            prices=prices,
            selected=selected,
            params=params,
            market_proxy=effective_proxy,
        )
        summary = summarize_variant(
            result,
            trades,
            selected,
            variant=f"top2_close_riskcap_{int(round(risk_cap * 100)):02d}",
            close_gap=float(args.gap),
            risk_cap_when_close=float(risk_cap),
        )
        rows.append(summary)

        out_dir = OUTPUT_DIR / summary["variant"]
        out_dir.mkdir(parents=True, exist_ok=True)
        write_dataframe_csv_atomic(result.reset_index(), out_dir / "backtest_nav.csv", index=False)
        write_dataframe_csv_atomic(trades, out_dir / "trades.csv", index=False)
        write_dataframe_csv_atomic(pd.DataFrame([summary]), out_dir / "summary.csv", index=False)

    out = pd.DataFrame(rows)
    base_row = out.iloc[0]
    for col in ["annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral", "trade_count", "close_days"]:
        out[f"{col}_diff_vs_base"] = out[col] - base_row[col]
    write_dataframe_csv_atomic(out, OUTPUT_DIR / "summary.csv", index=False)
    print(
        out[
            [
                "variant",
                "risk_cap_when_close",
                "close_gap",
                "close_days",
                "annualized_return",
                "sharpe_rf0",
                "max_drawdown",
                "max_drawdown_integral",
                "trade_count",
                "annualized_return_diff_vs_base",
                "sharpe_rf0_diff_vs_base",
                "max_drawdown_diff_vs_base",
                "max_drawdown_integral_diff_vs_base",
                "trade_count_diff_vs_base",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
