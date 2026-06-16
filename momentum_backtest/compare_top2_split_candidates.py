#!/usr/bin/env python3
"""对比当前正式基线下，前二信号接近时的分仓候选。"""

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
    build_signal_quality_score,
    build_strategy_summary,
    load_core_selected_and_prices,
    resolve_strategy_universe,
    write_dataframe_csv_atomic,
)
from compare_market_proxy_variants import build_proxy_catalog
from compare_hs300_regime_fixes import run_target_weights_strategy
from official_strategy_core import build_official_target_weights

OUTPUT_DIR = Path("momentum_backtest/output/research/top2_split_candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比前二信号接近时的分仓候选。")
    parser.add_argument("--gap", type=float, default=0.005, help="前二信号质量差小于等于该值时触发分仓。默认 0.5%。")
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.7,0.6,0.5",
        help="主信号分仓比例列表，次信号自动为 1-ratio，例如 0.7 表示 70/30。",
    )
    return parser.parse_args()


def parse_ratio_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("empty ratios")
    for v in values:
        if not (0.5 <= v < 1.0):
            raise ValueError(f"unsupported split ratio {v}; expected 0.5<=ratio<1.0")
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


def extract_target_weights(result: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in result.columns if c.startswith("target_weight_")]
    target = result[cols].copy()
    target.columns = [c.removeprefix("target_weight_") for c in cols]
    return target


def build_top2_context(prices: pd.DataFrame, params: dict[str, object]) -> tuple[pd.Series, pd.Series]:
    momentum, score = build_signal_quality_score(
        prices,
        lookback=25,
        method=str(params.get("signal_quality_method", "raw")),
        slope_penalty=float(params.get("signal_slope_penalty", 0.0)),
    )
    risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params["risk_codes"]],
        defensive_codes=[str(code) for code in params["defensive_codes"]],
    )
    risk_score = score[risk_codes]
    top1 = pd.Series(index=prices.index, dtype="object")
    top2 = pd.Series(index=prices.index, dtype="object")
    abs_cut = 0.05
    for dt in prices.index:
        valid = risk_score.loc[dt].dropna().sort_values(ascending=False)
        if valid.empty:
            continue
        first = str(valid.index[0])
        first_mom = float(momentum.loc[dt, first]) if pd.notna(momentum.loc[dt, first]) else float("nan")
        if pd.isna(first_mom) or first_mom <= abs_cut:
            continue
        top1.loc[dt] = first
        if len(valid) >= 2:
            second = str(valid.index[1])
            second_mom = float(momentum.loc[dt, second]) if pd.notna(momentum.loc[dt, second]) else float("nan")
            if pd.notna(second_mom) and second_mom > abs_cut:
                top2.loc[dt] = second
    return top1, top2


def apply_top2_split(
    target_weights: pd.DataFrame,
    base_result: pd.DataFrame,
    top1_risk: pd.Series,
    top2_risk: pd.Series,
    risk_codes: list[str],
    split_ratio: float,
) -> tuple[pd.DataFrame, pd.Series]:
    adjusted = target_weights.copy()
    trigger = base_result["top2_close_risk_cap_triggered"].fillna(False) if "top2_close_risk_cap_triggered" in base_result.columns else pd.Series(False, index=base_result.index)
    active = pd.Series(False, index=adjusted.index, dtype=bool)
    for dt in adjusted.index[trigger]:
        lead = top1_risk.loc[dt]
        second = top2_risk.loc[dt]
        if pd.isna(lead) or pd.isna(second):
            continue
        lead = str(lead)
        second = str(second)
        if lead not in risk_codes or second not in risk_codes:
            continue
        risk_weight = float(adjusted.loc[dt, risk_codes].sum())
        if risk_weight <= 1e-12:
            continue
        adjusted.loc[dt, risk_codes] = 0.0
        adjusted.loc[dt, lead] = risk_weight * split_ratio
        adjusted.loc[dt, second] = risk_weight * (1.0 - split_ratio)
        active.loc[dt] = True
    return adjusted, active


def run_variant_from_weights(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    weights: pd.DataFrame,
    base_result: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effective_momentum = base_result["effective_momentum"] if "effective_momentum" in base_result.columns else base_result["current_momentum"]
    result, trades = run_target_weights_strategy(prices, selected, weights, effective_momentum.rename("current_momentum"), 0.0003, 0.0002)
    for col in [
        "signal",
        "current_momentum",
        "effective_momentum",
        "base_target_exposure",
        "top2_close_risk_cap_triggered",
        "top2_close_gap",
        "top2_close_risk_cap",
        "stress_bond_trigger",
        "weak_market_trigger",
    ]:
        if col in base_result.columns:
            result[col] = base_result[col]
    result["target_exposure"] = weights.sum(axis=1)
    result = pd.concat([result, weights.add_prefix("target_weight_")], axis=1)
    return result, trades


def summarize_variant(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    variant: str,
    split_gap: float,
    split_ratio: str,
    split_days: int,
) -> dict[str, object]:
    summary = build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)
    summary["variant"] = variant
    summary["split_gap"] = split_gap
    summary["split_ratio"] = split_ratio
    summary["split_days"] = split_days
    return summary


def main() -> int:
    args = parse_args()
    ratios = parse_ratio_list(args.ratios)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected, prices, effective_proxy, base_params = load_official_inputs()
    params = dict(base_params)
    params["close_top2_gap"] = float(args.gap)
    base_weights, _, base_result = build_official_target_weights(prices, effective_proxy, params)

    rows: list[dict[str, object]] = []

    baseline_result, baseline_trades = run_variant_from_weights(prices, selected, base_weights, base_result)
    baseline_summary = summarize_variant(
        baseline_result,
        baseline_trades,
        selected,
        variant="baseline",
        split_gap=float(args.gap),
        split_ratio="100/0",
        split_days=0,
    )
    rows.append(baseline_summary)

    risk_codes = [str(code) for code in base_params["risk_codes"] if str(code) in base_weights.columns]
    top1_risk, top2_risk = build_top2_context(prices, params)
    for primary_ratio in ratios:
        split_weights, split_active = apply_top2_split(base_weights, base_result, top1_risk, top2_risk, risk_codes, float(primary_ratio))
        result, trades = run_variant_from_weights(prices, selected, split_weights, base_result)
        summary = summarize_variant(
            result,
            trades,
            selected,
            variant=f"top2_split_{int(round(primary_ratio * 100))}_{int(round((1 - primary_ratio) * 100))}",
            split_gap=float(args.gap),
            split_ratio=f"{int(round(primary_ratio * 100))}/{int(round((1 - primary_ratio) * 100))}",
            split_days=int(split_active.sum()),
        )
        rows.append(summary)

        out_dir = OUTPUT_DIR / summary["variant"]
        out_dir.mkdir(parents=True, exist_ok=True)
        write_dataframe_csv_atomic(result.reset_index(), out_dir / "backtest_nav.csv", index=False)
        write_dataframe_csv_atomic(trades, out_dir / "trades.csv", index=False)
        write_dataframe_csv_atomic(pd.DataFrame([summary]), out_dir / "summary.csv", index=False)

    out = pd.DataFrame(rows)
    base_row = out.iloc[0]
    for col in ["annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral", "trade_count", "split_days"]:
        out[f"{col}_diff_vs_base"] = out[col] - base_row[col]
    write_dataframe_csv_atomic(out, OUTPUT_DIR / "summary.csv", index=False)
    print(
        out[
            [
                "variant",
                "split_ratio",
                "split_gap",
                "split_days",
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
