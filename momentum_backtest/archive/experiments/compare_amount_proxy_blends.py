#!/usr/bin/env python3
"""对比不同量能代理混合方案，保持当前正式 breadth 定义不变。"""

from __future__ import annotations

import argparse

try:
    from ..runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from compare_goal_optimizations import load_market_volume_proxy
from compare_hs300_regime_fixes import load_cached_data, summarize
from compare_market_proxy_variants import build_risk_proxy_features
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    build_default_strategy_params,
    ensure_output_dirs,
    load_default_strategy_backtest_pool,
    resolve_strategy_universe,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_amount_proxy_blends")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比不同量能代理混合方案。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = load_default_strategy_backtest_pool()
    _, cached_prices = load_cached_data()
    start_ts = cached_prices.index.max() - pd.DateOffset(years=args.years)
    prices = cached_prices.loc[cached_prices.index >= start_ts].copy()
    selected_codes = selected["code"].astype(str).tolist()
    prices = prices[[code for code in selected_codes if code in prices.columns]].copy()

    base_proxy = load_market_volume_proxy(years=args.years, refresh=False).reindex(prices.index).ffill()
    risk_codes, _ = resolve_strategy_universe(prices)
    features = build_risk_proxy_features(prices, risk_codes=risk_codes)

    # 固定当前正式 breadth 口径，只测试量能代理本身是否值得替换。
    hybrid_breadth = 0.5 * base_proxy["market_breadth_proxy"] + 0.5 * features["breadth_blend"]

    variants: list[tuple[str, pd.Series, pd.Series, float, float]] = [
        (
            "current_a_share_amount",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.91,
            0.90,
        ),
    ]

    for etf_key, label in [
        ("above_ma20_ratio_20_60", "etf_ma20"),
        ("pos20_ratio_20_60", "etf_pos20"),
    ]:
        short_key = etf_key.replace("20_60", "5_20")
        for weight in [0.25, 0.50, 0.75]:
            variants.append(
                (
                    f"blend_{label}_{int(weight * 100):02d}",
                    (1 - weight) * base_proxy["market_amount_ratio_20_60"] + weight * features[etf_key],
                    (1 - weight) * base_proxy["market_amount_ratio_5_20"] + weight * features[short_key],
                    0.91,
                    0.90,
                )
            )

    variants.extend(
        [
            (
                "pure_etf_ma20_amount",
                features["above_ma20_ratio_20_60"],
                features["above_ma20_ratio_5_20"],
                0.96,
                0.95,
            ),
            (
                "pure_etf_pos20_amount",
                features["pos20_ratio_20_60"],
                features["pos20_ratio_5_20"],
                0.96,
                0.95,
            ),
        ]
    )

    rows: list[dict[str, object]] = []
    baseline_row: dict[str, float] | None = None

    for name, ratio_20_60, ratio_5_20, ratio_cut, short_cut in variants:
        proxy = pd.DataFrame(index=prices.index)
        proxy["market_amount_ratio_20_60"] = ratio_20_60
        proxy["market_amount_ratio_5_20"] = ratio_5_20
        proxy["market_breadth_proxy"] = hybrid_breadth

        params = build_default_strategy_params()
        # 传入已构造好的代理，避免 run_default_strategy_with_params 再次混合 breadth。
        params["proxy_kind"] = "baseline_sh_sz"
        params["volume_breadth_cut"] = -0.04
        params["volume_ratio_cut"] = ratio_cut
        params["volume_short_ratio_cut"] = short_cut

        result, trades = run_default_strategy_with_params(
            prices=prices,
            selected=selected,
            params=params,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            market_proxy=proxy,
        )
        summary = summarize(result, trades, selected)
        row = {
            "name": name,
            "annualized_return": float(summary["annualized_return"]),
            "sharpe_rf0": float(summary["sharpe_rf0"]),
            "max_drawdown": float(summary["max_drawdown"]),
            "max_drawdown_integral": float(summary["max_drawdown_integral"]),
            "trade_action_count": int(summary["trade_action_count"]),
            "volume_ratio_cut": ratio_cut,
            "volume_short_ratio_cut": short_cut,
        }
        if baseline_row is None:
            baseline_row = row.copy()
            row["annualized_diff"] = 0.0
            row["sharpe_diff"] = 0.0
            row["max_drawdown_diff"] = 0.0
            row["max_drawdown_integral_diff"] = 0.0
        else:
            row["annualized_diff"] = row["annualized_return"] - baseline_row["annualized_return"]
            row["sharpe_diff"] = row["sharpe_rf0"] - baseline_row["sharpe_rf0"]
            row["max_drawdown_diff"] = row["max_drawdown"] - baseline_row["max_drawdown"]
            row["max_drawdown_integral_diff"] = row["max_drawdown_integral"] - baseline_row["max_drawdown_integral"]
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    write_dataframe_csv_atomic(summary_df, SUMMARY_PATH, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nsummary saved to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
