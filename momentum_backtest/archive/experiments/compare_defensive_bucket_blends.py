#!/usr/bin/env python3
"""对比混合防守桶方案的历史实验脚本。

这个 baseline 继承自 cash overlay 历史搜索赢家，不是正式 28.2691 官方基线，
因此本脚本应保持历史语义，不做官方净值锚定。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

try:
    from ...runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from archive_data_loaders import (
    build_archive_flat_strategy_paths,
    filter_selected_price_columns,
    load_candidate_prices_with_checks,
    load_named_market_proxy,
)
from goal_optimization_common import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, send_improvement_notification
from hs300_regime_common import summarize
from context_loaders import append_overlay_baseline, load_cash_overlay_context, load_market_overlay_runtime
from overlay_candidate_catalog import CASH_CANDIDATES, TREASURY_10Y, TREASURY_30Y
from overlay_strategy_helpers import apply_simple_overlay, apply_split_defensive_overlay
from search_utils import (
    add_notify_cli_args,
    annotate_valid_improvements,
)
from variant_compare_helpers import (
    append_variant_result,
    build_baseline_preview_columns,
    build_compare_frame,
    filter_available_plot_lines,
    rerank_with_sort_notify_candidates,
    select_valid_change_rows,
    save_best_payload_and_notify_webhook,
    save_plot_and_print_baseline_preview,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    fetch_histories,
)


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_defensive_bucket_blends")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare treasury+cash defensive bucket blends.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Refresh ETF histories instead of using cached core data.")
    add_notify_cli_args(parser)
    return parser.parse_args()

def apply_blended_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    overlay_mode: str,
    risk_cap: float,
    treasury_code: str,
    cash_code: str,
    cash_share: float,
    drawdown_cut: float | None,
    ratio_cut: float | None,
    breadth_cut: float | None,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apply_split_defensive_overlay(
        prices,
        selected,
        proxy,
        params,
        primary_code=treasury_code,
        secondary_code=cash_code,
        secondary_share=cash_share,
        mode=overlay_mode,
        risk_cap=risk_cap,
        drawdown_cut=drawdown_cut,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )


def main() -> int:
    args = parse_args()

    runtime = load_market_overlay_runtime(load_cash_overlay_context)
    market_context = runtime["market_context"]
    overlay_context = runtime["overlay_context"]
    treasury_row = runtime["treasury_row"]
    params = runtime["params"]
    base_selected = runtime["base_selected"]
    cash_candidates = [row for row in CASH_CANDIDATES if row["code"] in {"511880", "511990"}]
    selected_for_prices, prices, available_candidates = load_candidate_prices_with_checks(
        base_selected,
        [treasury_row, *cash_candidates],
        years=args.years,
        refresh=args.refresh,
        fetch_fn=fetch_histories,
        label="defensive candidates",
        required_candidates=[treasury_row],
        required_label="required baseline treasury history",
    )
    cash_candidates = [row for row in available_candidates if str(row["code"]) != str(treasury_row["code"])]

    proxy = load_named_market_proxy(
        prices,
        years=args.years,
        refresh=args.refresh,
        proxy_kind=str(market_context["proxy_kind"]),
    )

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = build_compare_frame(prices)

    baseline_selected = pd.concat([base_selected, pd.DataFrame([treasury_row])], ignore_index=True)
    baseline_prices = filter_selected_price_columns(prices, baseline_selected)
    baseline_result, baseline_trades = apply_simple_overlay(
        prices=baseline_prices,
        selected=baseline_selected,
        proxy=proxy,
        params=params,
        treasury_code=overlay_context["treasury_code"],
        mode=overlay_context["mode"],
        risk_cap=overlay_context["risk_cap"],
        drawdown_cut=overlay_context["drawdown_cut"],
        ratio_cut=overlay_context["ratio_cut"],
        breadth_cut=overlay_context["breadth_cut"],
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = append_overlay_baseline(
        rows,
        descriptions,
        nav_compare,
        baseline_result=baseline_result,
        baseline_trades=baseline_trades,
        baseline_selected=baseline_selected,
        overlay_context=overlay_context,
    )

    for cash_row in cash_candidates:
        selected = pd.concat([base_selected, pd.DataFrame([treasury_row, cash_row])], ignore_index=True)
        candidate_prices = filter_selected_price_columns(prices, selected)
        for cash_share in (0.10, 0.20, 0.30, 0.40, 0.50):
            result, trades = apply_blended_overlay(
                prices=candidate_prices,
                selected=selected,
                proxy=proxy,
                params=params,
                overlay_mode=str(overlay_context["mode"]),
                risk_cap=float(overlay_context["risk_cap"]),
                treasury_code=overlay_context["treasury_code"],
                cash_code=str(cash_row["code"]),
                cash_share=float(cash_share),
                drawdown_cut=overlay_context["drawdown_cut"],
                ratio_cut=overlay_context["ratio_cut"],
                breadth_cut=overlay_context["breadth_cut"],
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
            )
            strategy = (
                f"{market_context['strategy']}__defbucket_{overlay_context['treasury_code']}_{cash_row['code']}"
                f"_cash{int(round(cash_share * 100)):02d}"
                f"_vr{int(round(float(overlay_context['ratio_cut']) * 100)):02d}"
                f"_vb{int(round((float(overlay_context['breadth_cut']) + 0.02) * 100)):02d}"
                f"_cap{int(round(float(overlay_context['risk_cap']) * 100)):02d}"
            )
            append_variant_result(
                rows,
                nav_compare,
                descriptions,
                strategy=strategy,
                result=result,
                summary=summarize(result, trades, selected),
                description=(
                    f"保持当前 hybrid market proxy + 弱市切防守框架不变，仅把防守桶改成 {treasury_row['name']} 与 {cash_row['name']} 的混合。"
                    f"触发条件仍为市场量能20/60<{float(overlay_context['ratio_cut']):.0%} 且广度<{float(overlay_context['breadth_cut']):.1%}，"
                    f"风险资产上限压到 {float(overlay_context['risk_cap']):.0%}，防守桶中现金占比 {float(cash_share):.0%}。"
                ),
                extra_fields={
                    "cash_code": str(cash_row["code"]),
                    "cash_share": float(cash_share),
                },
            )

    summary_df = annotate_valid_improvements(
        rows,
        baseline_summary,
        metric_tolerance=METRIC_TOLERANCE,
        strategy_name=str(overlay_context["strategy"]),
    )
    valid_df = select_valid_change_rows(summary_df)

    preview_columns = build_baseline_preview_columns(["strategy", "cash_code", "cash_share"])
    plot_lines = [
        (f"{overlay_context['strategy']}_nav", "Baseline", 2.2),
        (
            f"{market_context['strategy']}__defbucket_{overlay_context['treasury_code']}_511880_cash10_vr90_vb-1_cap20_nav",
            "511880 10%",
            1.8,
        ),
        (
            f"{market_context['strategy']}__defbucket_{overlay_context['treasury_code']}_511880_cash30_vr90_vb-1_cap20_nav",
            "511880 30%",
            1.8,
        ),
        (
            f"{market_context['strategy']}__defbucket_{overlay_context['treasury_code']}_511990_cash30_vr90_vb-1_cap20_nav",
            "511990 30%",
            1.8,
        ),
    ]
    available_plot_lines = filter_available_plot_lines(nav_compare, plot_lines)
    save_plot_and_print_baseline_preview(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        baseline_summary=baseline_summary,
        columns=preview_columns,
        plot_filename="comparison.png",
        title="Defensive Bucket Blends Comparison",
        lines=available_plot_lines,
        head=20,
    )

    save_best_payload_and_notify_webhook(
        args,
        best_path=BEST_PATH,
        notify_state_path=NOTIFY_STATE_PATH,
        baseline_summary=baseline_summary,
        improvements_df=valid_df,
        descriptions=descriptions,
        metric_tolerance=METRIC_TOLERANCE,
        default_webhook=DEFAULT_FEISHU_WEBHOOK,
        send_fn=send_improvement_notification,
        rerank_fn=rerank_with_sort_notify_candidates,
        suppress_exceptions=True,
        print_fn=print,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
