#!/usr/bin/env python3
"""Compare compact two-tier defensive overlays around the current treasury overlay winner.

This baseline inherits the simple bond overlay historical winner rather than the official
28.2691 formal baseline, so this script must keep its own research-chain semantics.
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
from context_loaders import (
    append_overlay_baseline,
    load_market_overlay_runtime,
    load_simple_overlay_context,
    run_two_tier_market_stress_overlay,
)
from goal_optimization_common import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, send_improvement_notification
from hs300_regime_common import summarize
from overlay_strategy_helpers import apply_simple_overlay
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
    save_plot_and_print_baseline_preview,
    save_best_payload_and_notify_webhook,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    fetch_histories,
)


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_two_tier_defense")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare compact two-tier defensive overlays.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Refresh ETF histories instead of using cached core data.")
    add_notify_cli_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runtime = load_market_overlay_runtime(load_simple_overlay_context)
    market_context = runtime["market_context"]
    overlay_context = runtime["overlay_context"]
    treasury_row = runtime["treasury_row"]
    params = runtime["params"]
    base_selected = runtime["base_selected"]
    selected_for_prices, prices, _ = load_candidate_prices_with_checks(
        base_selected,
        [treasury_row],
        years=args.years,
        refresh=args.refresh,
        fetch_fn=fetch_histories,
        label="treasury candidates",
        required_candidates=[treasury_row],
        required_label="required baseline treasury history",
    )

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
        mode="market_stress_only",
        risk_cap=overlay_context["risk_cap"],
        drawdown_cut=None,
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

    weak_ratio_candidates = [0.90, 0.905]
    weak_breadth_candidates = [-0.03, -0.025]
    weak_cap_candidates = [0.25, 0.30]
    extreme_ratio_candidates = [0.885, 0.89, 0.895]
    extreme_breadth_candidates = [-0.04, -0.035]
    extreme_cap_candidates = [0.10, 0.15]
    extreme_short_ratio_candidates: list[float | None] = [None, 0.95]

    for weak_ratio_cut in weak_ratio_candidates:
        for weak_breadth_cut in weak_breadth_candidates:
            for weak_cap in weak_cap_candidates:
                for extreme_ratio_cut in extreme_ratio_candidates:
                    if extreme_ratio_cut >= weak_ratio_cut:
                        continue
                    for extreme_breadth_cut in extreme_breadth_candidates:
                        if extreme_breadth_cut >= weak_breadth_cut:
                            continue
                        for extreme_cap in extreme_cap_candidates:
                            if extreme_cap >= weak_cap:
                                continue
                            for extreme_short_ratio_cut in extreme_short_ratio_candidates:
                                result, trades = run_two_tier_market_stress_overlay(
                                    prices=baseline_prices,
                                    selected=baseline_selected,
                                    proxy=proxy,
                                    params=params,
                                    treasury_code=overlay_context["treasury_code"],
                                    weak_ratio_cut=weak_ratio_cut,
                                    weak_breadth_cut=weak_breadth_cut,
                                    weak_cap=weak_cap,
                                    extreme_ratio_cut=extreme_ratio_cut,
                                    extreme_breadth_cut=extreme_breadth_cut,
                                    extreme_cap=extreme_cap,
                                    extreme_short_ratio_cut=extreme_short_ratio_cut,
                                    fee_rate=args.fee_rate,
                                    slippage_rate=args.slippage_rate,
                                )
                                short_tag = "na" if extreme_short_ratio_cut is None else f"{int(round(extreme_short_ratio_cut * 100)):02d}"
                                strategy = (
                                    f"{market_context['strategy']}__twotier_{overlay_context['treasury_code']}"
                                    f"_wvr{int(round(weak_ratio_cut * 1000)):03d}"
                                    f"_wvb{int(round((weak_breadth_cut + 0.02) * 1000)):03d}"
                                    f"_wcap{int(round(weak_cap * 100)):02d}"
                                    f"_evr{int(round(extreme_ratio_cut * 1000)):03d}"
                                    f"_evb{int(round((extreme_breadth_cut + 0.02) * 1000)):03d}"
                                    f"_ecap{int(round(extreme_cap * 100)):02d}"
                                    f"_evs{short_tag}"
                                )
                                short_desc = (
                                    "不加短期量能过滤"
                                    if extreme_short_ratio_cut is None
                                    else f"且极弱市要求5/20<{float(extreme_short_ratio_cut):.0%}"
                                )
                                append_variant_result(
                                    rows,
                                    nav_compare,
                                    descriptions,
                                    strategy=strategy,
                                    result=result,
                                    summary=summarize(result, trades, baseline_selected),
                                    description=(
                                        f"保持当前 hybrid market proxy + {treasury_row['name']} 防守结构不变，改成双层防守："
                                        f"弱市在量能20/60<{weak_ratio_cut:.1%} 且广度<{weak_breadth_cut:.1%} 时，把风险仓压到 {weak_cap:.0%}；"
                                        f"极弱市在量能20/60<{extreme_ratio_cut:.1%} 且广度<{extreme_breadth_cut:.1%} 时，把风险仓进一步压到 {extreme_cap:.0%}；"
                                        f"{short_desc}。"
                                    ),
                                    extra_fields={
                                        "weak_ratio_cut": weak_ratio_cut,
                                        "weak_breadth_cut": weak_breadth_cut,
                                        "weak_cap": weak_cap,
                                        "extreme_ratio_cut": extreme_ratio_cut,
                                        "extreme_breadth_cut": extreme_breadth_cut,
                                        "extreme_cap": extreme_cap,
                                        "extreme_short_ratio_cut": extreme_short_ratio_cut,
                                    },
                                )

    summary_df = annotate_valid_improvements(
        rows,
        baseline_summary,
        metric_tolerance=METRIC_TOLERANCE,
        strategy_name=str(overlay_context["strategy"]),
    )
    valid_df = select_valid_change_rows(summary_df)

    plot_lines = [
        (f"{overlay_context['strategy']}_nav", "Baseline", 2.2),
        (
            f"{market_context['strategy']}__twotier_{overlay_context['treasury_code']}_wvr900_wvb-10_wcap25_evr890_evb-15_ecap15_evsna_nav",
            "Two Tier 25/15",
            1.8,
        ),
        (
            f"{market_context['strategy']}__twotier_{overlay_context['treasury_code']}_wvr900_wvb-10_wcap25_evr890_evb-15_ecap10_evsna_nav",
            "Two Tier 25/10",
            1.8,
        ),
        (
            f"{market_context['strategy']}__twotier_{overlay_context['treasury_code']}_wvr905_wvb-05_wcap30_evr890_evb-15_ecap15_evs95_nav",
            "Two Tier 30/15 SR95",
            1.8,
        ),
    ]
    available_plot_lines = filter_available_plot_lines(nav_compare, plot_lines)
    save_plot_and_print_baseline_preview(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        baseline_summary=baseline_summary,
        columns=build_baseline_preview_columns(
            [
                "strategy",
                "weak_ratio_cut",
                "weak_breadth_cut",
                "weak_cap",
                "extreme_ratio_cut",
                "extreme_breadth_cut",
                "extreme_cap",
                "extreme_short_ratio_cut",
            ]
        ),
        plot_filename="comparison.png",
        title="Two-Tier Defense Comparison",
        lines=available_plot_lines,
        head=20,
    )

    save_best_payload_and_notify_webhook(
        args,
        best_path=BEST_PATH,
        notify_state_path=NOTIFY_STATE_PATH,
        descriptions=descriptions,
        baseline_summary=baseline_summary,
        improvements_df=valid_df,
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
