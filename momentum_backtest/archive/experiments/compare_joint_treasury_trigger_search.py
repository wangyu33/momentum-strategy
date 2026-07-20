#!/usr/bin/env python3
"""Joint search over treasury choice and nearby stress-bond trigger refinements.

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
    build_overlay_summary_fields,
    load_market_overlay_runtime,
    load_simple_overlay_context,
    run_market_stress_overlay,
)
from goal_optimization_common import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, send_improvement_notification
from hs300_regime_common import summarize
from overlay_candidate_catalog import TREASURY_CANDIDATES
from variant_compare_helpers import (
    append_variant_result,
    build_baseline_preview_columns,
    build_compare_frame,
    filter_available_plot_lines,
    save_plot_and_print_baseline_preview,
    save_best_payload_and_notify_ranked,
    save_best_payload_and_notify_ranked_webhook,
)
from search_utils import (
    add_notify_cli_args,
    extract_ranked_valid_improvements,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    fetch_histories,
)


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_joint_treasury_trigger_search")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly refine treasury choice and nearby stress-bond trigger parameters around the active baseline."
    )
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
    treasury_candidates = list(TREASURY_CANDIDATES)
    selected_for_prices, prices, treasury_candidates = load_candidate_prices_with_checks(
        base_selected,
        treasury_candidates,
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
    baseline_result, baseline_trades = run_market_stress_overlay(
        prices=baseline_prices,
        selected=baseline_selected,
        proxy=proxy,
        params=params,
        treasury_code=overlay_context["treasury_code"],
        risk_cap=overlay_context["risk_cap"],
        ratio_cut=overlay_context["ratio_cut"],
        breadth_cut=overlay_context["breadth_cut"],
        short_ratio_cut=None,
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

    ratio_candidates = [0.895, 0.90, 0.905]
    breadth_candidates = [-0.035, -0.03]
    risk_cap_candidates = [0.15, 0.18, 0.20, 0.22]
    short_ratio_candidates: list[float | None] = [None, 0.92]
    total_runs = len(treasury_candidates) * len(ratio_candidates) * len(breadth_candidates) * len(risk_cap_candidates) * len(short_ratio_candidates)
    run_idx = 0

    for treasury in treasury_candidates:
        selected = pd.concat([base_selected, pd.DataFrame([treasury])], ignore_index=True)
        candidate_prices = filter_selected_price_columns(prices, selected)
        for ratio_cut in ratio_candidates:
            for breadth_cut in breadth_candidates:
                for risk_cap in risk_cap_candidates:
                    for short_ratio_cut in short_ratio_candidates:
                        run_idx += 1
                        print(
                            f"[joint_treasury_trigger_search] {run_idx}/{total_runs} "
                            f"treasury={treasury['code']} ratio={ratio_cut:.3f} breadth={breadth_cut:.3f} "
                            f"cap={risk_cap:.2f} short={short_ratio_cut if short_ratio_cut is not None else 'na'}",
                            flush=True,
                        )
                        result, trades = run_market_stress_overlay(
                            prices=candidate_prices,
                            selected=selected,
                            proxy=proxy,
                            params=params,
                            treasury_code=str(treasury["code"]),
                            risk_cap=float(risk_cap),
                            ratio_cut=float(ratio_cut),
                            breadth_cut=float(breadth_cut),
                            short_ratio_cut=short_ratio_cut,
                            fee_rate=args.fee_rate,
                            slippage_rate=args.slippage_rate,
                        )
                        short_tag = "na" if short_ratio_cut is None else f"{int(round(short_ratio_cut * 100)):02d}"
                        strategy = (
                            f"{market_context['strategy']}__stressbond_{treasury['code']}"
                            f"_vr{int(round(ratio_cut * 1000)):03d}"
                            f"_vb{int(round((breadth_cut + 0.02) * 1000)):03d}"
                            f"_cap{int(round(risk_cap * 100)):02d}"
                            f"_vs{short_tag}"
                        )
                        short_desc = (
                            "不加短期量能过滤"
                            if short_ratio_cut is None
                            else f"且短期量能5/20<{float(short_ratio_cut):.0%}"
                        )
                        append_variant_result(
                            rows,
                            nav_compare,
                            descriptions,
                            strategy=strategy,
                            result=result,
                            summary=summarize(result, trades, selected),
                            description=(
                                f"保持当前 hybrid market proxy + 弱市切债框架不变，仅联合微调防守债券与触发阈值："
                                f"切债资产使用 {treasury['name']}({treasury['code']})，"
                                f"市场量能20/60<{float(ratio_cut):.1%}、广度<{float(breadth_cut):.1%}、{short_desc}，"
                                f"风险资产上限压到 {float(risk_cap):.0%}。"
                            ),
                            extra_fields={
                                **build_overlay_summary_fields(
                                    overlay_context,
                                    treasury_code=str(treasury["code"]),
                                    ratio_cut=ratio_cut,
                                    breadth_cut=breadth_cut,
                                    risk_cap=risk_cap,
                                ),
                                "short_ratio_cut": short_ratio_cut,
                            },
                        )

    summary_df, better_ranked_df = extract_ranked_valid_improvements(
        rows,
        baseline_summary,
        metric_tolerance=METRIC_TOLERANCE,
        strategy_name=str(overlay_context["strategy"]),
    )
    plot_lines = [
        (f"{overlay_context['strategy']}_nav", "Baseline", 2.2),
        (
            f"{market_context['strategy']}__stressbond_{overlay_context['treasury_code']}_vr900_vb000_cap18_vsna_nav",
            "10Y Cap18",
            1.8,
        ),
        (
            f"{market_context['strategy']}__stressbond_{overlay_context['treasury_code']}_vr895_vb-10_cap15_vsna_nav",
            "10Y Cap15",
            1.8,
        ),
        (
            f"{market_context['strategy']}__stressbond_{overlay_context['treasury_code']}_vr900_vb000_cap20_vs92_nav",
            "10Y Short92",
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
            ["strategy", "treasury_code", "ratio_cut", "breadth_cut", "risk_cap", "short_ratio_cut"]
        ),
        plot_filename="comparison.png",
        title="Joint Treasury Trigger Search Comparison",
        lines=available_plot_lines,
        head=20,
    )

    save_best_payload_and_notify_ranked_webhook(
        args,
        best_path=BEST_PATH,
        notify_state_path=NOTIFY_STATE_PATH,
        baseline_summary=baseline_summary,
        improvements_df=better_ranked_df,
        descriptions=descriptions,
        metric_tolerance=METRIC_TOLERANCE,
        default_webhook=DEFAULT_FEISHU_WEBHOOK,
        send_fn=send_improvement_notification,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
