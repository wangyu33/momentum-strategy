#!/usr/bin/env python3
"""Dual-objective local refine between the current winner and the lower-drawdown near neighbor."""

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

from archive_data_loaders import build_archive_flat_strategy_paths
from goal_optimization_common import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, send_improvement_notification
from hs300_regime_common import summarize
from current_stressbond_context import append_current_stressbond_baseline, load_current_stressbond_runtime
from stressbond_strategy_helpers import apply_stressbond_overlay
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
)
from search_utils import (
    add_notify_cli_args,
    extract_ranked_valid_improvements,
)
from variant_compare_helpers import (
    append_variant_result,
    build_baseline_preview_columns,
    build_compare_frame,
    filter_available_plot_lines,
    save_plot_and_print_baseline_preview,
    save_best_payload_and_notify_ranked,
    save_best_payload_and_notify_ranked_webhook,
)


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_stressbond_dual_objective_refine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-objective refine between current best and nearby lower-drawdown candidate.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    add_notify_cli_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runtime = load_current_stressbond_runtime(args.years)
    current_context = runtime["current_context"]
    treasury_row = runtime["treasury_row"]
    params = runtime["params"]
    selected = runtime["selected"]
    prices = runtime["prices"]
    proxy = runtime["proxy"]

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = build_compare_frame(prices)

    baseline_result, baseline_trades = apply_stressbond_overlay(
        prices=prices,
        selected=selected,
        proxy=proxy,
        params=params,
        treasury_code=str(current_context["treasury_code"]),
        risk_cap=float(current_context["risk_cap"]),
        ratio_cut=float(current_context["ratio_cut"]),
        breadth_cut=float(current_context["breadth_cut"]),
        vg_cap=float(current_context["vg_cap"]),
        overheat_cap=float(current_context["overheat_cap"]),
        overheat_hi_cap=float(current_context["overheat_hi_cap"]),
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_result, baseline_summary = append_current_stressbond_baseline(
        rows,
        descriptions,
        nav_compare,
        baseline_result=baseline_result,
        baseline_trades=baseline_trades,
        selected=selected,
        current_context=current_context,
    )

    risk_cap_candidates = [0.188, 0.189, 0.190, 0.191, 0.192]
    vg_cap_candidates = [0.284, 0.285, 0.286]
    overheat_cap_candidates = [0.067, 0.068, 0.069]
    overheat_hi_cap_candidates = [0.027, 0.028, 0.029]
    total_runs = len(risk_cap_candidates) * len(vg_cap_candidates) * len(overheat_cap_candidates) * len(overheat_hi_cap_candidates)
    run_idx = 0

    for risk_cap in risk_cap_candidates:
        for vg_cap in vg_cap_candidates:
            for overheat_cap in overheat_cap_candidates:
                for overheat_hi_cap in overheat_hi_cap_candidates:
                    if overheat_hi_cap > overheat_cap:
                        continue
                    run_idx += 1
                    print(
                        f"[stressbond_dual_objective_refine] {run_idx}/{total_runs} "
                        f"risk_cap={risk_cap:.3f} vg_cap={vg_cap:.3f} "
                        f"overheat={overheat_cap:.3f}/{overheat_hi_cap:.3f}",
                        flush=True,
                    )
                    result, trades = apply_stressbond_overlay(
                        prices=prices,
                        selected=selected,
                        proxy=proxy,
                        params=params,
                        treasury_code=str(current_context["treasury_code"]),
                        risk_cap=risk_cap,
                        ratio_cut=float(current_context["ratio_cut"]),
                        breadth_cut=float(current_context["breadth_cut"]),
                        vg_cap=vg_cap,
                        overheat_cap=overheat_cap,
                        overheat_hi_cap=overheat_hi_cap,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                    )
                    strategy = (
                        f"{current_context['strategy']}__dual"
                        f"_rc{int(round(risk_cap * 1000)):03d}"
                        f"_vg{int(round(vg_cap * 1000)):03d}"
                        f"_oc{int(round(overheat_cap * 1000)):03d}"
                        f"_oh{int(round(overheat_hi_cap * 1000)):03d}"
                        f"__stressbond_{current_context['treasury_code']}"
                        f"_vr{int(round(float(current_context['ratio_cut']) * 100)):02d}"
                        f"_vb{int(round((float(current_context['breadth_cut']) + 0.02) * 100)):02d}"
                    )
                    append_variant_result(
                        rows,
                        nav_compare,
                        descriptions,
                        strategy=strategy,
                        result=result,
                        summary=summarize(result, trades, selected),
                        description=(
                            f"在当前 {treasury_row['name']} 基线附近做双目标插值微调："
                            f"风险仓上限 {risk_cap:.1%}，弱量能上限 {vg_cap:.1%}，"
                            f"过热/极热上限 {overheat_cap:.1%}/{overheat_hi_cap:.1%}。"
                        ),
                        extra_fields={
                            "risk_cap": risk_cap,
                            "vg_cap": vg_cap,
                            "overheat_cap": overheat_cap,
                            "overheat_hi_cap": overheat_hi_cap,
                        },
                    )

    summary_df, better_ranked_df = extract_ranked_valid_improvements(
        rows,
        baseline_summary,
        metric_tolerance=METRIC_TOLERANCE,
        strategy_name=str(current_context["strategy"]),
    )
    plot_lines = [
        (f"{current_context['strategy']}_nav", "Baseline", 2.2),
        (
            f"{current_context['strategy']}__dual_rc188_vg284_oc067_oh027__stressbond_{current_context['treasury_code']}_vr90_vb-1_nav",
            "Dual 188/284",
            1.8,
        ),
        (
            f"{current_context['strategy']}__dual_rc190_vg285_oc068_oh028__stressbond_{current_context['treasury_code']}_vr90_vb-1_nav",
            "Dual 190/285",
            1.8,
        ),
        (
            f"{current_context['strategy']}__dual_rc192_vg286_oc069_oh029__stressbond_{current_context['treasury_code']}_vr90_vb-1_nav",
            "Dual 192/286",
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
            ["strategy", "risk_cap", "vg_cap", "overheat_cap", "overheat_hi_cap"]
        ),
        plot_filename="comparison.png",
        title="Stressbond Dual Objective Refine Comparison",
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
