#!/usr/bin/env python3
"""Refine stress-bond persistence state machine around the active default baseline."""

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
from current_stressbond_context import (
    append_current_stressbond_baseline,
    load_current_stressbond_runtime,
    run_current_stressbond_persistence_overlay,
)
from goal_optimization_common import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, send_improvement_notification
from hs300_regime_common import summarize
from search_utils import add_notify_cli_args, extract_ranked_valid_improvements
from variant_compare_helpers import (
    append_variant_result,
    build_baseline_preview_columns,
    build_compare_frame,
    filter_available_plot_lines,
    save_plot_and_print_baseline_preview,
    save_best_payload_and_notify_ranked,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
)


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_stressbond_persistence_current")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine stress-bond persistence around the active baseline.")
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

    baseline_result, baseline_trades = run_current_stressbond_persistence_overlay(
        prices=prices,
        selected=selected,
        proxy=proxy,
        params=params,
        treasury_code=str(current_context["treasury_code"]),
        risk_cap=float(current_context["risk_cap"]),
        ratio_cut=float(current_context["ratio_cut"]),
        breadth_cut=float(current_context["breadth_cut"]),
        enter_days=1,
        exit_days=1,
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

    enter_days_candidates = [1, 2, 3]
    exit_days_candidates = [1, 2, 3, 5]
    risk_cap_candidates = [0.184, 0.186, 0.188]
    total_runs = len(enter_days_candidates) * len(exit_days_candidates) * len(risk_cap_candidates)
    run_idx = 0

    for enter_days in enter_days_candidates:
        for exit_days in exit_days_candidates:
            for risk_cap in risk_cap_candidates:
                run_idx += 1
                print(
                    f"[stressbond_persistence_current] {run_idx}/{total_runs} "
                    f"enter={enter_days} exit={exit_days} cap={risk_cap:.3f}",
                    flush=True,
                )
                result, trades = run_current_stressbond_persistence_overlay(
                    prices=prices,
                    selected=selected,
                    proxy=proxy,
                    params=params,
                    treasury_code=str(current_context["treasury_code"]),
                    risk_cap=risk_cap,
                    ratio_cut=float(current_context["ratio_cut"]),
                    breadth_cut=float(current_context["breadth_cut"]),
                    enter_days=enter_days,
                    exit_days=exit_days,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                )
                strategy = (
                    f"{current_context['strategy']}__prs"
                    f"_rc{int(round(risk_cap * 1000)):03d}"
                    f"_en{enter_days:02d}_ex{exit_days:02d}"
                )
                append_variant_result(
                    rows,
                    nav_compare,
                    descriptions,
                    strategy=strategy,
                    result=result,
                    summary=summarize(result, trades, selected),
                    description=(
                        f"围绕当前 {treasury_row['name']} 弱市切债状态机继续细化："
                        f"风险仓上限 {risk_cap:.1%}，触发连续 {enter_days} 天后进入防守，"
                        f"信号消失连续 {exit_days} 天后退出防守。"
                    ),
                    extra_fields={
                        "risk_cap": risk_cap,
                        "enter_days": enter_days,
                        "exit_days": exit_days,
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
        (f"{current_context['strategy']}__prs_rc184_en01_ex01_nav", "PRS 184 1/1", 1.8),
        (f"{current_context['strategy']}__prs_rc186_en02_ex02_nav", "PRS 186 2/2", 1.8),
        (f"{current_context['strategy']}__prs_rc188_en03_ex05_nav", "PRS 188 3/5", 1.8),
    ]
    available_plot_lines = filter_available_plot_lines(nav_compare, plot_lines)
    save_plot_and_print_baseline_preview(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        baseline_summary=baseline_summary,
        columns=build_baseline_preview_columns(["strategy", "risk_cap", "enter_days", "exit_days"]),
        plot_filename="comparison.png",
        title="Stressbond Persistence Current Comparison",
        lines=available_plot_lines,
        head=20,
    )

    save_best_payload_and_notify_ranked(
        args,
        best_path=BEST_PATH,
        notify_state_path=NOTIFY_STATE_PATH,
        descriptions=descriptions,
        baseline_summary=baseline_summary,
        improvements_df=better_ranked_df,
        metric_tolerance=METRIC_TOLERANCE,
        default_webhook=DEFAULT_FEISHU_WEBHOOK,
        send_fn=send_improvement_notification,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
