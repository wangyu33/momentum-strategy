#!/usr/bin/env python3
"""Refine current stress-bond trigger thresholds around the active default baseline."""

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
from current_stressbond_context import (
    append_current_stressbond_baseline,
    load_current_stressbond_runtime,
    run_current_stressbond_overlay,
)
from variant_compare_helpers import (
    append_variant_result,
    build_baseline_preview_columns,
    build_compare_frame,
    filter_available_plot_lines,
    save_plot_and_print_baseline_preview,
    save_best_payload_and_notify_ranked,
)
from search_utils import (
    add_notify_cli_args,
    extract_ranked_valid_improvements,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
)


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_stressbond_trigger_refine_current")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine current stress-bond trigger thresholds around the active baseline.")
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

    baseline_result, baseline_trades = run_current_stressbond_overlay(
        prices=prices,
        selected=selected,
        proxy=proxy,
        params=params,
        treasury_code=str(current_context["treasury_code"]),
        risk_cap=float(current_context["risk_cap"]),
        ratio_cut=float(current_context["ratio_cut"]),
        breadth_cut=float(current_context["breadth_cut"]),
        short_ratio_cut=None,
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

    ratio_candidates = [0.895, 0.900, 0.905]
    breadth_candidates = [-0.035, -0.030, -0.025]
    risk_cap_candidates = [0.186, 0.188, 0.190]
    short_ratio_candidates: list[float | None] = [None, 0.92]
    total_runs = len(ratio_candidates) * len(breadth_candidates) * len(risk_cap_candidates) * len(short_ratio_candidates)
    run_idx = 0

    for ratio_cut in ratio_candidates:
        for breadth_cut in breadth_candidates:
            for risk_cap in risk_cap_candidates:
                for short_ratio_cut in short_ratio_candidates:
                    run_idx += 1
                    print(
                        f"[stressbond_trigger_refine_current] {run_idx}/{total_runs} "
                        f"ratio={ratio_cut:.3f} breadth={breadth_cut:.3f} cap={risk_cap:.3f} "
                        f"short={short_ratio_cut if short_ratio_cut is not None else 'na'}",
                        flush=True,
                    )
                    result, trades = run_current_stressbond_overlay(
                        prices=prices,
                        selected=selected,
                        proxy=proxy,
                        params=params,
                        treasury_code=str(current_context["treasury_code"]),
                        risk_cap=risk_cap,
                        ratio_cut=ratio_cut,
                        breadth_cut=breadth_cut,
                        short_ratio_cut=short_ratio_cut,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                    )
                    short_tag = "na" if short_ratio_cut is None else f"{int(round(short_ratio_cut * 1000)):03d}"
                    strategy = (
                        f"{current_context['strategy']}__trf"
                        f"_vr{int(round(ratio_cut * 1000)):03d}"
                        f"_vb{int(round((breadth_cut + 0.02) * 1000)):03d}"
                        f"_rc{int(round(risk_cap * 1000)):03d}"
                        f"_vs{short_tag}"
                    )
                    short_desc = (
                        "不加5/20短量能门槛"
                        if short_ratio_cut is None
                        else f"补充5/20<{short_ratio_cut:.0%}短量能门槛"
                    )
                    append_variant_result(
                        rows,
                        nav_compare,
                        descriptions,
                        strategy=strategy,
                        result=result,
                        summary=summarize(result, trades, selected),
                        description=(
                            f"围绕当前 {treasury_row['name']} 弱市切债触发继续细化："
                            f"20/60量能<{ratio_cut:.1%}，广度<{breadth_cut:.1%}，"
                            f"风险仓上限 {risk_cap:.1%}，{short_desc}。"
                        ),
                        extra_fields={
                            "ratio_cut": ratio_cut,
                            "breadth_cut": breadth_cut,
                            "risk_cap": risk_cap,
                            "short_ratio_cut": short_ratio_cut,
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
        (f"{current_context['strategy']}__trf_vr900_vb-10_rc188_vsna_nav", "RC188", 1.8),
        (f"{current_context['strategy']}__trf_vr900_vb-10_rc190_vsna_nav", "RC190", 1.8),
        (f"{current_context['strategy']}__trf_vr900_vb-10_rc188_vs920_nav", "RC188 SR92", 1.8),
    ]
    available_plot_lines = filter_available_plot_lines(nav_compare, plot_lines)
    save_plot_and_print_baseline_preview(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        baseline_summary=baseline_summary,
        columns=build_baseline_preview_columns(
            ["strategy", "ratio_cut", "breadth_cut", "risk_cap", "short_ratio_cut"]
        ),
        plot_filename="comparison.png",
        title="Stressbond Trigger Refine Current Comparison",
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
