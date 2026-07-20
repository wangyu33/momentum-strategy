#!/usr/bin/env python3
"""Repair the current best stress-bond near-miss by tightening upper-layer caps around lower risk_cap variants.

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

from archive_data_loaders import build_archive_flat_strategy_paths, load_candidate_prices_with_checks, load_named_market_proxy
from context_loaders import (
    append_overlay_baseline,
    build_overlay_summary_fields,
    load_market_overlay_runtime,
    load_simple_overlay_context,
)
from goal_optimization_common import (
    DEFAULT_FEISHU_WEBHOOK,
    DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER,
    DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN,
    METRIC_TOLERANCE,
    send_improvement_notification,
)
from hs300_regime_common import summarize
from search_utils import (
    add_notify_cli_args,
    extract_ranked_valid_improvements,
)
from stressbond_strategy_helpers import apply_stressbond_overlay
from variant_compare_helpers import (
    append_variant_result,
    build_baseline_preview_columns,
    build_compare_frame,
    filter_available_plot_lines,
    save_plot_and_print_baseline_preview,
    save_best_payload_and_notify_ranked,
    save_best_payload_and_notify_ranked_webhook,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    fetch_histories,
)


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_stressbond_nearmiss_repair")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair the current lower-risk-cap stress-bond near-miss by slightly tightening upper-layer caps."
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
    params["volume_guard_relief_buffer"] = float(
        params.get("volume_guard_relief_buffer", DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER)
    )
    params["volume_guard_soft_span"] = float(params.get("volume_guard_soft_span", DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN))
    base_selected = runtime["base_selected"]
    selected = pd.concat([base_selected, pd.DataFrame([treasury_row])], ignore_index=True)
    _, prices, _ = load_candidate_prices_with_checks(
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

    baseline_result, baseline_trades = apply_stressbond_overlay(
        prices=prices,
        selected=selected,
        proxy=proxy,
        params=params,
        treasury_code=overlay_context["treasury_code"],
        risk_cap=overlay_context["risk_cap"],
        ratio_cut=overlay_context["ratio_cut"],
        breadth_cut=overlay_context["breadth_cut"],
        vg_cap=float(params["volume_guard_cap"]),
        overheat_cap=float(params["overheat_max_exposure"]),
        overheat_hi_cap=float(params["overheat_high_max_exposure"]),
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = append_overlay_baseline(
        rows,
        descriptions,
        nav_compare,
        baseline_result=baseline_result,
        baseline_trades=baseline_trades,
        baseline_selected=selected,
        overlay_context=overlay_context,
    )
    baseline_summary.update(build_overlay_summary_fields(overlay_context))

    risk_cap_candidates = [0.15, 0.16, 0.17, 0.18]
    vg_cap_candidates = [0.29, 0.30]
    overheat_cap_candidates = [0.07, 0.08]
    overheat_hi_cap_candidates = [0.03, 0.04]
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
                        f"[stressbond_nearmiss_repair] {run_idx}/{total_runs} "
                        f"risk_cap={risk_cap:.2f} vg_cap={vg_cap:.2f} "
                        f"overheat={overheat_cap:.2f}/{overheat_hi_cap:.2f}",
                        flush=True,
                    )
                    result, trades = apply_stressbond_overlay(
                        prices=prices,
                        selected=selected,
                        proxy=proxy,
                        params=params,
                        treasury_code=overlay_context["treasury_code"],
                        risk_cap=risk_cap,
                        ratio_cut=overlay_context["ratio_cut"],
                        breadth_cut=overlay_context["breadth_cut"],
                        vg_cap=vg_cap,
                        overheat_cap=overheat_cap,
                        overheat_hi_cap=overheat_hi_cap,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                    )
                    strategy = (
                        f"{market_context['strategy']}__repair"
                        f"_rc{int(round(risk_cap * 100)):02d}"
                        f"_vg{int(round(vg_cap * 100)):02d}"
                        f"_oc{int(round(overheat_cap * 100)):02d}"
                        f"_oh{int(round(overheat_hi_cap * 100)):02d}"
                        f"__stressbond_{overlay_context['treasury_code']}"
                        f"_vr{int(round(float(overlay_context['ratio_cut']) * 100)):02d}"
                        f"_vb{int(round((float(overlay_context['breadth_cut']) + 0.02) * 100)):02d}"
                    )
                    append_variant_result(
                        rows,
                        nav_compare,
                        descriptions,
                        strategy=strategy,
                        result=result,
                        summary=summarize(result, trades, selected),
                        description=(
                            f"保持当前 hybrid market proxy + {treasury_row['name']} 弱市切债框架不变，"
                            f"仅围绕 lower risk cap near-miss 做上层收紧：风险仓上限 {risk_cap:.0%}，"
                            f"弱量能上限 {vg_cap:.0%}，过热上限 {overheat_cap:.0%}，极热上限 {overheat_hi_cap:.0%}。"
                        ),
                        extra_fields={
                            **build_overlay_summary_fields(overlay_context, risk_cap=risk_cap),
                            "vg_cap": vg_cap,
                            "overheat_cap": overheat_cap,
                            "overheat_hi_cap": overheat_hi_cap,
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
            f"{market_context['strategy']}__repair_rc15_vg29_oc07_oh03__stressbond_{overlay_context['treasury_code']}_vr89_vb00_nav",
            "RC15 VG29",
            1.8,
        ),
        (
            f"{market_context['strategy']}__repair_rc16_vg30_oc07_oh03__stressbond_{overlay_context['treasury_code']}_vr89_vb00_nav",
            "RC16 VG30",
            1.8,
        ),
        (
            f"{market_context['strategy']}__repair_rc18_vg30_oc08_oh04__stressbond_{overlay_context['treasury_code']}_vr89_vb00_nav",
            "RC18 VG30",
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
        title="Stressbond Near-Miss Repair Comparison",
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
