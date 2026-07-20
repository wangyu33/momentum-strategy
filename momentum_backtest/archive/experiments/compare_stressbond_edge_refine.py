#!/usr/bin/env python3
"""Fine-grid search around the strongest stress-bond near-miss candidate.

This baseline inherits an upstream stress-bond historical winner rather than the official
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

from archive_data_loaders import build_archive_flat_strategy_paths, load_named_market_proxy, load_recent_selected_prices
from goal_optimization_common import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, send_improvement_notification
from hs300_regime_common import summarize
from context_loaders import load_stressbond_nearmiss_context
from overlay_candidate_catalog import lookup_overlay_asset
from search_utils import (
    add_notify_cli_args,
    extract_ranked_valid_improvements,
)
from stressbond_strategy_helpers import apply_stressbond_overlay
from tail_risk_overlay_common import load_market_proxy_best_context
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
    load_fixed_etf_pool,
)


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_stressbond_edge_refine")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-grid refine the strongest stress-bond near-miss candidate.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    add_notify_cli_args(parser)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    market_context = load_market_proxy_best_context()
    overlay_context = load_stressbond_nearmiss_context()
    treasury_row = lookup_overlay_asset(overlay_context["treasury_code"])
    params = dict(market_context["params"])
    params["volume_ratio_cut"] = float(market_context["volume_ratio_cut"])
    params["volume_short_ratio_cut"] = float(market_context["volume_short_ratio_cut"])
    params["volume_breadth_cut"] = float(market_context["volume_breadth_cut"])
    drop_codes = [str(code) for code in params["drop_codes"]]

    base_selected = load_fixed_etf_pool()
    base_selected = base_selected[~base_selected["code"].astype(str).isin(drop_codes)].reset_index(drop=True)
    selected = pd.concat([base_selected, pd.DataFrame([treasury_row])], ignore_index=True)

    prices = load_recent_selected_prices(selected, args.years)
    proxy = load_named_market_proxy(prices, years=args.years, proxy_kind=str(market_context["proxy_kind"]))

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
    append_variant_result(
        rows,
        nav_compare,
        descriptions,
        strategy=str(overlay_context["strategy"]),
        result=baseline_result,
        summary=summarize(baseline_result, baseline_trades, selected),
        description=overlay_context["description"],
    )
    baseline_summary = rows[-1]

    risk_cap_candidates = [0.178, 0.180, 0.182, 0.185, 0.188, 0.190, 0.192]
    vg_cap_candidates = [0.285, 0.290, 0.295]
    overheat_cap_candidates = [0.068, 0.070, 0.072]
    overheat_hi_cap_candidates = [0.028, 0.030, 0.032]
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
                        f"[stressbond_edge_refine] {run_idx}/{total_runs} "
                        f"risk_cap={risk_cap:.3f} vg_cap={vg_cap:.3f} "
                        f"overheat={overheat_cap:.3f}/{overheat_hi_cap:.3f}",
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
                        f"{market_context['strategy']}__edge"
                        f"_rc{int(round(risk_cap * 1000)):03d}"
                        f"_vg{int(round(vg_cap * 1000)):03d}"
                        f"_oc{int(round(overheat_cap * 1000)):03d}"
                        f"_oh{int(round(overheat_hi_cap * 1000)):03d}"
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
                            f"围绕当前 {treasury_row['name']} 最接近有效的新候选继续细化："
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
        strategy_name=str(overlay_context["strategy"]),
    )
    plot_lines = [
        (f"{overlay_context['strategy']}_nav", "Baseline", 2.2),
        (f"{market_context['strategy']}__edge_rc180_vg290_oc070_oh030__stressbond_{overlay_context['treasury_code']}_vr89_vb00_nav", "Edge RC180", 1.8),
        (f"{market_context['strategy']}__edge_rc185_vg290_oc070_oh030__stressbond_{overlay_context['treasury_code']}_vr89_vb00_nav", "Edge RC185", 1.8),
        (f"{market_context['strategy']}__edge_rc188_vg295_oc072_oh032__stressbond_{overlay_context['treasury_code']}_vr89_vb00_nav", "Edge RC188", 1.8),
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
        title="Stressbond Edge Refine Comparison",
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
