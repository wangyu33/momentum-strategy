#!/usr/bin/env python3
"""对比现金类防守资产替代当前债券覆盖层的历史实验脚本。

这个 baseline 继承自 simple bond overlay 的历史搜索赢家，不是正式 28.2691
官方基线，因此本脚本应保持历史语义，不做官方净值锚定。
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
from context_loaders import (
    append_overlay_baseline,
    build_overlay_summary_fields,
    load_market_overlay_runtime,
    load_simple_overlay_context,
)
from overlay_candidate_catalog import CASH_CANDIDATES, TREASURY_10Y, TREASURY_30Y, lookup_overlay_asset
from overlay_strategy_helpers import apply_simple_overlay
from search_utils import add_notify_cli_args, annotate_valid_improvements
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


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_cash_overlay_candidates")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare cash-like overlays against the current notified defensive overlay.")
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
    params = runtime["params"]
    base_selected = runtime["base_selected"]
    candidate_rows = [lookup_overlay_asset(overlay_context["treasury_code"]), TREASURY_10Y, TREASURY_30Y, *CASH_CANDIDATES]
    unique_candidates: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    for row in candidate_rows:
        if row["code"] in seen_codes:
            continue
        seen_codes.add(row["code"])
        unique_candidates.append(row)

    selected_for_prices, prices, unique_candidates = load_candidate_prices_with_checks(
        base_selected,
        unique_candidates,
        years=args.years,
        refresh=args.refresh,
        fetch_fn=fetch_histories,
        label="cash candidates",
    )
    if not unique_candidates:
        raise RuntimeError("missing required cash overlay candidate histories")

    proxy = load_named_market_proxy(
        prices,
        years=args.years,
        refresh=args.refresh,
        proxy_kind=str(market_context["proxy_kind"]),
    )

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = build_compare_frame(prices)

    baseline_candidate = lookup_overlay_asset(overlay_context["treasury_code"])
    baseline_selected = pd.concat([base_selected, pd.DataFrame([baseline_candidate])], ignore_index=True)
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
    baseline_summary.update(
        build_overlay_summary_fields(
            overlay_context,
            treasury_code=str(baseline_candidate["code"]),
            defensive_code=str(baseline_candidate["code"]),
            include_mode=True,
            include_drawdown_cut=True,
        )
    )

    for candidate in unique_candidates:
        selected = pd.concat([base_selected, pd.DataFrame([candidate])], ignore_index=True)
        candidate_prices = filter_selected_price_columns(prices, selected)
        risk_caps = [overlay_context["risk_cap"]] if candidate["code"] == overlay_context["treasury_code"] else [
            overlay_context["risk_cap"],
            min(float(overlay_context["risk_cap"]) + 0.10, 0.40),
        ]
        for risk_cap in risk_caps:
            result, trades = apply_simple_overlay(
                prices=candidate_prices,
                selected=selected,
                proxy=proxy,
                params=params,
                treasury_code=str(candidate["code"]),
                mode=overlay_context["mode"],
                risk_cap=float(risk_cap),
                drawdown_cut=overlay_context["drawdown_cut"],
                ratio_cut=overlay_context["ratio_cut"],
                breadth_cut=overlay_context["breadth_cut"],
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
            )
            if candidate["code"] == overlay_context["treasury_code"] and abs(float(risk_cap) - float(overlay_context["risk_cap"])) < 1e-12:
                strategy = str(overlay_context["strategy"])
            else:
                strategy = (
                    f"{market_context['strategy']}__cashguard_{candidate['code']}"
                    f"_vr{int(round(float(overlay_context['ratio_cut']) * 100)):02d}"
                    f"_vb{int(round((float(overlay_context['breadth_cut']) + 0.02) * 100)):02d}"
                    f"_cap{int(round(float(risk_cap) * 100)):02d}"
                )
            if strategy != str(overlay_context["strategy"]):
                append_variant_result(
                    rows,
                    nav_compare,
                    descriptions,
                    strategy=strategy,
                    result=result,
                    summary=summarize(result, trades, selected),
                    description=(
                        f"保持当前 hybrid market proxy + 弱市切防守框架不变，仅把弱市防守资产替换为 {candidate['name']}。"
                        f"触发条件仍为市场量能20/60<{float(overlay_context['ratio_cut']):.0%} 且广度<{float(overlay_context['breadth_cut']):.1%}，"
                        f"风险资产上限压到 {float(risk_cap):.0%}。"
                    ),
                    extra_fields=build_overlay_summary_fields(
                        overlay_context,
                        treasury_code=str(candidate["code"]),
                        defensive_code=str(candidate["code"]),
                        risk_cap=risk_cap,
                        include_mode=True,
                        include_drawdown_cut=True,
                    ),
                )

    summary_df = annotate_valid_improvements(
        rows,
        baseline_summary,
        metric_tolerance=METRIC_TOLERANCE,
        strategy_name=str(overlay_context["strategy"]),
    )
    valid_df = select_valid_change_rows(summary_df)

    preview_columns = build_baseline_preview_columns(["strategy", "defensive_code", "risk_cap"])
    plot_lines = [
        (f"{overlay_context['strategy']}_nav", "Baseline", 2.2),
        (
            f"{market_context['strategy']}__cashguard_511880_vr90_vb-1_cap20_nav",
            "Cash 511880",
            1.8,
        ),
        (
            f"{market_context['strategy']}__cashguard_511990_vr90_vb-1_cap20_nav",
            "Cash 511990",
            1.8,
        ),
        (
            f"{market_context['strategy']}__cashguard_511260_vr90_vb-1_cap30_nav",
            "10Y Cap30",
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
        title="Cash Overlay Candidates Comparison",
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
