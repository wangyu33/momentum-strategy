#!/usr/bin/env python3
"""Test small smooth volume-guard refinements under the current stress-bond baseline.

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
from context_loaders import append_overlay_baseline, load_market_overlay_runtime, load_simple_overlay_context
from goal_optimization_common import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, build_dynamic_core_target_weights, send_improvement_notification
from hs300_regime_common import run_target_weights_strategy, summarize
from overlay_strategy_helpers import apply_risk_cap, apply_treasury_cap, build_market_stress_trigger
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
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    RISK_CODES,
    fetch_histories,
)


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_stressbond_smooth_guard")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search small smooth volume-guard refinements under current stress-bond baseline.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Refresh ETF histories instead of using cached core data.")
    add_notify_cli_args(parser)
    return parser.parse_args()

def build_smoothed_target_weights(
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    relief_buffer: float,
    soft_span: float,
    vg_cap: float,
    overheat_cap: float,
    overheat_hi_cap: float,
) -> tuple[pd.DataFrame, pd.Series]:
    aggressive_weights, aggressive_momentum, _, _, aggressive_trend = build_dynamic_core_target_weights(
        prices,
        lookback=25,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        core_weight=float(params["aggressive_core_weight"]),
        hs300_mom60_cut=0.05,
        hs300_mom120_cut=0.10,
        hs300_ma_window=90,
    )
    conservative_weights, conservative_momentum, _, _, _ = build_dynamic_core_target_weights(
        prices,
        lookback=25,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        core_weight=float(params["conservative_core_weight"]),
        hs300_mom60_cut=0.04,
        hs300_mom120_cut=0.10,
        hs300_ma_window=120,
    )
    aggressive_mask = aggressive_trend & (aggressive_momentum >= float(params["regime_momentum_cut"]))
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    mixed_momentum = conservative_momentum.copy()
    mixed_momentum.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    mixed_momentum = mixed_momentum.rename("current_momentum")

    proxy = proxy.reindex(prices.index).ffill()
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    row_risk_weight = mixed_target_weights[risk_codes].sum(axis=1)
    volume_weak_mask = (
        (proxy["market_amount_ratio_20_60"] < float(params["volume_ratio_cut"]))
        & (proxy["market_amount_ratio_5_20"] < float(params["volume_short_ratio_cut"]))
        & (proxy["market_breadth_proxy"] < float(params["volume_breadth_cut"]))
        & (mixed_momentum <= float(params["volume_guard_momentum_ceiling"]))
    ).fillna(False)

    dynamic_cap = pd.Series(vg_cap, index=prices.index, dtype="float64")
    if relief_buffer > 0:
        span = max(soft_span, 1e-6)
        weakness_20_60 = ((float(params["volume_ratio_cut"]) - proxy["market_amount_ratio_20_60"]) / span).clip(0.0, 1.0)
        weakness_5_20 = ((float(params["volume_short_ratio_cut"]) - proxy["market_amount_ratio_5_20"]) / span).clip(0.0, 1.0)
        weakness_breadth = ((float(params["volume_breadth_cut"]) - proxy["market_breadth_proxy"]) / span).clip(0.0, 1.0)
        momentum_divisor = max(abs(float(params["volume_guard_momentum_ceiling"])), 1e-6)
        weakness_momentum = ((float(params["volume_guard_momentum_ceiling"]) - mixed_momentum) / momentum_divisor).clip(0.0, 1.0)
        guard_strength = ((weakness_20_60 + weakness_5_20 + weakness_breadth + weakness_momentum) / 4.0).clip(0.0, 1.0)
        dynamic_cap = (vg_cap + relief_buffer * (1.0 - guard_strength)).clip(lower=vg_cap, upper=1.0)

    mixed_target_weights = apply_risk_cap(
        mixed_target_weights,
        risk_budget_codes=risk_codes,
        trigger_mask=volume_weak_mask,
        risk_cap=dynamic_cap,
    )

    base_result, _ = run_target_weights_strategy(
        prices,
        pd.DataFrame({"code": list(prices.columns), "theme": list(prices.columns), "name": list(prices.columns)}),
        mixed_target_weights,
        mixed_momentum,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
    )
    post_guard_risk = mixed_target_weights[risk_codes].sum(axis=1)
    reduce_mask = (
        (post_guard_risk > overheat_cap)
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (mixed_momentum >= float(params["overheat_momentum_cut"]))
    )
    mixed_target_weights = apply_risk_cap(
        mixed_target_weights,
        risk_budget_codes=risk_codes,
        trigger_mask=reduce_mask,
        risk_cap=overheat_cap,
    )

    post_overheat_risk = mixed_target_weights[risk_codes].sum(axis=1)
    high_reduce_mask = (
        (post_overheat_risk > overheat_hi_cap)
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (mixed_momentum >= float(params["overheat_high_momentum_cut"]))
    )
    mixed_target_weights = apply_risk_cap(
        mixed_target_weights,
        risk_budget_codes=risk_codes,
        trigger_mask=high_reduce_mask,
        risk_cap=overheat_hi_cap,
    )

    return mixed_target_weights, mixed_momentum


def apply_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    treasury_code: str,
    relief_buffer: float,
    soft_span: float,
    vg_cap: float,
    overheat_cap: float,
    overheat_hi_cap: float,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_weights, mixed_momentum = build_smoothed_target_weights(
        prices=prices,
        proxy=proxy,
        params=params,
        relief_buffer=relief_buffer,
        soft_span=soft_span,
        vg_cap=vg_cap,
        overheat_cap=overheat_cap,
        overheat_hi_cap=overheat_hi_cap,
    )
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    trigger_mask = build_market_stress_trigger(
        proxy,
        index=prices.index,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
    )
    overlaid_weights = apply_treasury_cap(
        base_weights,
        risk_budget_codes=risk_codes,
        treasury_code=treasury_code,
        trigger_mask=trigger_mask,
        risk_cap=risk_cap,
    )

    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)


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
    selected = pd.concat([base_selected, pd.DataFrame([treasury_row])], ignore_index=True)

    baseline_result, baseline_trades = apply_overlay(
        prices=prices,
        selected=selected,
        proxy=proxy,
        params=params,
        treasury_code=overlay_context["treasury_code"],
        relief_buffer=0.0,
        soft_span=0.04,
        vg_cap=float(params["volume_guard_cap"]),
        overheat_cap=float(params["overheat_max_exposure"]),
        overheat_hi_cap=float(params["overheat_high_max_exposure"]),
        risk_cap=overlay_context["risk_cap"],
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
        baseline_selected=selected,
        overlay_context=overlay_context,
    )

    total_runs = 3 * 3 * 3 * 3 * 3
    run_idx = 0
    for relief_buffer in (0.02, 0.04, 0.06):
        for soft_span in (0.03, 0.04, 0.05):
            for vg_cap in (0.30, 0.31, 0.32):
                for overheat_cap in (0.08, 0.09, 0.10):
                    for overheat_hi_cap in (0.04, 0.05, 0.06):
                        overheat_hi_cap = min(overheat_hi_cap, overheat_cap)
                        run_idx += 1
                        print(
                            f"[stressbond_smooth_guard] {run_idx}/{total_runs} "
                            f"gb={relief_buffer:.2f} gs={soft_span:.2f} vg={vg_cap:.2f} "
                            f"overheat={overheat_cap:.2f}/{overheat_hi_cap:.2f}",
                            flush=True,
                        )
                        result, trades = apply_overlay(
                            prices=prices,
                            selected=selected,
                            proxy=proxy,
                            params=params,
                            treasury_code=overlay_context["treasury_code"],
                            relief_buffer=relief_buffer,
                            soft_span=soft_span,
                            vg_cap=vg_cap,
                            overheat_cap=overheat_cap,
                            overheat_hi_cap=overheat_hi_cap,
                            risk_cap=overlay_context["risk_cap"],
                            ratio_cut=overlay_context["ratio_cut"],
                            breadth_cut=overlay_context["breadth_cut"],
                            fee_rate=args.fee_rate,
                            slippage_rate=args.slippage_rate,
                        )
                        strategy = (
                            f"{market_context['strategy']}__smooth"
                            f"_gb{int(round(relief_buffer * 100)):02d}"
                            f"_gs{int(round(soft_span * 100)):02d}"
                            f"_vg{int(round(vg_cap * 100)):02d}"
                            f"_cap{int(round(overheat_cap * 100)):02d}"
                            f"_hi{int(round(overheat_hi_cap * 100)):02d}"
                            f"__stressbond_{overlay_context['treasury_code']}"
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
                                f"保持当前 hybrid market proxy + {treasury_row['name']} 弱市切债结构不变，"
                                f"仅给弱量能保护增加平滑缓冲：基础弱量能上限 {vg_cap:.0%}，"
                                f"最多缓冲 +{relief_buffer:.0%}，软区间 {soft_span:.0%}；"
                                f"过热上限 {overheat_cap:.0%}，极热上限 {overheat_hi_cap:.0%}。"
                            ),
                            extra_fields={
                                "relief_buffer": relief_buffer,
                                "soft_span": soft_span,
                                "vg_cap": vg_cap,
                                "overheat_cap": overheat_cap,
                                "overheat_hi_cap": overheat_hi_cap,
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
            f"{market_context['strategy']}__smooth_gb02_gs03_vg30_cap08_hi04__stressbond_{overlay_context['treasury_code']}_vr90_vb-1_cap20_nav",
            "Smooth 2/3",
            1.8,
        ),
        (
            f"{market_context['strategy']}__smooth_gb04_gs04_vg31_cap09_hi05__stressbond_{overlay_context['treasury_code']}_vr90_vb-1_cap20_nav",
            "Smooth 4/4",
            1.8,
        ),
        (
            f"{market_context['strategy']}__smooth_gb06_gs05_vg32_cap10_hi06__stressbond_{overlay_context['treasury_code']}_vr90_vb-1_cap20_nav",
            "Smooth 6/5",
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
            ["strategy"],
            metric_columns=("annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"),
        ),
        plot_filename="comparison.png",
        title="Stressbond Smooth Guard Comparison",
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
