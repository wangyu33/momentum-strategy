#!/usr/bin/env python3
"""对比当前市场代理最佳策略之上的简单国债覆盖层规则。

这个 baseline 来自当前 market-proxy 历史研究上下文，不是正式 28.2691 官方基线，
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
from hs300_regime_common import load_cached_data, summarize
from market_proxy_common import evaluate_strategy
from overlay_candidate_catalog import TREASURY_10Y, TREASURY_CANDIDATES, lookup_overlay_asset
from overlay_strategy_helpers import apply_simple_overlay
from search_utils import (
    add_notify_cli_args,
    annotate_valid_improvements,
)
from tail_risk_overlay_common import load_market_proxy_best_context
from variant_compare_helpers import (
    append_best_payload_records,
    append_variant_result,
    build_baseline_preview_columns,
    build_compare_frame,
    filter_available_plot_lines,
    rerank_with_sort_notify_candidates,
    select_nonbaseline_rows,
    select_valid_change_rows,
    save_plot_and_print_baseline_preview,
    save_best_payload_and_notify_webhook,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    fetch_histories,
    load_fixed_etf_pool,
)


OUTPUT_DIR, BEST_PATH, NOTIFY_STATE_PATH = build_archive_flat_strategy_paths("compare_simple_bond_overlay")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比当前市场代理最佳策略之上的简单国债覆盖层规则。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取 ETF 历史数据，而不是复用本地缓存。")
    add_notify_cli_args(parser, default_enabled=True)
    return parser.parse_args()
def main() -> int:
    args = parse_args()
    context = load_market_proxy_best_context()
    params = dict(context["params"])
    params["volume_ratio_cut"] = float(context["volume_ratio_cut"])
    params["volume_short_ratio_cut"] = float(context["volume_short_ratio_cut"])
    params["volume_breadth_cut"] = float(context["volume_breadth_cut"])
    drop_codes = [str(code) for code in params["drop_codes"]]

    base_selected = load_fixed_etf_pool()
    base_selected = base_selected[~base_selected["code"].astype(str).isin(drop_codes)].reset_index(drop=True)
    treasury_candidates = list(TREASURY_CANDIDATES)
    # 这条历史链最初就是围绕 10 年国债基线扩展，显式写出而不是依赖候选顺序。
    baseline_treasury = lookup_overlay_asset(str(TREASURY_10Y["code"]))
    selected_for_prices, prices, available_treasury_candidates = load_candidate_prices_with_checks(
        base_selected,
        treasury_candidates,
        years=args.years,
        refresh=args.refresh,
        fetch_fn=fetch_histories,
        label="treasury candidates",
        required_candidates=[baseline_treasury],
        required_label="required baseline treasury history",
    )
    if not available_treasury_candidates:
        print("[warn] no treasury candidates available; only baseline result will be generated")

    proxy = load_named_market_proxy(
        prices,
        years=args.years,
        refresh=args.refresh,
        proxy_kind=str(context["proxy_kind"]),
        risk_codes=[str(code) for code in params.get("risk_codes", [])] if params.get("risk_codes") is not None else None,
    )

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = build_compare_frame(prices)

    baseline_selected = pd.concat([base_selected, pd.DataFrame([baseline_treasury])], ignore_index=True)
    baseline_selected = baseline_selected.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)
    baseline_prices = filter_selected_price_columns(prices, baseline_selected)
    baseline_result, baseline_trades = evaluate_strategy(
        selected=baseline_selected,
        prices=baseline_prices,
        market_proxy=proxy,
        params=params,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    append_variant_result(
        rows,
        nav_compare,
        descriptions,
        strategy=str(context["strategy"]),
        result=baseline_result,
        summary=summarize(baseline_result, baseline_trades, baseline_selected),
        description=str(context["description"]),
    )
    baseline_summary = rows[-1]

    for treasury in available_treasury_candidates:
        selected = pd.concat([base_selected, pd.DataFrame([treasury])], ignore_index=True)
        candidate_prices = filter_selected_price_columns(prices, selected)
        for risk_cap in (0.20, 0.30, 0.40, 0.50, 0.60):
            for drawdown_cut in (-0.03, -0.05, -0.08, -0.10):
                result, trades = apply_simple_overlay(
                    prices=candidate_prices,
                    selected=selected,
                    proxy=proxy,
                    params=params,
                    treasury_code=str(treasury["code"]),
                    mode="drawdown_only",
                    risk_cap=risk_cap,
                    drawdown_cut=drawdown_cut,
                    ratio_cut=None,
                    breadth_cut=None,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                )
                strategy = (
                    f"{context['strategy']}__drawbond_{treasury['code']}"
                    f"_dd{int(round(abs(drawdown_cut) * 100)):02d}_cap{int(round(risk_cap * 100)):02d}"
                )
                append_variant_result(
                    rows,
                    nav_compare,
                    descriptions,
                    strategy=strategy,
                    result=result,
                    summary=summarize(result, trades, selected),
                    description=(
                        f"保持最新 hybrid market proxy 策略不变，仅加简单回撤切债：当策略回撤<={drawdown_cut:.0%} 时，"
                        f"把风险资产上限压到 {risk_cap:.0%}，削减部分切到 {treasury['name']}。"
                    ),
                    extra_fields={
                        "treasury_code": str(treasury["code"]),
                        "mode": "drawdown_only",
                        "risk_cap": risk_cap,
                        "drawdown_cut": drawdown_cut,
                    },
                )
            for ratio_cut in (0.88, 0.90, 0.92):
                for breadth_cut in (-0.03, -0.02, -0.01):
                    result, trades = apply_simple_overlay(
                        prices=candidate_prices,
                        selected=selected,
                        proxy=proxy,
                        params=params,
                        treasury_code=str(treasury["code"]),
                        mode="market_stress_only",
                        risk_cap=risk_cap,
                        drawdown_cut=None,
                        ratio_cut=ratio_cut,
                        breadth_cut=breadth_cut,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                    )
                    strategy = (
                        f"{context['strategy']}__stressbond_{treasury['code']}"
                        f"_vr{int(round(ratio_cut * 100)):02d}_vb{int(round((breadth_cut + 0.02) * 100)):02d}"
                        f"_cap{int(round(risk_cap * 100)):02d}"
                    )
                    append_variant_result(
                        rows,
                        nav_compare,
                        descriptions,
                        strategy=strategy,
                        result=result,
                        summary=summarize(result, trades, selected),
                        description=(
                            f"保持最新 hybrid market proxy 策略不变，仅加简单市场压力切债：当市场量能20/60<{ratio_cut:.0%} 且广度<{breadth_cut:.1%} 时，"
                            f"把风险资产上限压到 {risk_cap:.0%}，削减部分切到 {treasury['name']}。"
                        ),
                        extra_fields={
                            "treasury_code": str(treasury["code"]),
                            "mode": "market_stress_only",
                            "risk_cap": risk_cap,
                            "ratio_cut": ratio_cut,
                            "breadth_cut": breadth_cut,
                        },
                    )

    summary_df = annotate_valid_improvements(
        rows,
        baseline_summary,
        metric_tolerance=METRIC_TOLERANCE,
        strategy_name=str(context["strategy"]),
    )
    valid_df = select_valid_change_rows(summary_df, rerank_fn=rerank_with_sort_notify_candidates)
    fallback_context_df = select_nonbaseline_rows(
        summary_df,
        baseline_strategy=str(context["strategy"]),
        rerank_fn=rerank_with_sort_notify_candidates,
    )

    plot_lines = [
        (f"{context['strategy']}_nav", "Baseline", 2.2),
        (f"{context['strategy']}__stressbond_{baseline_treasury['code']}_vr90_vb00_cap20_nav", "Stress Bond 20", 1.8),
        (f"{context['strategy']}__stressbond_{baseline_treasury['code']}_vr90_vb00_cap30_nav", "Stress Bond 30", 1.8),
        (f"{context['strategy']}__drawbond_{baseline_treasury['code']}_dd05_cap20_nav", "Drawdown Bond", 1.8),
    ]
    available_plot_lines = filter_available_plot_lines(nav_compare, plot_lines)
    save_plot_and_print_baseline_preview(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        baseline_summary=baseline_summary,
        columns=build_baseline_preview_columns(["strategy", "mode", "treasury_code"]),
        plot_filename="comparison.png",
        title="Simple Bond Overlay Comparison",
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
    append_best_payload_records(
        BEST_PATH,
        key="strict_improvements",
        records_df=fallback_context_df if valid_df.empty else pd.DataFrame(),
        limit=1,
        print_fn=print,
        message="[info] preserved fallback overlay context in best.json for downstream historical chains",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
