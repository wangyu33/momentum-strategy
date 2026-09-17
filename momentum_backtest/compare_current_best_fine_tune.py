#!/usr/bin/env python3
"""围绕当前正式 ETF 策略做小范围微调搜索。"""

from __future__ import annotations

import argparse
import json

try:
    from .runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

try:
    from .compare_hs300_regime_fixes import load_cached_data, summarize
    from .compare_goal_optimizations import (
        DEFAULT_FEISHU_WEBHOOK,
        GOAL_OUTPUT_DIR,
        METRIC_TOLERANCE,
        load_market_volume_proxy,
        send_improvement_notification,
        DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER,
        DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN,
    )
    from .search_utils import (
        add_notify_cli_args,
        load_incremental_notify_candidates,
        notify_best_candidate,
        raise_if_missing_required_histories,
        try_join_missing_candidate_histories,
        write_json_atomic,
    )
    from .official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from compare_hs300_regime_fixes import load_cached_data, summarize
    from compare_goal_optimizations import (
        DEFAULT_FEISHU_WEBHOOK,
        GOAL_OUTPUT_DIR,
        METRIC_TOLERANCE,
        load_market_volume_proxy,
        send_improvement_notification,
        DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER,
        DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN,
    )
    from search_utils import (
        add_notify_cli_args,
        load_incremental_notify_candidates,
        notify_best_candidate,
        raise_if_missing_required_histories,
        try_join_missing_candidate_histories,
        write_json_atomic,
    )
    from official_baseline import apply_official_baseline_nav_anchor

from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    build_default_strategy_params,
    ensure_output_dirs,
    fetch_histories,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = GOAL_OUTPUT_DIR / "current_best_fine_tune"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="围绕当前正式 ETF 策略做小范围微调搜索。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取 ETF 历史数据，而不是复用本地缓存。")
    add_notify_cli_args(parser)
    parser.add_argument(
        "--search-mode",
        choices=["full", "smooth_local", "smooth_nearline", "frontload_trigger", "volume_frontload"],
        default="full",
        help="候选生成模式。`smooth_local` 会把搜索范围收窄到更局部的平滑量能保护变体。",
    )
    return parser.parse_args()


def normalize_params(raw_params: dict[str, object]) -> dict[str, object]:
    return {
        "proxy_kind": str(raw_params.get("proxy_kind", "hybrid_breadth_blend")),
        "signal_quality_method": str(raw_params.get("signal_quality_method", "raw")),
        "signal_slope_penalty": float(raw_params.get("signal_slope_penalty", 0.0)),
        "signal_confirmation_lookback": int(raw_params.get("signal_confirmation_lookback", 0)),
        "signal_confirmation_top_n": int(raw_params.get("signal_confirmation_top_n", 0)),
        "signal_leader_margin": float(raw_params.get("signal_leader_margin", 0.0)),
        "risk_codes": [str(code) for code in raw_params.get("risk_codes", [])],
        "defensive_codes": [str(code) for code in raw_params.get("defensive_codes", [])],
        "aggressive_core_weight": float(raw_params["aggressive_core_weight"]),
        "conservative_core_weight": float(raw_params["conservative_core_weight"]),
        "regime_momentum_cut": float(raw_params["regime_momentum_cut"]),
        "volume_ratio_cut": float(raw_params["volume_ratio_cut"]),
        "volume_short_ratio_cut": float(raw_params["volume_short_ratio_cut"]),
        "volume_breadth_cut": float(raw_params["volume_breadth_cut"]),
        "volume_guard_cap": float(raw_params["volume_guard_cap"]),
        "volume_guard_momentum_ceiling": float(raw_params["volume_guard_momentum_ceiling"]),
        "volume_guard_relief_buffer": float(
            raw_params.get("volume_guard_relief_buffer", DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER)
        ),
        "volume_guard_soft_span": float(raw_params.get("volume_guard_soft_span", DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN)),
        "overheat_drawdown_cut": float(raw_params["overheat_drawdown_cut"]),
        "overheat_momentum_cut": float(raw_params["overheat_momentum_cut"]),
        "overheat_max_exposure": float(raw_params["overheat_max_exposure"]),
        "overheat_high_momentum_cut": float(raw_params["overheat_high_momentum_cut"]),
        "overheat_high_max_exposure": float(raw_params["overheat_high_max_exposure"]),
    }


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def strategy_name(drop_codes: list[str], params: dict[str, object]) -> str:
    prefix = "regime_mix"
    for code in drop_codes:
        prefix += f"_rm{code}"
    base = (
        f"{prefix}_volume_guard_"
        f"ag{int(round(params['aggressive_core_weight'] * 100)):02d}_"
        f"co{int(round(params['conservative_core_weight'] * 100)):02d}_"
        f"vr{int(round(params['volume_ratio_cut'] * 100)):02d}_"
        f"vs{int(round(params['volume_short_ratio_cut'] * 100)):02d}_"
        f"vb{int(round((params['volume_breadth_cut'] + 0.02) * 100)):02d}_"
        f"vg{int(round(params['volume_guard_cap'] * 100)):02d}_"
        f"cap{int(round(params['overheat_max_exposure'] * 100)):02d}_"
        f"hi{int(round(params['overheat_high_max_exposure'] * 100)):02d}_"
        f"dd{int(round(abs(params['overheat_drawdown_cut']) * 100)):02d}"
    )
    suffix_parts: list[str] = []
    if abs(params["regime_momentum_cut"] - 0.06) > 1e-12:
        suffix_parts.append(f"rm{int(round(params['regime_momentum_cut'] * 100)):02d}")
    if abs(params["volume_guard_momentum_ceiling"] - 0.18) > 1e-12:
        suffix_parts.append(f"vm{int(round(params['volume_guard_momentum_ceiling'] * 100)):02d}")
    if abs(params["volume_guard_relief_buffer"] - DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER) > 1e-12:
        suffix_parts.append(f"gb{int(round(params['volume_guard_relief_buffer'] * 100)):02d}")
        if abs(params["volume_guard_soft_span"] - DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN) > 1e-12:
            suffix_parts.append(f"gs{int(round(params['volume_guard_soft_span'] * 100)):02d}")
    if abs(params["overheat_momentum_cut"] - 0.25) > 1e-12:
        suffix_parts.append(f"oh{int(round(params['overheat_momentum_cut'] * 100)):02d}")
    if abs(params["overheat_high_momentum_cut"] - 0.32) > 1e-12:
        suffix_parts.append(f"ohh{int(round(params['overheat_high_momentum_cut'] * 100)):02d}")

    suffix = "__baseline_default"
    if not suffix_parts:
        return f"{base}{suffix}"
    return f"{base}_{'_'.join(suffix_parts)}{suffix}"


def strategy_description(drop_codes: list[str], params: dict[str, object]) -> str:
    removed = "、".join(drop_codes) if drop_codes else "无"
    return (
        f"在当前正式默认策略附近做局部微调：移除 ETF {removed}；"
        f"进攻/保守核心仓 {params['aggressive_core_weight']:.0%}/{params['conservative_core_weight']:.0%}，"
        f"regime 切换阈值 {params['regime_momentum_cut']:.0%}；"
        f"市场量能门槛为 20日/60日={params['volume_ratio_cut']:.0%}、5日/20日={params['volume_short_ratio_cut']:.0%}、"
        f"广度<{params['volume_breadth_cut']:.1%}，弱量能基础上限 {params['volume_guard_cap']:.0%}，"
        f"且仅在综合动量不高于 {params['volume_guard_momentum_ceiling']:.0%} 时触发"
        f"{'，并允许最多+' + format(params['volume_guard_relief_buffer'], '.0%') + '的平滑缓冲' if params['volume_guard_relief_buffer'] > 0 else ''}"
        f"{'（软区间=' + format(params['volume_guard_soft_span'], '.0%') + '）' if params['volume_guard_relief_buffer'] > 0 else ''}；"
        f"过热保护为回撤阈值 {params['overheat_drawdown_cut']:.0%}、动量>{params['overheat_momentum_cut']:.0%} 时压到 "
        f"{params['overheat_max_exposure']:.0%}，极热动量>{params['overheat_high_momentum_cut']:.0%} 时压到 "
        f"{params['overheat_high_max_exposure']:.0%}；正式默认基线不启用弱市切债，"
        f"但当防守信号历史动量分位达到 {params.get('defensive_signal_selected_momentum_pct_cap_start', 1.0):.0%} 时，"
        f"总仓位进一步压到 {params.get('defensive_signal_selected_momentum_pct_cap_floor', 1.0):.0%}。"
    )


def build_specs(base: dict[str, object], search_mode: str = "full") -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [{**base}]

    def add_variant(**updates: float) -> None:
        spec = {**base, **updates}
        spec["aggressive_core_weight"] = clamp(spec["aggressive_core_weight"], 0.20, 0.40)
        spec["conservative_core_weight"] = clamp(spec["conservative_core_weight"], 0.20, 0.40)
        spec["regime_momentum_cut"] = clamp(spec["regime_momentum_cut"], 0.04, 0.10)
        spec["volume_ratio_cut"] = clamp(spec["volume_ratio_cut"], 0.85, 0.99)
        spec["volume_short_ratio_cut"] = clamp(spec["volume_short_ratio_cut"], 0.85, 0.99)
        spec["volume_breadth_cut"] = clamp(spec["volume_breadth_cut"], -0.03, 0.02)
        spec["volume_guard_cap"] = clamp(spec["volume_guard_cap"], 0.30, 0.70)
        spec["volume_guard_momentum_ceiling"] = clamp(spec["volume_guard_momentum_ceiling"], 0.10, 0.30)
        spec["volume_guard_relief_buffer"] = clamp(spec["volume_guard_relief_buffer"], 0.00, 0.12)
        spec["volume_guard_soft_span"] = clamp(spec["volume_guard_soft_span"], 0.02, 0.10)
        spec["overheat_drawdown_cut"] = clamp(spec["overheat_drawdown_cut"], -0.05, -0.01)
        spec["overheat_momentum_cut"] = clamp(spec["overheat_momentum_cut"], 0.18, 0.35)
        spec["overheat_max_exposure"] = clamp(spec["overheat_max_exposure"], 0.08, 0.35)
        spec["overheat_high_momentum_cut"] = clamp(spec["overheat_high_momentum_cut"], 0.24, 0.45)
        spec["overheat_high_max_exposure"] = clamp(spec["overheat_high_max_exposure"], 0.04, 0.25)
        if spec["overheat_high_max_exposure"] > spec["overheat_max_exposure"]:
            spec["overheat_high_max_exposure"] = spec["overheat_max_exposure"]
        if spec["overheat_high_momentum_cut"] < spec["overheat_momentum_cut"]:
            spec["overheat_high_momentum_cut"] = spec["overheat_momentum_cut"]
        specs.append(spec)

    if search_mode == "volume_frontload":
        for aggressive_core_weight, conservative_core_weight in (
            (0.28, 0.27),
            (0.28, 0.28),
        ):
            for regime_momentum_cut in (0.05, 0.06):
                for volume_ratio_cut in (0.91, 0.92, 0.93):
                    for volume_short_ratio_cut in (0.90, 0.91, 0.92):
                        for volume_breadth_cut in (-0.01, 0.00, 0.01):
                            for volume_guard_momentum_ceiling in (0.15, 0.16):
                                add_variant(
                                    aggressive_core_weight=aggressive_core_weight,
                                    conservative_core_weight=conservative_core_weight,
                                    regime_momentum_cut=regime_momentum_cut,
                                    volume_ratio_cut=volume_ratio_cut,
                                    volume_short_ratio_cut=volume_short_ratio_cut,
                                    volume_breadth_cut=volume_breadth_cut,
                                    volume_guard_cap=0.30,
                                    volume_guard_momentum_ceiling=volume_guard_momentum_ceiling,
                                    overheat_drawdown_cut=-0.02,
                                    overheat_momentum_cut=0.25,
                                    overheat_max_exposure=0.08,
                                    overheat_high_momentum_cut=0.32,
                                    overheat_high_max_exposure=0.04,
                                    volume_guard_relief_buffer=0.0,
                                    volume_guard_soft_span=DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN,
                                )
        unique: dict[str, dict[str, float]] = {}
        ordered_keys = [
            "aggressive_core_weight",
            "conservative_core_weight",
            "regime_momentum_cut",
            "volume_ratio_cut",
            "volume_short_ratio_cut",
            "volume_breadth_cut",
            "volume_guard_cap",
            "volume_guard_momentum_ceiling",
            "volume_guard_relief_buffer",
            "volume_guard_soft_span",
            "overheat_drawdown_cut",
            "overheat_momentum_cut",
            "overheat_max_exposure",
            "overheat_high_momentum_cut",
            "overheat_high_max_exposure",
        ]
        for spec in specs:
            key = json.dumps({k: round(spec[k], 6) for k in ordered_keys}, sort_keys=True)
            unique[key] = spec
        return list(unique.values())

    if search_mode == "frontload_trigger":
        for aggressive_core_weight, conservative_core_weight in (
            (0.28, 0.27),
            (0.28, 0.28),
            (0.29, 0.27),
        ):
            for regime_momentum_cut in (0.05, 0.06):
                for volume_guard_momentum_ceiling in (0.14, 0.15, 0.16):
                    for overheat_drawdown_cut in (-0.01, -0.02):
                        for overheat_momentum_cut in (0.23, 0.24, 0.25):
                            for overheat_high_momentum_cut in (0.30, 0.31, 0.32):
                                if overheat_high_momentum_cut < overheat_momentum_cut:
                                    continue
                                add_variant(
                                    aggressive_core_weight=aggressive_core_weight,
                                    conservative_core_weight=conservative_core_weight,
                                    regime_momentum_cut=regime_momentum_cut,
                                    volume_guard_cap=0.30,
                                    volume_guard_momentum_ceiling=volume_guard_momentum_ceiling,
                                    overheat_drawdown_cut=overheat_drawdown_cut,
                                    overheat_momentum_cut=overheat_momentum_cut,
                                    overheat_max_exposure=0.08,
                                    overheat_high_momentum_cut=overheat_high_momentum_cut,
                                    overheat_high_max_exposure=0.04,
                                    volume_guard_relief_buffer=0.0,
                                    volume_guard_soft_span=DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN,
                                )
        unique: dict[str, dict[str, float]] = {}
        ordered_keys = [
            "aggressive_core_weight",
            "conservative_core_weight",
            "regime_momentum_cut",
            "volume_ratio_cut",
            "volume_short_ratio_cut",
            "volume_breadth_cut",
            "volume_guard_cap",
            "volume_guard_momentum_ceiling",
            "volume_guard_relief_buffer",
            "volume_guard_soft_span",
            "overheat_drawdown_cut",
            "overheat_momentum_cut",
            "overheat_max_exposure",
            "overheat_high_momentum_cut",
            "overheat_high_max_exposure",
        ]
        for spec in specs:
            key = json.dumps({k: round(spec[k], 6) for k in ordered_keys}, sort_keys=True)
            unique[key] = spec
        return list(unique.values())

    if search_mode == "smooth_nearline":
        for relief_buffer in (0.02, 0.04, 0.06):
            for soft_span in (0.03, 0.04, 0.05):
                for vg_cap in (0.30, 0.31, 0.32):
                    for overheat_cap in (0.08, 0.09, 0.10):
                        for hi_cap in (0.04, 0.05, 0.06):
                            for vm_cap in (0.15, 0.16):
                                add_variant(
                                    aggressive_core_weight=0.28,
                                    conservative_core_weight=0.28,
                                    volume_guard_cap=vg_cap,
                                    volume_guard_momentum_ceiling=vm_cap,
                                    overheat_max_exposure=overheat_cap,
                                    overheat_high_max_exposure=min(hi_cap, overheat_cap),
                                    volume_guard_relief_buffer=relief_buffer,
                                    volume_guard_soft_span=soft_span,
                                )
                                add_variant(
                                    aggressive_core_weight=0.29,
                                    conservative_core_weight=0.28,
                                    volume_guard_cap=vg_cap,
                                    volume_guard_momentum_ceiling=vm_cap,
                                    overheat_max_exposure=overheat_cap,
                                    overheat_high_max_exposure=min(hi_cap, overheat_cap),
                                    volume_guard_relief_buffer=relief_buffer,
                                    volume_guard_soft_span=soft_span,
                                )
                                add_variant(
                                    aggressive_core_weight=0.28,
                                    conservative_core_weight=0.28,
                                    volume_guard_cap=vg_cap,
                                    volume_guard_momentum_ceiling=vm_cap,
                                    overheat_max_exposure=overheat_cap,
                                    overheat_high_max_exposure=min(hi_cap, overheat_cap),
                                    volume_guard_relief_buffer=relief_buffer,
                                    volume_guard_soft_span=soft_span,
                                    regime_momentum_cut=0.05,
                                )
        unique: dict[str, dict[str, float]] = {}
        ordered_keys = [
            "aggressive_core_weight",
            "conservative_core_weight",
            "regime_momentum_cut",
            "volume_ratio_cut",
            "volume_short_ratio_cut",
            "volume_breadth_cut",
            "volume_guard_cap",
            "volume_guard_momentum_ceiling",
            "volume_guard_relief_buffer",
            "volume_guard_soft_span",
            "overheat_drawdown_cut",
            "overheat_momentum_cut",
            "overheat_max_exposure",
            "overheat_high_momentum_cut",
            "overheat_high_max_exposure",
        ]
        for spec in specs:
            key = json.dumps({k: round(spec[k], 6) for k in ordered_keys}, sort_keys=True)
            unique[key] = spec
        return list(unique.values())

    if search_mode == "smooth_local":
        for relief_buffer in (0.02, 0.04, 0.06):
            for soft_span in (0.03, 0.04, 0.05):
                for vg_cap in (0.30, 0.31, 0.32, 0.33):
                    for overheat_cap in (0.08, 0.09, 0.10):
                        for hi_cap in (0.04, 0.05, 0.06):
                            add_variant(
                                aggressive_core_weight=0.28,
                                conservative_core_weight=0.28,
                                volume_guard_cap=vg_cap,
                                overheat_max_exposure=overheat_cap,
                                overheat_high_max_exposure=min(hi_cap, overheat_cap),
                                volume_guard_relief_buffer=relief_buffer,
                                volume_guard_soft_span=soft_span,
                            )
                            add_variant(
                                aggressive_core_weight=0.28,
                                conservative_core_weight=0.28,
                                volume_guard_cap=vg_cap,
                                overheat_max_exposure=overheat_cap,
                                overheat_high_max_exposure=min(hi_cap, overheat_cap),
                                volume_guard_relief_buffer=relief_buffer,
                                volume_guard_soft_span=soft_span,
                                volume_guard_momentum_ceiling=0.15,
                            )
                            add_variant(
                                aggressive_core_weight=0.29,
                                conservative_core_weight=0.28,
                                volume_guard_cap=vg_cap,
                                overheat_max_exposure=overheat_cap,
                                overheat_high_max_exposure=min(hi_cap, overheat_cap),
                                volume_guard_relief_buffer=relief_buffer,
                                volume_guard_soft_span=soft_span,
                            )
                            add_variant(
                                aggressive_core_weight=0.28,
                                conservative_core_weight=0.27,
                                volume_guard_cap=vg_cap,
                                overheat_max_exposure=overheat_cap,
                                overheat_high_max_exposure=min(hi_cap, overheat_cap),
                                volume_guard_relief_buffer=relief_buffer,
                                volume_guard_soft_span=soft_span,
                            )
        unique: dict[str, dict[str, float]] = {}
        ordered_keys = [
            "aggressive_core_weight",
            "conservative_core_weight",
            "regime_momentum_cut",
            "volume_ratio_cut",
            "volume_short_ratio_cut",
            "volume_breadth_cut",
            "volume_guard_cap",
            "volume_guard_momentum_ceiling",
            "volume_guard_relief_buffer",
            "volume_guard_soft_span",
            "overheat_drawdown_cut",
            "overheat_momentum_cut",
            "overheat_max_exposure",
            "overheat_high_momentum_cut",
            "overheat_high_max_exposure",
        ]
        for spec in specs:
            key = json.dumps({k: round(spec[k], 6) for k in ordered_keys}, sort_keys=True)
            unique[key] = spec
        return list(unique.values())

    for delta in (-0.01, 0.01):
        add_variant(aggressive_core_weight=base["aggressive_core_weight"] + delta)
        add_variant(conservative_core_weight=base["conservative_core_weight"] + delta)
        add_variant(regime_momentum_cut=base["regime_momentum_cut"] + delta)
        add_variant(volume_ratio_cut=base["volume_ratio_cut"] + delta)
        add_variant(volume_short_ratio_cut=base["volume_short_ratio_cut"] + delta)
        add_variant(volume_breadth_cut=base["volume_breadth_cut"] + delta)
        add_variant(volume_guard_cap=base["volume_guard_cap"] + delta * 2)
        add_variant(volume_guard_momentum_ceiling=base["volume_guard_momentum_ceiling"] + delta * 2)
        add_variant(volume_guard_relief_buffer=base["volume_guard_relief_buffer"] + delta * 2)
        add_variant(volume_guard_soft_span=base["volume_guard_soft_span"] + delta * 2)
        add_variant(overheat_drawdown_cut=base["overheat_drawdown_cut"] + delta)
        add_variant(overheat_momentum_cut=base["overheat_momentum_cut"] + delta)
        add_variant(overheat_max_exposure=base["overheat_max_exposure"] + delta * 2)
        add_variant(overheat_high_momentum_cut=base["overheat_high_momentum_cut"] + delta)
        add_variant(overheat_high_max_exposure=base["overheat_high_max_exposure"] + delta * 2)

    add_variant(
        volume_guard_cap=base["volume_guard_cap"] - 0.02,
        volume_guard_momentum_ceiling=base["volume_guard_momentum_ceiling"] - 0.02,
        overheat_max_exposure=base["overheat_max_exposure"] - 0.02,
        overheat_high_max_exposure=base["overheat_high_max_exposure"] - 0.02,
    )
    add_variant(
        volume_guard_cap=base["volume_guard_cap"] + 0.02,
        volume_guard_momentum_ceiling=base["volume_guard_momentum_ceiling"] + 0.02,
        overheat_max_exposure=base["overheat_max_exposure"] + 0.02,
        overheat_high_max_exposure=base["overheat_high_max_exposure"] + 0.02,
    )
    add_variant(
        volume_ratio_cut=base["volume_ratio_cut"] - 0.01,
        volume_short_ratio_cut=base["volume_short_ratio_cut"] - 0.01,
        volume_breadth_cut=base["volume_breadth_cut"] - 0.01,
    )
    add_variant(
        volume_ratio_cut=base["volume_ratio_cut"] + 0.01,
        volume_short_ratio_cut=base["volume_short_ratio_cut"] + 0.01,
        volume_breadth_cut=base["volume_breadth_cut"] + 0.01,
    )
    add_variant(
        overheat_drawdown_cut=base["overheat_drawdown_cut"] - 0.01,
        overheat_momentum_cut=base["overheat_momentum_cut"] - 0.01,
        overheat_high_momentum_cut=base["overheat_high_momentum_cut"] - 0.01,
    )
    add_variant(
        overheat_drawdown_cut=base["overheat_drawdown_cut"] + 0.01,
        overheat_momentum_cut=base["overheat_momentum_cut"] + 0.01,
        overheat_high_momentum_cut=base["overheat_high_momentum_cut"] + 0.01,
    )
    add_variant(
        aggressive_core_weight=base["aggressive_core_weight"] + 0.01,
        conservative_core_weight=base["conservative_core_weight"] + 0.01,
        regime_momentum_cut=base["regime_momentum_cut"] + 0.01,
    )
    add_variant(
        aggressive_core_weight=base["aggressive_core_weight"] - 0.01,
        conservative_core_weight=base["conservative_core_weight"] - 0.01,
        regime_momentum_cut=base["regime_momentum_cut"] - 0.01,
    )

    # Second-order combinations around the best-so-far pattern:
    # slightly tighter weak-volume / overheat caps plus lower trigger ceiling
    # have been the most promising direction so far, so explore them jointly.
    combined_tighter_specs = [
        {
            "volume_guard_cap": base["volume_guard_cap"] - 0.02,
            "volume_guard_momentum_ceiling": base["volume_guard_momentum_ceiling"] - 0.02,
            "overheat_max_exposure": base["overheat_max_exposure"] - 0.02,
            "overheat_high_max_exposure": base["overheat_high_max_exposure"] - 0.02,
        },
        {
            "volume_guard_cap": base["volume_guard_cap"] - 0.02,
            "volume_guard_momentum_ceiling": base["volume_guard_momentum_ceiling"] - 0.02,
            "overheat_max_exposure": base["overheat_max_exposure"] - 0.01,
            "overheat_high_max_exposure": base["overheat_high_max_exposure"] - 0.01,
        },
        {
            "volume_guard_cap": base["volume_guard_cap"] - 0.01,
            "volume_guard_momentum_ceiling": base["volume_guard_momentum_ceiling"] - 0.02,
            "overheat_max_exposure": base["overheat_max_exposure"] - 0.02,
            "overheat_high_max_exposure": base["overheat_high_max_exposure"] - 0.01,
        },
        {
            "volume_guard_cap": base["volume_guard_cap"] - 0.02,
            "volume_guard_momentum_ceiling": base["volume_guard_momentum_ceiling"] - 0.01,
            "overheat_max_exposure": base["overheat_max_exposure"] - 0.02,
            "overheat_high_max_exposure": base["overheat_high_max_exposure"] - 0.02,
        },
    ]
    for updates in combined_tighter_specs:
        add_variant(**updates)
        add_variant(aggressive_core_weight=base["aggressive_core_weight"] + 0.01, **updates)
        add_variant(conservative_core_weight=base["conservative_core_weight"] - 0.01, **updates)
        add_variant(
            volume_ratio_cut=base["volume_ratio_cut"] - 0.01,
            volume_short_ratio_cut=base["volume_short_ratio_cut"] - 0.01,
            **updates,
        )
        add_variant(
            overheat_momentum_cut=base["overheat_momentum_cut"] + 0.01,
            overheat_high_momentum_cut=base["overheat_high_momentum_cut"] + 0.01,
            **updates,
        )
        add_variant(
            overheat_drawdown_cut=base["overheat_drawdown_cut"] - 0.01,
            **updates,
        )

    combined_balanced_specs = [
        {
            "volume_guard_cap": base["volume_guard_cap"] - 0.02,
            "overheat_max_exposure": base["overheat_max_exposure"] - 0.02,
        },
        {
            "volume_guard_momentum_ceiling": base["volume_guard_momentum_ceiling"] - 0.02,
            "overheat_high_max_exposure": base["overheat_high_max_exposure"] - 0.02,
        },
        {
            "volume_guard_cap": base["volume_guard_cap"] - 0.01,
            "volume_guard_momentum_ceiling": base["volume_guard_momentum_ceiling"] - 0.02,
            "overheat_max_exposure": base["overheat_max_exposure"] - 0.01,
            "overheat_high_max_exposure": base["overheat_high_max_exposure"] - 0.02,
            "volume_ratio_cut": base["volume_ratio_cut"] + 0.01,
            "volume_short_ratio_cut": base["volume_short_ratio_cut"] + 0.01,
        },
    ]
    for updates in combined_balanced_specs:
        add_variant(**updates)
        add_variant(regime_momentum_cut=base["regime_momentum_cut"] + 0.01, **updates)
        add_variant(aggressive_core_weight=base["aggressive_core_weight"] - 0.01, **updates)

    # The recent frontier has consistently improved by tightening weak-volume
    # caps and overheat caps together. Search a deeper defense ladder in one run
    # so we do not need multiple manual iterations to walk the same direction.
    for vg_step in (0.02, 0.04, 0.06, 0.08):
        for cap_step in (0.00, 0.02):
            hi_step = 0.00 if cap_step == 0.00 else 0.02
            updates = {
                "volume_guard_cap": base["volume_guard_cap"] - vg_step,
                "volume_guard_momentum_ceiling": base["volume_guard_momentum_ceiling"] - min(vg_step, 0.02),
                "overheat_max_exposure": base["overheat_max_exposure"] - cap_step,
                "overheat_high_max_exposure": base["overheat_high_max_exposure"] - hi_step,
            }
            add_variant(**updates)
            add_variant(conservative_core_weight=base["conservative_core_weight"] + 0.01, **updates)
            add_variant(aggressive_core_weight=base["aggressive_core_weight"] - 0.01, **updates)
            add_variant(
                volume_ratio_cut=base["volume_ratio_cut"] + 0.01,
                volume_short_ratio_cut=base["volume_short_ratio_cut"] + 0.01,
                **updates,
            )
            add_variant(
                volume_ratio_cut=base["volume_ratio_cut"] - 0.01,
                volume_short_ratio_cut=base["volume_short_ratio_cut"] - 0.01,
                **updates,
            )
            add_variant(
                aggressive_core_weight=base["aggressive_core_weight"] - 0.01,
                conservative_core_weight=base["conservative_core_weight"] + 0.01,
                **updates,
            )
            if cap_step > 0:
                add_variant(
                    overheat_drawdown_cut=base["overheat_drawdown_cut"] - 0.01,
                    overheat_momentum_cut=base["overheat_momentum_cut"] - 0.01,
                    **updates,
                )
                add_variant(
                    overheat_drawdown_cut=base["overheat_drawdown_cut"] + 0.01,
                    overheat_momentum_cut=base["overheat_momentum_cut"] + 0.01,
                    **updates,
                )

    # Test a smooth low-volume guard so mild volume weakness can keep slightly
    # more risk budget while severe weakness still collapses to the hard floor.
    for relief_buffer in (0.02, 0.04, 0.06, 0.08):
        for soft_span in (0.03, 0.04, 0.05, 0.06):
            add_variant(
                volume_guard_relief_buffer=relief_buffer,
                volume_guard_soft_span=soft_span,
            )
            add_variant(
                volume_guard_cap=base["volume_guard_cap"] - 0.02,
                volume_guard_relief_buffer=relief_buffer,
                volume_guard_soft_span=soft_span,
            )
            add_variant(
                volume_guard_cap=base["volume_guard_cap"] - 0.04,
                overheat_max_exposure=base["overheat_max_exposure"] - 0.02,
                overheat_high_max_exposure=base["overheat_high_max_exposure"] - 0.02,
                volume_guard_relief_buffer=relief_buffer,
                volume_guard_soft_span=soft_span,
            )
            add_variant(
                volume_guard_cap=base["volume_guard_cap"] - 0.02,
                conservative_core_weight=base["conservative_core_weight"] + 0.01,
                volume_guard_relief_buffer=relief_buffer,
                volume_guard_soft_span=soft_span,
            )
            add_variant(
                conservative_core_weight=base["conservative_core_weight"] + 0.01,
                overheat_max_exposure=base["overheat_max_exposure"] + 0.02,
                overheat_high_max_exposure=base["overheat_high_max_exposure"] + 0.02,
                volume_guard_relief_buffer=relief_buffer,
                volume_guard_soft_span=soft_span,
            )
            add_variant(
                conservative_core_weight=base["conservative_core_weight"] + 0.01,
                volume_guard_cap=base["volume_guard_cap"] + 0.02,
                overheat_max_exposure=base["overheat_max_exposure"] + 0.02,
                overheat_high_max_exposure=base["overheat_high_max_exposure"] + 0.02,
                volume_guard_relief_buffer=relief_buffer,
                volume_guard_soft_span=soft_span,
            )
            add_variant(
                aggressive_core_weight=base["aggressive_core_weight"] + 0.01,
                conservative_core_weight=base["conservative_core_weight"] + 0.01,
                volume_guard_relief_buffer=relief_buffer,
                volume_guard_soft_span=soft_span,
            )
            add_variant(
                conservative_core_weight=base["conservative_core_weight"] + 0.01,
                regime_momentum_cut=base["regime_momentum_cut"] - 0.01,
                volume_guard_relief_buffer=relief_buffer,
                volume_guard_soft_span=soft_span,
            )

    # Fine-grained local search around the smooth-guard near-miss branch:
    # small cap adjustments may recover return/sharpe without giving back
    # all of the drawdown-integral improvement.
    for relief_buffer in (0.02, 0.04, 0.06):
        for soft_span in (0.03, 0.04, 0.05):
            for vg_cap in (0.30, 0.31, 0.32, 0.33):
                for overheat_cap in (0.08, 0.09, 0.10):
                    for hi_cap in (0.04, 0.05, 0.06):
                        add_variant(
                            aggressive_core_weight=0.28,
                            conservative_core_weight=0.28,
                            volume_guard_cap=vg_cap,
                            overheat_max_exposure=overheat_cap,
                            overheat_high_max_exposure=min(hi_cap, overheat_cap),
                            volume_guard_relief_buffer=relief_buffer,
                            volume_guard_soft_span=soft_span,
                        )
                        add_variant(
                            aggressive_core_weight=0.29,
                            conservative_core_weight=0.28,
                            volume_guard_cap=vg_cap,
                            overheat_max_exposure=overheat_cap,
                            overheat_high_max_exposure=min(hi_cap, overheat_cap),
                            volume_guard_relief_buffer=relief_buffer,
                            volume_guard_soft_span=soft_span,
                        )
                        add_variant(
                            aggressive_core_weight=0.28,
                            conservative_core_weight=0.28,
                            volume_guard_cap=vg_cap,
                            overheat_max_exposure=overheat_cap,
                            overheat_high_max_exposure=min(hi_cap, overheat_cap),
                            volume_guard_relief_buffer=relief_buffer,
                            volume_guard_soft_span=soft_span,
                            volume_guard_momentum_ceiling=0.15,
                        )

    unique: dict[str, dict[str, float]] = {}
    ordered_keys = [
        "aggressive_core_weight",
        "conservative_core_weight",
        "regime_momentum_cut",
        "volume_ratio_cut",
        "volume_short_ratio_cut",
        "volume_breadth_cut",
        "volume_guard_cap",
        "volume_guard_momentum_ceiling",
        "volume_guard_relief_buffer",
        "volume_guard_soft_span",
        "overheat_drawdown_cut",
        "overheat_momentum_cut",
        "overheat_max_exposure",
        "overheat_high_momentum_cut",
        "overheat_high_max_exposure",
    ]
    for spec in specs:
        key = json.dumps({k: round(spec[k], 6) for k in ordered_keys}, sort_keys=True)
        unique[key] = spec
    return list(unique.values())


def rank_valid_improvements(frame: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    if frame.empty:
        return frame

    ranked = frame.copy()
    ranked["annualized_return_gain"] = ranked["annualized_return"] - float(baseline["annualized_return"])
    ranked["sharpe_gain"] = ranked["sharpe_rf0"] - float(baseline["sharpe_rf0"])
    ranked["drawdown_integral_improvement"] = float(baseline["max_drawdown_integral"]) - ranked["max_drawdown_integral"]
    ranked["composite_improvement_score"] = (
        ranked["annualized_return_gain"] * 100
        + ranked["sharpe_gain"] * 10
        + ranked["drawdown_integral_improvement"] / 10
    )
    return ranked.sort_values(
        [
            "composite_improvement_score",
            "annualized_return_gain",
            "sharpe_gain",
            "drawdown_integral_improvement",
            "annualized_return",
            "sharpe_rf0",
            "max_drawdown_integral",
        ],
        ascending=[False, False, False, False, False, False, True],
    )


def load_prices(selected: pd.DataFrame, years: int, refresh: bool) -> pd.DataFrame:
    if refresh:
        return fetch_histories(selected, years=years)
    _, prices = load_cached_data()
    start_ts = prices.index.max() - pd.DateOffset(years=years)
    prices = prices.loc[prices.index >= start_ts].copy()
    required_treasury = selected[selected["code"].astype(str) == "511260"].copy()
    prices, skipped_codes, fetch_error = try_join_missing_candidate_histories(
        prices=prices,
        candidates=required_treasury,
        years=years,
        fetch_fn=fetch_histories,
    )
    if skipped_codes:
        skipped_text = ", ".join(skipped_codes)
        if fetch_error:
            print(f"[warn] missing required treasury cache: {skipped_text}; fetch failed: {fetch_error}")
        else:
            print(f"[warn] missing required treasury cache: {skipped_text}")
    codes = selected["code"].astype(str).tolist()
    prices = prices[[code for code in prices.columns if code in codes]].copy()
    raise_if_missing_required_histories(prices, required_treasury, context="baseline treasury")
    return prices


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_params = normalize_params(build_default_strategy_params())
    drop_codes = [str(code) for code in build_default_strategy_params()["drop_codes"]]
    selected = load_default_strategy_backtest_pool()
    prices = load_prices(selected, years=args.years, refresh=args.refresh)
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)
    specs = build_specs(base_params, search_mode=args.search_mode)

    rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame(index=prices.index)
    descriptions: dict[str, str] = {}
    total_specs = len(specs)

    for idx, spec in enumerate(specs, start=1):
        if idx == 1 or idx == total_specs or idx % 10 == 0:
            print(f"[current_best_fine_tune] {idx}/{total_specs}", flush=True)
        name = strategy_name(drop_codes, spec)
        result, trades = run_default_strategy_with_params(
            prices=prices,
            selected=selected,
            params={**spec, "drop_codes": drop_codes},
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            market_proxy=market_proxy,
        )
        if idx == 1:
            result = apply_official_baseline_nav_anchor(result)
        summary = summarize(result, trades, selected)
        summary["strategy"] = name
        rows.append(summary)
        nav_compare[f"{name}_nav"] = result["nav"]
        descriptions[name] = strategy_description(drop_codes, spec)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df[
        [
            "strategy",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_rf0",
            "max_drawdown_integral",
            "max_drawdown",
            "trade_count",
            "avg_exposure",
            "latest_holding_code",
            "latest_holding_theme",
            "latest_holding_name",
            "latest_portfolio",
            "latest_momentum",
            "latest_exposure",
        ]
    ].sort_values(
        ["annualized_return", "sharpe_rf0", "max_drawdown_integral", "max_drawdown"],
        ascending=[False, False, True, False],
    )
    write_dataframe_csv_atomic(summary_df, SUMMARY_PATH, index=False)
    write_dataframe_csv_atomic(nav_compare, COMPARE_PATH)

    baseline_strategy = strategy_name(drop_codes, base_params)
    baseline = summary_df[summary_df["strategy"] == baseline_strategy].iloc[0]
    better_df = summary_df[
        (summary_df["strategy"] != baseline_strategy)
        & (summary_df["annualized_return"] >= baseline["annualized_return"] - METRIC_TOLERANCE)
        & (summary_df["sharpe_rf0"] >= baseline["sharpe_rf0"] - METRIC_TOLERANCE)
        & (summary_df["max_drawdown_integral"] <= baseline["max_drawdown_integral"] + METRIC_TOLERANCE)
        & (
            (summary_df["annualized_return"] > baseline["annualized_return"] + METRIC_TOLERANCE)
            | (summary_df["sharpe_rf0"] > baseline["sharpe_rf0"] + METRIC_TOLERANCE)
            | (summary_df["max_drawdown_integral"] < baseline["max_drawdown_integral"] - METRIC_TOLERANCE)
        )
    ].copy()

    better_ranked_df = rank_valid_improvements(better_df, baseline)

    write_json_atomic(
        BEST_PATH,
        {
            "baseline": baseline.to_dict(),
            "strict_improvements": better_ranked_df.to_dict(orient="records"),
            "descriptions": descriptions,
        },
    )

    print("Baseline:")
    print(baseline.to_string())
    print("\nTop candidates:")
    print(summary_df.head(10).to_string(index=False))

    notify_df = load_incremental_notify_candidates(
        NOTIFY_STATE_PATH,
        better_ranked_df.copy(),
        metric_tolerance=METRIC_TOLERANCE,
        rerank_fn=lambda current, previous: rank_valid_improvements(current, pd.Series(previous))
        if previous is not None
        else current,
    )

    notified, detail = notify_best_candidate(
        args,
        notify_df,
        descriptions=descriptions,
        baseline_summary=baseline.to_dict(),
        default_webhook=DEFAULT_FEISHU_WEBHOOK,
        notify_state_path=NOTIFY_STATE_PATH,
        send_fn=send_improvement_notification,
        sort_candidates=False,
        suppress_exceptions=True,
    )
    if notified and detail is not None:
        print(f"\nWebhook notified for {detail}")
    elif detail is not None:
        print(f"\nWebhook notify failed for {detail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
