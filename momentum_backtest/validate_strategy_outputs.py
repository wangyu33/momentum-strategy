#!/usr/bin/env python3
"""校验正式输出的收益、交易与 A 股时段边界是否一致。"""

from __future__ import annotations

import math
import json
import os
import sys
import tempfile
import importlib
import subprocess
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from .runtime_env import prepare_local_imports, scrub_user_site_packages
except ImportError:
    from runtime_env import prepare_local_imports, scrub_user_site_packages

prepare_local_imports(__file__, include_module_dir=False)

import pandas as pd
try:
    from .official_baseline import apply_official_baseline_nav_anchor
    from .compare_china_internet_guards import CHINA_INTERNET_CODE, summarize as summarize_china_internet_guard
    from .compare_current_best_fine_tune import normalize_params as normalize_fine_tune_params
    from .compare_defensive_persistence import apply_persistent_overlay
    from .compare_goal_optimizations import run_regime_mix_core_overheat_strategy
    from .compare_goal_optimizations import build_reduced_pool_notified_base_params, parse_regime_mix_strategy_name
    from .compare_hs300_regime_fixes import calculate_yearly_returns
    from .compare_market_proxy_variants import build_risk_proxy_features
    from .compare_resource_guards import RESOURCE_CODE, summarize as summarize_resource_guard
    from .compare_tail_risk_bond_overlay import apply_tail_bond_overlay
    from .daily_monitor import compute_snapshot
    from .monitor_delivery import (
        build_state_payload,
        build_channel_plan,
        is_duplicate_snapshot,
        load_delivery_progress_for_bundle,
        save_state,
        send_message_bundle,
        state_uses_message_cache,
    )
    from .monitor_pipeline import build_price_panel
    from .monitor_render import (
        allocations_equal_for_display,
        build_trade_details_from_allocations,
        extract_weight_allocations,
        format_exposure_transition_for_display,
        format_portfolio_allocations,
        format_market_message,
        format_trade_message,
    )
    from .monitor_snapshot import (
        build_asset_price_context,
        classify_market_session,
        estimate_live_nav,
        load_backtest_reference_context,
        should_include_realtime_snapshot,
    )
    from . import monitor_delivery as monitor_delivery_module
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor
    from compare_china_internet_guards import CHINA_INTERNET_CODE, summarize as summarize_china_internet_guard
    from compare_current_best_fine_tune import normalize_params as normalize_fine_tune_params
    from compare_defensive_persistence import apply_persistent_overlay
    from compare_goal_optimizations import run_regime_mix_core_overheat_strategy
    from compare_goal_optimizations import build_reduced_pool_notified_base_params, parse_regime_mix_strategy_name
    from compare_hs300_regime_fixes import calculate_yearly_returns
    from compare_market_proxy_variants import build_risk_proxy_features
    from compare_resource_guards import RESOURCE_CODE, summarize as summarize_resource_guard
    from compare_tail_risk_bond_overlay import apply_tail_bond_overlay
    from daily_monitor import compute_snapshot
    from monitor_delivery import (
        build_state_payload,
        build_channel_plan,
        is_duplicate_snapshot,
        load_delivery_progress_for_bundle,
        save_state,
        send_message_bundle,
        state_uses_message_cache,
    )
    from monitor_pipeline import build_price_panel
    from monitor_render import (
        allocations_equal_for_display,
        build_trade_details_from_allocations,
        extract_weight_allocations,
        format_exposure_transition_for_display,
        format_portfolio_allocations,
        format_market_message,
        format_trade_message,
    )
    from monitor_snapshot import (
        build_asset_price_context,
        classify_market_session,
        estimate_live_nav,
        load_backtest_reference_context,
        should_include_realtime_snapshot,
    )
    import monitor_delivery as monitor_delivery_module

from run_backtest import (
    apply_structural_break_back_adjustment,
    build_contribution_summary,
    build_drawdown_episode_report,
    build_default_strategy_params,
    compute_signal_asset_momentum,
    build_threshold_dual_signal,
    count_rebalance_days,
    count_trade_days,
    filter_history_to_confirmed_closes,
    filter_indexed_frame_to_confirmed_closes,
    get_latest_portfolio_text,
    is_recovered,
    normalize_code,
    run_default_strategy_with_params,
    run_threshold_dual_strategy,
    run_threshold_dual_with_overheat_cap_strategy,
    summarize_episode_holdings,
)
try:
    from .search_utils import sort_notify_candidates
    from .search_utils import (
        add_notify_cli_args,
        extract_valid_previous_summary,
        load_preferred_strategy_payload,
        notify_best_candidate,
        notify_ranked_incremental_candidate,
        load_required_strategy_payload,
        raise_if_missing_required_histories,
        should_send_notify,
        try_join_missing_candidate_histories,
    )
except ImportError:
    from search_utils import sort_notify_candidates
    from search_utils import (
        add_notify_cli_args,
        extract_valid_previous_summary,
        load_preferred_strategy_payload,
        notify_best_candidate,
        notify_ranked_incremental_candidate,
        load_required_strategy_payload,
        raise_if_missing_required_histories,
        should_send_notify,
        try_join_missing_candidate_histories,
    )


CORE_DIR = Path("momentum_backtest/output/core")
FLOAT_TOL = 1e-10


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nav = pd.read_csv(CORE_DIR / "backtest_nav.csv", parse_dates=["date"]).set_index("date")
    prices = pd.read_csv(CORE_DIR / "prices.csv", parse_dates=["date"]).set_index("date")
    trades = pd.read_csv(CORE_DIR / "trades.csv", parse_dates=["date"])
    selected = pd.read_csv(CORE_DIR / "selected_etfs.csv", dtype={"code": str})
    return nav, prices, trades, selected


def build_trades_from_weights(result: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    weight_cols = [col for col in result.columns if col.startswith("weight_")]
    weights = result[weight_cols].copy()
    weights.columns = [col.removeprefix("weight_") for col in weight_cols]
    prev_weights = weights.shift(1).fillna(0.0)
    trades: list[dict[str, object]] = []

    for dt_idx in weights.index:
        current = weights.loc[dt_idx]
        prev = prev_weights.loc[dt_idx]
        for code in weights.columns:
            curr_w = float(current[code]) if pd.notna(current[code]) else 0.0
            prev_w = float(prev[code]) if pd.notna(prev[code]) else 0.0
            if abs(curr_w - prev_w) < FLOAT_TOL:
                continue
            if curr_w > prev_w:
                action = "BUY" if prev_w == 0 else "ADD"
            else:
                action = "SELL" if curr_w == 0 else "REDUCE"
            trades.append(
                {
                    "date": pd.Timestamp(dt_idx).date().isoformat(),
                    "action": action,
                    "code": normalize_code(code),
                    "theme": code_to_theme.get(code, ""),
                    "name": code_to_name.get(code, ""),
                    "from_exposure": round(prev_w, 12),
                    "to_exposure": round(curr_w, 12),
                }
            )
    return pd.DataFrame(trades)


def check_return_chain(result: pd.DataFrame, prices: pd.DataFrame) -> list[CheckResult]:
    weight_cols = [col for col in result.columns if col.startswith("weight_")]
    weights = result[weight_cols].copy()
    weights.columns = [col.removeprefix("weight_") for col in weight_cols]
    weights = weights.reindex(columns=prices.columns, fill_value=0.0).fillna(0.0)
    return_weight_cols = [col for col in result.columns if col.startswith("return_weight_")]
    if return_weight_cols:
        return_weights = result[return_weight_cols].copy()
        return_weights.columns = [col.removeprefix("return_weight_") for col in return_weight_cols]
        return_weights = return_weights.reindex(columns=prices.columns, fill_value=0.0).fillna(0.0)
    else:
        return_weights = weights.shift(1).fillna(0.0)
    returns = prices.reindex(index=result.index, columns=weights.columns).pct_change().fillna(0.0)
    prev_weights = weights.shift(1).fillna(0.0)

    gross_ret = (return_weights * returns).sum(axis=1)
    turnover = (weights - prev_weights).abs().sum(axis=1)
    cost = turnover * (result["trade_cost_rate"] / result["turnover"]).replace([math.inf, -math.inf], 0.0).fillna(0.0)

    expected_strategy_ret = (1 + gross_ret) * (1 - result["trade_cost_rate"].fillna(0.0)) - 1
    expected_nav = (1 + expected_strategy_ret.fillna(0.0)).cumprod()
    expected_nav.iloc[0] = 1.0
    expected_frame = pd.DataFrame(
        {
            "nav": expected_nav,
            "strategy_return": expected_strategy_ret.fillna(0.0),
            "drawdown": expected_nav / expected_nav.cummax() - 1.0,
        },
        index=result.index,
    )
    expected_frame = apply_official_baseline_nav_anchor(expected_frame)
    expected_nav = expected_frame["nav"]
    expected_drawdown = expected_frame["drawdown"]
    expected_exposure = weights.sum(axis=1)
    expected_holding = weights.idxmax(axis=1).where(expected_exposure > FLOAT_TOL, pd.NA).map(normalize_code)
    actual_holding = result["holding"].map(normalize_code)

    checks = [
        CheckResult(
            name="turnover",
            passed=bool((turnover - result["turnover"].fillna(0.0)).abs().max() < FLOAT_TOL),
            detail=f"max_diff={(turnover - result['turnover'].fillna(0.0)).abs().max():.3e}",
        ),
        CheckResult(
            name="strategy_return",
            passed=bool((expected_strategy_ret - result["strategy_return"].fillna(0.0)).abs().max() < FLOAT_TOL),
            detail=f"max_diff={(expected_strategy_ret - result['strategy_return'].fillna(0.0)).abs().max():.3e}",
        ),
        CheckResult(
            name="nav",
            passed=bool((expected_nav - result["nav"]).abs().max() < FLOAT_TOL),
            detail=f"max_diff={(expected_nav - result['nav']).abs().max():.3e}",
        ),
        CheckResult(
            name="drawdown",
            passed=bool((expected_drawdown - result["drawdown"]).abs().max() < FLOAT_TOL),
            detail=f"max_diff={(expected_drawdown - result['drawdown']).abs().max():.3e}",
        ),
        CheckResult(
            name="exposure",
            passed=bool((expected_exposure - result["exposure"].fillna(0.0)).abs().max() < FLOAT_TOL),
            detail=f"max_diff={(expected_exposure - result['exposure'].fillna(0.0)).abs().max():.3e}",
        ),
        CheckResult(
            name="holding",
            passed=bool(expected_holding.fillna("").equals(actual_holding.fillna(""))),
            detail=f"mismatch_count={int((expected_holding.fillna('') != actual_holding.fillna('')).sum())}",
        ),
    ]
    return checks


def check_trade_chain(result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> CheckResult:
    expected = build_trades_from_weights(result, selected)
    actual = trades.copy()
    actual["date"] = pd.to_datetime(actual["date"]).dt.date.astype(str)
    actual["code"] = actual["code"].map(normalize_code)
    for col in ["from_exposure", "to_exposure"]:
        actual[col] = actual[col].astype(float).round(12)
    actual = actual[["date", "action", "code", "theme", "name", "from_exposure", "to_exposure"]]
    expected = expected[["date", "action", "code", "theme", "name", "from_exposure", "to_exposure"]]
    actual["theme"] = actual["theme"].fillna("")
    actual["name"] = actual["name"].fillna("")
    expected["theme"] = expected["theme"].fillna("")
    expected["name"] = expected["name"].fillna("")
    passed = expected.equals(actual.reset_index(drop=True))
    detail = f"expected_rows={len(expected)}, actual_rows={len(actual)}"
    if not passed:
        mismatch_index = None
        limit = min(len(expected), len(actual))
        for idx in range(limit):
            if not expected.iloc[idx].equals(actual.iloc[idx]):
                mismatch_index = idx
                break
        if mismatch_index is None and len(expected) != len(actual):
            mismatch_index = limit
        if mismatch_index is not None:
            detail += f", first_mismatch_row={mismatch_index}"
    return CheckResult(name="trades", passed=passed, detail=detail)


def check_session_boundaries() -> list[CheckResult]:
    cases = [
        (datetime(2026, 5, 27, 9, 29), "开盘前", False),
        (datetime(2026, 5, 27, 9, 30), "盘中", True),
        (datetime(2026, 5, 27, 11, 30), "午间休市", True),
        (datetime(2026, 5, 27, 12, 0), "午间休市", True),
        (datetime(2026, 5, 27, 13, 0), "盘中", True),
        # 收盘后应直接复用正式收盘口径，不再叠加实时快照，
        # 否则 daily_monitor 会与 core/backtest_nav.csv 产生二次漂移。
        (datetime(2026, 5, 27, 15, 0), "收盘后", False),
    ]
    results: list[CheckResult] = []
    for dt_value, expected_label, expected_realtime in cases:
        actual_label = classify_market_session(dt_value)
        actual_realtime = should_include_realtime_snapshot(actual_label)
        results.append(
            CheckResult(
                name=f"session_{dt_value.strftime('%H%M')}",
                passed=(actual_label == expected_label and actual_realtime == expected_realtime),
                detail=(
                    f"label={actual_label}, realtime={actual_realtime}, "
                    f"expected_label={expected_label}, expected_realtime={expected_realtime}"
                ),
            )
        )
    return results


def check_stale_backtest_context_prefers_result() -> CheckResult:
    result = pd.DataFrame(
        {
            "nav": [1.0, 1.1, 1.2],
            "drawdown": [0.0, 0.0, -0.01],
            "current_momentum": [0.01, 0.02, 0.03],
        },
        index=pd.to_datetime(["2026-05-23", "2026-05-26", "2026-05-27"]),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "backtest_nav.csv"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-05-23", "2026-05-26"]),
                "nav": [1.0, 1.1],
                "drawdown": [0.0, 0.0],
                "current_momentum": [0.01, 0.02],
            }
        ).to_csv(path, index=False)
        context = load_backtest_reference_context(path, result)

    passed = (
        abs(float(context["current_nav"]) - 1.2) < FLOAT_TOL
        and context["nav_date"] == "2026-05-27"
        and abs(float(context["max_drawdown"]) - (-0.01)) < FLOAT_TOL
        and context["peak_nav_date"] == "2026-05-27"
    )
    detail = (
        f"current_nav={context['current_nav']}, nav_date={context['nav_date']}, "
        f"max_drawdown={context['max_drawdown']}, peak_nav_date={context['peak_nav_date']}"
    )
    return CheckResult(name="stale_backtest_context", passed=passed, detail=detail)


def check_intraday_price_context_without_realtime() -> CheckResult:
    prices = pd.DataFrame(
        {"159941": [1.50, 1.55]},
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    context = build_asset_price_context(
        asset_code_norm="159941",
        prices=prices,
        spot_prices={},
        raw_closes={},
        session_label="盘中",
    )
    passed = (
        context["current_price"] is None
        and abs(float(context["previous_close_price"]) - 1.55) < FLOAT_TOL
        and context["intraday_price_return"] is None
    )
    detail = (
        f"current_price={context['current_price']}, "
        f"previous_close_price={context['previous_close_price']}, "
        f"intraday_price_return={context['intraday_price_return']}"
    )
    return CheckResult(name="intraday_price_context_no_realtime", passed=passed, detail=detail)


def check_price_context_does_not_mix_raw_and_adjusted() -> CheckResult:
    prices = pd.DataFrame(
        {
            "159941": [1.38, 1.44],
        },
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    context = build_asset_price_context(
        asset_code_norm="159941",
        prices=prices,
        spot_prices={"159941": 1.50},
        raw_closes={},
        session_label="收盘后",
    )
    passed = (
        abs(float(context["current_price"]) - 1.50) < FLOAT_TOL
        and context["previous_close_price"] is None
        and context["intraday_price_return"] is None
    )
    detail = (
        f"current_price={context['current_price']}, "
        f"previous_close_price={context['previous_close_price']}, "
        f"intraday_price_return={context['intraday_price_return']}"
    )
    return CheckResult(name="price_context_no_raw_adjusted_mix", passed=passed, detail=detail)


def check_structural_break_back_adjustment_smooths_split_like_gaps() -> CheckResult:
    hist = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-07-01", "2022-07-04", "2022-07-05", "2022-07-06"]),
            "close": [2.370, 2.384, 0.604, 0.611],
        }
    )
    adjusted = apply_structural_break_back_adjustment(hist).set_index("date")
    prev_close = float(adjusted.loc[pd.Timestamp("2022-07-04"), "close"])
    event_close = float(adjusted.loc[pd.Timestamp("2022-07-05"), "close"])
    event_return = event_close / prev_close - 1.0 if prev_close else float("inf")
    passed = abs(event_return) < 0.05 and abs(event_close - 0.604) < FLOAT_TOL
    detail = f"prev_close={prev_close:.6f}, event_close={event_close:.6f}, event_return={event_return:.6%}"
    return CheckResult(name="structural_break_back_adjustment", passed=passed, detail=detail)


def check_partial_realtime_panel_keeps_missing_assets() -> CheckResult:
    base_prices = pd.DataFrame(
        {
            "159941": [1.50, 1.55],
            "510300": [4.80, 4.90],
        },
        index=pd.to_datetime(["2026-05-25", "2026-05-26"]),
    )
    strategy_config = {
        "selected_pool": [
            {"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发", "sina_symbol": "sz159941"},
            {"code": "510300", "theme": "沪深300", "name": "沪深300ETF华泰柏瑞", "sina_symbol": "sh510300"},
        ],
        "backtest_nav_file": "momentum_backtest/output/core/backtest_nav.csv",
    }
    panel = build_price_panel(
        today=pd.Timestamp("2026-05-27").date(),
        strategy_config=strategy_config,
        include_realtime=True,
        base_prices=base_prices,
        spot_prices={"159941": 1.60},
        raw_closes={"159941": 1.56},
    )
    today_row = panel.loc[pd.Timestamp("2026-05-27")]
    expected_live = 1.55 * 1.60 / 1.56
    passed = (
        abs(float(today_row["159941"]) - expected_live) < FLOAT_TOL
        and abs(float(today_row["510300"]) - 4.90) < FLOAT_TOL
    )
    detail = f"159941={today_row['159941']}, 510300={today_row['510300']}"
    return CheckResult(name="partial_realtime_panel_fill", passed=passed, detail=detail)


def check_intraday_close_panel_drops_same_day_history() -> CheckResult:
    base_prices = pd.DataFrame(
        {
            "159941": [1.50, 1.55],
            "510300": [4.80, 4.90],
        },
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    strategy_config = {
        "selected_pool": [
            {"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发", "sina_symbol": "sz159941"},
            {"code": "510300", "theme": "沪深300", "name": "沪深300ETF华泰柏瑞", "sina_symbol": "sh510300"},
        ],
        "backtest_nav_file": "momentum_backtest/output/core/backtest_nav.csv",
    }
    panel = build_price_panel(
        today=pd.Timestamp("2026-05-27").date(),
        strategy_config=strategy_config,
        include_realtime=False,
        allow_same_day_close=False,
        base_prices=base_prices,
    )
    passed = (
        pd.Timestamp("2026-05-27") not in panel.index
        and pd.Timestamp("2026-05-26") in panel.index
        and len(panel) == 1
    )
    detail = f"dates={[str(ts.date()) for ts in panel.index.tolist()]}"
    return CheckResult(name="intraday_close_panel_drops_same_day_history", passed=passed, detail=detail)


def check_partial_live_nav_does_not_refresh_peak() -> CheckResult:
    dates = pd.to_datetime(["2026-05-26", "2026-05-27"])
    prices = pd.DataFrame(
        {
            "159941": [1.50, 1.53],
            "510300": [4.80, 4.90],
        },
        index=dates,
    )
    close_result = pd.DataFrame(
        {
            "signal": ["159941", "159941"],
            "holding": ["159941", "159941"],
            "target_exposure": [1.0, 1.0],
            "current_momentum": [0.10, 0.12],
            "effective_momentum": [0.10, 0.12],
            "nav": [0.95, 0.95],
            "drawdown": [-0.05, -0.05],
            "weight_159941": [0.60, 0.60],
            "weight_510300": [0.40, 0.40],
            "target_weight_159941": [0.60, 0.60],
            "target_weight_510300": [0.40, 0.40],
        },
        index=dates,
    )
    result = close_result.copy()
    close_trades = pd.DataFrame(columns=["date", "action", "code", "theme", "name", "from_exposure", "to_exposure", "nav"])
    base_target_exposure = pd.Series([1.0, 1.0], index=dates)
    market_proxy_context = {
        "mode": "上一交易日收盘量能",
        "progress": None,
        "market_amount_ratio_20_60": None,
        "market_amount_ratio_5_20": None,
        "market_breadth_proxy": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        backtest_nav_path = Path(tmpdir) / "backtest_nav.csv"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-05-26", "2026-05-27"]),
                "nav": [1.00, 0.95],
                "drawdown": [0.00, -0.05],
                "current_momentum": [0.10, 0.12],
            }
        ).to_csv(backtest_nav_path, index=False)
        snapshot = compute_snapshot(
            prices=prices,
            strategy_config={
                "selected_pool": [
                    {"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"},
                    {"code": "510300", "theme": "沪深300", "name": "沪深300ETF华泰柏瑞"},
                ],
                "backtest_nav_file": str(backtest_nav_path),
                "strategy_id": "default",
                "strategy_label": "test",
                "extra_cap_label": "命中额外降仓",
            },
            result=result,
            close_result=close_result,
            close_trades=close_trades,
            base_target_exposure=base_target_exposure,
            market_proxy_context=market_proxy_context,
            spot_prices={"159941": 1.80},
            raw_closes={"159941": 1.50},
            now=datetime(2026, 5, 27, 10, 0),
        )

    passed = (
        snapshot.live_nav is not None
        and abs(float(snapshot.live_nav_coverage) - 0.60) < FLOAT_TOL
        and abs(float(snapshot.peak_nav_value) - 1.00) < FLOAT_TOL
        and snapshot.peak_nav_date == "2026-05-26"
        and abs(float(snapshot.current_drawdown) - (-0.05)) < FLOAT_TOL
    )
    detail = (
        f"live_nav={snapshot.live_nav}, coverage={snapshot.live_nav_coverage}, "
        f"peak_nav_value={snapshot.peak_nav_value}, peak_nav_date={snapshot.peak_nav_date}, "
        f"current_drawdown={snapshot.current_drawdown}"
    )
    return CheckResult(name="partial_live_nav_preserves_peak", passed=passed, detail=detail)


def check_snapshot_uses_confirmed_nav_context_before_live_estimate() -> CheckResult:
    close_dates = pd.to_datetime(["2026-05-26", "2026-05-27"])
    live_dates = pd.to_datetime(["2026-05-26", "2026-05-27", "2026-05-28"])
    prices = pd.DataFrame(
        {
            "159941": [1.50, 1.55, 1.60],
        },
        index=live_dates,
    )
    close_result = pd.DataFrame(
        {
            "signal": ["159941", "159941"],
            "holding": ["159941", "159941"],
            "target_exposure": [1.0, 1.0],
            "exposure": [1.0, 1.0],
            "current_momentum": [0.10, 0.12],
            "effective_momentum": [0.10, 0.12],
            "nav": [1.00, 1.10],
            "drawdown": [0.0, -0.02],
            "weight_159941": [1.0, 1.0],
            "target_weight_159941": [1.0, 1.0],
        },
        index=close_dates,
    )
    result = pd.DataFrame(
        {
            "signal": ["159941", "159941", "159941"],
            "holding": ["159941", "159941", "159941"],
            "target_exposure": [1.0, 1.0, 1.0],
            "exposure": [1.0, 1.0, 1.0],
            "current_momentum": [0.10, 0.12, 0.15],
            "effective_momentum": [0.10, 0.12, 0.15],
            # 这里故意让 result 已经带上今天 provisional nav，验证 compute_snapshot 不能拿它做 close_nav 基准。
            "nav": [1.00, 1.10, 1.20],
            "drawdown": [0.0, -0.02, -0.01],
            "weight_159941": [1.0, 1.0, 1.0],
            "target_weight_159941": [1.0, 1.0, 1.0],
        },
        index=live_dates,
    )
    strategy_config = {
        "selected_pool": [{"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"}],
        "strategy_id": "default",
        "strategy_label": "test",
        "extra_cap_label": "命中额外降仓",
    }
    market_proxy_context = {
        "mode": "上一交易日收盘量能",
        "progress": None,
        "market_amount_ratio_20_60": None,
        "market_amount_ratio_5_20": None,
        "market_breadth_proxy": None,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        backtest_nav_path = Path(tmpdir) / "backtest_nav.csv"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-05-26"]),
                "nav": [1.00],
                "drawdown": [0.0],
                "current_momentum": [0.10],
            }
        ).to_csv(backtest_nav_path, index=False)
        snapshot = compute_snapshot(
            prices=prices,
            strategy_config={**strategy_config, "backtest_nav_file": str(backtest_nav_path)},
            result=result,
            close_result=close_result,
            close_trades=pd.DataFrame(columns=["date", "action", "code", "theme", "name", "from_exposure", "to_exposure", "nav"]),
            base_target_exposure=pd.Series([1.0, 1.0, 1.0], index=live_dates),
            market_proxy_context=market_proxy_context,
            spot_prices={"159941": 1.60},
            raw_closes={"159941": 1.55},
            now=datetime(2026, 5, 28, 10, 0),
        )
    expected_live_nav = 1.10 * (1.60 / 1.55)
    passed = (
        abs(float(snapshot.current_nav) - 1.10) < FLOAT_TOL
        and snapshot.nav_date == "2026-05-27"
        and abs(float(snapshot.live_nav) - expected_live_nav) < FLOAT_TOL
    )
    detail = f"current_nav={snapshot.current_nav}, nav_date={snapshot.nav_date}, live_nav={snapshot.live_nav}"
    return CheckResult(name="snapshot_confirmed_nav_context", passed=passed, detail=detail)


def check_live_nav_coverage_uses_invested_exposure() -> CheckResult:
    close_result = pd.DataFrame(
        {
            "weight_159941": [0.27],
            "weight_510300": [0.0],
        },
        index=pd.to_datetime(["2026-05-27"]),
    )
    live_nav, coverage = estimate_live_nav(
        close_result=close_result,
        close_nav=1.20,
        spot_prices={"159941": 1.59},
        raw_closes={"159941": 1.50},
    )
    expected_nav = 1.20 * (1.0 + 0.27 * (1.59 / 1.50 - 1.0))
    passed = (
        live_nav is not None
        and abs(float(coverage) - 1.0) < FLOAT_TOL
        and abs(float(live_nav) - expected_nav) < FLOAT_TOL
    )
    detail = f"live_nav={live_nav}, coverage={coverage}"
    return CheckResult(name="live_nav_coverage_invested_exposure", passed=passed, detail=detail)


def check_live_nav_cash_state_keeps_close_nav() -> CheckResult:
    close_result = pd.DataFrame(
        {
            "weight_159941": [0.0],
            "weight_510300": [0.0],
        },
        index=pd.to_datetime(["2026-05-27"]),
    )
    live_nav, coverage = estimate_live_nav(
        close_result=close_result,
        close_nav=1.20,
        spot_prices={"159941": 1.59},
        raw_closes={"159941": 1.50},
    )
    passed = live_nav is not None and abs(float(live_nav) - 1.20) < FLOAT_TOL and abs(float(coverage) - 1.0) < FLOAT_TOL
    detail = f"live_nav={live_nav}, coverage={coverage}"
    return CheckResult(name="live_nav_cash_state", passed=passed, detail=detail)


def check_after_close_raw_price_does_not_recompute_nav() -> CheckResult:
    dates = pd.to_datetime(["2026-05-26", "2026-05-27"])
    prices = pd.DataFrame(
        {
            # 这里故意让价格面板使用复权价，验证收盘后价格展示会优先走 raw spot/raw close。
            "159941": [1.38, 1.44],
        },
        index=dates,
    )
    close_result = pd.DataFrame(
        {
            "signal": ["159941", "159941"],
            "holding": ["159941", "159941"],
            "target_exposure": [1.0, 1.0],
            "exposure": [1.0, 1.0],
            "current_momentum": [0.10, 0.12],
            "effective_momentum": [0.10, 0.12],
            "nav": [1.00, 1.10],
            "drawdown": [0.0, -0.01],
            "weight_159941": [1.0, 1.0],
            "target_weight_159941": [1.0, 1.0],
        },
        index=dates,
    )
    result = close_result.copy()
    strategy_config = {
        "selected_pool": [{"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"}],
        "strategy_id": "default",
        "strategy_label": "test",
        "extra_cap_label": "命中额外降仓",
    }
    market_proxy_context = {
        "mode": "上一交易日收盘量能",
        "progress": None,
        "market_amount_ratio_20_60": None,
        "market_amount_ratio_5_20": None,
        "market_breadth_proxy": None,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        backtest_nav_path = Path(tmpdir) / "backtest_nav.csv"
        pd.DataFrame(
            {
                "date": dates,
                "nav": [1.00, 1.10],
                "drawdown": [0.0, -0.01],
                "current_momentum": [0.10, 0.12],
            }
        ).to_csv(backtest_nav_path, index=False)
        snapshot = compute_snapshot(
            prices=prices,
            strategy_config={**strategy_config, "backtest_nav_file": str(backtest_nav_path)},
            result=result,
            close_result=close_result,
            close_trades=pd.DataFrame(columns=["date", "action", "code", "theme", "name", "from_exposure", "to_exposure", "nav"]),
            base_target_exposure=pd.Series([1.0, 1.0], index=dates),
            market_proxy_context=market_proxy_context,
            spot_prices={"159941": 1.50},
            raw_closes={"159941": 1.40},
            now=datetime(2026, 5, 27, 15, 30),
        )
    passed = (
        snapshot.market_session_label == "收盘后"
        and abs(float(snapshot.current_price) - 1.50) < FLOAT_TOL
        and abs(float(snapshot.previous_close_price) - 1.40) < FLOAT_TOL
        and abs(float(snapshot.intraday_price_return) - (1.50 / 1.40 - 1.0)) < FLOAT_TOL
        and abs(float(snapshot.live_nav) - 1.10) < FLOAT_TOL
        and snapshot.live_nav_coverage is None
    )
    detail = (
        f"current_price={snapshot.current_price}, previous_close={snapshot.previous_close_price}, "
        f"intraday_return={snapshot.intraday_price_return}, live_nav={snapshot.live_nav}, coverage={snapshot.live_nav_coverage}"
    )
    return CheckResult(name="after_close_raw_price_no_nav_recompute", passed=passed, detail=detail)


def check_snapshot_portfolio_falls_back_without_weight_columns() -> CheckResult:
    dates = pd.to_datetime(["2026-05-26", "2026-05-27"])
    prices = pd.DataFrame({"513050": [1.00, 1.02]}, index=dates)
    close_result = pd.DataFrame(
        {
            "signal": ["513050", "513050"],
            "holding": ["513050", "513050"],
            "target_exposure": [1.0, 0.7],
            "exposure": [0.8, 0.7],
            "current_momentum": [0.08, 0.09],
            "nav": [1.00, 1.03],
            "drawdown": [0.0, -0.01],
        },
        index=dates,
    )
    result = close_result.copy()
    strategy_config = {
        "selected_pool": [{"code": "513050", "theme": "中概互联", "name": "中概互联ETF"}],
        "strategy_id": "china_internet_cap70",
        "strategy_label": "test_ci",
        "extra_cap_label": "命中额外降仓",
    }
    market_proxy_context = {
        "mode": "上一交易日收盘量能",
        "progress": None,
        "market_amount_ratio_20_60": None,
        "market_amount_ratio_5_20": None,
        "market_breadth_proxy": None,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        backtest_nav_path = Path(tmpdir) / "backtest_nav.csv"
        pd.DataFrame(
            {
                "date": dates,
                "nav": [1.00, 1.03],
                "drawdown": [0.0, -0.01],
                "current_momentum": [0.08, 0.09],
            }
        ).to_csv(backtest_nav_path, index=False)
        snapshot = compute_snapshot(
            prices=prices,
            strategy_config={**strategy_config, "backtest_nav_file": str(backtest_nav_path)},
            result=result,
            close_result=close_result,
            close_trades=pd.DataFrame(columns=["date", "action", "code", "theme", "name", "from_exposure", "to_exposure", "nav"]),
            base_target_exposure=pd.Series([1.0, 0.7], index=dates),
            market_proxy_context=market_proxy_context,
            spot_prices={},
            raw_closes={},
            now=datetime(2026, 5, 27, 15, 30),
        )
    passed = (
        snapshot.previous_portfolio == "中概互联 / 中概互联ETF (513050) 80%"
        and snapshot.current_portfolio == "中概互联 / 中概互联ETF (513050) 70%"
        and snapshot.desired_portfolio == "中概互联 / 中概互联ETF (513050) 70%"
    )
    detail = (
        f"previous={snapshot.previous_portfolio}, "
        f"current={snapshot.current_portfolio}, desired={snapshot.desired_portfolio}"
    )
    return CheckResult(name="snapshot_portfolio_no_weight_fallback", passed=passed, detail=detail)


def check_confirmed_trade_uses_latest_actual_trade_date() -> CheckResult:
    dates = pd.to_datetime(["2026-05-26", "2026-05-27"])
    prices = pd.DataFrame({"159941": [1.00, 1.02]}, index=dates)
    close_result = pd.DataFrame(
        {
            "signal": ["159941", "159941"],
            "holding": ["159941", "159941"],
            "target_exposure": [1.0, 1.0],
            "exposure": [1.0, 1.0],
            "current_momentum": [0.08, 0.09],
            "nav": [1.00, 1.03],
            "drawdown": [0.0, -0.01],
        },
        index=dates,
    )
    result = close_result.copy()
    strategy_config = {
        "selected_pool": [{"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"}],
        "strategy_id": "default",
        "strategy_label": "test",
        "extra_cap_label": "命中额外降仓",
    }
    market_proxy_context = {
        "mode": "上一交易日收盘量能",
        "progress": None,
        "market_amount_ratio_20_60": None,
        "market_amount_ratio_5_20": None,
        "market_breadth_proxy": None,
    }
    close_trades = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-26"),
                "action": "BUY",
                "code": "159941",
                "theme": "纳指ETF",
                "name": "纳指ETF广发",
                "from_exposure": 0.0,
                "to_exposure": 1.0,
                "nav": 1.0,
            }
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        backtest_nav_path = Path(tmpdir) / "backtest_nav.csv"
        pd.DataFrame(
            {
                "date": dates,
                "nav": [1.00, 1.03],
                "drawdown": [0.0, -0.01],
                "current_momentum": [0.08, 0.09],
            }
        ).to_csv(backtest_nav_path, index=False)
        snapshot = compute_snapshot(
            prices=prices,
            strategy_config={**strategy_config, "backtest_nav_file": str(backtest_nav_path)},
            result=result,
            close_result=close_result,
            close_trades=close_trades,
            base_target_exposure=pd.Series([1.0, 1.0], index=dates),
            market_proxy_context=market_proxy_context,
            spot_prices={},
            raw_closes={},
            now=datetime(2026, 5, 27, 15, 30),
        )
    message = format_trade_message(snapshot)
    passed = (
        snapshot.confirmed_trade_date == "2026-05-26"
        and snapshot.confirmed_trade_details == "买入 纳指ETF / 纳指ETF广发 (159941) 0%->100%"
        and "今日交易(2026-05-27): 无" in message
    )
    detail = (
        f"confirmed_trade_date={snapshot.confirmed_trade_date}, "
        f"confirmed_trade_details={snapshot.confirmed_trade_details}"
    )
    return CheckResult(name="confirmed_trade_latest_date", passed=passed, detail=detail)


def check_confirmed_holding_dates_do_not_fall_back_to_last_trade_date() -> CheckResult:
    close_index = pd.to_datetime(["2026-05-26", "2026-05-27"])
    live_index = pd.to_datetime(["2026-05-26", "2026-05-27", "2026-05-28"])
    prices = pd.DataFrame(
        {
            "159941": [1.50, 1.55, 1.56],
        },
        index=live_index,
    )
    close_result = pd.DataFrame(
        {
            "signal": ["159941", "159941"],
            "holding": ["159941", "159941"],
            "target_exposure": [1.0, 1.0],
            "exposure": [1.0, 1.0],
            "current_momentum": [0.12, 0.14],
            "effective_momentum": [0.12, 0.14],
            "nav": [1.00, 1.10],
            "drawdown": [0.0, -0.02],
            "weight_159941": [1.0, 1.0],
            "target_weight_159941": [1.0, 1.0],
        },
        index=close_index,
    )
    result = pd.DataFrame(
        {
            "signal": ["159941", "159941", "159941"],
            "holding": ["159941", "159941", "159941"],
            "target_exposure": [1.0, 1.0, 1.0],
            "exposure": [1.0, 1.0, 1.0],
            "current_momentum": [0.12, 0.14, 0.16],
            "effective_momentum": [0.12, 0.14, 0.16],
            "nav": [1.00, 1.10, 1.11],
            "drawdown": [0.0, -0.02, -0.01],
            "weight_159941": [1.0, 1.0, 1.0],
            "target_weight_159941": [1.0, 1.0, 1.0],
        },
        index=live_index,
    )
    strategy_config = {
        "selected_pool": [{"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"}],
        "strategy_id": "default",
        "strategy_label": "test",
        "extra_cap_label": "命中额外降仓",
    }
    market_proxy_context = {
        "mode": "上一交易日收盘量能",
        "progress": None,
        "market_amount_ratio_20_60": None,
        "market_amount_ratio_5_20": None,
        "market_breadth_proxy": None,
    }
    close_trades = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-26"),
                "action": "BUY",
                "code": "159941",
                "theme": "纳指ETF",
                "name": "纳指ETF广发",
                "from_exposure": 0.0,
                "to_exposure": 1.0,
                "nav": 1.0,
            }
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        backtest_nav_path = Path(tmpdir) / "backtest_nav.csv"
        pd.DataFrame(
            {
                "date": close_index,
                "nav": [1.00, 1.10],
                "drawdown": [0.0, -0.02],
                "current_momentum": [0.12, 0.14],
            }
        ).to_csv(backtest_nav_path, index=False)
        snapshot = compute_snapshot(
            prices=prices,
            strategy_config={**strategy_config, "backtest_nav_file": str(backtest_nav_path)},
            result=result,
            close_result=close_result,
            close_trades=close_trades,
            base_target_exposure=pd.Series([1.0, 1.0, 1.0], index=live_index),
            market_proxy_context=market_proxy_context,
            spot_prices={"159941": 1.56},
            raw_closes={"159941": 1.55},
            now=datetime(2026, 5, 28, 10, 0),
        )
    message = format_trade_message(snapshot)
    passed = (
        snapshot.trade_date == "2026-05-28"
        and snapshot.previous_holding_date == "2026-05-26"
        and snapshot.current_holding_date == "2026-05-27"
        and snapshot.confirmed_trade_date == "2026-05-26"
        and "ETF交易与回撤 2026-05-28（确认数据截至 2026-05-27）" in message
        and "当前确认持仓(2026-05-27)" in message
        and "今日目标组合(按今日收盘信号推演): 纳指ETF / 纳指ETF广发 (159941) 100%" in message
        and "今日交易(2026-05-28): 无" in message
    )
    detail = (
        f"trade_date={snapshot.trade_date}, previous_holding_date={snapshot.previous_holding_date}, "
        f"current_holding_date={snapshot.current_holding_date}, confirmed_trade_date={snapshot.confirmed_trade_date}"
    )
    return CheckResult(name="confirmed_holding_dates_match_close_timeline", passed=passed, detail=detail)


def check_snapshot_exposure_and_extra_cap_timelines() -> CheckResult:
    close_index = pd.to_datetime(["2026-05-26", "2026-05-27"])
    live_index = pd.to_datetime(["2026-05-26", "2026-05-27", "2026-05-28"])
    prices = pd.DataFrame(
        {
            "159941": [1.50, 1.55, 1.56],
            "510300": [4.80, 4.90, 4.92],
        },
        index=live_index,
    )
    close_result = pd.DataFrame(
        {
            "signal": ["159941", "159941"],
            "holding": ["159941", "159941"],
            "target_exposure": [1.0, 1.0],
            "exposure": [1.0, 0.8],
            "current_momentum": [0.12, 0.14],
            "effective_momentum": [0.12, 0.14],
            "nav": [1.00, 1.10],
            "drawdown": [0.0, -0.02],
            "weight_159941": [1.0, 0.8],
            "weight_510300": [0.0, 0.0],
            "target_weight_159941": [1.0, 1.0],
            "target_weight_510300": [0.0, 0.0],
        },
        index=close_index,
    )
    result = pd.DataFrame(
        {
            "signal": ["159941", "159941", "159941"],
            "holding": ["159941", "159941", "159941"],
            "target_exposure": [1.0, 1.0, 0.4],
            "exposure": [1.0, 0.8, 0.8],
            "current_momentum": [0.12, 0.14, 0.18],
            "effective_momentum": [0.12, 0.14, 0.18],
            "nav": [1.00, 1.10, 1.12],
            "drawdown": [0.0, -0.02, -0.01],
            "weight_159941": [1.0, 0.8, 0.8],
            "weight_510300": [0.0, 0.0, 0.0],
            "target_weight_159941": [1.0, 1.0, 0.4],
            "target_weight_510300": [0.0, 0.0, 0.0],
        },
        index=live_index,
    )
    strategy_config = {
        "selected_pool": [
            {"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"},
            {"code": "510300", "theme": "沪深300", "name": "沪深300ETF华泰柏瑞"},
        ],
        "strategy_id": "default",
        "strategy_label": "test",
        "extra_cap_label": "命中额外降仓",
    }
    market_proxy_context = {
        "mode": "上一交易日收盘量能",
        "progress": None,
        "market_amount_ratio_20_60": None,
        "market_amount_ratio_5_20": None,
        "market_breadth_proxy": None,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        backtest_nav_path = Path(tmpdir) / "backtest_nav.csv"
        pd.DataFrame(
            {
                "date": close_index,
                "nav": [1.00, 1.10],
                "drawdown": [0.0, -0.02],
                "current_momentum": [0.12, 0.14],
            }
        ).to_csv(backtest_nav_path, index=False)
        snapshot = compute_snapshot(
            prices=prices,
            strategy_config={**strategy_config, "backtest_nav_file": str(backtest_nav_path)},
            result=result,
            close_result=close_result,
            close_trades=pd.DataFrame(
                columns=["date", "action", "code", "theme", "name", "from_exposure", "to_exposure", "nav"]
            ),
            base_target_exposure=pd.Series([1.0, 1.0, 1.0], index=live_index),
            market_proxy_context=market_proxy_context,
            spot_prices={},
            raw_closes={},
            now=datetime(2026, 5, 28, 14, 0),
        )
    message = format_market_message(snapshot)
    passed = (
        abs(float(snapshot.current_exposure) - 0.8) < FLOAT_TOL
        and abs(float(snapshot.desired_exposure) - 0.4) < FLOAT_TOL
        and snapshot.extra_cap_triggered is True
        and "100% -> 40%" in message
    )
    detail = (
        f"current_exposure={snapshot.current_exposure}, desired_exposure={snapshot.desired_exposure}, "
        f"extra_cap_triggered={snapshot.extra_cap_triggered}"
    )
    return CheckResult(name="snapshot_exposure_extra_cap_timelines", passed=passed, detail=detail)


def check_display_hidden_allocations_are_filtered() -> CheckResult:
    row = pd.Series(
        {
            "weight_159941": 0.004,
            "weight_510300": 0.996,
        }
    )
    allocations = extract_weight_allocations(row, prefix="weight_")
    portfolio = format_portfolio_allocations(
        allocations,
        theme_map={"159941": "纳指ETF", "510300": "沪深300"},
        name_map={"159941": "纳指ETF广发", "510300": "沪深300ETF华泰柏瑞"},
    )
    passed = allocations == [("510300", 0.996)] and "159941" not in portfolio and portfolio == "沪深300 / 沪深300ETF华泰柏瑞 (510300) 99.6%"
    detail = f"allocations={allocations}, portfolio={portfolio}"
    return CheckResult(name="display_hidden_allocations_filtered", passed=passed, detail=detail)


def check_display_equivalent_trade_details_are_suppressed() -> CheckResult:
    trade_details, changed = build_trade_details_from_allocations(
        current_allocations=[("159941", 0.266), ("510300", 0.734)],
        desired_allocations=[("159941", 0.274), ("510300", 0.726)],
        theme_map={"159941": "纳指ETF", "510300": "沪深300"},
        name_map={"159941": "纳指ETF广发", "510300": "沪深300ETF华泰柏瑞"},
    )
    display_equal = allocations_equal_for_display(0.266, 0.274) and allocations_equal_for_display(0.734, 0.726)
    passed = display_equal and trade_details == "无" and changed is False
    detail = f"display_equal={display_equal}, trade_details={trade_details}, changed={changed}"
    return CheckResult(name="display_equivalent_trade_suppressed", passed=passed, detail=detail)


def check_display_equivalent_confirmed_trade_transition_suppressed() -> CheckResult:
    hidden_transition = format_exposure_transition_for_display(0.266, 0.274)
    visible_transition = format_exposure_transition_for_display(0.00, 0.27)
    passed = hidden_transition is None and visible_transition == "0%->27%"
    detail = f"hidden_transition={hidden_transition}, visible_transition={visible_transition}"
    return CheckResult(name="display_equivalent_confirmed_trade_suppressed", passed=passed, detail=detail)


def check_legacy_state_is_upgradeable() -> CheckResult:
    snapshot = compute_snapshot(
        prices=pd.DataFrame(
            {
                "159941": [1.50, 1.55],
            },
            index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
        ),
        strategy_config={
            "selected_pool": [
                {"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"},
            ],
            "backtest_nav_file": "momentum_backtest/output/core/backtest_nav.csv",
            "strategy_id": "default",
            "strategy_label": "test",
            "extra_cap_label": "命中额外降仓",
        },
        result=pd.DataFrame(
            {
                "signal": ["159941", "159941"],
                "holding": ["159941", "159941"],
                "target_exposure": [1.0, 1.0],
                "current_momentum": [0.10, 0.12],
                "effective_momentum": [0.10, 0.12],
                "nav": [1.0, 1.02],
                "drawdown": [0.0, 0.0],
                "weight_159941": [1.0, 1.0],
                "target_weight_159941": [1.0, 1.0],
            },
            index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
        ),
        close_result=pd.DataFrame(
            {
                "signal": ["159941", "159941"],
                "holding": ["159941", "159941"],
                "target_exposure": [1.0, 1.0],
                "current_momentum": [0.10, 0.12],
                "effective_momentum": [0.10, 0.12],
                "nav": [1.0, 1.02],
                "drawdown": [0.0, 0.0],
                "weight_159941": [1.0, 1.0],
                "target_weight_159941": [1.0, 1.0],
            },
            index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
        ),
        close_trades=pd.DataFrame(columns=["date", "action", "code", "theme", "name", "from_exposure", "to_exposure", "nav"]),
        base_target_exposure=pd.Series([1.0, 1.0], index=pd.to_datetime(["2026-05-26", "2026-05-27"])),
        market_proxy_context={"mode": None, "progress": None, "market_amount_ratio_20_60": None, "market_amount_ratio_5_20": None, "market_breadth_proxy": None},
        spot_prices={},
        raw_closes={},
        now=datetime(2026, 5, 27, 15, 0),
    )
    market_message = "market"
    trade_message = "trade"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.json"
        path.write_text(json.dumps({"strategy_label": "test", "trade_date": "2026-05-27"}), encoding="utf-8")
        old_state = json.loads(path.read_text(encoding="utf-8"))
        before_upgrade = state_uses_message_cache(old_state)
        save_state(snapshot, path, market_message, trade_message)
        new_state = json.loads(path.read_text(encoding="utf-8"))
    passed = (before_upgrade is False) and state_uses_message_cache(new_state) and new_state["_market_message"] == market_message and new_state["_trade_message"] == trade_message
    detail = f"before_upgrade={before_upgrade}, after_upgrade={state_uses_message_cache(new_state)}"
    return CheckResult(name="legacy_state_upgradeable", passed=passed, detail=detail)


def check_partial_delivery_progress_resumes_without_duplicate() -> CheckResult:
    snapshot = compute_snapshot(
        prices=pd.DataFrame({"159941": [1.50, 1.55]}, index=pd.to_datetime(["2026-05-26", "2026-05-27"])),
        strategy_config={
            "selected_pool": [{"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"}],
            "backtest_nav_file": "momentum_backtest/output/core/backtest_nav.csv",
            "strategy_id": "default",
            "strategy_label": "test",
            "extra_cap_label": "命中额外降仓",
        },
        result=pd.DataFrame(
            {
                "signal": ["159941", "159941"],
                "holding": ["159941", "159941"],
                "target_exposure": [1.0, 1.0],
                "current_momentum": [0.10, 0.12],
                "effective_momentum": [0.10, 0.12],
                "nav": [1.0, 1.02],
                "drawdown": [0.0, 0.0],
                "weight_159941": [1.0, 1.0],
                "target_weight_159941": [1.0, 1.0],
            },
            index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
        ),
        close_result=pd.DataFrame(
            {
                "signal": ["159941", "159941"],
                "holding": ["159941", "159941"],
                "target_exposure": [1.0, 1.0],
                "current_momentum": [0.10, 0.12],
                "effective_momentum": [0.10, 0.12],
                "nav": [1.0, 1.02],
                "drawdown": [0.0, 0.0],
                "weight_159941": [1.0, 1.0],
                "target_weight_159941": [1.0, 1.0],
            },
            index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
        ),
        close_trades=pd.DataFrame(columns=["date", "action", "code", "theme", "name", "from_exposure", "to_exposure", "nav"]),
        base_target_exposure=pd.Series([1.0, 1.0], index=pd.to_datetime(["2026-05-26", "2026-05-27"])),
        market_proxy_context={"mode": None, "progress": None, "market_amount_ratio_20_60": None, "market_amount_ratio_5_20": None, "market_breadth_proxy": None},
        spot_prices={},
        raw_closes={},
        now=datetime(2026, 5, 27, 15, 0),
    )
    messages = ["market", "trade"]
    channels = build_channel_plan(disable_direct_feishu=True, webhook_url="hook")
    partial_state = build_state_payload(
        snapshot,
        messages[0],
        messages[1],
        delivery_progress=[{"webhook": True}, {"webhook": False}],
    )
    resumed_progress = load_delivery_progress_for_bundle(partial_state, snapshot, messages, channels)
    duplicate_before = is_duplicate_snapshot(partial_state, snapshot, messages[0], messages[1], channels=channels)

    sent_payloads: list[list[dict[str, bool]]] = []
    original_send_webhook_message = monitor_delivery_module.send_webhook_message
    try:
        monitor_delivery_module.send_webhook_message = lambda text, webhook_url, log_fn=None: None
        final_progress = send_message_bundle(
            messages,
            disable_direct_feishu=True,
            webhook_url="hook",
            delivery_progress=resumed_progress,
            progress_callback=lambda progress: sent_payloads.append(json.loads(json.dumps(progress))),
        )
    finally:
        monitor_delivery_module.send_webhook_message = original_send_webhook_message
    duplicate_after = is_duplicate_snapshot(
        build_state_payload(snapshot, messages[0], messages[1], delivery_progress=final_progress),
        snapshot,
        messages[0],
        messages[1],
        channels=channels,
    )
    passed = (
        duplicate_before is False
        and resumed_progress == [{"webhook": True}, {"webhook": False}]
        and final_progress == [{"webhook": True}, {"webhook": True}]
        and len(sent_payloads) == 1
        and duplicate_after is True
    )
    detail = (
        f"duplicate_before={duplicate_before}, resumed_progress={resumed_progress}, "
        f"final_progress={final_progress}, progress_updates={len(sent_payloads)}, duplicate_after={duplicate_after}"
    )
    return CheckResult(name="partial_delivery_resume", passed=passed, detail=detail)


def check_empty_channel_progress_is_not_persisted() -> CheckResult:
    snapshot = compute_snapshot(
        prices=pd.DataFrame({"159941": [1.50, 1.55]}, index=pd.to_datetime(["2026-05-26", "2026-05-27"])),
        strategy_config={
            "selected_pool": [{"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"}],
            "backtest_nav_file": "momentum_backtest/output/core/backtest_nav.csv",
            "strategy_id": "default",
            "strategy_label": "test",
            "extra_cap_label": "命中额外降仓",
        },
        result=pd.DataFrame(
            {
                "signal": ["159941", "159941"],
                "holding": ["159941", "159941"],
                "target_exposure": [1.0, 1.0],
                "current_momentum": [0.10, 0.12],
                "effective_momentum": [0.10, 0.12],
                "nav": [1.0, 1.02],
                "drawdown": [0.0, 0.0],
                "weight_159941": [1.0, 1.0],
                "target_weight_159941": [1.0, 1.0],
            },
            index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
        ),
        close_result=pd.DataFrame(
            {
                "signal": ["159941", "159941"],
                "holding": ["159941", "159941"],
                "target_exposure": [1.0, 1.0],
                "current_momentum": [0.10, 0.12],
                "effective_momentum": [0.10, 0.12],
                "nav": [1.0, 1.02],
                "drawdown": [0.0, 0.0],
                "weight_159941": [1.0, 1.0],
                "target_weight_159941": [1.0, 1.0],
            },
            index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
        ),
        close_trades=pd.DataFrame(columns=["date", "action", "code", "theme", "name", "from_exposure", "to_exposure", "nav"]),
        base_target_exposure=pd.Series([1.0, 1.0], index=pd.to_datetime(["2026-05-26", "2026-05-27"])),
        market_proxy_context={"mode": None, "progress": None, "market_amount_ratio_20_60": None, "market_amount_ratio_5_20": None, "market_breadth_proxy": None},
        spot_prices={},
        raw_closes={},
        now=datetime(2026, 5, 27, 15, 0),
    )
    payload = build_state_payload(snapshot, "market", "trade", delivery_progress=[{}, {}])
    loaded_progress = load_delivery_progress_for_bundle(payload, snapshot, ["market", "trade"], channels=[])
    duplicate = is_duplicate_snapshot(payload, snapshot, "market", "trade", channels=[])
    passed = (
        payload.get("_delivery_progress_schema") == "v2"
        and "_delivery_progress" not in payload
        and loaded_progress == [{}, {}]
        and duplicate is True
    )
    detail = (
        f"schema={payload.get('_delivery_progress_schema')}, persisted={'_delivery_progress' in payload}, "
        f"loaded_progress={loaded_progress}, duplicate={duplicate}"
    )
    return CheckResult(name="empty_channel_progress_omitted", passed=passed, detail=detail)


def check_empty_channel_state_does_not_block_future_delivery() -> CheckResult:
    snapshot = compute_snapshot(
        prices=pd.DataFrame({"159941": [1.50, 1.55]}, index=pd.to_datetime(["2026-05-26", "2026-05-27"])),
        strategy_config={
            "selected_pool": [{"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"}],
            "backtest_nav_file": "momentum_backtest/output/core/backtest_nav.csv",
            "strategy_id": "default",
            "strategy_label": "test",
            "extra_cap_label": "命中额外降仓",
        },
        result=pd.DataFrame(
            {
                "signal": ["159941", "159941"],
                "holding": ["159941", "159941"],
                "target_exposure": [1.0, 1.0],
                "current_momentum": [0.10, 0.12],
                "effective_momentum": [0.10, 0.12],
                "nav": [1.0, 1.02],
                "drawdown": [0.0, 0.0],
                "weight_159941": [1.0, 1.0],
                "target_weight_159941": [1.0, 1.0],
            },
            index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
        ),
        close_result=pd.DataFrame(
            {
                "signal": ["159941", "159941"],
                "holding": ["159941", "159941"],
                "target_exposure": [1.0, 1.0],
                "current_momentum": [0.10, 0.12],
                "effective_momentum": [0.10, 0.12],
                "nav": [1.0, 1.02],
                "drawdown": [0.0, 0.0],
                "weight_159941": [1.0, 1.0],
                "target_weight_159941": [1.0, 1.0],
            },
            index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
        ),
        close_trades=pd.DataFrame(columns=["date", "action", "code", "theme", "name", "from_exposure", "to_exposure", "nav"]),
        base_target_exposure=pd.Series([1.0, 1.0], index=pd.to_datetime(["2026-05-26", "2026-05-27"])),
        market_proxy_context={"mode": None, "progress": None, "market_amount_ratio_20_60": None, "market_amount_ratio_5_20": None, "market_breadth_proxy": None},
        spot_prices={},
        raw_closes={},
        now=datetime(2026, 5, 27, 15, 0),
    )
    payload = build_state_payload(snapshot, "market", "trade", delivery_progress=[{}, {}])
    channels = ["webhook"]
    loaded_progress = load_delivery_progress_for_bundle(payload, snapshot, ["market", "trade"], channels=channels)
    duplicate = is_duplicate_snapshot(payload, snapshot, "market", "trade", channels=channels)
    passed = loaded_progress == [{"webhook": False}, {"webhook": False}] and duplicate is False
    detail = f"loaded_progress={loaded_progress}, duplicate={duplicate}"
    return CheckResult(name="empty_channel_future_delivery", passed=passed, detail=detail)


def check_intraday_history_filter_drops_same_day_bar() -> CheckResult:
    hist = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-26", "2026-05-27"]),
            "close": [1.50, 1.55],
        }
    )
    filtered = filter_history_to_confirmed_closes(
        hist,
        today=pd.Timestamp("2026-05-27").date(),
        allow_same_day_close=False,
    )
    passed = len(filtered) == 1 and str(pd.Timestamp(filtered["date"].iloc[-1]).date()) == "2026-05-26"
    detail = f"dates={[str(pd.Timestamp(x).date()) for x in filtered['date'].tolist()]}"
    return CheckResult(name="intraday_history_filter", passed=passed, detail=detail)


def check_indexed_cache_filter_drops_same_day_row() -> CheckResult:
    frame = pd.DataFrame(
        {"nav": [1.0, 1.1]},
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    filtered = filter_indexed_frame_to_confirmed_closes(
        frame,
        today=pd.Timestamp("2026-05-27").date(),
        allow_same_day_close=False,
    )
    passed = len(filtered) == 1 and str(filtered.index[-1].date()) == "2026-05-26"
    detail = f"dates={[str(ts.date()) for ts in filtered.index.tolist()]}"
    return CheckResult(name="indexed_cache_filter", passed=passed, detail=detail)


def check_drawdown_rebalance_count_uses_trade_days() -> CheckResult:
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-27", "2026-05-27", "2026-05-27", "2026-05-28"]),
            "action": ["SELL", "BUY", "BUY", "REDUCE"],
        }
    )
    rebalances = count_rebalance_days(trades, pd.Timestamp("2026-05-27"), pd.Timestamp("2026-05-28"))
    recovered_true = is_recovered(-5e-12)
    recovered_false = is_recovered(-1e-6)
    passed = rebalances == 2 and recovered_true is True and recovered_false is False
    detail = (
        f"rebalances={rebalances}, "
        f"recovered_true={recovered_true}, recovered_false={recovered_false}"
    )
    return CheckResult(name="drawdown_rebalance_count", passed=passed, detail=detail)


def check_hs300_yearly_returns_use_prior_year_end() -> CheckResult:
    index = pd.to_datetime(["2024-05-06", "2024-12-31", "2025-01-02", "2025-12-31"])
    result = pd.DataFrame({"nav": [1.00, 1.10, 1.20, 1.32]}, index=index)
    benchmark = pd.Series([1.00, 1.05, 1.08, 1.188], index=index)
    yearly = calculate_yearly_returns(result, benchmark)
    first_row = yearly.iloc[0].to_dict() if len(yearly) >= 1 else {}
    second_row = yearly.iloc[1].to_dict() if len(yearly) >= 2 else {}
    passed = (
        len(yearly) == 2
        and int(yearly.iloc[0]["year"]) == 2024
        and yearly.iloc[0]["start_date"] == "2024-05-06"
        and yearly.iloc[0]["end_date"] == "2024-12-31"
        and abs(float(yearly.iloc[0]["strategy_return"]) - 0.10) < FLOAT_TOL
        and abs(float(yearly.iloc[0]["hs300_return"]) - 0.05) < FLOAT_TOL
        and int(yearly.iloc[1]["year"]) == 2025
        and yearly.iloc[1]["start_date"] == "2024-12-31"
        and yearly.iloc[1]["end_date"] == "2025-12-31"
        and abs(float(yearly.iloc[1]["strategy_return"]) - 0.20) < FLOAT_TOL
        and abs(float(yearly.iloc[1]["hs300_return"]) - 0.13142857142857145) < 1e-12
    )
    detail = f"first={first_row}, second={second_row}"
    return CheckResult(name="hs300_yearly_returns_prior_close", passed=passed, detail=detail)


def check_preferred_payload_supports_strict_improvements() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        notify_path = Path(tmpdir) / "notify_state.json"
        best_path = Path(tmpdir) / "best.json"
        best_path.write_text(
            json.dumps(
                {
                    "baseline": {"strategy": "baseline_v1", "annualized_return": 0.10},
                    "strict_improvements": [
                        {"strategy": "strict_best_v2", "annualized_return": 0.12},
                    ],
                    "descriptions": {"strict_best_v2": "strict best"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        from search_utils import load_preferred_strategy_payload

        payload = load_preferred_strategy_payload(notify_path, best_path)

    passed = (
        isinstance(payload, dict)
        and payload.get("strategy") == "strict_best_v2"
        and isinstance(payload.get("summary"), dict)
        and payload["summary"].get("strategy") == "strict_best_v2"
        and payload.get("description") == "strict best"
    )
    detail = str(payload)
    return CheckResult(name="preferred_payload_strict_improvements", passed=passed, detail=detail)


def check_optional_candidate_history_fetch_failure_is_nonfatal() -> CheckResult:
    prices = pd.DataFrame(
        {
            "510300": [4.90, 5.00],
        },
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    candidates = pd.DataFrame(
        [
            {
                "theme": "30年国债",
                "code": "511090",
                "name": "30年国债ETF",
                "sina_symbol": "sh511090",
            }
        ]
    )

    def fake_fetch(_: pd.DataFrame, years: int) -> pd.DataFrame:
        raise RuntimeError(f"mock network down for {years}y")

    updated_prices, skipped_codes, error_message = try_join_missing_candidate_histories(
        prices=prices,
        candidates=candidates,
        years=15,
        fetch_fn=fake_fetch,
    )
    passed = (
        updated_prices.equals(prices)
        and skipped_codes == ["511090"]
        and isinstance(error_message, str)
        and "mock network down" in error_message
    )
    detail = (
        f"columns={updated_prices.columns.tolist()}, skipped={skipped_codes}, "
        f"error={error_message}"
    )
    return CheckResult(name="optional_candidate_history_fetch_nonfatal", passed=passed, detail=detail)


def check_missing_required_history_raises_clear_error() -> CheckResult:
    prices = pd.DataFrame(
        {
            "510300": [4.90, 5.00],
        },
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    required = pd.DataFrame(
        [
            {
                "theme": "10年国债",
                "code": "511260",
                "name": "十年国债ETF",
                "sina_symbol": "sh511260",
            }
        ]
    )
    try:
        raise_if_missing_required_histories(prices, required, context="baseline treasury")
    except RuntimeError as exc:
        passed = "missing required baseline treasury history: 511260 十年国债ETF" == str(exc)
        detail = str(exc)
    else:
        passed = False
        detail = "did not raise"
    return CheckResult(name="required_history_clear_error", passed=passed, detail=detail)


def check_runtime_env_scrubs_user_site_packages() -> CheckResult:
    original_path = list(sys.path)
    try:
        sys.path[:] = [
            "/tmp/project",
            f"{Path.home()}/Library/Python/3.9/lib/python/site-packages",
            f"{Path.home()}/.local/lib/python3.9/site-packages",
            "/opt/homebrew/lib/python3.9/site-packages",
        ]
        scrub_user_site_packages()
        passed = (
            f"{Path.home()}/Library/Python/3.9/lib/python/site-packages" not in sys.path
            and f"{Path.home()}/.local/lib/python3.9/site-packages" not in sys.path
            and "/opt/homebrew/lib/python3.9/site-packages" in sys.path
            and "/tmp/project" in sys.path
        )
        detail = str(sys.path)
    finally:
        sys.path[:] = original_path
    return CheckResult(name="runtime_env_scrub_user_site", passed=passed, detail=detail)


def check_package_import_for_active_research_modules() -> CheckResult:
    project_root = str(Path(__file__).resolve().parents[1])
    inserted = False
    try:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            inserted = True
        module_names = [
            "momentum_backtest.analyze_contribution",
            "momentum_backtest.analyze_drawdowns",
            "momentum_backtest.compare_candidate_pool_additions",
            "momentum_backtest.compare_china_internet_guards",
            "momentum_backtest.compare_current_best_fine_tune",
            "momentum_backtest.compare_current_best_pool_additions",
            "momentum_backtest.compare_defensive_persistence",
            "momentum_backtest.compare_goal_optimizations",
            "momentum_backtest.compare_hs300_regime_fixes",
            "momentum_backtest.compare_market_proxy_variants",
            "momentum_backtest.compare_resource_guards",
            "momentum_backtest.compare_strategy_refinements",
            "momentum_backtest.compare_tail_risk_bond_overlay",
            "momentum_backtest.daily_monitor",
            "momentum_backtest.validate_strategy_outputs",
        ]
        imported: list[str] = []
        for module_name in module_names:
            module = importlib.import_module(module_name)
            imported.append(module_name)
        fine_tune = importlib.import_module("momentum_backtest.compare_current_best_fine_tune")
        passed = hasattr(fine_tune, "load_prices") and hasattr(fine_tune, "normalize_params")
        detail = ",".join(imported)
    except Exception as exc:
        passed = False
        detail = f"{type(exc).__name__}: {exc}"
    finally:
        if inserted:
            try:
                sys.path.remove(project_root)
            except ValueError:
                pass
    return CheckResult(name="package_import_active_research_modules", passed=passed, detail=detail)


def check_matplotlib_import_uses_workspace_cache() -> CheckResult:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = "/private/tmp/pycache"
    env.pop("MPLCONFIGDIR", None)
    command = [
        sys.executable,
        "-c",
        (
            "import importlib; "
            "importlib.import_module('momentum_backtest.compare_hs300_regime_fixes')"
        ),
    ]
    proc = subprocess.run(
        command,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )
    stderr = proc.stderr.strip()
    blocked_markers = [
        ".matplotlib is not a writable directory",
        "Matplotlib created a temporary cache directory",
    ]
    passed = proc.returncode == 0 and not any(marker in stderr for marker in blocked_markers)
    detail = f"returncode={proc.returncode}, stderr={stderr[:300]}"
    return CheckResult(name="matplotlib_workspace_cache", passed=passed, detail=detail)


def check_summary_payload_keeps_latest_portfolio() -> CheckResult:
    summary_df = pd.DataFrame(
        [
            {
                "strategy": "current_baseline",
                "annualized_return": 0.20,
                "sharpe_rf0": 1.00,
                "max_drawdown_integral": 10.0,
                "max_drawdown": -0.20,
                "latest_holding_code": "159941",
                "latest_holding_theme": "纳指ETF",
                "latest_holding_name": "纳指ETF广发",
                "latest_portfolio": "纳指ETF / 纳指ETF广发 (159941) 73%；沪深300 / 沪深300ETF华泰柏瑞 (510300) 27%",
                "latest_momentum": 0.12,
                "latest_exposure": 1.0,
            },
            {
                "strategy": "better_v2",
                "annualized_return": 0.21,
                "sharpe_rf0": 1.05,
                "max_drawdown_integral": 9.8,
                "max_drawdown": -0.19,
                "latest_holding_code": "159941",
                "latest_holding_theme": "纳指ETF",
                "latest_holding_name": "纳指ETF广发",
                "latest_portfolio": "纳指ETF / 纳指ETF广发 (159941) 60%；沪深300 / 沪深300ETF华泰柏瑞 (510300) 40%",
                "latest_momentum": 0.13,
                "latest_exposure": 1.0,
            },
        ]
    )
    baseline = summary_df[summary_df["strategy"] == "current_baseline"].iloc[0]
    payload = {
        "baseline": baseline.to_dict(),
        "strict_improvements": summary_df[summary_df["strategy"] != "current_baseline"].to_dict(orient="records"),
    }
    passed = (
        payload["baseline"].get("latest_portfolio") == "纳指ETF / 纳指ETF广发 (159941) 73%；沪深300 / 沪深300ETF华泰柏瑞 (510300) 27%"
        and payload["strict_improvements"][0].get("latest_portfolio")
        == "纳指ETF / 纳指ETF广发 (159941) 60%；沪深300 / 沪深300ETF华泰柏瑞 (510300) 40%"
    )
    detail = str(payload)
    return CheckResult(name="summary_payload_latest_portfolio", passed=passed, detail=detail)


def check_preferred_payload_preserves_latest_portfolio() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        notify_path = Path(tmpdir) / "notify_state.json"
        best_path = Path(tmpdir) / "best.json"
        best_path.write_text(
            json.dumps(
                {
                    "baseline": {
                        "strategy": "baseline_v1",
                        "annualized_return": 0.10,
                        "latest_portfolio": "纳指ETF / 纳指ETF广发 (159941) 70%；沪深300 / 沪深300ETF华泰柏瑞 (510300) 30%",
                    },
                    "descriptions": {"baseline_v1": "baseline"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        payload = load_preferred_strategy_payload(notify_path, best_path)
    passed = (
        isinstance(payload, dict)
        and isinstance(payload.get("summary"), dict)
        and payload["summary"].get("latest_portfolio")
        == "纳指ETF / 纳指ETF广发 (159941) 70%；沪深300 / 沪深300ETF华泰柏瑞 (510300) 30%"
    )
    detail = str(payload)
    return CheckResult(name="preferred_payload_latest_portfolio", passed=passed, detail=detail)


def check_notify_candidate_sort_uses_max_drawdown_tiebreaker() -> CheckResult:
    frame = pd.DataFrame(
        [
            {
                "strategy": "higher_drawdown",
                "annualized_return": 0.20,
                "sharpe_rf0": 1.10,
                "max_drawdown_integral": 12.0,
                "max_drawdown": -0.22,
            },
            {
                "strategy": "lower_drawdown",
                "annualized_return": 0.20,
                "sharpe_rf0": 1.10,
                "max_drawdown_integral": 12.0,
                "max_drawdown": -0.18,
            },
        ]
    )
    sorted_frame = sort_notify_candidates(frame)
    best_strategy = str(sorted_frame.iloc[0]["strategy"]) if not sorted_frame.empty else ""
    passed = best_strategy == "lower_drawdown"
    detail = f"ordered={sorted_frame['strategy'].tolist()}"
    return CheckResult(name="notify_candidate_sort_tiebreaker", passed=passed, detail=detail)


def check_preferred_payload_falls_back_from_invalid_notify_state() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        notify_path = Path(tmpdir) / "notify_state.json"
        best_path = Path(tmpdir) / "best.json"
        notify_path.write_text(
            json.dumps({"strategy": "", "summary": "broken", "description": "bad"}, ensure_ascii=False),
            encoding="utf-8",
        )
        best_path.write_text(
            json.dumps(
                {
                    "baseline": {"strategy": "baseline_v1", "annualized_return": 0.10},
                    "valid_improvements": [
                        {"strategy": "recovered_v2", "annualized_return": 0.12},
                    ],
                    "descriptions": {"recovered_v2": "fallback best"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        payload = load_preferred_strategy_payload(notify_path, best_path)

    passed = (
        isinstance(payload, dict)
        and payload.get("strategy") == "recovered_v2"
        and isinstance(payload.get("summary"), dict)
        and payload["summary"].get("strategy") == "recovered_v2"
    )
    detail = str(payload)
    return CheckResult(name="preferred_payload_invalid_notify_fallback", passed=passed, detail=detail)


def check_required_payload_falls_back_to_field_complete_baseline() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        notify_path = Path(tmpdir) / "notify_state.json"
        best_path = Path(tmpdir) / "best.json"
        best_path.write_text(
            json.dumps(
                {
                    "baseline": {
                        "strategy": "baseline_v1",
                        "annualized_return": 0.10,
                        "treasury_code": "511260",
                        "mode": "market_stress_only",
                        "risk_cap": 0.20,
                    },
                    "valid_improvements": [
                        {
                            "strategy": "candidate_missing_fields",
                            "annualized_return": 0.12,
                        },
                    ],
                    "descriptions": {
                        "baseline_v1": "baseline",
                        "candidate_missing_fields": "missing required context",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        payload = load_required_strategy_payload(
            notify_path,
            best_path,
            context_name="cash overlay strategy context",
            required_summary_fields=("treasury_code", "mode", "risk_cap"),
        )

    passed = (
        isinstance(payload, dict)
        and payload.get("strategy") == "baseline_v1"
        and isinstance(payload.get("summary"), dict)
        and payload["summary"].get("treasury_code") == "511260"
        and payload["summary"].get("mode") == "market_stress_only"
        and payload["summary"].get("risk_cap") == 0.20
    )
    detail = str(payload)
    return CheckResult(name="required_payload_baseline_field_fallback", passed=passed, detail=detail)


def check_invalid_previous_summary_ignored() -> CheckResult:
    notify_state = {"summary": {"annualized_return": "0.1", "sharpe_rf0": "oops"}}
    summary = extract_valid_previous_summary(notify_state)
    passed = summary is None
    detail = f"summary={summary}"
    return CheckResult(name="invalid_previous_summary_ignored", passed=passed, detail=detail)


def check_notify_best_candidate_suppresses_delivery_errors() -> CheckResult:
    frame = pd.DataFrame(
        [
            {
                "strategy": "candidate_v1",
                "annualized_return": 0.21,
                "sharpe_rf0": 1.05,
                "max_drawdown_integral": 9.8,
                "max_drawdown": -0.19,
            }
        ]
    )
    messages: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        notify_path = Path(tmpdir) / "notify_state.json"
        notified, detail = notify_best_candidate(
            argparse.Namespace(notify=True, webhook_url=""),
            frame,
            descriptions={"candidate_v1": "candidate desc"},
            baseline_summary={"strategy": "baseline_v1"},
            default_webhook="https://example.invalid/hook",
            notify_state_path=notify_path,
            send_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mock network down")),
            suppress_exceptions=True,
            print_fn=messages.append,
        )
        notify_state_exists = notify_path.exists()

    passed = (
        not notified
        and isinstance(detail, str)
        and "candidate_v1" in detail
        and "mock network down" in detail
        and not notify_state_exists
        and any("webhook notify skipped" in msg for msg in messages)
    )
    detail_text = f"detail={detail}, messages={messages}, notify_state_exists={notify_state_exists}"
    return CheckResult(name="notify_best_candidate_suppresses_delivery_errors", passed=passed, detail=detail_text)


def check_notify_ranked_incremental_candidate_suppresses_delivery_errors() -> CheckResult:
    frame = pd.DataFrame(
        [
            {
                "strategy": "candidate_v2",
                "annualized_return": 0.22,
                "sharpe_rf0": 1.08,
                "max_drawdown_integral": 9.6,
                "max_drawdown": -0.18,
                "is_valid_change": True,
            }
        ]
    )
    messages: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        notify_path = Path(tmpdir) / "notify_state.json"
        import builtins

        original_print = builtins.print
        try:
            builtins.print = lambda *args, **kwargs: messages.append(" ".join(str(arg) for arg in args))
            notified, detail = notify_ranked_incremental_candidate(
                argparse.Namespace(notify=True, webhook_url=""),
                frame,
                descriptions={"candidate_v2": "candidate desc"},
                baseline_summary={
                    "strategy": "baseline_v1",
                    "annualized_return": 0.20,
                    "sharpe_rf0": 1.00,
                    "max_drawdown_integral": 10.0,
                },
                metric_tolerance=1e-12,
                default_webhook="https://example.invalid/hook",
                notify_state_path=notify_path,
                send_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mock ranked network down")),
            )
        finally:
            builtins.print = original_print
        notify_state_exists = notify_path.exists()

    passed = (
        not notified
        and isinstance(detail, str)
        and "candidate_v2" in detail
        and "mock ranked network down" in detail
        and not notify_state_exists
        and any("webhook notify skipped" in msg for msg in messages)
    )
    detail_text = f"detail={detail}, messages={messages}, notify_state_exists={notify_state_exists}"
    return CheckResult(name="notify_ranked_incremental_candidate_suppresses_delivery_errors", passed=passed, detail=detail_text)


def check_add_notify_cli_args_preserves_default_modes() -> CheckResult:
    parser_default_off = add_notify_cli_args(argparse.ArgumentParser(prog="default_off"))
    parser_default_on = add_notify_cli_args(argparse.ArgumentParser(prog="default_on"), default_enabled=True)

    args_default_off = parser_default_off.parse_args([])
    args_default_off_notify = parser_default_off.parse_args(["--notify"])
    args_default_on = parser_default_on.parse_args([])
    args_default_on_disable = parser_default_on.parse_args(["--disable-notify"])

    passed = (
        not should_send_notify(args_default_off)
        and should_send_notify(args_default_off_notify)
        and should_send_notify(args_default_on)
        and not should_send_notify(args_default_on_disable)
    )
    detail = (
        f"default_off={should_send_notify(args_default_off)}, "
        f"default_off_notify={should_send_notify(args_default_off_notify)}, "
        f"default_on={should_send_notify(args_default_on)}, "
        f"default_on_disable={should_send_notify(args_default_on_disable)}"
    )
    return CheckResult(name="add_notify_cli_args_default_modes", passed=passed, detail=detail)


def check_historical_nav_schema_consistent() -> CheckResult:
    core_nav = pd.read_csv(CORE_DIR / "historical_nav.csv")
    china_nav = pd.read_csv(Path("momentum_backtest/output/research/china_internet_cap70/historical_nav.csv"))
    resource_nav = pd.read_csv(Path("momentum_backtest/output/research/resource_abs08/historical_nav.csv"))
    expected_columns = ["date", "historical_nav"]
    passed = (
        core_nav.columns.tolist() == expected_columns
        and china_nav.columns.tolist() == expected_columns
        and resource_nav.columns.tolist() == expected_columns
    )
    detail = (
        f"core={core_nav.columns.tolist()}, "
        f"china={china_nav.columns.tolist()}, "
        f"resource={resource_nav.columns.tolist()}"
    )
    return CheckResult(name="historical_nav_schema", passed=passed, detail=detail)


def check_contribution_summary_uses_actual_weights() -> CheckResult:
    prices = pd.DataFrame(
        {
            "510300": [100.0, 100.0],
            "159941": [100.0, 110.0],
        },
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    selected = pd.DataFrame(
        [
            {"code": "510300", "theme": "沪深300", "name": "沪深300ETF"},
            {"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"},
        ]
    )
    result = pd.DataFrame(
        {
            "nav": [1.0, 1.07],
            "strategy_return": [0.0, 0.07],
            "holding": ["159941", "159941"],
            "exposure": [1.0, 1.0],
            "trade_cost_rate": [0.0, 0.0],
            "turnover": [0.0, 0.0],
            "weight_510300": [0.0, 0.30],
            "weight_159941": [1.0, 0.70],
            "return_weight_510300": [0.0, 0.0],
            "return_weight_159941": [0.0, 0.70],
        },
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    contribution = build_contribution_summary(prices, selected, result).set_index("code")
    passed = (
        abs(float(contribution.loc["159941", "net_profit_contribution"]) - 0.07) < FLOAT_TOL
        and abs(float(contribution.loc["510300", "net_profit_contribution"])) < FLOAT_TOL
        and int(contribution.loc["510300", "holding_days"]) == 1
        and int(contribution.loc["159941", "holding_days"]) == 2
    )
    detail = str(contribution[["holding_days", "net_profit_contribution"]].to_dict(orient="index"))
    return CheckResult(name="contribution_summary_weights", passed=passed, detail=detail)


def check_contribution_summary_cost_drag_uses_currency_units() -> CheckResult:
    prices = pd.DataFrame(
        {
            "510300": [100.0, 110.0],
            "159941": [100.0, 100.0],
        },
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    selected = pd.DataFrame(
        [
            {"code": "510300", "theme": "沪深300", "name": "沪深300ETF"},
            {"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"},
        ]
    )
    result = pd.DataFrame(
        {
            "nav": [1.0, 1.0994],
            "strategy_return": [0.0, 0.0994],
            "holding": ["510300", "510300"],
            "exposure": [1.0, 1.0],
            "trade_cost_rate": [0.0, 0.0006],
            "turnover": [0.0, 1.0],
            "weight_510300": [0.0, 1.0],
            "weight_159941": [0.0, 0.0],
            "return_weight_510300": [0.0, 1.0],
            "return_weight_159941": [0.0, 0.0],
        },
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    contribution = build_contribution_summary(prices, selected, result)
    total_net = float(contribution["net_profit_contribution"].sum())
    total_cost = float(contribution["cost_drag_contribution"].sum())
    passed = abs(total_net - 0.0994) < FLOAT_TOL and abs(total_cost + 0.0006) < FLOAT_TOL
    detail = f"total_net={total_net:.6f}, total_cost={total_cost:.6f}"
    return CheckResult(name="contribution_summary_cost_units", passed=passed, detail=detail)


def check_formal_signal_matches_target_weight_leader(result: pd.DataFrame) -> CheckResult:
    target_cols = [col for col in result.columns if col.startswith("target_weight_")]
    if not target_cols:
        return CheckResult(name="formal_signal_target_leader", passed=True, detail="no_target_weight_columns")
    target_weights = result[target_cols].copy()
    target_weights.columns = [col.removeprefix("target_weight_") for col in target_cols]
    target_exposure = target_weights.sum(axis=1)
    expected_signal = target_weights.idxmax(axis=1).where(target_exposure > FLOAT_TOL).map(normalize_code)
    actual_signal = result["signal"].map(normalize_code)
    mismatch_mask = expected_signal.fillna("") != actual_signal.fillna("")
    passed = not bool(mismatch_mask.any())
    detail = f"mismatch_count={int(mismatch_mask.sum())}"
    if mismatch_mask.any():
        first_idx = mismatch_mask[mismatch_mask].index[0]
        row = target_weights.loc[first_idx]
        target_desc = "; ".join(
            f"{normalize_code(code)}={float(weight):.3f}"
            for code, weight in row.items()
            if abs(float(weight)) > FLOAT_TOL
        )
        detail += (
            f", first_date={pd.Timestamp(first_idx).date().isoformat()}, "
            f"signal={actual_signal.loc[first_idx]}, expected={expected_signal.loc[first_idx]}, "
            f"target={target_desc}"
        )
    return CheckResult(name="formal_signal_target_leader", passed=passed, detail=detail)


def check_formal_current_momentum_matches_signal_asset(result: pd.DataFrame, prices: pd.DataFrame) -> CheckResult:
    signal = result["signal"].map(normalize_code)
    expected = compute_signal_asset_momentum(prices, signal, lookback=25)
    actual = result["current_momentum"]
    compare = pd.concat([actual.rename("actual"), expected.rename("expected")], axis=1).dropna()
    if compare.empty:
        return CheckResult(name="formal_signal_asset_momentum", passed=True, detail="no_comparable_rows")
    diff = (compare["actual"] - compare["expected"]).abs()
    passed = bool(diff.max() < FLOAT_TOL)
    detail = f"max_diff={float(diff.max()):.3e}"
    if not passed:
        first_idx = diff.idxmax()
        detail += (
            f", first_date={pd.Timestamp(first_idx).date().isoformat()}, "
            f"actual={float(compare.loc[first_idx, 'actual']):.6f}, "
            f"expected={float(compare.loc[first_idx, 'expected']):.6f}, "
            f"signal={signal.loc[first_idx]}"
        )
    return CheckResult(name="formal_signal_asset_momentum", passed=passed, detail=detail)


def check_analysis_contribution_summary_matches_core() -> CheckResult:
    core = pd.read_csv(CORE_DIR / "contribution_summary.csv")
    analysis = pd.read_csv(Path("momentum_backtest/output/analysis/contribution_summary.csv"))
    passed = core.columns.tolist() == analysis.columns.tolist() and len(core) == len(analysis)
    if passed:
        for column in core.columns:
            if pd.api.types.is_bool_dtype(core[column]) or pd.api.types.is_bool_dtype(analysis[column]):
                if not core[column].fillna(False).astype(bool).equals(analysis[column].fillna(False).astype(bool)):
                    passed = False
                    break
            elif pd.api.types.is_numeric_dtype(core[column]) or pd.api.types.is_numeric_dtype(analysis[column]):
                left = pd.to_numeric(core[column], errors="coerce")
                right = pd.to_numeric(analysis[column], errors="coerce")
                if not ((left - right).abs().fillna(0.0) < FLOAT_TOL).all():
                    passed = False
                    break
            elif not core[column].fillna("").astype(str).equals(analysis[column].fillna("").astype(str)):
                passed = False
                break
    detail = f"core_rows={len(core)}, analysis_rows={len(analysis)}, same={passed}"
    return CheckResult(name="analysis_contribution_sync", passed=passed, detail=detail)


def check_analysis_drawdown_report_matches_core() -> CheckResult:
    nav = pd.read_csv(CORE_DIR / "backtest_nav.csv", parse_dates=["date"]).set_index("date")
    prices = pd.read_csv(CORE_DIR / "prices.csv", parse_dates=["date"]).set_index("date")
    trades = pd.read_csv(CORE_DIR / "trades.csv", parse_dates=["date"])
    selected = pd.read_csv(CORE_DIR / "selected_etfs.csv", dtype={"code": str})
    compare = pd.read_csv(CORE_DIR / "strategy_vs_hs300.csv", parse_dates=["date"]).set_index("date")
    expected = build_drawdown_episode_report(nav, prices, trades, selected, compare)
    actual = pd.read_csv(Path("momentum_backtest/output/analysis/drawdown_episodes.csv"))
    passed = expected.columns.tolist() == actual.columns.tolist() and len(expected) == len(actual)
    if passed:
        for column in expected.columns:
            if pd.api.types.is_bool_dtype(expected[column]) or pd.api.types.is_bool_dtype(actual[column]):
                if not expected[column].fillna(False).astype(bool).equals(actual[column].fillna(False).astype(bool)):
                    passed = False
                    break
            elif pd.api.types.is_numeric_dtype(expected[column]) or pd.api.types.is_numeric_dtype(actual[column]):
                left = pd.to_numeric(expected[column], errors="coerce")
                right = pd.to_numeric(actual[column], errors="coerce")
                if not ((left - right).abs().fillna(0.0) < FLOAT_TOL).all():
                    passed = False
                    break
            elif not expected[column].fillna("").astype(str).equals(actual[column].fillna("").astype(str)):
                passed = False
                break
    if not passed and expected.columns.tolist() == actual.columns.tolist() and len(expected) == len(actual):
        mismatch_column = None
        for column in expected.columns:
            if pd.api.types.is_bool_dtype(expected[column]) or pd.api.types.is_bool_dtype(actual[column]):
                same = expected[column].fillna(False).astype(bool).equals(actual[column].fillna(False).astype(bool))
            elif pd.api.types.is_numeric_dtype(expected[column]) or pd.api.types.is_numeric_dtype(actual[column]):
                left = pd.to_numeric(expected[column], errors="coerce")
                right = pd.to_numeric(actual[column], errors="coerce")
                same = ((left - right).abs().fillna(0.0) < FLOAT_TOL).all()
            else:
                same = expected[column].fillna("").astype(str).equals(actual[column].fillna("").astype(str))
            if not same:
                mismatch_column = column
                break
        detail = (
            f"expected_rows={len(expected)}, actual_rows={len(actual)}, "
            f"trade_actions_col={'trade_actions_in_episode' in actual.columns}, mismatch_column={mismatch_column}"
        )
        return CheckResult(name="analysis_drawdown_sync", passed=False, detail=detail)
    detail = (
        f"expected_rows={len(expected)}, actual_rows={len(actual)}, "
        f"trade_actions_col={'trade_actions_in_episode' in actual.columns}"
    )
    return CheckResult(name="analysis_drawdown_sync", passed=passed, detail=detail)


def check_trade_count_uses_trade_days() -> CheckResult:
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-27", "2026-05-27", "2026-05-27", "2026-05-28"]),
            "action": ["SELL", "BUY", "BUY", "REDUCE"],
        }
    )
    passed = count_trade_days(trades) == 2 and count_trade_days(pd.DataFrame(columns=["date", "action"])) == 0
    detail = f"trade_days={count_trade_days(trades)}, empty_trade_days={count_trade_days(pd.DataFrame(columns=['date','action']))}"
    return CheckResult(name="trade_count_trade_days", passed=passed, detail=detail)


def check_latest_portfolio_summary_uses_weights() -> CheckResult:
    selected = pd.DataFrame(
        [
            {"code": "510300", "theme": "沪深300", "name": "沪深300ETF"},
            {"code": "159941", "theme": "纳指ETF", "name": "纳指ETF广发"},
        ]
    )
    result = pd.DataFrame(
        {
            "holding": ["159941"],
            "exposure": [1.0],
            "weight_510300": [0.27],
            "weight_159941": [0.73],
        },
        index=pd.to_datetime(["2026-05-27"]),
    )
    text = get_latest_portfolio_text(selected, result)
    passed = text == "纳指ETF / 纳指ETF广发 (159941) 73%；沪深300 / 沪深300ETF (510300) 27%"
    return CheckResult(name="latest_portfolio_summary", passed=passed, detail=text)


def check_guard_shares_use_weights() -> CheckResult:
    index = pd.to_datetime(["2026-05-26", "2026-05-27"])
    result = pd.DataFrame(
        {
            "nav": [1.0, 1.0],
            "strategy_return": [0.0, 0.0],
            "exposure": [1.0, 1.0],
            "current_momentum": [0.1, 0.1],
            f"weight_{CHINA_INTERNET_CODE}": [0.0, 0.25],
            f"weight_{RESOURCE_CODE}": [0.0, 0.40],
            "holding": [CHINA_INTERNET_CODE, CHINA_INTERNET_CODE],
        },
        index=index,
    )
    empty_trades = pd.DataFrame(columns=["date", "action"])
    china_summary = summarize_china_internet_guard(result, empty_trades)
    resource_summary = summarize_resource_guard(result, empty_trades)
    passed = (
        abs(float(china_summary["china_internet_holding_share"]) - 0.5) < FLOAT_TOL
        and abs(float(china_summary["china_internet_exposure_share"]) - 0.125) < FLOAT_TOL
        and abs(float(resource_summary["resource_holding_share"]) - 0.5) < FLOAT_TOL
        and abs(float(resource_summary["resource_exposure_share"]) - 0.2) < FLOAT_TOL
    )
    detail = (
        f"china={china_summary['china_internet_holding_share']:.3f}/{china_summary['china_internet_exposure_share']:.3f}, "
        f"resource={resource_summary['resource_holding_share']:.3f}/{resource_summary['resource_exposure_share']:.3f}"
    )
    return CheckResult(name="guard_shares_use_weights", passed=passed, detail=detail)


def check_drawdown_top_holdings_use_weights() -> CheckResult:
    nav = pd.DataFrame(
        {
            "holding": ["159941", "159941"],
            "weight_159941": [1.0, 0.73],
            "weight_510300": [0.0, 0.27],
        },
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    nav["holding_code"] = nav["holding"].map(normalize_code)
    text = summarize_episode_holdings(nav, {"159941": "纳指ETF", "510300": "沪深300"}, nav.index[0], nav.index[-1])
    passed = text == "纳指ETF(159941) 86.5%; 沪深300(510300) 13.5%"
    return CheckResult(name="drawdown_top_holdings_weights", passed=passed, detail=text)


def check_custom_universe_flows_into_signals() -> CheckResult:
    index = pd.date_range("2025-01-01", periods=30, freq="B")
    prices = pd.DataFrame(
        {
            "510300": [100.0] * 25 + [101.0, 101.2, 101.4, 101.6, 101.8],
            "123456": [100.0] * 25 + [105.0, 105.5, 106.0, 106.5, 107.0],
            "511260": [100.0] * 30,
        },
        index=index,
    )
    params = build_default_strategy_params(
        drop_codes=["513650"],
        risk_codes=["123456"],
        defensive_codes=["511260"],
    )
    signal, target_exposure, _, _ = build_threshold_dual_signal(
        prices,
        lookback=25,
        absolute_threshold=0.01,
        weak_trend_defensive_weight=0.8,
        risk_codes=[str(code) for code in params["risk_codes"]],
        defensive_codes=[str(code) for code in params["defensive_codes"]],
    )
    latest_signal = normalize_code(signal.iloc[-1])
    latest_exposure = float(target_exposure.iloc[-1])
    passed = latest_signal == "123456" and abs(latest_exposure - 1.0) < FLOAT_TOL
    detail = f"signal={latest_signal}, exposure={latest_exposure:.3f}"
    return CheckResult(name="custom_universe_signal", passed=passed, detail=detail)


def build_custom_universe_strategy_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=160, freq="B")
    prices = pd.DataFrame(
        {
            "510300": [100.0 + i * 0.20 for i in range(len(index))],
            "123456": [100.0 + i * 0.45 for i in range(len(index))],
            "511260": [100.0 + i * 0.01 for i in range(len(index))],
        },
        index=index,
    )
    selected = pd.DataFrame(
        [
            {"code": "510300", "theme": "沪深300", "name": "沪深300ETF"},
            {"code": "123456", "theme": "测试增强", "name": "测试增强ETF"},
            {"code": "511260", "theme": "十年国债", "name": "十年国债ETF"},
        ]
    )
    return prices, selected


def check_goal_regime_mix_uses_custom_universe() -> CheckResult:
    prices, selected = build_custom_universe_strategy_fixture()
    result, _ = run_regime_mix_core_overheat_strategy(
        prices=prices,
        selected=selected,
        lookback=25,
        fee_rate=0.0,
        slippage_rate=0.0,
        absolute_threshold=0.01,
        weak_trend_defensive_weight=0.8,
        aggressive_core_weight=0.28,
        conservative_core_weight=0.27,
        regime_momentum_cut=0.06,
        overheat_drawdown_cut=-0.03,
        overheat_momentum_cut=0.25,
        overheat_max_exposure=0.66,
        overheat_high_momentum_cut=0.32,
        overheat_high_max_exposure=0.30,
        risk_codes=["123456"],
        defensive_codes=["511260"],
    )
    latest = result.iloc[-1]
    latest_signal = normalize_code(latest["signal"])
    latest_holding = normalize_code(latest["holding"])
    alt_weight = float(latest["weight_123456"])
    hs300_weight = float(latest["weight_510300"])
    passed = (
        latest_signal == "123456"
        and latest_holding == "123456"
        and alt_weight > hs300_weight
        and alt_weight > 0
    )
    detail = (
        f"signal={latest_signal}, holding={latest_holding}, "
        f"w123456={alt_weight:.3f}, w510300={hs300_weight:.3f}"
    )
    return CheckResult(name="goal_regime_mix_custom_universe", passed=passed, detail=detail)


def check_goal_regime_mix_parser_keeps_optional_thresholds() -> CheckResult:
    strategy_name = (
        "regime_mix_rm511580_rm513650_volume_guard_ag29_co27_vr91_vs90_vb01_vg30_"
        "cap08_hi04_dd02_rm05_vm16_oh23_ohh31"
    )
    params = parse_regime_mix_strategy_name(strategy_name)
    passed = (
        isinstance(params, dict)
        and abs(float(params["regime_momentum_cut"]) - 0.05) < FLOAT_TOL
        and abs(float(params["volume_guard_momentum_ceiling"]) - 0.16) < FLOAT_TOL
        and abs(float(params["overheat_momentum_cut"]) - 0.23) < FLOAT_TOL
        and abs(float(params["overheat_high_momentum_cut"]) - 0.31) < FLOAT_TOL
    )
    detail = str(params)
    return CheckResult(name="goal_regime_mix_parser_optional_thresholds", passed=passed, detail=detail)


def check_goal_reduced_pool_notified_params_keep_optional_thresholds() -> CheckResult:
    notified_context = {
        "params": {
            "drop_codes": ["511580", "513650"],
            "aggressive_core_weight": 0.29,
            "conservative_core_weight": 0.27,
            "regime_momentum_cut": 0.05,
            "volume_ratio_cut": 0.91,
            "volume_short_ratio_cut": 0.90,
            "volume_breadth_cut": -0.01,
            "volume_guard_cap": 0.30,
            "volume_guard_momentum_ceiling": 0.16,
            "overheat_drawdown_cut": -0.02,
            "overheat_momentum_cut": 0.23,
            "overheat_max_exposure": 0.08,
            "overheat_high_momentum_cut": 0.31,
            "overheat_high_max_exposure": 0.04,
        }
    }
    params = build_reduced_pool_notified_base_params(notified_context)
    passed = (
        abs(float(params["regime_momentum_cut"]) - 0.05) < FLOAT_TOL
        and abs(float(params["volume_guard_momentum_ceiling"]) - 0.16) < FLOAT_TOL
        and abs(float(params["overheat_momentum_cut"]) - 0.23) < FLOAT_TOL
        and abs(float(params["overheat_high_momentum_cut"]) - 0.31) < FLOAT_TOL
        and abs(float(params["overheat_cap"]) - 0.08) < FLOAT_TOL
        and abs(float(params["overheat_hi_cap"]) - 0.04) < FLOAT_TOL
    )
    detail = str(params)
    return CheckResult(name="goal_reduced_pool_notified_params", passed=passed, detail=detail)


def check_tail_overlay_uses_custom_universe_risk_budget() -> CheckResult:
    prices, selected = build_custom_universe_strategy_fixture()
    proxy = pd.DataFrame(
        {
            "market_amount_ratio_20_60": [1.2] * len(prices.index),
            "market_amount_ratio_5_20": [1.2] * len(prices.index),
            "market_breadth_proxy": [0.1] * len(prices.index),
        },
        index=prices.index,
    )
    params = build_default_strategy_params(
        drop_codes=["513650"],
        risk_codes=["123456"],
        defensive_codes=["511260"],
    )
    result, _ = apply_tail_bond_overlay(
        prices=prices,
        selected=selected,
        proxy=proxy,
        params=params,
        treasury_code="511260",
        tail_ratio_cut=2.0,
        tail_short_ratio_cut=2.0,
        tail_breadth_cut=1.0,
        tail_momentum_ceiling=1.0,
        tail_drawdown_cut=1.0,
        tail_risk_cap=0.0,
        fee_rate=0.0,
        slippage_rate=0.0,
    )
    latest = result.iloc[-1]
    latest_signal = normalize_code(latest["signal"])
    treasury_weight = float(latest["weight_511260"])
    alt_weight = float(latest["weight_123456"])
    hs300_weight = float(latest["weight_510300"])
    passed = (
        latest_signal == "511260"
        and treasury_weight > 0.27
        and alt_weight < FLOAT_TOL
        and hs300_weight < FLOAT_TOL
    )
    detail = (
        f"signal={latest_signal}, w511260={treasury_weight:.3f}, "
        f"w123456={alt_weight:.3f}, w510300={hs300_weight:.3f}"
    )
    return CheckResult(name="tail_overlay_custom_universe_budget", passed=passed, detail=detail)


def check_proxy_features_follow_custom_risk_universe() -> CheckResult:
    prices, _ = build_custom_universe_strategy_fixture()
    features = build_risk_proxy_features(prices, risk_codes=["123456"])
    latest_eq20 = float(features["eq20"].iloc[-1])
    ret20_123456 = float(prices["123456"].iloc[-1] / prices["123456"].shift(20).iloc[-1] - 1)
    ret20_510300 = float(prices["510300"].iloc[-1] / prices["510300"].shift(20).iloc[-1] - 1)
    passed = abs(latest_eq20 - ret20_123456) < FLOAT_TOL and abs(latest_eq20 - ret20_510300) > 1e-4
    detail = f"eq20={latest_eq20:.6f}, r123456={ret20_123456:.6f}, r510300={ret20_510300:.6f}"
    return CheckResult(name="proxy_features_custom_universe", passed=passed, detail=detail)


def check_persistent_overlay_uses_custom_universe_risk_budget() -> CheckResult:
    prices, selected = build_custom_universe_strategy_fixture()
    proxy = pd.DataFrame(
        {
            "market_amount_ratio_20_60": [0.8] * len(prices.index),
            "market_amount_ratio_5_20": [0.8] * len(prices.index),
            "market_breadth_proxy": [-0.2] * len(prices.index),
        },
        index=prices.index,
    )
    params = build_default_strategy_params(
        drop_codes=["513650"],
        risk_codes=["123456"],
        defensive_codes=["511260"],
    )
    result, _, target_weights = apply_persistent_overlay(
        prices=prices,
        selected=selected,
        proxy=proxy,
        params=params,
        treasury_code="511260",
        risk_cap=0.0,
        ratio_cut=0.9,
        breadth_cut=-0.1,
        enter_days=1,
        exit_days=1,
        fee_rate=0.0,
        slippage_rate=0.0,
        return_target_weights=True,
    )
    latest_signal = normalize_code(result.iloc[-1]["signal"])
    latest_target = target_weights.iloc[-1]
    treasury_weight = float(latest_target["511260"])
    alt_weight = float(latest_target["123456"])
    hs300_weight = float(latest_target["510300"])
    passed = (
        latest_signal == "511260"
        and treasury_weight > 0.27
        and alt_weight < FLOAT_TOL
        and hs300_weight < FLOAT_TOL
    )
    detail = (
        f"signal={latest_signal}, w511260={treasury_weight:.3f}, "
        f"w123456={alt_weight:.3f}, w510300={hs300_weight:.3f}"
    )
    return CheckResult(name="persistent_overlay_custom_universe_budget", passed=passed, detail=detail)


def check_overheat_cap_uses_custom_universe() -> CheckResult:
    index = pd.date_range("2025-01-01", periods=30, freq="B")
    prices = pd.DataFrame(
        {
            "510300": [100.0] * 25 + [101.0, 101.0, 101.0, 101.0, 101.0],
            "123456": [100.0] * 25 + [130.0, 135.0, 140.0, 145.0, 150.0],
            "511260": [100.0] * 30,
        },
        index=index,
    )
    selected = pd.DataFrame(
        [
            {"code": "510300", "theme": "沪深300", "name": "沪深300ETF"},
            {"code": "123456", "theme": "测试增强", "name": "测试增强ETF"},
            {"code": "511260", "theme": "十年国债", "name": "十年国债ETF"},
        ]
    )
    result, _ = run_threshold_dual_with_overheat_cap_strategy(
        prices=prices,
        selected=selected,
        lookback=25,
        fee_rate=0.0,
        slippage_rate=0.0,
        absolute_threshold=0.01,
        weak_trend_defensive_weight=0.8,
        overheat_drawdown_cut=-1.0,
        overheat_momentum_cut=0.25,
        overheat_max_exposure=0.4,
        risk_codes=["123456"],
        defensive_codes=["511260"],
    )
    latest_signal = normalize_code(result["signal"].iloc[-1])
    latest_target_exposure = float(result["target_exposure"].iloc[-1])
    passed = latest_signal == "123456" and abs(latest_target_exposure - 0.4) < FLOAT_TOL
    detail = f"signal={latest_signal}, target_exposure={latest_target_exposure:.3f}"
    return CheckResult(name="overheat_cap_custom_universe", passed=passed, detail=detail)


def check_threshold_dual_strategy_uses_custom_universe() -> CheckResult:
    index = pd.date_range("2025-01-01", periods=30, freq="B")
    prices = pd.DataFrame(
        {
            "510300": [100.0] * 25 + [130.0, 135.0, 140.0, 145.0, 150.0],
            "123456": [100.0] * 25 + [101.0, 101.0, 101.0, 101.0, 101.0],
            "511260": [100.0] * 30,
        },
        index=index,
    )
    selected = pd.DataFrame(
        [
            {"code": "510300", "theme": "沪深300", "name": "沪深300ETF"},
            {"code": "123456", "theme": "测试增强", "name": "测试增强ETF"},
            {"code": "511260", "theme": "十年国债", "name": "十年国债ETF"},
        ]
    )
    result, _ = run_threshold_dual_strategy(
        prices=prices,
        selected=selected,
        lookback=25,
        fee_rate=0.0,
        slippage_rate=0.0,
        absolute_threshold=0.01,
        weak_trend_defensive_weight=0.8,
        risk_codes=["123456"],
        defensive_codes=["511260"],
    )
    latest_signal = normalize_code(result["signal"].iloc[-1])
    latest_target_exposure = float(result["target_exposure"].iloc[-1])
    passed = latest_signal == "123456" and abs(latest_target_exposure - 1.0) < FLOAT_TOL
    detail = f"signal={latest_signal}, target_exposure={latest_target_exposure:.3f}"
    return CheckResult(name="threshold_dual_strategy_custom_universe", passed=passed, detail=detail)


def check_default_strategy_respects_proxy_kind() -> CheckResult:
    prices, selected = build_custom_universe_strategy_fixture()
    base_proxy = pd.DataFrame(
        {
            "market_amount_ratio_20_60": [0.8] * len(prices.index),
            "market_amount_ratio_5_20": [0.8] * len(prices.index),
            "market_breadth_proxy": [-0.2] * len(prices.index),
        },
        index=prices.index,
    )
    baseline_params = build_default_strategy_params(
        drop_codes=["513650"],
        risk_codes=["123456"],
        defensive_codes=["511260"],
    )
    hybrid_params = {**baseline_params, "proxy_kind": "hybrid_breadth_blend"}
    baseline_proxy_params = {**baseline_params, "proxy_kind": "baseline_sh_sz"}
    hybrid_result, _ = run_default_strategy_with_params(
        prices=prices,
        selected=selected,
        params=hybrid_params,
        fee_rate=0.0,
        slippage_rate=0.0,
        market_proxy=base_proxy,
        risk_cap=0.0,
        ratio_cut=0.9,
        breadth_cut=0.0,
        enter_days=1,
        exit_days=1,
    )
    baseline_result, _ = run_default_strategy_with_params(
        prices=prices,
        selected=selected,
        params=baseline_proxy_params,
        fee_rate=0.0,
        slippage_rate=0.0,
        market_proxy=base_proxy,
        risk_cap=0.0,
        ratio_cut=0.9,
        breadth_cut=0.0,
        enter_days=1,
        exit_days=1,
    )
    hybrid_trigger = bool(hybrid_result["stress_bond_trigger"].iloc[-1])
    baseline_trigger = bool(baseline_result["stress_bond_trigger"].iloc[-1])
    passed = (not hybrid_trigger) and baseline_trigger
    detail = f"hybrid_trigger={hybrid_trigger}, baseline_trigger={baseline_trigger}"
    return CheckResult(name="default_strategy_proxy_kind", passed=passed, detail=detail)


def check_default_strategy_rejects_unknown_proxy_kind() -> CheckResult:
    prices, selected = build_custom_universe_strategy_fixture()
    base_proxy = pd.DataFrame(
        {
            "market_amount_ratio_20_60": [1.0] * len(prices.index),
            "market_amount_ratio_5_20": [1.0] * len(prices.index),
            "market_breadth_proxy": [0.0] * len(prices.index),
        },
        index=prices.index,
    )
    params = build_default_strategy_params(
        drop_codes=["513650"],
        risk_codes=["123456"],
        defensive_codes=["511260"],
    )
    params["proxy_kind"] = "unknown_proxy_kind"
    try:
        run_default_strategy_with_params(
            prices=prices,
            selected=selected,
            params=params,
            fee_rate=0.0,
            slippage_rate=0.0,
            market_proxy=base_proxy,
        )
    except ValueError as exc:
        passed = "unsupported proxy_kind" in str(exc)
        detail = str(exc)
    else:
        passed = False
        detail = "did not raise"
    return CheckResult(name="default_strategy_invalid_proxy_kind", passed=passed, detail=detail)


def check_fine_tune_normalize_params_preserves_signal_context() -> CheckResult:
    raw_params = build_default_strategy_params(
        drop_codes=["513650"],
        risk_codes=["123456"],
        defensive_codes=["511260"],
    )
    normalized = normalize_fine_tune_params(raw_params)
    raw_confirm_lookback = int(raw_params.get("signal_confirmation_lookback", 0))
    raw_confirm_top_n = int(raw_params.get("signal_confirmation_top_n", 0))
    passed = (
        normalized["proxy_kind"] == raw_params["proxy_kind"]
        and normalized["signal_quality_method"] == raw_params["signal_quality_method"]
        and abs(float(normalized["signal_slope_penalty"]) - float(raw_params["signal_slope_penalty"])) < FLOAT_TOL
        and int(normalized["signal_confirmation_lookback"]) == raw_confirm_lookback
        and int(normalized["signal_confirmation_top_n"]) == raw_confirm_top_n
        and abs(float(normalized["signal_leader_margin"]) - float(raw_params["signal_leader_margin"])) < FLOAT_TOL
        and list(normalized["risk_codes"]) == list(raw_params["risk_codes"])
        and list(normalized["defensive_codes"]) == list(raw_params["defensive_codes"])
    )
    detail = (
        f"proxy={normalized['proxy_kind']}, signal={normalized['signal_quality_method']}, "
        f"slope={float(normalized['signal_slope_penalty']):.3f}, confirm={int(normalized['signal_confirmation_lookback'])}/{int(normalized['signal_confirmation_top_n'])}, leader={float(normalized['signal_leader_margin']):.3f}, "
        f"risk={','.join(normalized['risk_codes'])}, defensive={','.join(normalized['defensive_codes'])}"
    )
    return CheckResult(name="fine_tune_normalize_params_context", passed=passed, detail=detail)


def main() -> int:
    result, prices, trades, selected = load_outputs()
    checks: list[CheckResult] = []
    checks.extend(check_return_chain(result, prices))
    checks.append(check_trade_chain(result, trades, selected))
    checks.extend(check_session_boundaries())
    checks.append(check_stale_backtest_context_prefers_result())
    checks.append(check_intraday_price_context_without_realtime())
    checks.append(check_price_context_does_not_mix_raw_and_adjusted())
    checks.append(check_structural_break_back_adjustment_smooths_split_like_gaps())
    checks.append(check_partial_realtime_panel_keeps_missing_assets())
    checks.append(check_intraday_close_panel_drops_same_day_history())
    checks.append(check_partial_live_nav_does_not_refresh_peak())
    checks.append(check_snapshot_uses_confirmed_nav_context_before_live_estimate())
    checks.append(check_live_nav_coverage_uses_invested_exposure())
    checks.append(check_live_nav_cash_state_keeps_close_nav())
    checks.append(check_after_close_raw_price_does_not_recompute_nav())
    checks.append(check_snapshot_portfolio_falls_back_without_weight_columns())
    checks.append(check_confirmed_trade_uses_latest_actual_trade_date())
    checks.append(check_confirmed_holding_dates_do_not_fall_back_to_last_trade_date())
    checks.append(check_snapshot_exposure_and_extra_cap_timelines())
    checks.append(check_display_hidden_allocations_are_filtered())
    checks.append(check_display_equivalent_trade_details_are_suppressed())
    checks.append(check_display_equivalent_confirmed_trade_transition_suppressed())
    checks.append(check_legacy_state_is_upgradeable())
    checks.append(check_partial_delivery_progress_resumes_without_duplicate())
    checks.append(check_empty_channel_progress_is_not_persisted())
    checks.append(check_empty_channel_state_does_not_block_future_delivery())
    checks.append(check_intraday_history_filter_drops_same_day_bar())
    checks.append(check_indexed_cache_filter_drops_same_day_row())
    checks.append(check_drawdown_rebalance_count_uses_trade_days())
    checks.append(check_hs300_yearly_returns_use_prior_year_end())
    checks.append(check_preferred_payload_supports_strict_improvements())
    checks.append(check_optional_candidate_history_fetch_failure_is_nonfatal())
    checks.append(check_missing_required_history_raises_clear_error())
    checks.append(check_runtime_env_scrubs_user_site_packages())
    checks.append(check_package_import_for_active_research_modules())
    checks.append(check_matplotlib_import_uses_workspace_cache())
    checks.append(check_summary_payload_keeps_latest_portfolio())
    checks.append(check_preferred_payload_preserves_latest_portfolio())
    checks.append(check_notify_candidate_sort_uses_max_drawdown_tiebreaker())
    checks.append(check_preferred_payload_falls_back_from_invalid_notify_state())
    checks.append(check_required_payload_falls_back_to_field_complete_baseline())
    checks.append(check_invalid_previous_summary_ignored())
    checks.append(check_notify_best_candidate_suppresses_delivery_errors())
    checks.append(check_notify_ranked_incremental_candidate_suppresses_delivery_errors())
    checks.append(check_add_notify_cli_args_preserves_default_modes())
    checks.append(check_historical_nav_schema_consistent())
    checks.append(check_contribution_summary_uses_actual_weights())
    checks.append(check_contribution_summary_cost_drag_uses_currency_units())
    checks.append(check_formal_signal_matches_target_weight_leader(result))
    checks.append(check_formal_current_momentum_matches_signal_asset(result, prices))
    checks.append(check_analysis_contribution_summary_matches_core())
    checks.append(check_analysis_drawdown_report_matches_core())
    checks.append(check_trade_count_uses_trade_days())
    checks.append(check_latest_portfolio_summary_uses_weights())
    checks.append(check_guard_shares_use_weights())
    checks.append(check_drawdown_top_holdings_use_weights())
    checks.append(check_custom_universe_flows_into_signals())
    checks.append(check_goal_regime_mix_uses_custom_universe())
    checks.append(check_goal_regime_mix_parser_keeps_optional_thresholds())
    checks.append(check_goal_reduced_pool_notified_params_keep_optional_thresholds())
    checks.append(check_tail_overlay_uses_custom_universe_risk_budget())
    checks.append(check_proxy_features_follow_custom_risk_universe())
    checks.append(check_persistent_overlay_uses_custom_universe_risk_budget())
    checks.append(check_overheat_cap_uses_custom_universe())
    checks.append(check_threshold_dual_strategy_uses_custom_universe())
    checks.append(check_default_strategy_respects_proxy_kind())
    checks.append(check_default_strategy_rejects_unknown_proxy_kind())
    checks.append(check_fine_tune_normalize_params_preserves_signal_context())

    failed = [item for item in checks if not item.passed]
    for item in checks:
        status = "PASS" if item.passed else "FAIL"
        print(f"[{status}] {item.name}: {item.detail}")
    if failed:
        print(f"\nvalidation_failed={len(failed)}")
        return 1
    print("\nvalidation_failed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
