#!/usr/bin/env python3
"""archive 历史研究链共用的上游策略上下文读取 helper。"""

from __future__ import annotations

from typing import Callable

try:
    from .archive_data_loaders import build_archive_flat_strategy_paths
except ImportError:
    from archive_data_loaders import build_archive_flat_strategy_paths

try:
    from .overlay_candidate_catalog import lookup_overlay_asset
except ImportError:
    from overlay_candidate_catalog import lookup_overlay_asset

try:
    from .hs300_regime_common import summarize
except ImportError:
    from hs300_regime_common import summarize

try:
    from .tail_risk_overlay_common import load_market_proxy_best_context
except ImportError:
    from tail_risk_overlay_common import load_market_proxy_best_context

try:
    from .variant_compare_helpers import append_variant_result
except ImportError:
    from variant_compare_helpers import append_variant_result

try:
    from .overlay_strategy_helpers import apply_market_stress_treasury_overlay, apply_two_tier_market_stress_treasury_overlay
except ImportError:
    from overlay_strategy_helpers import apply_market_stress_treasury_overlay, apply_two_tier_market_stress_treasury_overlay

try:
    from ...run_backtest import load_fixed_etf_pool
except ImportError:
    from run_backtest import load_fixed_etf_pool

try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

try:
    from ...search_utils import load_required_strategy_payload
except ImportError:
    from search_utils import load_required_strategy_payload

SIMPLE_OVERLAY_DIR, SIMPLE_OVERLAY_BEST_PATH, SIMPLE_OVERLAY_NOTIFY_PATH = build_archive_flat_strategy_paths(
    "compare_simple_bond_overlay"
)
CASH_OVERLAY_DIR, CASH_OVERLAY_BEST_PATH, CASH_OVERLAY_NOTIFY_PATH = build_archive_flat_strategy_paths(
    "compare_cash_overlay_candidates"
)
STRESSBOND_NEARMISS_DIR, STRESSBOND_NEARMISS_BEST_PATH, STRESSBOND_NEARMISS_NOTIFY_PATH = build_archive_flat_strategy_paths(
    "compare_stressbond_nearmiss_repair"
)


def _build_overlay_context(payload: dict[str, object]) -> dict[str, object]:
    summary = dict(payload["summary"])
    return {
        "strategy": str(payload["strategy"]),
        "summary": summary,
        "description": str(payload["description"]),
        "mode": str(summary["mode"]) if summary.get("mode") is not None else "",
        "risk_cap": float(summary["risk_cap"]),
        "drawdown_cut": None if summary.get("drawdown_cut") is None else float(summary["drawdown_cut"]),
        "ratio_cut": None if summary.get("ratio_cut") is None else float(summary["ratio_cut"]),
        "breadth_cut": None if summary.get("breadth_cut") is None else float(summary["breadth_cut"]),
        "treasury_code": str(summary["treasury_code"]),
    }


def load_simple_overlay_context() -> dict[str, object]:
    payload = load_required_strategy_payload(
        SIMPLE_OVERLAY_NOTIFY_PATH,
        SIMPLE_OVERLAY_BEST_PATH,
        context_name="simple bond overlay strategy context",
        required_summary_fields=("risk_cap", "treasury_code"),
    )
    return _build_overlay_context(payload)


def load_cash_overlay_context() -> dict[str, object]:
    payload = load_required_strategy_payload(
        CASH_OVERLAY_NOTIFY_PATH,
        CASH_OVERLAY_BEST_PATH,
        context_name="cash overlay strategy context",
        required_summary_fields=("treasury_code", "mode", "risk_cap"),
    )
    return _build_overlay_context(payload)


def load_stressbond_nearmiss_context() -> dict[str, object]:
    payload = load_required_strategy_payload(
        STRESSBOND_NEARMISS_NOTIFY_PATH,
        STRESSBOND_NEARMISS_BEST_PATH,
        context_name="stressbond near-miss strategy context",
        required_summary_fields=("treasury_code", "risk_cap", "ratio_cut", "breadth_cut"),
    )
    summary = dict(payload["summary"])
    return {
        "strategy": str(payload["strategy"]),
        "summary": summary,
        "description": str(payload["description"]),
        "treasury_code": str(summary["treasury_code"]),
        "risk_cap": float(summary["risk_cap"]),
        "ratio_cut": float(summary["ratio_cut"]),
        "breadth_cut": float(summary["breadth_cut"]),
    }


def load_market_overlay_runtime(
    overlay_context_loader: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """统一装配覆盖层历史链共用的 market/overlay 上下文。"""
    market_context = load_market_proxy_best_context()
    overlay_context = overlay_context_loader()
    treasury_row = lookup_overlay_asset(str(overlay_context["treasury_code"]))
    params = dict(market_context["params"])
    params["volume_ratio_cut"] = float(market_context["volume_ratio_cut"])
    params["volume_short_ratio_cut"] = float(market_context["volume_short_ratio_cut"])
    params["volume_breadth_cut"] = float(market_context["volume_breadth_cut"])
    drop_codes = [str(code) for code in params["drop_codes"]]

    base_selected = load_fixed_etf_pool()
    base_selected = base_selected[~base_selected["code"].astype(str).isin(drop_codes)].reset_index(drop=True)
    return {
        "market_context": market_context,
        "overlay_context": overlay_context,
        "treasury_row": treasury_row,
        "params": params,
        "base_selected": base_selected,
    }


def build_overlay_summary_fields(
    overlay_context: dict[str, object],
    *,
    treasury_code: str | None = None,
    defensive_code: str | None = None,
    risk_cap: float | None = None,
    ratio_cut: float | None = None,
    breadth_cut: float | None = None,
    include_mode: bool = False,
    include_drawdown_cut: bool = False,
) -> dict[str, object]:
    """统一生成 overlay 历史链常见的摘要字段。"""
    fields: dict[str, object] = {
        "treasury_code": str(treasury_code or overlay_context["treasury_code"]),
        "risk_cap": float(overlay_context["risk_cap"] if risk_cap is None else risk_cap),
    }
    base_ratio_cut = overlay_context.get("ratio_cut")
    if ratio_cut is None:
        ratio_cut = None if base_ratio_cut is None else float(base_ratio_cut)
    if ratio_cut is not None:
        fields["ratio_cut"] = float(ratio_cut)
    base_breadth_cut = overlay_context.get("breadth_cut")
    if breadth_cut is None:
        breadth_cut = None if base_breadth_cut is None else float(base_breadth_cut)
    if breadth_cut is not None:
        fields["breadth_cut"] = float(breadth_cut)
    if defensive_code is not None:
        fields["defensive_code"] = str(defensive_code)
    if include_mode:
        fields["mode"] = str(overlay_context["mode"])
    if include_drawdown_cut:
        fields["drawdown_cut"] = overlay_context.get("drawdown_cut")
    return fields


def append_overlay_baseline(
    rows: list[dict[str, object]],
    descriptions: dict[str, str],
    nav_compare: "pd.DataFrame | None",
    *,
    baseline_result: "pd.DataFrame",
    baseline_trades: "pd.DataFrame",
    baseline_selected: "pd.DataFrame",
    overlay_context: dict[str, object],
) -> dict[str, object]:
    """统一追加覆盖层历史链的 baseline 行。"""
    baseline_result = apply_official_baseline_nav_anchor(baseline_result)
    return append_variant_result(
        rows,
        nav_compare,
        descriptions,
        strategy=str(overlay_context["strategy"]),
        result=baseline_result,
        summary=summarize(baseline_result, baseline_trades, baseline_selected),
        description=str(overlay_context["description"]),
    )


def run_market_stress_overlay(
    prices: "pd.DataFrame",
    selected: "pd.DataFrame",
    proxy: "pd.DataFrame",
    params: dict[str, float | list[str]],
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    fee_rate: float,
    slippage_rate: float,
    short_ratio_cut: float | None = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """运行覆盖层历史链共用的单层弱市切债变体。"""
    return apply_market_stress_treasury_overlay(
        prices,
        selected,
        proxy,
        params,
        treasury_code=treasury_code,
        risk_cap=risk_cap,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
        short_ratio_cut=short_ratio_cut,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )


def run_two_tier_market_stress_overlay(
    prices: "pd.DataFrame",
    selected: "pd.DataFrame",
    proxy: "pd.DataFrame",
    params: dict[str, float | list[str]],
    treasury_code: str,
    weak_ratio_cut: float,
    weak_breadth_cut: float,
    weak_cap: float,
    extreme_ratio_cut: float,
    extreme_breadth_cut: float,
    extreme_cap: float,
    fee_rate: float,
    slippage_rate: float,
    extreme_short_ratio_cut: float | None = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """运行覆盖层历史链共用的双层弱市切债变体。"""
    return apply_two_tier_market_stress_treasury_overlay(
        prices,
        selected,
        proxy,
        params,
        treasury_code=treasury_code,
        weak_ratio_cut=weak_ratio_cut,
        weak_breadth_cut=weak_breadth_cut,
        weak_cap=weak_cap,
        extreme_ratio_cut=extreme_ratio_cut,
        extreme_breadth_cut=extreme_breadth_cut,
        extreme_cap=extreme_cap,
        extreme_short_ratio_cut=extreme_short_ratio_cut,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
