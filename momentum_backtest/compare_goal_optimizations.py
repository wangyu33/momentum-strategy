#!/usr/bin/env python3
"""围绕当前正式基线搜索增量优化方案。"""

from __future__ import annotations

import argparse
import re
import sys
try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import akshare as ak
import pandas as pd

try:
    from .compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
    from .daily_monitor import DEFAULT_FEISHU_WEBHOOK
    from .monitor_delivery import send_webhook_message
    from .search_utils import (
        add_notify_cli_args,
        load_incremental_notify_candidates,
        load_preferred_strategy_payload,
        notify_best_candidate,
        raise_if_missing_required_histories,
        sort_notify_candidates,
        try_join_missing_candidate_histories,
        write_json_atomic,
    )
except ImportError:
    from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
    from daily_monitor import DEFAULT_FEISHU_WEBHOOK
    from monitor_delivery import send_webhook_message
    from search_utils import (
        add_notify_cli_args,
        load_incremental_notify_candidates,
        load_preferred_strategy_payload,
        notify_best_candidate,
        raise_if_missing_required_histories,
        sort_notify_candidates,
        try_join_missing_candidate_histories,
        write_json_atomic,
    )

from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
    DEFAULT_OVERHEAT_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    RESEARCH_OUTPUT_DIR,
    build_threshold_dual_signal,
    ensure_output_dirs,
    fetch_histories,
    load_default_strategy_backtest_pool,
    resolve_strategy_universe,
    run_default_strategy,
    write_dataframe_csv_atomic,
)


GOAL_OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "goal_optimizations"
SUMMARY_PATH = GOAL_OUTPUT_DIR / "goal_optimization_summary.csv"
COMPARE_PATH = GOAL_OUTPUT_DIR / "goal_optimization_nav_compare.csv"
BEST_PATH = GOAL_OUTPUT_DIR / "goal_optimization_best.json"
NOTIFY_STATE_PATH = GOAL_OUTPUT_DIR / "goal_optimization_notify_state.json"
MARKET_VOLUME_CACHE_PATH = GOAL_OUTPUT_DIR / "market_volume_proxy.csv"
MARKET_VOLUME_ERROR_PATH = GOAL_OUTPUT_DIR / "market_volume_proxy_error.json"
LEGACY_MARKET_VOLUME_ERROR_PATH = GOAL_OUTPUT_DIR / "market_volume_proxy_error.txt"
METRIC_TOLERANCE = 1e-12
DEFAULT_REGIME_MIX_REGIME_MOMENTUM_CUT = 0.06
DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING = 0.18
DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT = 0.32
DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER = 0.0
DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN = 0.04


def extract_drop_codes_from_strategy_name(strategy_name: str) -> list[str]:
    return re.findall(r"rm(\d{6})", strategy_name)


def parse_regime_mix_strategy_name(strategy_name: str) -> dict[str, float | list[str]] | None:
    if "regime_mix" not in strategy_name or "volume_guard" not in strategy_name:
        return None

    def extract_pct(tag: str) -> float | None:
        match = re.search(rf"{tag}(\d{{2}})(?!\d)", strategy_name)
        if match is None:
            return None
        return int(match.group(1)) / 100

    aggressive_core_weight = extract_pct("ag")
    conservative_core_weight = extract_pct("co")
    volume_ratio_cut = extract_pct("vr")
    volume_short_ratio_cut = extract_pct("vs")
    volume_guard_cap = extract_pct("vg")
    overheat_cap = extract_pct("cap")
    overheat_hi_cap = extract_pct("hi")
    regime_momentum_cut = extract_pct("rm")
    volume_guard_momentum_ceiling = extract_pct("vm")
    overheat_momentum_cut = extract_pct("oh")
    overheat_high_momentum_cut = extract_pct("ohh")
    volume_guard_relief_buffer = extract_pct("gb")
    volume_guard_soft_span = extract_pct("gs")

    vb_match = re.search(r"vb(\d{2})", strategy_name)
    dd_match = re.search(r"dd(\d{2})", strategy_name)
    if None in [
        aggressive_core_weight,
        conservative_core_weight,
        volume_ratio_cut,
        volume_short_ratio_cut,
        volume_guard_cap,
        overheat_cap,
        overheat_hi_cap,
    ]:
        return None
    if vb_match is None or dd_match is None:
        return None

    return {
        "drop_codes": extract_drop_codes_from_strategy_name(strategy_name),
        "aggressive_core_weight": aggressive_core_weight,
        "conservative_core_weight": conservative_core_weight,
        "regime_momentum_cut": (
            regime_momentum_cut if regime_momentum_cut is not None else DEFAULT_REGIME_MIX_REGIME_MOMENTUM_CUT
        ),
        "volume_ratio_cut": volume_ratio_cut,
        "volume_short_ratio_cut": volume_short_ratio_cut,
        "volume_breadth_cut": int(vb_match.group(1)) / 100 - 0.02,
        "volume_guard_cap": volume_guard_cap,
        "volume_guard_momentum_ceiling": (
            volume_guard_momentum_ceiling
            if volume_guard_momentum_ceiling is not None
            else DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING
        ),
        "volume_guard_relief_buffer": (
            volume_guard_relief_buffer
            if volume_guard_relief_buffer is not None
            else DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER
        ),
        "volume_guard_soft_span": (
            volume_guard_soft_span if volume_guard_soft_span is not None else DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN
        ),
        "overheat_drawdown_cut": -(int(dd_match.group(1)) / 100),
        "overheat_momentum_cut": (
            overheat_momentum_cut if overheat_momentum_cut is not None else DEFAULT_OVERHEAT_MOMENTUM_CUT
        ),
        "overheat_max_exposure": overheat_cap,
        "overheat_high_momentum_cut": (
            overheat_high_momentum_cut
            if overheat_high_momentum_cut is not None
            else DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT
        ),
        "overheat_high_max_exposure": overheat_hi_cap,
    }


def build_reduced_pool_notified_base_params(notified_context: dict[str, object] | None) -> dict[str, float]:
    """从已通知策略恢复 remove_511580_rm513650 家族继续搜参所需的完整基线参数。"""
    base_rm_pool_params = {
        "aggressive_core_weight": 0.28,
        "conservative_core_weight": 0.27,
        "regime_momentum_cut": 0.06,
        "volume_ratio_cut": 0.91,
        "volume_short_ratio_cut": 0.90,
        "volume_breadth_cut": -0.01,
        "volume_guard_cap": 0.48,
        "volume_guard_momentum_ceiling": 0.16,
        "overheat_drawdown_cut": -0.02,
        "overheat_momentum_cut": 0.25,
        "overheat_cap": 0.20,
        "overheat_high_momentum_cut": 0.32,
        "overheat_hi_cap": 0.12,
    }
    if notified_context is None:
        return base_rm_pool_params

    params = notified_context["params"]
    drop_codes = params.get("drop_codes", [])
    if drop_codes != ["511580", "513650"]:
        return base_rm_pool_params

    return {
        "aggressive_core_weight": float(params["aggressive_core_weight"]),
        "conservative_core_weight": float(params["conservative_core_weight"]),
        "regime_momentum_cut": float(params["regime_momentum_cut"]),
        "volume_ratio_cut": float(params["volume_ratio_cut"]),
        "volume_short_ratio_cut": float(params["volume_short_ratio_cut"]),
        "volume_breadth_cut": float(params["volume_breadth_cut"]),
        "volume_guard_cap": float(params["volume_guard_cap"]),
        "volume_guard_momentum_ceiling": float(params["volume_guard_momentum_ceiling"]),
        "overheat_drawdown_cut": float(params["overheat_drawdown_cut"]),
        "overheat_momentum_cut": float(params["overheat_momentum_cut"]),
        "overheat_cap": float(params["overheat_max_exposure"]),
        "overheat_high_momentum_cut": float(params["overheat_high_momentum_cut"]),
        "overheat_hi_cap": float(params["overheat_high_max_exposure"]),
    }


def load_notified_regime_mix_context() -> dict[str, object] | None:
    payload = load_preferred_strategy_payload(NOTIFY_STATE_PATH, BEST_PATH)
    if payload is None:
        return None
    strategy_name = str(payload.get("strategy", ""))
    params = parse_regime_mix_strategy_name(strategy_name)
    if params is None:
        return None
    return {
        "strategy": strategy_name,
        "summary": payload.get("summary", {}),
        "description": payload.get("description", ""),
        "params": params,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="围绕当前正式基线搜索增量优化方案。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    add_notify_cli_args(parser)
    parser.add_argument("--refresh", action="store_true", help="重新抓取价格，而不是复用 core 缓存。")
    return parser.parse_args()


def fetch_market_volume_proxy(years: int) -> pd.DataFrame:
    end_date = pd.Timestamp.today().normalize()
    start_date = (end_date - pd.DateOffset(years=years + 1)).strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")

    def load_one(symbol: str, prefix: str) -> pd.DataFrame:
        errors: list[str] = []
        loaders = [
            ("tx", lambda: ak.stock_zh_index_daily_tx(symbol=symbol, start_date=start_date, end_date=end_date_str)),
            ("em", lambda: ak.stock_zh_index_daily_em(symbol=symbol, start_date=start_date, end_date=end_date_str)),
        ]
        for source_name, loader in loaders:
            try:
                df = loader()
                if df is None or df.empty:
                    errors.append(f"{source_name}: empty")
                    continue
                df = df.rename(columns={"close": f"{prefix}_close", "amount": f"{prefix}_amount"})[
                    ["date", f"{prefix}_close", f"{prefix}_amount"]
                ].copy()
                df["date"] = pd.to_datetime(df["date"])
                df[f"{prefix}_close"] = pd.to_numeric(df[f"{prefix}_close"], errors="coerce")
                df[f"{prefix}_amount"] = pd.to_numeric(df[f"{prefix}_amount"], errors="coerce")
                return df
            except Exception as exc:
                errors.append(f"{source_name}: {exc}")
        raise RuntimeError(f"failed to load {symbol}: {' | '.join(errors)}")

    sh = load_one("sh000001", "sh")
    sz = load_one("sz399106", "sz")

    proxy = sh.merge(sz, on="date", how="inner").dropna().sort_values("date")
    proxy["market_amount"] = proxy["sh_amount"] + proxy["sz_amount"]
    proxy["market_amount_ma20"] = proxy["market_amount"].rolling(20).mean()
    proxy["market_amount_ma60"] = proxy["market_amount"].rolling(60).mean()
    proxy["market_amount_ratio_20_60"] = proxy["market_amount_ma20"] / proxy["market_amount_ma60"]
    proxy["market_amount_ratio_5_20"] = proxy["market_amount"].rolling(5).mean() / proxy["market_amount_ma20"]
    proxy["market_breadth_proxy"] = (
        proxy["sh_close"] / proxy["sh_close"].shift(20) - 1
        + proxy["sz_close"] / proxy["sz_close"].shift(20) - 1
    ) / 2
    return proxy.set_index("date")


def load_market_volume_proxy(years: int, refresh: bool) -> pd.DataFrame:
    GOAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    required_cols = {
        "date",
        "market_amount",
        "market_amount_ma20",
        "market_amount_ma60",
        "market_amount_ratio_20_60",
        "market_amount_ratio_5_20",
        "market_breadth_proxy",
    }
    end_date = pd.Timestamp.today().normalize()
    required_start = end_date - pd.DateOffset(years=years + 1)
    stale_cutoff = end_date - pd.Timedelta(days=7)
    if not refresh and MARKET_VOLUME_CACHE_PATH.exists():
        try:
            proxy = pd.read_csv(MARKET_VOLUME_CACHE_PATH, parse_dates=["date"])
            if not required_cols.issubset(proxy.columns):
                raise ValueError("market volume cache missing required columns")
            if proxy.empty:
                raise ValueError("market volume cache is empty")
            if pd.Timestamp(proxy["date"].min()).normalize() > required_start.normalize():
                raise ValueError("market volume cache coverage is too short")
            if pd.Timestamp(proxy["date"].max()).normalize() < stale_cutoff.normalize():
                raise ValueError("market volume cache is stale")
            return proxy.set_index("date")
        except Exception:
            pass

    proxy = fetch_market_volume_proxy(years=years)
    proxy_to_save = proxy.reset_index()
    write_dataframe_csv_atomic(proxy_to_save, MARKET_VOLUME_CACHE_PATH, index=False)
    if MARKET_VOLUME_ERROR_PATH.exists():
        MARKET_VOLUME_ERROR_PATH.unlink()
    if LEGACY_MARKET_VOLUME_ERROR_PATH.exists():
        LEGACY_MARKET_VOLUME_ERROR_PATH.unlink()
    return proxy


def build_parametrized_hs300_trend_filter(
    prices: pd.DataFrame,
    mom60_cut: float,
    mom120_cut: float,
    ma_window: int,
) -> pd.Series:
    hs300_code = "510300"
    close = prices[hs300_code]
    mom60 = close / close.shift(60) - 1
    mom120 = close / close.shift(120) - 1
    ma = close.rolling(ma_window).mean()
    return ((mom60 > mom60_cut) & (mom120 > mom120_cut) & (close > ma)).rename("hs300_trend_filter")


def build_hs300_strength_core_weight(
    prices: pd.DataFrame,
    base_core_weight: float,
    mom60_cut: float,
    mom120_cut: float,
    ma_window: int,
    medium_multiplier: float = 0.5,
    strong_multiplier: float = 1.0,
    extreme_multiplier: float = 1.25,
) -> pd.Series:
    hs300_code = "510300"
    close = prices[hs300_code]
    mom60 = close / close.shift(60) - 1
    mom120 = close / close.shift(120) - 1
    ma = close.rolling(ma_window).mean()
    above_ma = close > ma

    medium_mask = (mom60 > mom60_cut) & (mom120 > mom120_cut) & above_ma
    strong_mask = (mom60 > mom60_cut + 0.02) & (mom120 > mom120_cut + 0.04) & above_ma
    extreme_mask = (mom60 > mom60_cut + 0.05) & (mom120 > mom120_cut + 0.08) & above_ma

    weight = pd.Series(0.0, index=prices.index, dtype="float64")
    weight.loc[medium_mask] = base_core_weight * medium_multiplier
    weight.loc[strong_mask] = base_core_weight * strong_multiplier
    weight.loc[extreme_mask] = base_core_weight * extreme_multiplier
    return weight.clip(lower=0.0, upper=0.60).rename("dynamic_core_weight")


def build_hs300_relative_lead_filter(
    prices: pd.DataFrame,
    candidate_signal: pd.Series,
    lookback: int,
    hs300_mom_cut: float,
    lead_margin_cut: float,
    ma_window: int,
) -> pd.Series:
    hs300_code = "510300"
    hs300_close = prices[hs300_code]
    hs300_mom = hs300_close / hs300_close.shift(lookback) - 1
    hs300_ma = hs300_close.rolling(ma_window).mean()
    signal_mom = pd.Series(index=prices.index, dtype="float64")
    for dt_idx in prices.index:
        signal_code = candidate_signal.loc[dt_idx]
        if pd.isna(signal_code) or signal_code not in prices.columns:
            signal_mom.loc[dt_idx] = float("nan")
            continue
        value = prices.loc[dt_idx, signal_code] / prices.loc[prices.index[prices.index.get_loc(dt_idx) - lookback], signal_code] - 1 if prices.index.get_loc(dt_idx) >= lookback else float("nan")
        signal_mom.loc[dt_idx] = value
    return (
        (hs300_mom > hs300_mom_cut)
        & ((hs300_mom - signal_mom.fillna(-999)) > lead_margin_cut)
        & (hs300_close > hs300_ma)
    ).rename("hs300_relative_lead_filter")


def build_dynamic_core_target_weights(
    prices: pd.DataFrame,
    lookback: int,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    core_weight: float,
    hs300_mom60_cut: float,
    hs300_mom120_cut: float,
    hs300_ma_window: int,
    signal_quality_method: str = "raw",
    slope_penalty: float = 0.0,
    leader_margin: float = 0.0,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        signal_quality_method=signal_quality_method,
        slope_penalty=slope_penalty,
        leader_margin=leader_margin,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    trend_filter = build_parametrized_hs300_trend_filter(
        prices,
        mom60_cut=hs300_mom60_cut,
        mom120_cut=hs300_mom120_cut,
        ma_window=hs300_ma_window,
    )
    hs300_code = "510300"
    hs300_mom60 = prices[hs300_code] / prices[hs300_code].shift(60) - 1
    dynamic_core_weight = trend_filter.astype(float) * core_weight
    satellite_scale = 1.0 - dynamic_core_weight

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_weights[hs300_code] += dynamic_core_weight
    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_scale.loc[code_mask] * target_exposure.loc[code_mask]

    blended_momentum = (satellite_scale * current_momentum + dynamic_core_weight * hs300_mom60).rename("current_momentum")
    return target_weights, blended_momentum, signal, target_exposure, trend_filter


def resolve_goal_strategy_universe(
    prices: pd.DataFrame,
    *,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """解析研究策略有效信号池，并把 HS300 核心仓纳入风险预算统计。"""
    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    risk_budget_codes = list(active_risk_codes)
    if "510300" in prices.columns and "510300" not in risk_budget_codes:
        risk_budget_codes.append("510300")
    return active_risk_codes, active_defensive_codes, risk_budget_codes


def run_dynamic_core_satellite_overheat_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    core_weight: float,
    overheat_drawdown_cut: float,
    overheat_momentum_cut: float,
    overheat_max_exposure: float,
    overheat_high_momentum_cut: float,
    overheat_high_max_exposure: float,
    hs300_mom60_cut: float = 0.05,
    hs300_mom120_cut: float = 0.10,
    hs300_ma_window: int = 120,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_risk_codes, active_defensive_codes, risk_budget_codes = resolve_goal_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    trend_filter = build_parametrized_hs300_trend_filter(
        prices,
        mom60_cut=hs300_mom60_cut,
        mom120_cut=hs300_mom120_cut,
        ma_window=hs300_ma_window,
    )
    hs300_code = "510300"
    hs300_mom60 = prices[hs300_code] / prices[hs300_code].shift(60) - 1
    dynamic_core_weight = trend_filter.astype(float) * core_weight
    satellite_scale = 1.0 - dynamic_core_weight

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_weights[hs300_code] += dynamic_core_weight
    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_scale.loc[code_mask] * target_exposure.loc[code_mask]

    blended_momentum = (satellite_scale * current_momentum + dynamic_core_weight * hs300_mom60).rename("current_momentum")

    base_result, _ = run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )

    row_risk_weight = target_weights[risk_budget_codes].sum(axis=1)
    reduce_mask = (
        (row_risk_weight > overheat_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (blended_momentum >= overheat_momentum_cut)
    )

    capped_target_weights = target_weights.copy()
    if reduce_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[reduce_mask] = overheat_max_exposure / row_risk_weight.loc[reduce_mask]
        capped_target_weights.loc[reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            reduce_mask, risk_budget_codes
        ].mul(scale.loc[reduce_mask], axis=0)

    high_reduce_mask = (
        (capped_target_weights[risk_budget_codes].sum(axis=1) > overheat_high_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (blended_momentum >= overheat_high_momentum_cut)
    )
    if high_reduce_mask.any():
        high_scale = pd.Series(1.0, index=prices.index, dtype="float64")
        high_scale.loc[high_reduce_mask] = (
            overheat_high_max_exposure / capped_target_weights[risk_budget_codes].sum(axis=1).loc[high_reduce_mask]
        )
        capped_target_weights.loc[high_reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            high_reduce_mask, risk_budget_codes
        ].mul(high_scale.loc[high_reduce_mask], axis=0)

    return run_target_weights_strategy(
        prices,
        selected,
        capped_target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )


def run_dynamic_core_bear_guard_overheat_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    core_weight: float,
    bear_guard_momentum_cut: float,
    bear_guard_max_exposure: float,
    overheat_drawdown_cut: float,
    overheat_momentum_cut: float,
    overheat_max_exposure: float,
    overheat_high_momentum_cut: float,
    overheat_high_max_exposure: float,
    hs300_mom60_cut: float = 0.05,
    hs300_mom120_cut: float = 0.10,
    hs300_ma_window: int = 120,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_risk_codes, active_defensive_codes, risk_budget_codes = resolve_goal_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    trend_filter = build_parametrized_hs300_trend_filter(
        prices,
        mom60_cut=hs300_mom60_cut,
        mom120_cut=hs300_mom120_cut,
        ma_window=hs300_ma_window,
    )
    hs300_code = "510300"
    hs300_mom60 = prices[hs300_code] / prices[hs300_code].shift(60) - 1
    dynamic_core_weight = trend_filter.astype(float) * core_weight
    satellite_scale = 1.0 - dynamic_core_weight

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_weights[hs300_code] += dynamic_core_weight
    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_scale.loc[code_mask] * target_exposure.loc[code_mask]

    row_risk_weight = target_weights[risk_budget_codes].sum(axis=1)
    bear_guard_mask = (
        (~trend_filter)
        & (row_risk_weight > bear_guard_max_exposure)
        & (current_momentum <= bear_guard_momentum_cut)
    )
    guarded_target_weights = target_weights.copy()
    if bear_guard_mask.any():
        guard_scale = pd.Series(1.0, index=prices.index, dtype="float64")
        guard_scale.loc[bear_guard_mask] = bear_guard_max_exposure / row_risk_weight.loc[bear_guard_mask]
        guarded_target_weights.loc[bear_guard_mask, risk_budget_codes] = guarded_target_weights.loc[
            bear_guard_mask, risk_budget_codes
        ].mul(guard_scale.loc[bear_guard_mask], axis=0)

    blended_momentum = (
        guarded_target_weights[risk_budget_codes].sum(axis=1) * current_momentum
        + guarded_target_weights[hs300_code] * hs300_mom60
    ).rename("current_momentum")

    base_result, _ = run_target_weights_strategy(
        prices,
        selected,
        guarded_target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )

    guarded_risk_weight = guarded_target_weights[risk_budget_codes].sum(axis=1)
    reduce_mask = (
        (guarded_risk_weight > overheat_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (blended_momentum >= overheat_momentum_cut)
    )

    capped_target_weights = guarded_target_weights.copy()
    if reduce_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[reduce_mask] = overheat_max_exposure / guarded_risk_weight.loc[reduce_mask]
        capped_target_weights.loc[reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            reduce_mask, risk_budget_codes
        ].mul(scale.loc[reduce_mask], axis=0)

    high_reduce_mask = (
        (capped_target_weights[risk_budget_codes].sum(axis=1) > overheat_high_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (blended_momentum >= overheat_high_momentum_cut)
    )
    if high_reduce_mask.any():
        high_scale = pd.Series(1.0, index=prices.index, dtype="float64")
        high_scale.loc[high_reduce_mask] = (
            overheat_high_max_exposure / capped_target_weights[risk_budget_codes].sum(axis=1).loc[high_reduce_mask]
        )
        capped_target_weights.loc[high_reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            high_reduce_mask, risk_budget_codes
        ].mul(high_scale.loc[high_reduce_mask], axis=0)

    return run_target_weights_strategy(
        prices,
        selected,
        capped_target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )


def run_dynamic_core_short_brake_overheat_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    core_weight: float,
    short_lb: int,
    short_momentum_cut: float,
    brake_momentum_ceiling: float,
    brake_defensive_weight: float,
    overheat_drawdown_cut: float,
    overheat_momentum_cut: float,
    overheat_max_exposure: float,
    overheat_high_momentum_cut: float,
    overheat_high_max_exposure: float,
    hs300_mom60_cut: float = 0.05,
    hs300_mom120_cut: float = 0.10,
    hs300_ma_window: int = 120,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_risk_codes, active_defensive_codes, risk_budget_codes = resolve_goal_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    trend_filter = build_parametrized_hs300_trend_filter(
        prices,
        mom60_cut=hs300_mom60_cut,
        mom120_cut=hs300_mom120_cut,
        ma_window=hs300_ma_window,
    )
    hs300_code = "510300"
    hs300_mom60 = prices[hs300_code] / prices[hs300_code].shift(60) - 1
    short_momentum = prices / prices.shift(short_lb) - 1
    long_momentum = prices / prices.shift(lookback) - 1
    defensive_mom = long_momentum[active_defensive_codes]
    defensive_winner = defensive_mom.idxmax(axis=1, skipna=True)
    defensive_best = defensive_mom.max(axis=1, skipna=True)

    braked_signal = signal.copy()
    braked_exposure = target_exposure.copy()
    braked_momentum = current_momentum.copy()

    for dt_idx in prices.index:
        asset = signal.loc[dt_idx]
        if pd.isna(asset) or asset not in active_risk_codes:
            continue
        if bool(trend_filter.loc[dt_idx]):
            continue
        raw_mom = current_momentum.loc[dt_idx]
        if pd.isna(raw_mom) or float(raw_mom) > brake_momentum_ceiling:
            continue
        asset_short_mom = short_momentum.loc[dt_idx, asset]
        if pd.isna(asset_short_mom) or float(asset_short_mom) > short_momentum_cut:
            continue

        def_asset = defensive_winner.loc[dt_idx]
        def_score = defensive_best.loc[dt_idx]
        braked_signal.loc[dt_idx] = def_asset if pd.notna(def_asset) else pd.NA
        braked_exposure.loc[dt_idx] = (
            brake_defensive_weight if pd.notna(def_score) and float(def_score) > 0 else 0.0
        )
        braked_momentum.loc[dt_idx] = float(def_score) if pd.notna(def_score) else float("nan")

    dynamic_core_weight = trend_filter.astype(float) * core_weight
    satellite_scale = 1.0 - dynamic_core_weight

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_weights[hs300_code] += dynamic_core_weight
    for code in prices.columns:
        code_mask = braked_signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_scale.loc[code_mask] * braked_exposure.loc[code_mask]

    blended_momentum = (satellite_scale * braked_momentum + dynamic_core_weight * hs300_mom60).rename("current_momentum")
    base_result, _ = run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )

    row_risk_weight = target_weights[risk_budget_codes].sum(axis=1)
    reduce_mask = (
        (row_risk_weight > overheat_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (blended_momentum >= overheat_momentum_cut)
    )

    capped_target_weights = target_weights.copy()
    if reduce_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[reduce_mask] = overheat_max_exposure / row_risk_weight.loc[reduce_mask]
        capped_target_weights.loc[reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            reduce_mask, risk_budget_codes
        ].mul(scale.loc[reduce_mask], axis=0)

    high_reduce_mask = (
        (capped_target_weights[risk_budget_codes].sum(axis=1) > overheat_high_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (blended_momentum >= overheat_high_momentum_cut)
    )
    if high_reduce_mask.any():
        high_scale = pd.Series(1.0, index=prices.index, dtype="float64")
        high_scale.loc[high_reduce_mask] = (
            overheat_high_max_exposure / capped_target_weights[risk_budget_codes].sum(axis=1).loc[high_reduce_mask]
        )
        capped_target_weights.loc[high_reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            high_reduce_mask, risk_budget_codes
        ].mul(high_scale.loc[high_reduce_mask], axis=0)

    return run_target_weights_strategy(
        prices,
        selected,
        capped_target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )


def run_regime_mix_core_overheat_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    aggressive_core_weight: float,
    conservative_core_weight: float,
    regime_momentum_cut: float,
    overheat_drawdown_cut: float,
    overheat_momentum_cut: float,
    overheat_max_exposure: float,
    overheat_high_momentum_cut: float,
    overheat_high_max_exposure: float,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_risk_codes, active_defensive_codes, risk_budget_codes = resolve_goal_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    aggressive_weights, aggressive_momentum, _, _, aggressive_trend = build_dynamic_core_target_weights(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        core_weight=aggressive_core_weight,
        hs300_mom60_cut=0.05,
        hs300_mom120_cut=0.10,
        hs300_ma_window=90,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    conservative_weights, conservative_momentum, _, _, _ = build_dynamic_core_target_weights(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        core_weight=conservative_core_weight,
        hs300_mom60_cut=0.04,
        hs300_mom120_cut=0.10,
        hs300_ma_window=120,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )

    aggressive_mask = (aggressive_trend) & (aggressive_momentum >= regime_momentum_cut)
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    mixed_momentum = conservative_momentum.copy()
    mixed_momentum.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    mixed_momentum = mixed_momentum.rename("current_momentum")

    base_result, _ = run_target_weights_strategy(
        prices,
        selected,
        mixed_target_weights,
        mixed_momentum,
        fee_rate,
        slippage_rate,
    )

    row_risk_weight = mixed_target_weights[risk_budget_codes].sum(axis=1)
    reduce_mask = (
        (row_risk_weight > overheat_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (mixed_momentum >= overheat_momentum_cut)
    )
    capped_target_weights = mixed_target_weights.copy()
    if reduce_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[reduce_mask] = overheat_max_exposure / row_risk_weight.loc[reduce_mask]
        capped_target_weights.loc[reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            reduce_mask, risk_budget_codes
        ].mul(scale.loc[reduce_mask], axis=0)

    high_reduce_mask = (
        (capped_target_weights[risk_budget_codes].sum(axis=1) > overheat_high_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (mixed_momentum >= overheat_high_momentum_cut)
    )
    if high_reduce_mask.any():
        high_scale = pd.Series(1.0, index=prices.index, dtype="float64")
        high_scale.loc[high_reduce_mask] = (
            overheat_high_max_exposure / capped_target_weights[risk_budget_codes].sum(axis=1).loc[high_reduce_mask]
        )
        capped_target_weights.loc[high_reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            high_reduce_mask, risk_budget_codes
        ].mul(high_scale.loc[high_reduce_mask], axis=0)

    return run_target_weights_strategy(
        prices,
        selected,
        capped_target_weights,
        mixed_momentum,
        fee_rate,
        slippage_rate,
    )


def run_regime_mix_core_volume_guard_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    market_volume_proxy: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    aggressive_core_weight: float,
    conservative_core_weight: float,
    regime_momentum_cut: float,
    volume_ratio_cut: float,
    volume_short_ratio_cut: float,
    volume_breadth_cut: float,
    volume_guard_cap: float,
    volume_guard_momentum_ceiling: float,
    overheat_drawdown_cut: float,
    overheat_momentum_cut: float,
    overheat_max_exposure: float,
    overheat_high_momentum_cut: float,
    overheat_high_max_exposure: float,
    volume_guard_relief_buffer: float = DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER,
    volume_guard_soft_span: float = DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_risk_codes, active_defensive_codes, risk_budget_codes = resolve_goal_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    aggressive_weights, aggressive_momentum, _, _, aggressive_trend = build_dynamic_core_target_weights(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        core_weight=aggressive_core_weight,
        hs300_mom60_cut=0.05,
        hs300_mom120_cut=0.10,
        hs300_ma_window=90,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    conservative_weights, conservative_momentum, _, _, _ = build_dynamic_core_target_weights(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        core_weight=conservative_core_weight,
        hs300_mom60_cut=0.04,
        hs300_mom120_cut=0.10,
        hs300_ma_window=120,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    aggressive_mask = aggressive_trend & (aggressive_momentum >= regime_momentum_cut)
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    mixed_momentum = conservative_momentum.copy()
    mixed_momentum.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    mixed_momentum = mixed_momentum.rename("current_momentum")

    proxy = market_volume_proxy.reindex(prices.index).ffill()
    if not proxy.empty:
        row_risk_weight = mixed_target_weights[risk_budget_codes].sum(axis=1)
        volume_weak_mask = (
            (proxy["market_amount_ratio_20_60"] < volume_ratio_cut)
            & (proxy["market_amount_ratio_5_20"] < volume_short_ratio_cut)
            & (proxy["market_breadth_proxy"] < volume_breadth_cut)
            & (mixed_momentum <= volume_guard_momentum_ceiling)
        ).fillna(False)

        if volume_guard_relief_buffer > 0:
            soft_span = max(volume_guard_soft_span, 1e-6)
            weakness_20_60 = ((volume_ratio_cut - proxy["market_amount_ratio_20_60"]) / soft_span).clip(0.0, 1.0)
            weakness_5_20 = ((volume_short_ratio_cut - proxy["market_amount_ratio_5_20"]) / soft_span).clip(0.0, 1.0)
            weakness_breadth = ((volume_breadth_cut - proxy["market_breadth_proxy"]) / soft_span).clip(0.0, 1.0)
            momentum_divisor = max(abs(volume_guard_momentum_ceiling), 1e-6)
            weakness_momentum = ((volume_guard_momentum_ceiling - mixed_momentum) / momentum_divisor).clip(0.0, 1.0)
            guard_strength = ((weakness_20_60 + weakness_5_20 + weakness_breadth + weakness_momentum) / 4.0).clip(
                0.0, 1.0
            )
            dynamic_cap = (
                volume_guard_cap + volume_guard_relief_buffer * (1.0 - guard_strength)
            ).clip(lower=volume_guard_cap, upper=1.0)
        else:
            dynamic_cap = pd.Series(volume_guard_cap, index=prices.index, dtype="float64")

        guard_mask = volume_weak_mask & (row_risk_weight > dynamic_cap)
        if guard_mask.any():
            scale = pd.Series(1.0, index=prices.index, dtype="float64")
            scale.loc[guard_mask] = dynamic_cap.loc[guard_mask] / row_risk_weight.loc[guard_mask]
            mixed_target_weights.loc[guard_mask, risk_budget_codes] = mixed_target_weights.loc[
                guard_mask, risk_budget_codes
            ].mul(scale.loc[guard_mask], axis=0)

    base_result, _ = run_target_weights_strategy(
        prices,
        selected,
        mixed_target_weights,
        mixed_momentum,
        fee_rate,
        slippage_rate,
    )
    row_risk_weight = mixed_target_weights[risk_budget_codes].sum(axis=1)
    reduce_mask = (
        (row_risk_weight > overheat_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (mixed_momentum >= overheat_momentum_cut)
    )
    capped_target_weights = mixed_target_weights.copy()
    if reduce_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[reduce_mask] = overheat_max_exposure / row_risk_weight.loc[reduce_mask]
        capped_target_weights.loc[reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            reduce_mask, risk_budget_codes
        ].mul(scale.loc[reduce_mask], axis=0)

    high_reduce_mask = (
        (capped_target_weights[risk_budget_codes].sum(axis=1) > overheat_high_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (mixed_momentum >= overheat_high_momentum_cut)
    )
    if high_reduce_mask.any():
        high_scale = pd.Series(1.0, index=prices.index, dtype="float64")
        high_scale.loc[high_reduce_mask] = (
            overheat_high_max_exposure / capped_target_weights[risk_budget_codes].sum(axis=1).loc[high_reduce_mask]
        )
        capped_target_weights.loc[high_reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            high_reduce_mask, risk_budget_codes
        ].mul(high_scale.loc[high_reduce_mask], axis=0)

    return run_target_weights_strategy(
        prices,
        selected,
        capped_target_weights,
        mixed_momentum,
        fee_rate,
        slippage_rate,
    )


def run_relative_lead_core_overheat_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    core_weight: float,
    overheat_drawdown_cut: float,
    overheat_momentum_cut: float,
    overheat_max_exposure: float,
    overheat_high_momentum_cut: float,
    overheat_high_max_exposure: float,
    hs300_mom_cut: float,
    lead_margin_cut: float,
    ma_window: int,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_risk_codes, active_defensive_codes, risk_budget_codes = resolve_goal_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    hs300_code = "510300"
    hs300_mom = prices[hs300_code] / prices[hs300_code].shift(lookback) - 1
    lead_filter = build_hs300_relative_lead_filter(
        prices,
        candidate_signal=signal,
        lookback=lookback,
        hs300_mom_cut=hs300_mom_cut,
        lead_margin_cut=lead_margin_cut,
        ma_window=ma_window,
    )
    dynamic_core_weight = lead_filter.astype(float) * core_weight
    satellite_scale = 1.0 - dynamic_core_weight

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_weights[hs300_code] += dynamic_core_weight
    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_scale.loc[code_mask] * target_exposure.loc[code_mask]

    blended_momentum = (satellite_scale * current_momentum + dynamic_core_weight * hs300_mom).rename("current_momentum")
    base_result, _ = run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )

    row_risk_weight = target_weights[risk_budget_codes].sum(axis=1)
    reduce_mask = (
        (row_risk_weight > overheat_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (blended_momentum >= overheat_momentum_cut)
    )
    capped_target_weights = target_weights.copy()
    if reduce_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[reduce_mask] = overheat_max_exposure / row_risk_weight.loc[reduce_mask]
        capped_target_weights.loc[reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            reduce_mask, risk_budget_codes
        ].mul(scale.loc[reduce_mask], axis=0)

    high_reduce_mask = (
        (capped_target_weights[risk_budget_codes].sum(axis=1) > overheat_high_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (blended_momentum >= overheat_high_momentum_cut)
    )
    if high_reduce_mask.any():
        high_scale = pd.Series(1.0, index=prices.index, dtype="float64")
        high_scale.loc[high_reduce_mask] = (
            overheat_high_max_exposure / capped_target_weights[risk_budget_codes].sum(axis=1).loc[high_reduce_mask]
        )
        capped_target_weights.loc[high_reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            high_reduce_mask, risk_budget_codes
        ].mul(high_scale.loc[high_reduce_mask], axis=0)

    return run_target_weights_strategy(
        prices,
        selected,
        capped_target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )


def run_strength_adaptive_core_overheat_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    base_core_weight: float,
    overheat_drawdown_cut: float,
    overheat_momentum_cut: float,
    overheat_max_exposure: float,
    overheat_high_momentum_cut: float,
    overheat_high_max_exposure: float,
    hs300_mom60_cut: float = 0.05,
    hs300_mom120_cut: float = 0.10,
    hs300_ma_window: int = 120,
    medium_multiplier: float = 0.5,
    strong_multiplier: float = 1.0,
    extreme_multiplier: float = 1.25,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_risk_codes, active_defensive_codes, risk_budget_codes = resolve_goal_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    hs300_code = "510300"
    hs300_mom60 = prices[hs300_code] / prices[hs300_code].shift(60) - 1
    dynamic_core_weight = build_hs300_strength_core_weight(
        prices,
        base_core_weight=base_core_weight,
        mom60_cut=hs300_mom60_cut,
        mom120_cut=hs300_mom120_cut,
        ma_window=hs300_ma_window,
        medium_multiplier=medium_multiplier,
        strong_multiplier=strong_multiplier,
        extreme_multiplier=extreme_multiplier,
    )
    satellite_scale = 1.0 - dynamic_core_weight

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_weights[hs300_code] += dynamic_core_weight
    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_scale.loc[code_mask] * target_exposure.loc[code_mask]

    blended_momentum = (satellite_scale * current_momentum + dynamic_core_weight * hs300_mom60).rename("current_momentum")
    base_result, _ = run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )

    row_risk_weight = target_weights[risk_budget_codes].sum(axis=1)
    reduce_mask = (
        (row_risk_weight > overheat_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (blended_momentum >= overheat_momentum_cut)
    )

    capped_target_weights = target_weights.copy()
    if reduce_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[reduce_mask] = overheat_max_exposure / row_risk_weight.loc[reduce_mask]
        capped_target_weights.loc[reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            reduce_mask, risk_budget_codes
        ].mul(scale.loc[reduce_mask], axis=0)

    high_reduce_mask = (
        (capped_target_weights[risk_budget_codes].sum(axis=1) > overheat_high_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (blended_momentum >= overheat_high_momentum_cut)
    )
    if high_reduce_mask.any():
        high_scale = pd.Series(1.0, index=prices.index, dtype="float64")
        high_scale.loc[high_reduce_mask] = (
            overheat_high_max_exposure / capped_target_weights[risk_budget_codes].sum(axis=1).loc[high_reduce_mask]
        )
        capped_target_weights.loc[high_reduce_mask, risk_budget_codes] = capped_target_weights.loc[
            high_reduce_mask, risk_budget_codes
        ].mul(high_scale.loc[high_reduce_mask], axis=0)

    return run_target_weights_strategy(
        prices,
        selected,
        capped_target_weights,
        blended_momentum,
        fee_rate,
        slippage_rate,
    )


def build_candidates(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    args: argparse.Namespace,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame, str]]:
    candidates: list[tuple[str, pd.DataFrame, pd.DataFrame, str]] = []
    notified_context = load_notified_regime_mix_context()
    market_volume_proxy = pd.DataFrame()
    try:
        market_volume_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)
    except Exception as exc:
        write_json_atomic(MARKET_VOLUME_ERROR_PATH, {"error": str(exc)})
        print(f"[goal_optimizations] market volume proxy unavailable: {exc}", file=sys.stderr)
        market_volume_proxy = pd.DataFrame()

    baseline_result, baseline_trades = run_default_strategy(
        prices,
        selected,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    candidates.append(
        (
            "current_baseline",
            baseline_result,
            baseline_trades,
            "当前正式基线：regime mix + A股量能防守 + 过热双层降仓 + 十年国债弱市防守。",
        )
    )

    config_specs = [
        {
            "core_weight": 0.10,
            "cap": DEFAULT_OVERHEAT_MAX_EXPOSURE,
            "hi_cap": DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
            "mom60_cut": 0.05,
            "mom120_cut": 0.10,
            "ma_window": 120,
        },
        {
            "core_weight": 0.15,
            "cap": DEFAULT_OVERHEAT_MAX_EXPOSURE,
            "hi_cap": DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
            "mom60_cut": 0.05,
            "mom120_cut": 0.10,
            "ma_window": 120,
        },
        {
            "core_weight": 0.20,
            "cap": DEFAULT_OVERHEAT_MAX_EXPOSURE,
            "hi_cap": DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
            "mom60_cut": 0.05,
            "mom120_cut": 0.10,
            "ma_window": 120,
        },
        {
            "core_weight": 0.25,
            "cap": DEFAULT_OVERHEAT_MAX_EXPOSURE,
            "hi_cap": DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
            "mom60_cut": 0.05,
            "mom120_cut": 0.10,
            "ma_window": 120,
        },
        {"core_weight": 0.26, "cap": 0.35, "hi_cap": 0.30, "mom60_cut": 0.05, "mom120_cut": 0.10, "ma_window": 120},
        {"core_weight": 0.28, "cap": 0.35, "hi_cap": 0.30, "mom60_cut": 0.05, "mom120_cut": 0.10, "ma_window": 120},
        {"core_weight": 0.30, "cap": 0.35, "hi_cap": 0.30, "mom60_cut": 0.05, "mom120_cut": 0.10, "ma_window": 120},
        {"core_weight": 0.28, "cap": 0.35, "hi_cap": 0.30, "mom60_cut": 0.04, "mom120_cut": 0.08, "ma_window": 120},
        {"core_weight": 0.28, "cap": 0.35, "hi_cap": 0.30, "mom60_cut": 0.04, "mom120_cut": 0.10, "ma_window": 120},
        {"core_weight": 0.28, "cap": 0.35, "hi_cap": 0.30, "mom60_cut": 0.06, "mom120_cut": 0.10, "ma_window": 120},
        {"core_weight": 0.28, "cap": 0.35, "hi_cap": 0.30, "mom60_cut": 0.05, "mom120_cut": 0.12, "ma_window": 120},
        {"core_weight": 0.28, "cap": 0.35, "hi_cap": 0.30, "mom60_cut": 0.05, "mom120_cut": 0.10, "ma_window": 90},
        {"core_weight": 0.28, "cap": 0.35, "hi_cap": 0.30, "mom60_cut": 0.05, "mom120_cut": 0.10, "ma_window": 150},
    ]

    for spec in config_specs:
        core_weight = float(spec["core_weight"])
        overheat_cap = float(spec["cap"])
        overheat_hi_cap = float(spec["hi_cap"])
        mom60_cut = float(spec["mom60_cut"])
        mom120_cut = float(spec["mom120_cut"])
        ma_window = int(spec["ma_window"])
        result, trades = run_dynamic_core_satellite_overheat_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
            weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
            core_weight=core_weight,
            overheat_drawdown_cut=DEFAULT_OVERHEAT_DRAWDOWN_CUT,
            overheat_momentum_cut=DEFAULT_OVERHEAT_MOMENTUM_CUT,
            overheat_max_exposure=overheat_cap,
            overheat_high_momentum_cut=DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
            overheat_high_max_exposure=overheat_hi_cap,
            hs300_mom60_cut=mom60_cut,
            hs300_mom120_cut=mom120_cut,
            hs300_ma_window=ma_window,
        )
        candidates.append(
            (
                f"adaptive_core_overheat_{int(round(core_weight * 100)):02d}_cap{int(round(overheat_cap * 100)):02d}_"
                f"hi{int(round(overheat_hi_cap * 100)):02d}_m60{int(round(mom60_cut * 100)):02d}_"
                f"m120{int(round(mom120_cut * 100)):02d}_ma{ma_window}",
                result,
                trades,
                f"在当前过热抑制基线之上，仅在 HS300 60/120 日趋势共振时加入 {core_weight:.0%} 的 HS300 核心仓；"
                f"第一层过热风险资产上限收紧到 {overheat_cap:.0%}，第二层极热上限为 {overheat_hi_cap:.0%}，"
                f"HS300 趋势过滤条件为 60 日动量>{mom60_cut:.0%}、120 日动量>{mom120_cut:.0%} 且价格站上 {ma_window} 日均线；"
                "其余仓位继续按原动量轮动与过热降仓执行。",
            )
        )

    # Fine-tune the current best family around the notified winner:
    # 28% HS300 core, 60d/120d trend gate = 4%/10%, MA120.
    fine_tune_specs = []
    for overheat_cap, overheat_hi_cap in [(0.33, 0.25), (0.35, 0.30), (0.37, 0.30)]:
        for overheat_momentum_cut in [0.22, 0.23, 0.24, 0.26]:
            for overheat_high_momentum_cut in [0.30, 0.32, 0.34, 0.38]:
                fine_tune_specs.append(
                    {
                        "core_weight": 0.28,
                        "cap": overheat_cap,
                        "hi_cap": overheat_hi_cap,
                        "mom60_cut": 0.04,
                        "mom120_cut": 0.10,
                        "ma_window": 120,
                        "overheat_momentum_cut": overheat_momentum_cut,
                        "overheat_high_momentum_cut": overheat_high_momentum_cut,
                    }
                )

    for spec in fine_tune_specs:
        result, trades = run_dynamic_core_satellite_overheat_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
            weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
            core_weight=float(spec["core_weight"]),
            overheat_drawdown_cut=DEFAULT_OVERHEAT_DRAWDOWN_CUT,
            overheat_momentum_cut=float(spec["overheat_momentum_cut"]),
            overheat_max_exposure=float(spec["cap"]),
            overheat_high_momentum_cut=float(spec["overheat_high_momentum_cut"]),
            overheat_high_max_exposure=float(spec["hi_cap"]),
            hs300_mom60_cut=float(spec["mom60_cut"]),
            hs300_mom120_cut=float(spec["mom120_cut"]),
            hs300_ma_window=int(spec["ma_window"]),
        )
        candidates.append(
            (
                f"adaptive_core_overheat_{int(round(float(spec['core_weight']) * 100)):02d}_"
                f"cap{int(round(float(spec['cap']) * 100)):02d}_"
                f"hi{int(round(float(spec['hi_cap']) * 100)):02d}_"
                f"m60{int(round(float(spec['mom60_cut']) * 100)):02d}_"
                f"m120{int(round(float(spec['mom120_cut']) * 100)):02d}_"
                f"ma{int(spec['ma_window'])}_"
                f"oh{int(round(float(spec['overheat_momentum_cut']) * 100)):02d}_"
                f"ohh{int(round(float(spec['overheat_high_momentum_cut']) * 100)):02d}",
                result,
                trades,
                f"在当前已通知最优框架上继续细调过热保护：HS300 核心仓 {float(spec['core_weight']):.0%}，"
                f"趋势门槛维持 60 日动量>{float(spec['mom60_cut']):.0%}、120 日动量>{float(spec['mom120_cut']):.0%} 且站上 "
                f"{int(spec['ma_window'])} 日均线；第一层过热在动量>{float(spec['overheat_momentum_cut']):.0%} 时将风险资产仓位压到 "
                f"{float(spec['cap']):.0%}，第二层极热在动量>{float(spec['overheat_high_momentum_cut']):.0%} 时进一步压到 "
                f"{float(spec['hi_cap']):.0%}。",
            )
        )

    # Fine-tune the higher-return MA90 family with tighter overheat controls
    # to see whether drawdown can be compressed enough while keeping its edge.
    ma90_rescue_specs = []
    for overheat_drawdown_cut in [-0.02, -0.03]:
        for overheat_cap, overheat_hi_cap in [(0.30, 0.20), (0.30, 0.25), (0.33, 0.25), (0.35, 0.25)]:
            for overheat_momentum_cut in [0.20, 0.22, 0.24, 0.25]:
                for overheat_high_momentum_cut in [0.28, 0.30, 0.32, 0.35]:
                    ma90_rescue_specs.append(
                        {
                            "core_weight": 0.28,
                            "cap": overheat_cap,
                            "hi_cap": overheat_hi_cap,
                            "mom60_cut": 0.05,
                            "mom120_cut": 0.10,
                            "ma_window": 90,
                            "overheat_drawdown_cut": overheat_drawdown_cut,
                            "overheat_momentum_cut": overheat_momentum_cut,
                            "overheat_high_momentum_cut": overheat_high_momentum_cut,
                        }
                    )

    for spec in ma90_rescue_specs:
        result, trades = run_dynamic_core_satellite_overheat_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
            weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
            core_weight=float(spec["core_weight"]),
            overheat_drawdown_cut=float(spec["overheat_drawdown_cut"]),
            overheat_momentum_cut=float(spec["overheat_momentum_cut"]),
            overheat_max_exposure=float(spec["cap"]),
            overheat_high_momentum_cut=float(spec["overheat_high_momentum_cut"]),
            overheat_high_max_exposure=float(spec["hi_cap"]),
            hs300_mom60_cut=float(spec["mom60_cut"]),
            hs300_mom120_cut=float(spec["mom120_cut"]),
            hs300_ma_window=int(spec["ma_window"]),
        )
        candidates.append(
            (
                f"adaptive_core_overheat_{int(round(float(spec['core_weight']) * 100)):02d}_"
                f"cap{int(round(float(spec['cap']) * 100)):02d}_"
                f"hi{int(round(float(spec['hi_cap']) * 100)):02d}_"
                f"m60{int(round(float(spec['mom60_cut']) * 100)):02d}_"
                f"m120{int(round(float(spec['mom120_cut']) * 100)):02d}_"
                f"ma{int(spec['ma_window'])}_"
                f"dd{int(round(abs(float(spec['overheat_drawdown_cut'])) * 100)):02d}_"
                f"oh{int(round(float(spec['overheat_momentum_cut']) * 100)):02d}_"
                f"ohh{int(round(float(spec['overheat_high_momentum_cut']) * 100)):02d}",
                result,
                trades,
                f"沿用更高弹性的 HS300 MA90 核心仓框架：HS300 核心仓 {float(spec['core_weight']):.0%}，"
                f"趋势门槛为 60 日动量>{float(spec['mom60_cut']):.0%}、120 日动量>{float(spec['mom120_cut']):.0%} 且站上 "
                f"{int(spec['ma_window'])} 日均线；当回撤不低于 {float(spec['overheat_drawdown_cut']):.0%} 且动量>{float(spec['overheat_momentum_cut']):.0%} 时，"
                f"将风险资产压到 {float(spec['cap']):.0%}，极热条件动量>{float(spec['overheat_high_momentum_cut']):.0%} 时进一步压到 "
                f"{float(spec['hi_cap']):.0%}。",
            )
        )

    bear_guard_specs = []
    for ma_window, mom60_cut, overheat_cap, overheat_hi_cap, overheat_high_cut in [
        (120, 0.04, 0.35, 0.30, 0.35),
        (90, 0.05, 0.30, 0.20, 0.32),
    ]:
        for bear_guard_momentum_cut in [0.02, 0.04, 0.06]:
            for bear_guard_max_exposure in [0.60, 0.70, 0.80]:
                bear_guard_specs.append(
                    {
                        "core_weight": 0.28,
                        "mom60_cut": mom60_cut,
                        "mom120_cut": 0.10,
                        "ma_window": ma_window,
                        "bear_guard_momentum_cut": bear_guard_momentum_cut,
                        "bear_guard_max_exposure": bear_guard_max_exposure,
                        "overheat_drawdown_cut": DEFAULT_OVERHEAT_DRAWDOWN_CUT,
                        "overheat_momentum_cut": DEFAULT_OVERHEAT_MOMENTUM_CUT,
                        "overheat_cap": overheat_cap,
                        "overheat_high_momentum_cut": overheat_high_cut,
                        "overheat_hi_cap": overheat_hi_cap,
                    }
                )

    for spec in bear_guard_specs:
        result, trades = run_dynamic_core_bear_guard_overheat_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
            weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
            core_weight=float(spec["core_weight"]),
            bear_guard_momentum_cut=float(spec["bear_guard_momentum_cut"]),
            bear_guard_max_exposure=float(spec["bear_guard_max_exposure"]),
            overheat_drawdown_cut=float(spec["overheat_drawdown_cut"]),
            overheat_momentum_cut=float(spec["overheat_momentum_cut"]),
            overheat_max_exposure=float(spec["overheat_cap"]),
            overheat_high_momentum_cut=float(spec["overheat_high_momentum_cut"]),
            overheat_high_max_exposure=float(spec["overheat_hi_cap"]),
            hs300_mom60_cut=float(spec["mom60_cut"]),
            hs300_mom120_cut=float(spec["mom120_cut"]),
            hs300_ma_window=int(spec["ma_window"]),
        )
        candidates.append(
            (
                f"bear_guard_core_{int(round(float(spec['core_weight']) * 100)):02d}_"
                f"m60{int(round(float(spec['mom60_cut']) * 100)):02d}_"
                f"m120{int(round(float(spec['mom120_cut']) * 100)):02d}_"
                f"ma{int(spec['ma_window'])}_"
                f"bgm{int(round(float(spec['bear_guard_momentum_cut']) * 100)):02d}_"
                f"bgx{int(round(float(spec['bear_guard_max_exposure']) * 100)):02d}_"
                f"cap{int(round(float(spec['overheat_cap']) * 100)):02d}_"
                f"hi{int(round(float(spec['overheat_hi_cap']) * 100)):02d}",
                result,
                trades,
                f"在 HS300 核心仓框架外增加弱市风险上限：当 HS300 趋势门槛失效且当前动量不高于 "
                f"{float(spec['bear_guard_momentum_cut']):.0%} 时，风险资产总仓位先压到 {float(spec['bear_guard_max_exposure']):.0%}；"
                f"随后继续保留原有过热保护，第一层上限 {float(spec['overheat_cap']):.0%}，第二层极热上限 "
                f"{float(spec['overheat_hi_cap']):.0%}。HS300 过滤条件为 60 日动量>{float(spec['mom60_cut']):.0%}、"
                f"120 日动量>{float(spec['mom120_cut']):.0%} 且站上 {int(spec['ma_window'])} 日均线。",
            )
        )

    short_brake_specs = []
    for ma_window, mom60_cut, overheat_cap, overheat_hi_cap, overheat_high_cut in [
        (120, 0.04, 0.35, 0.30, 0.35),
        (90, 0.05, 0.30, 0.20, 0.32),
    ]:
        for short_lb in [3, 5]:
            for short_momentum_cut in [-0.02, -0.01, 0.0]:
                for brake_momentum_ceiling in [0.04, 0.06, 0.08]:
                    for brake_defensive_weight in [0.8, 1.0]:
                        short_brake_specs.append(
                            {
                                "core_weight": 0.28,
                                "mom60_cut": mom60_cut,
                                "mom120_cut": 0.10,
                                "ma_window": ma_window,
                                "short_lb": short_lb,
                                "short_momentum_cut": short_momentum_cut,
                                "brake_momentum_ceiling": brake_momentum_ceiling,
                                "brake_defensive_weight": brake_defensive_weight,
                                "overheat_drawdown_cut": DEFAULT_OVERHEAT_DRAWDOWN_CUT,
                                "overheat_momentum_cut": DEFAULT_OVERHEAT_MOMENTUM_CUT,
                                "overheat_cap": overheat_cap,
                                "overheat_high_momentum_cut": overheat_high_cut,
                                "overheat_hi_cap": overheat_hi_cap,
                            }
                        )

    for spec in short_brake_specs:
        result, trades = run_dynamic_core_short_brake_overheat_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
            weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
            core_weight=float(spec["core_weight"]),
            short_lb=int(spec["short_lb"]),
            short_momentum_cut=float(spec["short_momentum_cut"]),
            brake_momentum_ceiling=float(spec["brake_momentum_ceiling"]),
            brake_defensive_weight=float(spec["brake_defensive_weight"]),
            overheat_drawdown_cut=float(spec["overheat_drawdown_cut"]),
            overheat_momentum_cut=float(spec["overheat_momentum_cut"]),
            overheat_max_exposure=float(spec["overheat_cap"]),
            overheat_high_momentum_cut=float(spec["overheat_high_momentum_cut"]),
            overheat_high_max_exposure=float(spec["overheat_hi_cap"]),
            hs300_mom60_cut=float(spec["mom60_cut"]),
            hs300_mom120_cut=float(spec["mom120_cut"]),
            hs300_ma_window=int(spec["ma_window"]),
        )
        candidates.append(
            (
                f"short_brake_core_{int(round(float(spec['core_weight']) * 100)):02d}_"
                f"m60{int(round(float(spec['mom60_cut']) * 100)):02d}_"
                f"m120{int(round(float(spec['mom120_cut']) * 100)):02d}_"
                f"ma{int(spec['ma_window'])}_"
                f"s{int(spec['short_lb']):02d}_"
                f"sc{int(round((float(spec['short_momentum_cut']) + 0.10) * 100)):02d}_"
                f"mc{int(round(float(spec['brake_momentum_ceiling']) * 100)):02d}_"
                f"dw{int(round(float(spec['brake_defensive_weight']) * 100)):02d}_"
                f"cap{int(round(float(spec['overheat_cap']) * 100)):02d}_"
                f"hi{int(round(float(spec['overheat_hi_cap']) * 100)):02d}",
                result,
                trades,
                f"在 HS300 核心仓框架中加入短周期刹车：若 HS300 趋势过滤失效，且风险信号的 {int(spec['short_lb'])} 日动量"
                f"不高于 {float(spec['short_momentum_cut']):.0%}、同时 25 日信号动量不高于 {float(spec['brake_momentum_ceiling']):.0%}，"
                f"则不继续持有风险资产，改切到防守资产并给 {float(spec['brake_defensive_weight']):.0%} 仓位；其余部分仍按 "
                f"HS300 核心仓 + 过热降仓执行。",
            )
        )

    regime_mix_specs = []
    for aggressive_core_weight in [0.28, 0.30]:
        for conservative_core_weight in [0.24, 0.28]:
            for regime_momentum_cut in [0.06, 0.08, 0.10, 0.12]:
                for overheat_cap, overheat_hi_cap, overheat_high_cut in [(0.30, 0.20, 0.32), (0.33, 0.25, 0.32), (0.35, 0.30, 0.35)]:
                    regime_mix_specs.append(
                        {
                            "aggressive_core_weight": aggressive_core_weight,
                            "conservative_core_weight": conservative_core_weight,
                            "regime_momentum_cut": regime_momentum_cut,
                            "overheat_drawdown_cut": DEFAULT_OVERHEAT_DRAWDOWN_CUT,
                            "overheat_momentum_cut": DEFAULT_OVERHEAT_MOMENTUM_CUT,
                            "overheat_cap": overheat_cap,
                            "overheat_high_momentum_cut": overheat_high_cut,
                            "overheat_hi_cap": overheat_hi_cap,
                        }
                    )

    for spec in regime_mix_specs:
        result, trades = run_regime_mix_core_overheat_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
            weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
            aggressive_core_weight=float(spec["aggressive_core_weight"]),
            conservative_core_weight=float(spec["conservative_core_weight"]),
            regime_momentum_cut=float(spec["regime_momentum_cut"]),
            overheat_drawdown_cut=float(spec["overheat_drawdown_cut"]),
            overheat_momentum_cut=float(spec["overheat_momentum_cut"]),
            overheat_max_exposure=float(spec["overheat_cap"]),
            overheat_high_momentum_cut=float(spec["overheat_high_momentum_cut"]),
            overheat_high_max_exposure=float(spec["overheat_hi_cap"]),
        )
        candidates.append(
            (
                f"regime_mix_core_ag{int(round(float(spec['aggressive_core_weight']) * 100)):02d}_"
                f"co{int(round(float(spec['conservative_core_weight']) * 100)):02d}_"
                f"rm{int(round(float(spec['regime_momentum_cut']) * 100)):02d}_"
                f"cap{int(round(float(spec['overheat_cap']) * 100)):02d}_"
                f"hi{int(round(float(spec['overheat_hi_cap']) * 100)):02d}",
                result,
                trades,
                f"采用进攻/防守双框架切换：强势区间使用 MA90 的进攻型 HS300 核心仓框架，弱势区间回到 MA120 的保守框架；"
                f"当进攻框架的综合动量达到 {float(spec['regime_momentum_cut']):.0%} 且 HS300 趋势过滤通过时，切入进攻态。"
                f"进攻核心仓 {float(spec['aggressive_core_weight']):.0%}，保守核心仓 {float(spec['conservative_core_weight']):.0%}，"
                f"过热上限 {float(spec['overheat_cap']):.0%} / {float(spec['overheat_hi_cap']):.0%}。",
            )
        )

    if not market_volume_proxy.empty:
        volume_guard_specs = []
        for volume_ratio_cut in [0.92, 0.95, 0.98]:
            for volume_short_ratio_cut in [0.90, 0.95]:
                for volume_breadth_cut in [-0.02, 0.00]:
                    for volume_guard_cap in [0.70, 0.80]:
                        volume_guard_specs.append(
                            {
                                "aggressive_core_weight": 0.30,
                                "conservative_core_weight": 0.28,
                                "regime_momentum_cut": 0.06,
                                "volume_ratio_cut": volume_ratio_cut,
                                "volume_short_ratio_cut": volume_short_ratio_cut,
                                "volume_breadth_cut": volume_breadth_cut,
                                "volume_guard_cap": volume_guard_cap,
                                "volume_guard_momentum_ceiling": 0.18,
                                "overheat_drawdown_cut": DEFAULT_OVERHEAT_DRAWDOWN_CUT,
                                "overheat_momentum_cut": DEFAULT_OVERHEAT_MOMENTUM_CUT,
                                "overheat_cap": 0.30,
                                "overheat_high_momentum_cut": 0.32,
                                "overheat_hi_cap": 0.20,
                            }
                        )

        for spec in volume_guard_specs:
            result, trades = run_regime_mix_core_volume_guard_strategy(
                prices,
                selected,
                market_volume_proxy=market_volume_proxy,
                lookback=args.lookback,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
                absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
                weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
                aggressive_core_weight=float(spec["aggressive_core_weight"]),
                conservative_core_weight=float(spec["conservative_core_weight"]),
                regime_momentum_cut=float(spec["regime_momentum_cut"]),
                volume_ratio_cut=float(spec["volume_ratio_cut"]),
                volume_short_ratio_cut=float(spec["volume_short_ratio_cut"]),
                volume_breadth_cut=float(spec["volume_breadth_cut"]),
                volume_guard_cap=float(spec["volume_guard_cap"]),
                volume_guard_momentum_ceiling=float(spec["volume_guard_momentum_ceiling"]),
                overheat_drawdown_cut=float(spec["overheat_drawdown_cut"]),
                overheat_momentum_cut=float(spec["overheat_momentum_cut"]),
                overheat_max_exposure=float(spec["overheat_cap"]),
                overheat_high_momentum_cut=float(spec["overheat_high_momentum_cut"]),
                overheat_high_max_exposure=float(spec["overheat_hi_cap"]),
            )
            candidates.append(
                (
                    f"regime_mix_volume_guard_ag{int(round(float(spec['aggressive_core_weight']) * 100)):02d}_"
                    f"co{int(round(float(spec['conservative_core_weight']) * 100)):02d}_"
                    f"rm{int(round(float(spec['regime_momentum_cut']) * 100)):02d}_"
                    f"vr{int(round(float(spec['volume_ratio_cut']) * 100)):02d}_"
                    f"vs{int(round(float(spec['volume_short_ratio_cut']) * 100)):02d}_"
                    f"vg{int(round(float(spec['volume_guard_cap']) * 100)):02d}",
                    result,
                    trades,
                    f"在当前已通知最优的 regime-mix 框架外，再叠加市场级 A 股量能过滤：当沪深综合成交额 20 日/60 日均值比"
                    f"低于 {float(spec['volume_ratio_cut']):.0%}、5 日/20 日比低于 {float(spec['volume_short_ratio_cut']):.0%}，且市场广度代理弱于 "
                    f"{float(spec['volume_breadth_cut']):.0%} 时，把风险资产总仓位压到 {float(spec['volume_guard_cap']):.0%}；"
                    f"其余逻辑仍按进攻/防守双框架 + 过热降仓执行。",
                )
            )

        if "511580" in selected["code"].astype(str).tolist():
            reduced_pool_specs = [
                {
                    "name_prefix": "regime_mix_rm511580_volume_guard",
                    "description_prefix": "在当前 remove_511580 版本基础上，再叠加市场级 A 股量能过滤：移除政金债 ETF(511580)，仅保留黄金和红利低波做防守。",
                    "drop_codes": ["511580"],
                    "specs": [
                        {
                            "aggressive_core_weight": 0.28,
                            "conservative_core_weight": 0.27,
                            "volume_ratio_cut": 0.91,
                            "volume_short_ratio_cut": 0.90,
                            "volume_breadth_cut": -0.01,
                            "volume_guard_cap": 0.52,
                            "overheat_drawdown_cut": -0.02,
                            "overheat_cap": 0.22,
                            "overheat_hi_cap": 0.14,
                        },
                        {
                            "aggressive_core_weight": 0.28,
                            "conservative_core_weight": 0.28,
                            "volume_ratio_cut": 0.91,
                            "volume_short_ratio_cut": 0.90,
                            "volume_breadth_cut": 0.00,
                            "volume_guard_cap": 0.56,
                            "overheat_cap": 0.30,
                            "overheat_hi_cap": 0.20,
                        },
                        {
                            "aggressive_core_weight": 0.28,
                            "conservative_core_weight": 0.27,
                            "volume_ratio_cut": 0.91,
                            "volume_short_ratio_cut": 0.90,
                            "volume_breadth_cut": 0.00,
                            "volume_guard_cap": 0.60,
                            "overheat_cap": 0.30,
                            "overheat_hi_cap": 0.20,
                        },
                        {
                            "aggressive_core_weight": 0.30,
                            "conservative_core_weight": 0.28,
                            "volume_ratio_cut": 0.92,
                            "volume_short_ratio_cut": 0.90,
                            "volume_breadth_cut": 0.00,
                            "volume_guard_cap": 0.70,
                            "overheat_cap": 0.30,
                            "overheat_hi_cap": 0.20,
                        },
                        {
                            "aggressive_core_weight": 0.30,
                            "conservative_core_weight": 0.28,
                            "volume_ratio_cut": 0.95,
                            "volume_short_ratio_cut": 0.90,
                            "volume_breadth_cut": 0.00,
                            "volume_guard_cap": 0.70,
                            "overheat_cap": 0.30,
                            "overheat_hi_cap": 0.20,
                        },
                        {
                            "aggressive_core_weight": 0.30,
                            "conservative_core_weight": 0.28,
                            "volume_ratio_cut": 0.92,
                            "volume_short_ratio_cut": 0.90,
                            "volume_breadth_cut": 0.00,
                            "volume_guard_cap": 0.80,
                            "overheat_cap": 0.30,
                            "overheat_hi_cap": 0.20,
                        },
                        {
                            "aggressive_core_weight": 0.30,
                            "conservative_core_weight": 0.28,
                            "volume_ratio_cut": 0.92,
                            "volume_short_ratio_cut": 0.95,
                            "volume_breadth_cut": 0.00,
                            "volume_guard_cap": 0.70,
                            "overheat_cap": 0.30,
                            "overheat_hi_cap": 0.20,
                        },
                    ],
                },
                {
                    "name_prefix": "regime_mix_rm511580_rm513650_volume_guard",
                    "description_prefix": "在当前 remove_511580 + remove_513650 版本基础上，再叠加市场级 A 股量能过滤：移除政金债 ETF(511580) 和 标普500ETF(513650)，仅保留黄金和红利低波做防守，同时把海外权益集中到纳指/H股/德国等更有区分度的方向。",
                    "drop_codes": ["511580", "513650"],
                    "specs": [],
                },
            ]
            base_rm_pool_params = build_reduced_pool_notified_base_params(notified_context)
            reduced_pool_specs[1]["specs"] = [
                {
                    **base_rm_pool_params,
                },
                {
                    **base_rm_pool_params,
                    "conservative_core_weight": max(0.20, base_rm_pool_params["conservative_core_weight"] - 0.01),
                    "volume_guard_cap": max(0.30, base_rm_pool_params["volume_guard_cap"] - 0.02),
                    "overheat_cap": max(0.10, base_rm_pool_params["overheat_cap"] - 0.01),
                    "overheat_hi_cap": max(0.08, base_rm_pool_params["overheat_hi_cap"] - 0.01),
                },
                {
                    **base_rm_pool_params,
                    "conservative_core_weight": min(0.40, base_rm_pool_params["conservative_core_weight"] + 0.01),
                    "volume_guard_cap": min(0.80, base_rm_pool_params["volume_guard_cap"] + 0.02),
                    "overheat_cap": min(0.40, base_rm_pool_params["overheat_cap"] + 0.01),
                    "overheat_hi_cap": min(0.25, base_rm_pool_params["overheat_hi_cap"] + 0.01),
                },
                {
                    **base_rm_pool_params,
                    "volume_ratio_cut": max(0.85, base_rm_pool_params["volume_ratio_cut"] - 0.01),
                    "volume_short_ratio_cut": max(0.85, base_rm_pool_params["volume_short_ratio_cut"] - 0.01),
                },
                {
                    **base_rm_pool_params,
                    "volume_ratio_cut": min(0.99, base_rm_pool_params["volume_ratio_cut"] + 0.01),
                    "volume_short_ratio_cut": min(0.99, base_rm_pool_params["volume_short_ratio_cut"] + 0.01),
                },
                {
                    **base_rm_pool_params,
                    "aggressive_core_weight": min(0.40, base_rm_pool_params["aggressive_core_weight"] + 0.01),
                    "conservative_core_weight": min(0.40, base_rm_pool_params["conservative_core_weight"] + 0.01),
                },
                {
                    **base_rm_pool_params,
                    "aggressive_core_weight": max(0.20, base_rm_pool_params["aggressive_core_weight"] - 0.01),
                    "conservative_core_weight": max(0.20, base_rm_pool_params["conservative_core_weight"] - 0.01),
                },
                {
                    **base_rm_pool_params,
                    "volume_breadth_cut": max(-0.03, base_rm_pool_params["volume_breadth_cut"] - 0.01),
                },
                {
                    **base_rm_pool_params,
                    "volume_breadth_cut": min(0.02, base_rm_pool_params["volume_breadth_cut"] + 0.01),
                },
            ]
            for pool_spec in reduced_pool_specs:
                reduced_selected = selected[~selected["code"].astype(str).isin(pool_spec["drop_codes"])].reset_index(drop=True)
                reduced_codes = reduced_selected["code"].astype(str).tolist()
                reduced_prices = prices[[code for code in prices.columns if code in reduced_codes]].copy()
                for spec in pool_spec["specs"]:
                    result, trades = run_regime_mix_core_volume_guard_strategy(
                        reduced_prices,
                        reduced_selected,
                        market_volume_proxy=market_volume_proxy,
                        lookback=args.lookback,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
                        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
                        aggressive_core_weight=float(spec.get("aggressive_core_weight", 0.30)),
                        conservative_core_weight=float(spec.get("conservative_core_weight", 0.28)),
                        regime_momentum_cut=float(spec.get("regime_momentum_cut", 0.06)),
                        volume_ratio_cut=float(spec["volume_ratio_cut"]),
                        volume_short_ratio_cut=float(spec["volume_short_ratio_cut"]),
                        volume_breadth_cut=float(spec["volume_breadth_cut"]),
                        volume_guard_cap=float(spec["volume_guard_cap"]),
                        volume_guard_momentum_ceiling=float(spec.get("volume_guard_momentum_ceiling", 0.18)),
                        overheat_drawdown_cut=float(spec.get("overheat_drawdown_cut", DEFAULT_OVERHEAT_DRAWDOWN_CUT)),
                        overheat_momentum_cut=float(spec.get("overheat_momentum_cut", DEFAULT_OVERHEAT_MOMENTUM_CUT)),
                        overheat_max_exposure=float(spec.get("overheat_cap", 0.30)),
                        overheat_high_momentum_cut=float(spec.get("overheat_high_momentum_cut", 0.32)),
                        overheat_high_max_exposure=float(spec.get("overheat_hi_cap", 0.20)),
                    )
                    candidates.append(
                        (
                            f"{pool_spec['name_prefix']}_"
                            f"ag{int(round(float(spec.get('aggressive_core_weight', 0.30)) * 100)):02d}_"
                            f"co{int(round(float(spec.get('conservative_core_weight', 0.28)) * 100)):02d}_"
                            f"vr{int(round(float(spec['volume_ratio_cut']) * 100)):02d}_"
                            f"vs{int(round(float(spec['volume_short_ratio_cut']) * 100)):02d}_"
                            f"vb{int(round((float(spec['volume_breadth_cut']) + 0.02) * 100)):02d}_"
                            f"vg{int(round(float(spec['volume_guard_cap']) * 100)):02d}_"
                            f"cap{int(round(float(spec.get('overheat_cap', 0.30)) * 100)):02d}_"
                            f"hi{int(round(float(spec.get('overheat_hi_cap', 0.20)) * 100)):02d}_"
                            f"dd{int(round(abs(float(spec.get('overheat_drawdown_cut', DEFAULT_OVERHEAT_DRAWDOWN_CUT))) * 100)):02d}_"
                            f"rm{int(round(float(spec.get('regime_momentum_cut', 0.06)) * 100)):02d}_"
                            f"vm{int(round(float(spec.get('volume_guard_momentum_ceiling', 0.18)) * 100)):02d}_"
                            f"oh{int(round(float(spec.get('overheat_momentum_cut', DEFAULT_OVERHEAT_MOMENTUM_CUT)) * 100)):02d}_"
                            f"ohh{int(round(float(spec.get('overheat_high_momentum_cut', 0.32)) * 100)):02d}",
                            result,
                            trades,
                            f"{pool_spec['description_prefix']}进攻/保守核心仓分别为 "
                            f"{float(spec.get('aggressive_core_weight', 0.30)):.0%}/{float(spec.get('conservative_core_weight', 0.28)):.0%}。"
                            f"当沪深综合成交额 20 日/60 日均值比低于 "
                            f"{float(spec['volume_ratio_cut']):.0%}、5 日/20 日比低于 {float(spec['volume_short_ratio_cut']):.0%}，"
                            f"且市场广度代理弱于 {float(spec['volume_breadth_cut']):.1%} 时，把风险资产总仓位压到 "
                            f"{float(spec['volume_guard_cap']):.0%}；过热降仓上限 {float(spec.get('overheat_cap', 0.30)):.0%}/"
                            f"{float(spec.get('overheat_hi_cap', 0.20)):.0%}，进攻切换阈值 {float(spec.get('regime_momentum_cut', 0.06)):.0%}，"
                            f"弱市量能防守只在动量不高于 {float(spec.get('volume_guard_momentum_ceiling', 0.18)):.0%} 时生效；"
                            f"过热触发回撤阈值 {float(spec.get('overheat_drawdown_cut', DEFAULT_OVERHEAT_DRAWDOWN_CUT)):.0%}，"
                            f"第一层过热动量阈值 {float(spec.get('overheat_momentum_cut', DEFAULT_OVERHEAT_MOMENTUM_CUT)):.0%}，"
                            f"第二层极热阈值 {float(spec.get('overheat_high_momentum_cut', 0.32)):.0%}。",
                        )
                    )

    strength_specs = [
        {
            "base_core_weight": 0.28,
            "cap": 0.35,
            "hi_cap": 0.30,
            "mom60_cut": 0.04,
            "mom120_cut": 0.10,
            "ma_window": 120,
            "medium_multiplier": 0.45,
            "strong_multiplier": 1.0,
            "extreme_multiplier": 1.25,
        },
        {
            "base_core_weight": 0.28,
            "cap": 0.35,
            "hi_cap": 0.30,
            "mom60_cut": 0.04,
            "mom120_cut": 0.10,
            "ma_window": 120,
            "medium_multiplier": 0.50,
            "strong_multiplier": 1.0,
            "extreme_multiplier": 1.35,
        },
        {
            "base_core_weight": 0.30,
            "cap": 0.35,
            "hi_cap": 0.30,
            "mom60_cut": 0.04,
            "mom120_cut": 0.10,
            "ma_window": 120,
            "medium_multiplier": 0.45,
            "strong_multiplier": 1.0,
            "extreme_multiplier": 1.25,
        },
        {
            "base_core_weight": 0.28,
            "cap": 0.35,
            "hi_cap": 0.30,
            "mom60_cut": 0.04,
            "mom120_cut": 0.10,
            "ma_window": 90,
            "medium_multiplier": 0.45,
            "strong_multiplier": 1.0,
            "extreme_multiplier": 1.25,
        },
    ]
    for spec in strength_specs:
        result, trades = run_strength_adaptive_core_overheat_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
            weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
            base_core_weight=float(spec["base_core_weight"]),
            overheat_drawdown_cut=DEFAULT_OVERHEAT_DRAWDOWN_CUT,
            overheat_momentum_cut=DEFAULT_OVERHEAT_MOMENTUM_CUT,
            overheat_max_exposure=float(spec["cap"]),
            overheat_high_momentum_cut=DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
            overheat_high_max_exposure=float(spec["hi_cap"]),
            hs300_mom60_cut=float(spec["mom60_cut"]),
            hs300_mom120_cut=float(spec["mom120_cut"]),
            hs300_ma_window=int(spec["ma_window"]),
            medium_multiplier=float(spec["medium_multiplier"]),
            strong_multiplier=float(spec["strong_multiplier"]),
            extreme_multiplier=float(spec["extreme_multiplier"]),
        )
        candidates.append(
            (
                f"strength_adaptive_core_{int(round(float(spec['base_core_weight']) * 100)):02d}_"
                f"cap{int(round(float(spec['cap']) * 100)):02d}_hi{int(round(float(spec['hi_cap']) * 100)):02d}_"
                f"m60{int(round(float(spec['mom60_cut']) * 100)):02d}_m120{int(round(float(spec['mom120_cut']) * 100)):02d}_"
                f"ma{int(spec['ma_window'])}_med{int(round(float(spec['medium_multiplier']) * 100)):02d}_"
                f"ext{int(round(float(spec['extreme_multiplier']) * 100)):02d}",
                result,
                trades,
                f"在当前过热抑制基线之上引入 HS300 强度自适应核心仓：基础核心仓 {float(spec['base_core_weight']):.0%}，"
                f"当 HS300 满足 60 日动量>{float(spec['mom60_cut']):.0%}、120 日动量>{float(spec['mom120_cut']):.0%} 且站上 "
                f"{int(spec['ma_window'])} 日均线后，按中等趋势 {float(spec['medium_multiplier']):.0%}、强趋势 "
                f"{float(spec['strong_multiplier']):.0%}、极强趋势 {float(spec['extreme_multiplier']):.0%} 分层提高核心仓；"
                f"第一层过热风险资产上限为 {float(spec['cap']):.0%}，第二层极热上限为 {float(spec['hi_cap']):.0%}。",
            )
        )

    relative_lead_specs = [
        {"core_weight": 0.28, "cap": 0.35, "hi_cap": 0.30, "hs300_mom_cut": 0.04, "lead_margin_cut": 0.02, "ma_window": 120},
        {"core_weight": 0.28, "cap": 0.35, "hi_cap": 0.30, "hs300_mom_cut": 0.04, "lead_margin_cut": 0.03, "ma_window": 120},
        {"core_weight": 0.30, "cap": 0.35, "hi_cap": 0.30, "hs300_mom_cut": 0.04, "lead_margin_cut": 0.02, "ma_window": 120},
        {"core_weight": 0.28, "cap": 0.35, "hi_cap": 0.30, "hs300_mom_cut": 0.05, "lead_margin_cut": 0.02, "ma_window": 90},
    ]
    for spec in relative_lead_specs:
        result, trades = run_relative_lead_core_overheat_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
            weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
            core_weight=float(spec["core_weight"]),
            overheat_drawdown_cut=DEFAULT_OVERHEAT_DRAWDOWN_CUT,
            overheat_momentum_cut=DEFAULT_OVERHEAT_MOMENTUM_CUT,
            overheat_max_exposure=float(spec["cap"]),
            overheat_high_momentum_cut=DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
            overheat_high_max_exposure=float(spec["hi_cap"]),
            hs300_mom_cut=float(spec["hs300_mom_cut"]),
            lead_margin_cut=float(spec["lead_margin_cut"]),
            ma_window=int(spec["ma_window"]),
        )
        candidates.append(
            (
                f"relative_lead_core_{int(round(float(spec['core_weight']) * 100)):02d}_cap{int(round(float(spec['cap']) * 100)):02d}_"
                f"hi{int(round(float(spec['hi_cap']) * 100)):02d}_hsmom{int(round(float(spec['hs300_mom_cut']) * 100)):02d}_"
                f"lead{int(round(float(spec['lead_margin_cut']) * 100)):02d}_ma{int(spec['ma_window'])}",
                result,
                trades,
                f"在当前过热抑制基线之上，仅当 HS300 近 {args.lookback} 日动量超过 {float(spec['hs300_mom_cut']):.0%}、"
                f"且相对当前动量胜出至少 {float(spec['lead_margin_cut']):.0%}、同时站上 {int(spec['ma_window'])} 日均线时，"
                f"加入 {float(spec['core_weight']):.0%} 的 HS300 核心仓；第一层过热上限 {float(spec['cap']):.0%}，第二层极热上限 {float(spec['hi_cap']):.0%}。",
            )
        )

    return candidates


def send_improvement_notification(
    baseline_summary: dict[str, float | int | str],
    _best_name: str,
    best_summary: dict[str, float | int | str],
    description: str,
    webhook_url: str,
) -> None:
    ann_diff = float(best_summary["annualized_return"]) - float(baseline_summary["annualized_return"])
    sharpe_diff = float(best_summary["sharpe_rf0"]) - float(baseline_summary["sharpe_rf0"])
    dd_int_diff = float(best_summary["max_drawdown_integral"]) - float(baseline_summary["max_drawdown_integral"])

    ann_diff_text = f"{ann_diff:+.2%}"
    sharpe_diff_text = f"{sharpe_diff:+.3f}"
    dd_int_diff_text = f"{dd_int_diff:+.4f}"
    message = (
        "ETF策略更新\n"
        f"核心变化: {description}\n"
        f"改进幅度: 年化 {ann_diff_text} | Sharpe {sharpe_diff_text} | 全历史回撤积分 {dd_int_diff_text}\n"
        f"当前结果: 年化 {best_summary['annualized_return']:.2%} | Sharpe {best_summary['sharpe_rf0']:.3f} | 全历史回撤积分 {best_summary['max_drawdown_integral']:.4f}"
    )
    send_webhook_message(message, webhook_url)


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    GOAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = load_default_strategy_backtest_pool()
    if args.refresh:
        prices = fetch_histories(selected, years=args.years)
    else:
        _, prices = load_cached_data()
        start_ts = prices.index.max() - pd.DateOffset(years=args.years)
        prices = prices.loc[prices.index >= start_ts].copy()
        required_treasury = selected[selected["code"].astype(str) == "511260"].copy()
        prices, skipped_codes, fetch_error = try_join_missing_candidate_histories(
            prices=prices,
            candidates=required_treasury,
            years=args.years,
            fetch_fn=fetch_histories,
        )
        if skipped_codes:
            skipped_text = ", ".join(skipped_codes)
            if fetch_error:
                print(f"[warn] missing required treasury cache: {skipped_text}; fetch failed: {fetch_error}")
            else:
                print(f"[warn] missing required treasury cache: {skipped_text}")
    raise_if_missing_required_histories(
        prices,
        selected[selected["code"].astype(str) == "511260"].copy(),
        context="baseline treasury",
    )
    candidates = build_candidates(prices, selected, args)

    rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame(index=prices.index)
    descriptions: dict[str, str] = {}
    for name, result, trades, description in candidates:
        summary = summarize(result, trades, selected)
        summary["strategy"] = name
        rows.append(summary)
        nav_compare[f"{name}_nav"] = result["nav"]
        descriptions[name] = description

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
    ].sort_values(["annualized_return", "sharpe_rf0"], ascending=False)
    write_dataframe_csv_atomic(summary_df, SUMMARY_PATH, index=False)
    write_dataframe_csv_atomic(nav_compare, COMPARE_PATH)

    baseline = summary_df[summary_df["strategy"] == "current_baseline"].iloc[0]
    better_df = summary_df[
        (summary_df["strategy"] != "current_baseline")
        & (summary_df["annualized_return"] >= baseline["annualized_return"] - METRIC_TOLERANCE)
        & (summary_df["sharpe_rf0"] >= baseline["sharpe_rf0"] - METRIC_TOLERANCE)
        & (summary_df["max_drawdown_integral"] <= baseline["max_drawdown_integral"] + METRIC_TOLERANCE)
        & (
            (summary_df["annualized_return"] > baseline["annualized_return"] + METRIC_TOLERANCE)
            | (summary_df["sharpe_rf0"] > baseline["sharpe_rf0"] + METRIC_TOLERANCE)
            | (summary_df["max_drawdown_integral"] < baseline["max_drawdown_integral"] - METRIC_TOLERANCE)
        )
    ].copy()
    better_df = sort_notify_candidates(better_df)

    result_payload: dict[str, object] = {
        "baseline": baseline.to_dict(),
        "strict_improvements": better_df.to_dict(orient="records"),
        "descriptions": descriptions,
    }
    write_json_atomic(BEST_PATH, result_payload)

    print("Baseline:")
    print(baseline.to_string())
    print("\nTop candidates:")
    print(summary_df.head(10).to_string(index=False))

    notify_df = load_incremental_notify_candidates(
        NOTIFY_STATE_PATH,
        better_df.copy(),
        metric_tolerance=METRIC_TOLERANCE,
    )

    notified, detail = notify_best_candidate(
        args,
        notify_df,
        descriptions=descriptions,
        baseline_summary=baseline.to_dict(),
        default_webhook=DEFAULT_FEISHU_WEBHOOK,
        notify_state_path=NOTIFY_STATE_PATH,
        send_fn=send_improvement_notification,
        suppress_exceptions=True,
    )
    if notified and detail is not None:
        print(f"\nWebhook notified for {detail}")
    elif detail is not None:
        print(f"\nWebhook notify failed for {detail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
