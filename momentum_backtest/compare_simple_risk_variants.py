#!/usr/bin/env python3
"""在简单质量动量基线上统一评估风控/过热处理变体。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import warnings

import pandas as pd

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
    from .core.backtest import (
        build_strategy_summary,
        build_trades_from_weight_frame,
        compute_signal_asset_momentum,
        finalize_position_columns,
        recompute_return_chain,
        run_threshold_dual_strategy,
        run_target_weights_strategy,
    )
    from .core.config import (
        DEFAULT_DUAL_CASH_EXIT_THRESHOLD,
        DEFAULT_FEE_RATE,
        DEFAULT_LOOKBACK,
        RISK_CODES,
        DEFAULT_SLIPPAGE_RATE,
        DEFAULT_STRESS_BOND_BREADTH_CUT,
        DEFAULT_STRESS_BOND_CODE,
        DEFAULT_STRESS_BOND_ENTER_DAYS,
        DEFAULT_STRESS_BOND_EXIT_DAYS,
        DEFAULT_STRESS_BOND_FILL_RESIDUAL_CASH,
        DEFAULT_STRESS_BOND_RATIO_CUT,
        OUTPUT_DIR,
    )
    from .core.data import load_core_selected_and_prices
    from .core.proxy import MARKET_VOLUME_CACHE_PATH
    from .core.reporting import get_latest_portfolio_text
    from .core.signals import build_asset_own_momentum_percentile, build_signal_quality_score
    from .core.strategy import build_default_strategy_params, run_default_strategy_with_params, run_simple_quality_momentum_strategy
    from .core.version_replay import run_historical_version_with_current_engine
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports
    from core.backtest import (
        build_strategy_summary,
        build_trades_from_weight_frame,
        compute_signal_asset_momentum,
        finalize_position_columns,
        recompute_return_chain,
        run_threshold_dual_strategy,
        run_target_weights_strategy,
    )
    from core.config import (
        DEFAULT_DUAL_CASH_EXIT_THRESHOLD,
        DEFAULT_FEE_RATE,
        DEFAULT_LOOKBACK,
        RISK_CODES,
        DEFAULT_SLIPPAGE_RATE,
        DEFAULT_STRESS_BOND_BREADTH_CUT,
        DEFAULT_STRESS_BOND_CODE,
        DEFAULT_STRESS_BOND_ENTER_DAYS,
        DEFAULT_STRESS_BOND_EXIT_DAYS,
        DEFAULT_STRESS_BOND_FILL_RESIDUAL_CASH,
        DEFAULT_STRESS_BOND_RATIO_CUT,
        OUTPUT_DIR,
    )
    from core.data import load_core_selected_and_prices
    from core.proxy import MARKET_VOLUME_CACHE_PATH
    from core.reporting import get_latest_portfolio_text
    from core.signals import build_asset_own_momentum_percentile, build_signal_quality_score
    from core.strategy import build_default_strategy_params, run_default_strategy_with_params, run_simple_quality_momentum_strategy
    from core.version_replay import run_historical_version_with_current_engine

try:
    from .replay_historical_versions import load_market_proxy_from_file
except ImportError:
    from replay_historical_versions import load_market_proxy_from_file


prepare_local_imports(__file__)
configure_matplotlib_env()
warnings.filterwarnings("ignore")


DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "analysis" / "simple_risk_variants_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一回测简单质量动量基线上的风控/过热处理变体。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="汇总表输出路径。",
    )
    return parser.parse_args()


def build_target_weights_variant(
    *,
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    base_result: pd.DataFrame,
    base_target_weights: pd.DataFrame,
    variant_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    returns = prices.pct_change(fill_method=None)
    total_vol20 = returns.rolling(20).std(ddof=0) * math.sqrt(252)
    downside_vol20 = returns.where(returns < 0, 0.0).rolling(20).std(ddof=0) * math.sqrt(252)
    momentum_pct = build_asset_own_momentum_percentile(
        prices,
        lookback=DEFAULT_LOOKBACK,
        state_lookback=756,
        min_periods=120,
    )
    raw_momentum = prices / prices.shift(DEFAULT_LOOKBACK) - 1
    ret5 = prices / prices.shift(5) - 1
    quality_score = build_signal_quality_score(
        prices,
        lookback=DEFAULT_LOOKBACK,
        method="slope",
        slope_penalty=0.85,
    )[1]
    quality_mean_5 = quality_score.rolling(5).mean()
    quality_mean_prev5 = quality_mean_5.shift(5)

    delta = prices.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_gain = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = down.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0.0, other=float("nan"))
    rsi14 = 100 - 100 / (1 + rs)

    ma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std(ddof=0)
    upper20 = ma20 + 2 * std20
    lower20 = ma20 - 2 * std20
    bandwidth = ((upper20 - lower20) / ma20.replace(0.0, pd.NA)).replace([float("inf"), -float("inf")], pd.NA)
    bandwidth_pct120 = bandwidth.rolling(120, min_periods=30).apply(
        lambda arr: (
            (pd.Series(arr).dropna() < pd.Series(arr).dropna().iloc[-1]).sum()
            + 0.5 * (pd.Series(arr).dropna() == pd.Series(arr).dropna().iloc[-1]).sum()
        )
        / max(len(pd.Series(arr).dropna()), 1)
        if len(pd.Series(arr).dropna()) > 0
        else float("nan"),
        raw=False,
    )

    signal = base_result["signal"].astype("object")
    scale = pd.Series(1.0, index=prices.index, dtype="float64", name=f"{variant_name}_scale")
    triggered = pd.Series(False, index=prices.index, dtype=bool, name=f"{variant_name}_triggered")

    breakout = (prices > upper20) & (rsi14 > 65)
    breakout_recent_5 = breakout.rolling(5, min_periods=1).max().fillna(0.0) > 0
    rsi_recent_peak_10 = rsi14.shift(1).rolling(10).max()
    touched_upper_recent_5 = ((prices / upper20) >= 1.0).shift(1).rolling(5).max().fillna(0.0) > 0

    for dt_idx in prices.index:
        code = signal.loc[dt_idx]
        if pd.isna(code) or str(code) not in prices.columns:
            continue
        code = str(code)

        pct = momentum_pct.loc[dt_idx, code]
        total_vol = total_vol20.loc[dt_idx, code]
        down_vol = downside_vol20.loc[dt_idx, code]
        rsi = rsi14.loc[dt_idx, code]
        quality_now = quality_mean_5.loc[dt_idx, code]
        quality_prev = quality_mean_prev5.loc[dt_idx, code]
        mom5 = ret5.loc[dt_idx, code]
        mom25 = raw_momentum.loc[dt_idx, code]
        is_squeeze = pd.notna(bandwidth_pct120.loc[dt_idx, code]) and float(bandwidth_pct120.loc[dt_idx, code]) <= 0.20
        breakout_confirm = bool(breakout.loc[dt_idx, code]) if code in breakout.columns else False
        breakout_window = bool(breakout_recent_5.loc[dt_idx, code]) if code in breakout_recent_5.columns else False
        rollover = (
            pd.notna(rsi_recent_peak_10.loc[dt_idx, code])
            and pd.notna(rsi)
            and float(rsi_recent_peak_10.loc[dt_idx, code]) >= 75.0
            and float(rsi) < 70.0
            and bool(touched_upper_recent_5.loc[dt_idx, code])
            and pd.notna(upper20.loc[dt_idx, code])
            and float(prices.loc[dt_idx, code]) < float(upper20.loc[dt_idx, code])
        )
        overheat_expansion = (
            pd.notna(bandwidth_pct120.loc[dt_idx, code])
            and pd.notna(rsi)
            and float(bandwidth_pct120.loc[dt_idx, code]) >= 0.80
            and float(rsi) >= 75.0
        )

        if variant_name == "vol_scale_20pct":
            if pd.notna(total_vol) and float(total_vol) > 0:
                scale.loc[dt_idx] = min(1.0, max(0.35, 0.20 / float(total_vol)))
        elif variant_name == "downside_vol_scale_20pct":
            if pd.notna(down_vol) and float(down_vol) > 0:
                scale.loc[dt_idx] = min(1.0, max(0.35, 0.20 / float(down_vol)))
        elif variant_name == "pct97_decay_rsi":
            if pd.notna(pct) and pd.notna(rsi_recent_peak_10.loc[dt_idx, code]) and pd.notna(rsi):
                if float(pct) >= 0.97 and float(rsi_recent_peak_10.loc[dt_idx, code]) >= 75.0 and float(rsi) < 70.0:
                    scale.loc[dt_idx] = 0.50
        elif variant_name == "pct97_decay_quality":
            if pd.notna(pct) and pd.notna(quality_now) and pd.notna(quality_prev) and pd.notna(mom5) and pd.notna(mom25):
                if float(pct) >= 0.97 and float(quality_now) < float(quality_prev) and float(mom5) < float(mom25) / 5.0:
                    scale.loc[dt_idx] = 0.50
        elif variant_name == "pct97_99_floor70":
            if pd.notna(pct) and float(pct) >= 0.97:
                progress = min(max((float(pct) - 0.97) / 0.02, 0.0), 1.0)
                scale.loc[dt_idx] = 1.0 + (0.70 - 1.0) * progress
        elif variant_name == "pct97_99_floor80":
            if pd.notna(pct) and float(pct) >= 0.97:
                progress = min(max((float(pct) - 0.97) / 0.02, 0.0), 1.0)
                scale.loc[dt_idx] = 1.0 + (0.80 - 1.0) * progress
        elif variant_name == "boll_squeeze_breakout_confirm":
            if is_squeeze and not breakout_window:
                scale.loc[dt_idx] = 0.70
            elif is_squeeze and breakout_confirm:
                scale.loc[dt_idx] = 1.00
        elif variant_name == "boll_rsi_rollover_cap":
            if rollover:
                scale.loc[dt_idx] = 0.50
        elif variant_name == "state_squeeze_breakout_rollover":
            if rollover:
                scale.loc[dt_idx] = 0.50
            elif is_squeeze and not breakout_window:
                scale.loc[dt_idx] = 0.70
            elif overheat_expansion:
                scale.loc[dt_idx] = 0.80
            else:
                scale.loc[dt_idx] = 1.00
        else:
            raise KeyError(f"unsupported variant: {variant_name}")

        triggered.loc[dt_idx] = abs(float(scale.loc[dt_idx]) - 1.0) > 1e-12

    target_weights = base_target_weights.mul(scale, axis=0)
    return target_weights, triggered


def run_variant_from_target_weights(
    *,
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    target_weights: pd.DataFrame,
    base_result: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result, _ = run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        base_result["current_momentum"],
        fee_rate,
        slippage_rate,
    )
    return_weight_cols = [col for col in result.columns if col.startswith("weight_")]
    return_weights = result[return_weight_cols].copy()
    return_weights.columns = [col.removeprefix("weight_") for col in return_weight_cols]
    result = finalize_position_columns(
        result,
        selected,
        confirmed_weights=target_weights,
        return_weights=return_weights,
        return_holding=result["holding"] if "holding" in result.columns else None,
        return_exposure=result["exposure"] if "exposure" in result.columns else None,
    )
    result = recompute_return_chain(result, prices, fee_rate=fee_rate, slippage_rate=slippage_rate)
    trades = build_trades_from_weight_frame(target_weights, result["nav"], selected)
    signal_momentum = compute_signal_asset_momentum(prices, result["signal"], DEFAULT_LOOKBACK)
    result["current_momentum"] = signal_momentum
    result["effective_momentum"] = signal_momentum
    result["signal_asset_momentum"] = signal_momentum
    result["target_exposure"] = target_weights.sum(axis=1).rename("target_exposure")
    result["base_target_exposure"] = target_weights.sum(axis=1).rename("base_target_exposure")
    return result, trades


def build_threshold_dual_overlay_variant(
    *,
    prices: pd.DataFrame,
    base_result: pd.DataFrame,
    base_target_weights: pd.DataFrame,
    variant_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    returns = prices.pct_change(fill_method=None)
    total_vol20 = returns.rolling(20).std(ddof=0) * math.sqrt(252)
    momentum_pct_5y = build_asset_own_momentum_percentile(
        prices,
        lookback=DEFAULT_LOOKBACK,
        state_lookback=1260,
        min_periods=1260,
    )
    signal = base_result["signal"].astype("object")
    target_exposure = pd.to_numeric(base_result["target_exposure"], errors="coerce").fillna(0.0)
    current_momentum = pd.to_numeric(base_result["current_momentum"], errors="coerce")

    hs300 = pd.to_numeric(prices["510300"], errors="coerce") if "510300" in prices.columns else prices.mean(axis=1)
    hs300_ret = hs300.pct_change(fill_method=None)
    hs300_vol20 = hs300_ret.rolling(20).std(ddof=0) * math.sqrt(252)
    hs300_ret_3 = hs300 / hs300.shift(3) - 1
    hs300_ret_5 = hs300 / hs300.shift(5) - 1
    hs300_drawdown_60 = hs300 / hs300.rolling(60, min_periods=20).max() - 1

    mom60 = prices / prices.shift(60) - 1
    mom120 = prices / prices.shift(120) - 1
    low9 = prices.rolling(9, min_periods=9).min()
    high9 = prices.rolling(9, min_periods=9).max()
    rsv = ((prices - low9) / (high9 - low9).replace(0.0, pd.NA) * 100.0).clip(lower=0.0, upper=100.0)
    k_value = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=3).mean()
    d_value = k_value.ewm(alpha=1 / 3, adjust=False, min_periods=3).mean()
    j_value = 3.0 * k_value - 2.0 * d_value
    delta = prices.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_gain = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = down.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0.0, other=float("nan"))
    rsi14 = 100.0 - 100.0 / (1.0 + rs)
    ma20 = prices.rolling(20, min_periods=20).mean()
    std20 = prices.rolling(20, min_periods=20).std(ddof=0)
    upper20 = ma20 + 2.0 * std20
    lower20 = ma20 - 2.0 * std20
    bandwidth = ((upper20 - lower20) / ma20.replace(0.0, pd.NA)).replace([float("inf"), -float("inf")], pd.NA)
    bandwidth_pct120 = bandwidth.rolling(120, min_periods=30).apply(
        lambda arr: (
            (pd.Series(arr).dropna() < pd.Series(arr).dropna().iloc[-1]).sum()
            + 0.5 * (pd.Series(arr).dropna() == pd.Series(arr).dropna().iloc[-1]).sum()
        )
        / max(len(pd.Series(arr).dropna()), 1)
        if len(pd.Series(arr).dropna()) > 0
        else float("nan"),
        raw=False,
    )

    scale = pd.Series(1.0, index=prices.index, dtype="float64", name=f"{variant_name}_scale")
    triggered = pd.Series(False, index=prices.index, dtype=bool, name=f"{variant_name}_triggered")
    risk_codes = set(RISK_CODES)

    for dt_idx in prices.index:
        code = signal.loc[dt_idx]
        if pd.isna(code):
            continue
        code = str(code)
        if code not in prices.columns:
            continue

        exposure = float(target_exposure.loc[dt_idx]) if pd.notna(target_exposure.loc[dt_idx]) else 0.0
        signal_mom = float(current_momentum.loc[dt_idx]) if pd.notna(current_momentum.loc[dt_idx]) else float("nan")
        current_scale = 1.0

        if variant_name in {"threshold_weak_vol_scale", "threshold_combo_guard"}:
            asset_vol = total_vol20.loc[dt_idx, code] if code in total_vol20.columns else float("nan")
            if exposure > 1e-12 and pd.notna(signal_mom) and 0.0 < signal_mom <= 0.05 and pd.notna(asset_vol) and float(asset_vol) > 0:
                current_scale = min(current_scale, min(1.0, max(0.35, 0.20 / float(asset_vol))))

        if variant_name in {"threshold_panic_rebound_cap", "threshold_combo_guard"}:
            if (
                code in risk_codes
                and exposure >= 0.999999
                and pd.notna(hs300_vol20.loc[dt_idx])
                and pd.notna(hs300_ret_5.loc[dt_idx])
                and pd.notna(hs300_drawdown_60.loc[dt_idx])
                and float(hs300_vol20.loc[dt_idx]) >= 0.24
                and float(hs300_ret_5.loc[dt_idx]) >= 0.05
                and float(hs300_drawdown_60.loc[dt_idx]) <= -0.10
            ):
                current_scale = min(current_scale, 0.50)

        if variant_name in {"threshold_trend_consistency", "threshold_combo_guard"}:
            mom60_value = mom60.loc[dt_idx, code] if code in mom60.columns else float("nan")
            mom120_value = mom120.loc[dt_idx, code] if code in mom120.columns else float("nan")
            if exposure > 1e-12 and (
                pd.isna(mom60_value)
                or pd.isna(mom120_value)
                or float(mom60_value) <= 0.0
                or float(mom120_value) <= 0.0
            ):
                current_scale = min(current_scale, 0.50 if exposure >= 0.999999 else 0.70)

        if variant_name == "threshold_pct_5y_85_90_97":
            pct_value = momentum_pct_5y.loc[dt_idx, code] if code in momentum_pct_5y.columns else float("nan")
            if exposure > 1e-12 and pd.notna(pct_value):
                pct = float(pct_value)
                if pct >= 0.97:
                    current_scale = min(current_scale, 0.50)
                elif pct >= 0.90:
                    progress = (pct - 0.90) / 0.07
                    current_scale = min(current_scale, 0.80 + (0.50 - 0.80) * progress)
                elif pct >= 0.85:
                    progress = (pct - 0.85) / 0.05
                    current_scale = min(current_scale, 1.00 + (0.80 - 1.00) * progress)
        if variant_name == "threshold_pct_5y_90_95_99_soft":
            pct_value = momentum_pct_5y.loc[dt_idx, code] if code in momentum_pct_5y.columns else float("nan")
            if exposure > 1e-12 and pd.notna(pct_value):
                pct = float(pct_value)
                if pct >= 0.99:
                    current_scale = min(current_scale, 0.60)
                elif pct >= 0.95:
                    progress = (pct - 0.95) / 0.04
                    current_scale = min(current_scale, 0.85 + (0.60 - 0.85) * progress)
                elif pct >= 0.90:
                    progress = (pct - 0.90) / 0.05
                    current_scale = min(current_scale, 1.00 + (0.85 - 1.00) * progress)
        if variant_name == "threshold_pct_5y_88_93_98_soft":
            pct_value = momentum_pct_5y.loc[dt_idx, code] if code in momentum_pct_5y.columns else float("nan")
            if exposure > 1e-12 and pd.notna(pct_value):
                pct = float(pct_value)
                if pct >= 0.98:
                    current_scale = min(current_scale, 0.60)
                elif pct >= 0.93:
                    progress = (pct - 0.93) / 0.05
                    current_scale = min(current_scale, 0.85 + (0.60 - 0.85) * progress)
                elif pct >= 0.88:
                    progress = (pct - 0.88) / 0.05
                    current_scale = min(current_scale, 1.00 + (0.85 - 1.00) * progress)

        rsi_value = rsi14.loc[dt_idx, code] if code in rsi14.columns else float("nan")
        k_val = k_value.loc[dt_idx, code] if code in k_value.columns else float("nan")
        d_val = d_value.loc[dt_idx, code] if code in d_value.columns else float("nan")
        j_val = j_value.loc[dt_idx, code] if code in j_value.columns else float("nan")
        price_now = prices.loc[dt_idx, code]
        upper_now = upper20.loc[dt_idx, code] if code in upper20.columns else float("nan")
        bandwidth_pct = bandwidth_pct120.loc[dt_idx, code] if code in bandwidth_pct120.columns else float("nan")
        boll_hot = (
            pd.notna(price_now)
            and pd.notna(upper_now)
            and pd.notna(bandwidth_pct)
            and float(price_now) >= float(upper_now)
            and float(bandwidth_pct) >= 0.80
        )
        boll_extreme_hot = (
            pd.notna(price_now)
            and pd.notna(upper_now)
            and pd.notna(bandwidth_pct)
            and float(price_now) >= float(upper_now)
            and float(bandwidth_pct) >= 0.90
        )
        rsi_hot = pd.notna(rsi_value) and float(rsi_value) >= 75.0
        kdj_hot = (
            pd.notna(k_val)
            and pd.notna(d_val)
            and pd.notna(j_val)
            and float(k_val) >= 80.0
            and float(d_val) >= 75.0
            and float(j_val) >= 95.0
        )

        if variant_name == "threshold_rsi_hot_cap":
            if exposure > 1e-12 and rsi_hot:
                current_scale = min(current_scale, 0.80)
        if variant_name == "threshold_kdj_hot_cap":
            if exposure > 1e-12 and kdj_hot:
                current_scale = min(current_scale, 0.80)
        if variant_name == "threshold_boll_hot_cap":
            if exposure > 1e-12 and boll_hot:
                current_scale = min(current_scale, 0.80)
        if variant_name == "threshold_rsi_kdj_boll_combo":
            hot_count = int(rsi_hot) + int(kdj_hot) + int(boll_hot)
            if exposure > 1e-12 and hot_count >= 2:
                current_scale = min(current_scale, 0.60)
        if variant_name == "boll_baseline_extreme_cap":
            if exposure > 1e-12 and boll_extreme_hot:
                current_scale = min(current_scale, 0.70)
        if variant_name == "boll_baseline_panic_sensitive":
            if (
                code in risk_codes
                and exposure >= 0.799999
                and pd.notna(hs300_vol20.loc[dt_idx])
                and pd.notna(hs300_ret_3.loc[dt_idx])
                and pd.notna(hs300_drawdown_60.loc[dt_idx])
                and float(hs300_vol20.loc[dt_idx]) >= 0.20
                and float(hs300_ret_3.loc[dt_idx]) >= 0.03
                and float(hs300_drawdown_60.loc[dt_idx]) <= -0.08
            ):
                current_scale = min(current_scale, 0.70)
        if variant_name == "boll_baseline_combo":
            if exposure > 1e-12 and boll_extreme_hot:
                current_scale = min(current_scale, 0.70)
            if (
                code in risk_codes
                and exposure >= 0.799999
                and pd.notna(hs300_vol20.loc[dt_idx])
                and pd.notna(hs300_ret_3.loc[dt_idx])
                and pd.notna(hs300_drawdown_60.loc[dt_idx])
                and float(hs300_vol20.loc[dt_idx]) >= 0.20
                and float(hs300_ret_3.loc[dt_idx]) >= 0.03
                and float(hs300_drawdown_60.loc[dt_idx]) <= -0.08
            ):
                current_scale = min(current_scale, 0.70)

        scale.loc[dt_idx] = current_scale
        triggered.loc[dt_idx] = abs(current_scale - 1.0) > 1e-12

    target_weights = base_target_weights.mul(scale, axis=0)
    return target_weights, triggered


def build_summary_row(
    *,
    version: str,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    selected: pd.DataFrame,
    trigger_days: int,
) -> dict[str, object]:
    summary = build_strategy_summary(
        result,
        trades,
        selected=selected,
        include_max_drawdown_integral=True,
        get_latest_portfolio_text=get_latest_portfolio_text,
    )
    return {
        "version": version,
        "annualized_return": float(summary["annualized_return"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "sharpe_rf0": float(summary["sharpe_rf0"]),
        "trigger_days": int(trigger_days),
        "avg_exposure": float(summary["avg_exposure"]),
        "drawdown_integral": float(summary["full_history_drawdown_integral"]),
        "end_nav": float(result["nav"].iloc[-1]),
        "trade_count": int(summary["trade_count"]),
        "latest_portfolio": str(summary.get("latest_portfolio", "")),
    }


def main() -> int:
    args = parse_args()
    selected, prices = load_core_selected_and_prices()

    base_result, base_trades = run_simple_quality_momentum_strategy(
        prices,
        selected,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    base_target_weights = base_result[[col for col in base_result.columns if col.startswith("target_weight_")]].copy()
    base_target_weights.columns = [col.removeprefix("target_weight_") for col in base_target_weights.columns]

    rows: list[dict[str, object]] = [
        build_summary_row(
            version="simple_baseline",
            result=base_result,
            trades=base_trades,
            selected=selected,
            trigger_days=0,
        )
    ]

    threshold_dual_result, threshold_dual_trades = run_threshold_dual_strategy(
        prices,
        selected,
        lookback=DEFAULT_LOOKBACK,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        cash_exit_threshold=DEFAULT_DUAL_CASH_EXIT_THRESHOLD,
    )
    threshold_dual_trigger_days = int(
        (
            (threshold_dual_result["target_exposure"] > 0)
            & (threshold_dual_result["target_exposure"] < 0.999999)
        ).sum()
    )
    rows.append(
        build_summary_row(
            version="threshold_dual_5pct",
            result=threshold_dual_result,
            trades=threshold_dual_trades,
            selected=selected,
            trigger_days=threshold_dual_trigger_days,
        )
    )
    threshold_dual_target_weights = threshold_dual_result[
        [col for col in threshold_dual_result.columns if col.startswith("weight_")]
    ].copy()
    threshold_dual_target_weights.columns = [col.removeprefix("weight_") for col in threshold_dual_target_weights.columns]

    for cash_exit_threshold in (0.0, 0.01):
        cash_result, cash_trades = run_threshold_dual_strategy(
            prices,
            selected,
            lookback=DEFAULT_LOOKBACK,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            cash_exit_threshold=cash_exit_threshold,
        )
        cash_trigger_days = int((cash_result["target_exposure"] <= 1e-12).sum())
        rows.append(
            build_summary_row(
                version=f"threshold_dual_cash_{int(cash_exit_threshold * 100):02d}pct",
                result=cash_result,
                trades=cash_trades,
                selected=selected,
                trigger_days=cash_trigger_days,
            )
        )

    regime_params = build_default_strategy_params()
    regime_params.update(
        {
            "volume_guard_cap": 1.0,
            "close_top2_gap": 0.0,
            "close_top2_risk_cap": 1.0,
            "pre_overheat_end_exposure": 1.0,
            "overheat_max_exposure": 1.0,
            "overheat_high_max_exposure": 1.0,
            "signal_selected_volatility_cap_start": 1.0,
            "signal_selected_volatility_cap_end": 1.0,
            "signal_selected_volatility_cap_floor": 1.0,
            "signal_selected_momentum_pct_cap_start": 1.0,
            "signal_selected_momentum_pct_cap_end": 1.0,
            "signal_selected_momentum_pct_cap_floor": 1.0,
            "defensive_signal_selected_momentum_pct_cap_start": 1.0,
            "defensive_signal_selected_momentum_pct_cap_floor": 1.0,
            "target_min_rebalance_threshold": 0.0,
        }
    )
    regime_result, regime_trades = run_default_strategy_with_params(
        prices,
        selected,
        params=regime_params,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        market_proxy=load_market_proxy_from_file(MARKET_VOLUME_CACHE_PATH),
        treasury_code=DEFAULT_STRESS_BOND_CODE,
        risk_cap=1.0,
        ratio_cut=DEFAULT_STRESS_BOND_RATIO_CUT,
        breadth_cut=DEFAULT_STRESS_BOND_BREADTH_CUT,
        enter_days=DEFAULT_STRESS_BOND_ENTER_DAYS,
        exit_days=DEFAULT_STRESS_BOND_EXIT_DAYS,
        fill_residual_cash_to_treasury=DEFAULT_STRESS_BOND_FILL_RESIDUAL_CASH,
    )
    regime_trigger_days = int(
        (
            (regime_result["target_exposure"] > 0)
            & (regime_result["target_exposure"] < 0.999999)
        ).sum()
    )
    rows.append(
        build_summary_row(
            version="regime_dualcore_5pct",
            result=regime_result,
            trades=regime_trades,
            selected=selected,
            trigger_days=regime_trigger_days,
        )
    )

    threshold_overlay_variants = [
        "threshold_weak_vol_scale",
        "threshold_panic_rebound_cap",
        "threshold_trend_consistency",
        "threshold_pct_5y_85_90_97",
        "threshold_pct_5y_90_95_99_soft",
        "threshold_pct_5y_88_93_98_soft",
        "threshold_rsi_hot_cap",
        "threshold_kdj_hot_cap",
        "threshold_boll_hot_cap",
        "threshold_rsi_kdj_boll_combo",
        "threshold_combo_guard",
    ]
    for variant_name in threshold_overlay_variants:
        target_weights, triggered = build_threshold_dual_overlay_variant(
            prices=prices,
            base_result=threshold_dual_result,
            base_target_weights=threshold_dual_target_weights,
            variant_name=variant_name,
        )
        result, trades = run_variant_from_target_weights(
            prices=prices,
            selected=selected,
            target_weights=target_weights,
            base_result=threshold_dual_result,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        rows.append(
            build_summary_row(
                version=variant_name,
                result=result,
                trades=trades,
                selected=selected,
                trigger_days=int(triggered.fillna(False).sum()),
            )
        )

    variant_names = [
        "vol_scale_20pct",
        "downside_vol_scale_20pct",
        "pct97_decay_rsi",
        "pct97_decay_quality",
        "pct97_99_floor70",
        "pct97_99_floor80",
        "boll_squeeze_breakout_confirm",
        "boll_rsi_rollover_cap",
        "state_squeeze_breakout_rollover",
    ]

    for variant_name in variant_names:
        target_weights, triggered = build_target_weights_variant(
            prices=prices,
            selected=selected,
            base_result=base_result,
            base_target_weights=base_target_weights,
            variant_name=variant_name,
        )
        result, trades = run_variant_from_target_weights(
            prices=prices,
            selected=selected,
            target_weights=target_weights,
            base_result=base_result,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        rows.append(
            build_summary_row(
                version=variant_name,
                result=result,
                trades=trades,
                selected=selected,
                trigger_days=int(triggered.fillna(False).sum()),
            )
        )

    market_proxy = load_market_proxy_from_file(MARKET_VOLUME_CACHE_PATH)
    v8_result, v8_trades, _ = run_historical_version_with_current_engine(
        prices,
        selected,
        "v8acc7b6",
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        market_proxy=market_proxy,
    )
    rows.append(
        build_summary_row(
            version="v8acc7b6_ref",
            result=v8_result,
            trades=v8_trades,
            selected=selected,
            trigger_days=0,
        )
    )

    summary_df = pd.DataFrame(rows).sort_values("annualized_return", ascending=False).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
