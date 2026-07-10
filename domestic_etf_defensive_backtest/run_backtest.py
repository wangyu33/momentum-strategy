#!/usr/bin/env python3
"""运行国内 ETF 防守模板回测，支持基础版与更保守版。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from momentum_backtest.runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from momentum_backtest.compare_goal_optimizations import load_market_volume_proxy
from momentum_backtest.compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy
from momentum_backtest.compare_market_proxy_variants import build_proxy_catalog
from momentum_backtest.run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    build_benchmark_nav,
    build_contribution_overview,
    build_contribution_summary,
    compute_signal_asset_momentum,
    configure_matplotlib,
    save_figure_atomic,
    save_contribution_chart,
    compute_signal_asset_momentum,
    build_persistent_trigger_mask,
    build_signal_quality_score,
    build_strategy_summary,
    build_top2_close_risk_flag,
    build_yearly_return_rows,
    choose_signal_winner_with_margin,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    resolve_strategy_universe,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("domestic_etf_defensive_backtest/output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行国内 ETF 防守模板回测。")
    parser.add_argument("--preset", choices=["all", "base", "conservative"], default="all", help="回测哪个模板。")
    parser.add_argument("--years", type=int, default=10, help="回测最近多少年。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取缺失历史数据与市场代理。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def build_rows() -> dict[str, dict[str, str]]:
    rows = {str(row["code"]): dict(row) for row in load_fixed_etf_pool().to_dict("records")}
    rows["511260"] = {
        "theme": "十年国债ETF",
        "code": "511260",
        "name": "十年国债ETF",
        "sina_symbol": "sh511260",
    }
    rows["CASH"] = {
        "theme": "现金替代",
        "code": "CASH",
        "name": "现金替代",
        "sina_symbol": "",
    }
    return rows


def build_presets() -> dict[str, dict[str, object]]:
    return {
        "base": {
            "description": "国内ETF防守基础版：日频信号，防守池承接，极弱市切十年国债。",
            "risk_codes": ["510300", "159949", "159954", "159941", "513880"],
            "defensive_codes": ["518880", "512890", "159985"],
            "lookback": 25,
            "signal_quality_method": "slope",
            "slope_penalty": 0.85,
            "leader_margin": 0.0,
            "absolute_threshold": 0.05,
            "weak_trend_defensive_weight": 0.80,
            "top2_close_gap": 0.005,
            "top2_risk_cap": 0.70,
            "weak_market": {
                "ratio_cut": 0.91,
                "short_ratio_cut": 0.90,
                "breadth_cut": -0.04,
                "momentum_ceiling": 0.16,
                "risk_cap": 0.0,
                "bucket_mode": "defensive_winner",
            },
            "stress": {
                "ratio_cut": 0.90,
                "breadth_cut": -0.031,
                "risk_cap": 0.0,
                "enter_days": 1,
                "exit_days": 1,
                "bucket": {"511260": 1.0},
            },
            "overheat": {
                "drawdown_cut": -0.02,
                "pre_start_cut": 0.15,
                "pre_end_cut": 0.20,
                "pre_end_exposure": 0.90,
                "overheat_cut": 0.25,
                "overheat_cap": 0.30,
                "high_cut": 0.30,
                "high_cap": 0.10,
            },
            "defensive_full_exposure_cap": 1.0,
            "residual_bucket": {},
            "rebalance_mode": "daily",
        },
        "conservative": {
            "description": "国内ETF更保守版：加入 511580、现金替代、周频调仓与更早压仓。",
            "risk_codes": ["510300", "159949", "159954", "159941", "513880"],
            "defensive_codes": ["518880", "512890", "159985", "511580"],
            "lookback": 25,
            "signal_quality_method": "slope",
            "slope_penalty": 0.85,
            "leader_margin": 0.003,
            "absolute_threshold": 0.06,
            "weak_trend_defensive_weight": 0.70,
            "top2_close_gap": 0.007,
            "top2_risk_cap": 0.50,
            "weak_market": {
                "ratio_cut": 0.93,
                "short_ratio_cut": 0.92,
                "breadth_cut": -0.02,
                "momentum_ceiling": 0.18,
                "risk_cap": 0.20,
                "bucket_mode": "conservative_weak_bucket",
                "bucket": {"511580": 0.6, "CASH": 0.4},
            },
            "stress": {
                "ratio_cut": 0.91,
                "breadth_cut": -0.02,
                "risk_cap": 0.0,
                "enter_days": 2,
                "exit_days": 2,
                "bucket": {"511260": 0.5, "511580": 0.3, "CASH": 0.2},
            },
            "overheat": {
                "drawdown_cut": -0.015,
                "pre_start_cut": 0.12,
                "pre_end_cut": 0.18,
                "pre_end_exposure": 0.80,
                "overheat_cut": 0.22,
                "overheat_cap": 0.20,
                "high_cut": 0.27,
                "high_cap": 0.08,
            },
            "defensive_full_exposure_cap": 0.90,
            "residual_bucket": {"CASH": 1.0},
            "rebalance_mode": "weekly",
        },
    }


def make_selected(rows: dict[str, dict[str, str]], preset_names: list[str], presets: dict[str, dict[str, object]]) -> pd.DataFrame:
    codes: list[str] = []
    for name in preset_names:
        preset = presets[name]
        codes.extend(str(code) for code in preset["risk_codes"])
        codes.extend(str(code) for code in preset["defensive_codes"])
        codes.extend(str(code) for code in preset["stress"].get("bucket", {}).keys())
        codes.extend(str(code) for code in preset["residual_bucket"].keys())
        weak_market = preset["weak_market"]
        codes.extend(str(code) for code in weak_market.get("bucket", {}).keys())
    seen = []
    for code in codes:
        if code not in seen:
            seen.append(code)
    return pd.DataFrame([rows[code] for code in seen])


def merge_price_panels(base_prices: pd.DataFrame, extra_prices: pd.DataFrame) -> pd.DataFrame:
    if extra_prices.empty:
        return base_prices
    merged = base_prices.join(extra_prices, how="outer", rsuffix="_extra")
    dup_cols = [col for col in merged.columns if col.endswith("_extra")]
    for dup in dup_cols:
        base_col = dup.removesuffix("_extra")
        merged[base_col] = merged[base_col].combine_first(merged[dup])
        merged = merged.drop(columns=[dup])
    return merged.sort_index().ffill()


def load_prices(selected: pd.DataFrame, years: int, refresh: bool) -> pd.DataFrame:
    _, cached_prices = load_cached_data()
    prices = cached_prices.copy()
    real_selected = selected[selected["code"] != "CASH"].copy()
    need_codes = [code for code in real_selected["code"].astype(str).tolist() if code not in prices.columns]
    if refresh or need_codes:
        extra_selected = real_selected[real_selected["code"].astype(str).isin(need_codes)] if need_codes else real_selected
        extra_prices = fetch_histories(extra_selected, years=years)
        prices = merge_price_panels(prices, extra_prices)

    start_ts = prices.index.max() - pd.DateOffset(years=years)
    prices = prices.loc[prices.index >= start_ts].copy()
    prices["CASH"] = 1.0

    missing = [code for code in real_selected["code"].astype(str) if code not in prices.columns]
    if missing:
        raise RuntimeError(f"missing required histories: {', '.join(missing)}")

    return prices[[code for code in selected["code"].astype(str) if code in prices.columns]].copy()


def load_proxy(prices: pd.DataFrame, years: int, refresh: bool, risk_codes: list[str]) -> pd.DataFrame:
    base_proxy = load_market_volume_proxy(years=years, refresh=refresh)
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(base_proxy, prices, risk_codes=risk_codes)
    }
    return proxy_catalog["hybrid_breadth_blend"].reindex(prices.index).ffill()


def build_rebalance_mask(index: pd.DatetimeIndex, mode: str) -> pd.Series:
    mask = pd.Series(True, index=index, dtype=bool)
    if mode == "weekly":
        week_end_dates = pd.Series(index=index, data=index).groupby(index.to_period("W-FRI")).max().tolist()
        mask[:] = index.isin(week_end_dates)
        if not mask.empty:
            mask.iloc[0] = True
    return mask


def apply_rebalance_mode(target_weights: pd.DataFrame, mode: str) -> pd.DataFrame:
    mask = build_rebalance_mask(target_weights.index, mode)
    adjusted = target_weights.where(mask, pd.NA).ffill().fillna(0.0)
    return adjusted.astype(float)


def build_signal_frame(prices: pd.DataFrame, preset: dict[str, object]) -> pd.DataFrame:
    momentum, score = build_signal_quality_score(
        prices,
        lookback=int(preset["lookback"]),
        method=str(preset["signal_quality_method"]),
        slope_penalty=float(preset["slope_penalty"]),
    )
    risk_codes, defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in preset["risk_codes"]],
        defensive_codes=[str(code) for code in preset["defensive_codes"]],
    )
    risk_score = score[risk_codes]
    defensive_score = score[defensive_codes]

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    risk_winner = pd.Series(index=prices.index, dtype="object", name="risk_winner")
    defensive_winner = pd.Series(index=prices.index, dtype="object", name="defensive_winner")
    risk_momentum = pd.Series(index=prices.index, dtype="float64", name="risk_momentum")
    defensive_momentum = pd.Series(index=prices.index, dtype="float64", name="defensive_momentum")

    prev_risk: str | None = None
    prev_def: str | None = None
    for dt_idx in prices.index:
        r_asset = choose_signal_winner_with_margin(risk_score.loc[dt_idx], prev_risk, float(preset["leader_margin"]))
        d_asset = choose_signal_winner_with_margin(defensive_score.loc[dt_idx], prev_def, 0.0)
        r_mom = float(momentum.loc[dt_idx, r_asset]) if r_asset and pd.notna(momentum.loc[dt_idx, r_asset]) else float("nan")
        d_mom = float(momentum.loc[dt_idx, d_asset]) if d_asset and pd.notna(momentum.loc[dt_idx, d_asset]) else float("nan")
        risk_winner.loc[dt_idx] = r_asset if r_asset else pd.NA
        defensive_winner.loc[dt_idx] = d_asset if d_asset else pd.NA
        risk_momentum.loc[dt_idx] = r_mom
        defensive_momentum.loc[dt_idx] = d_mom

        if pd.notna(r_mom) and r_mom > float(preset["absolute_threshold"]):
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = r_mom
        elif pd.notna(r_mom) and r_mom > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = float(preset["weak_trend_defensive_weight"])
            current_momentum.loc[dt_idx] = r_mom
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_mom) and d_mom > 0 else 0.0
            current_momentum.loc[dt_idx] = d_mom if pd.notna(d_mom) else float("nan")
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")

        prev_risk = str(r_asset) if r_asset else None
        prev_def = str(d_asset) if d_asset else None

    return pd.DataFrame(
        {
            "signal": signal,
            "target_exposure": target_exposure,
            "current_momentum": current_momentum,
            "risk_winner": risk_winner,
            "defensive_winner": defensive_winner,
            "risk_momentum": risk_momentum,
            "defensive_momentum": defensive_momentum,
        }
    )


def add_bucket_weights(target_weights: pd.DataFrame, date: pd.Timestamp, bucket: dict[str, float], extra_weight: float) -> None:
    if extra_weight <= 1e-12:
        return
    total = sum(float(v) for v in bucket.values())
    if total <= 0:
        return
    for code, ratio in bucket.items():
        target_weights.loc[date, str(code)] += extra_weight * float(ratio) / total


def build_target_weights(prices: pd.DataFrame, proxy: pd.DataFrame, preset: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = build_signal_frame(prices, preset)
    risk_codes = [str(code) for code in preset["risk_codes"]]
    defensive_codes = [str(code) for code in preset["defensive_codes"]]
    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for dt_idx in prices.index:
        signal = frame.loc[dt_idx, "signal"]
        exposure = float(frame.loc[dt_idx, "target_exposure"]) if pd.notna(frame.loc[dt_idx, "target_exposure"]) else 0.0
        if pd.notna(signal) and str(signal) in target_weights.columns and exposure > 0:
            target_weights.loc[dt_idx, str(signal)] = exposure

        if exposure < 1.0:
            add_bucket_weights(target_weights, dt_idx, dict(preset["residual_bucket"]), 1.0 - exposure)

        if pd.notna(signal) and str(signal) in defensive_codes:
            cap = float(preset.get("defensive_full_exposure_cap", 1.0))
            current = float(target_weights.loc[dt_idx, str(signal)])
            if current > cap:
                reduced = current - cap
                target_weights.loc[dt_idx, str(signal)] = cap
                add_bucket_weights(target_weights, dt_idx, dict(preset["residual_bucket"]), reduced)

    close_flag = build_top2_close_risk_flag(
        prices,
        lookback=int(preset["lookback"]),
        absolute_threshold=float(preset["absolute_threshold"]),
        signal_quality_method=str(preset["signal_quality_method"]),
        slope_penalty=float(preset["slope_penalty"]),
        risk_codes=risk_codes,
        close_gap=float(preset["top2_close_gap"]),
    )
    for dt_idx in prices.index[close_flag.fillna(False)]:
        risk_weight = float(target_weights.loc[dt_idx, risk_codes].sum())
        cap = float(preset["top2_risk_cap"])
        if risk_weight > cap > 0:
            scale = cap / risk_weight
            before = risk_weight
            target_weights.loc[dt_idx, risk_codes] = target_weights.loc[dt_idx, risk_codes] * scale
            add_bucket_weights(target_weights, dt_idx, dict(preset["residual_bucket"]), before - cap)

    weak_cfg = dict(preset["weak_market"])
    weak_mask = (
        (proxy["market_amount_ratio_20_60"] < float(weak_cfg["ratio_cut"]))
        & (proxy["market_amount_ratio_5_20"] < float(weak_cfg["short_ratio_cut"]))
        & (proxy["market_breadth_proxy"] < float(weak_cfg["breadth_cut"]))
        & (frame["current_momentum"] <= float(weak_cfg["momentum_ceiling"]))
    ).fillna(False)
    for dt_idx in prices.index[weak_mask]:
        risk_weight = float(target_weights.loc[dt_idx, risk_codes].sum())
        cap = float(weak_cfg["risk_cap"])
        if risk_weight > cap:
            scale = 0.0 if cap <= 0 else cap / risk_weight
            before = risk_weight
            target_weights.loc[dt_idx, risk_codes] = target_weights.loc[dt_idx, risk_codes] * scale
            released = before - cap
            if str(weak_cfg.get("bucket_mode")) == "defensive_winner":
                d_asset = frame.loc[dt_idx, "defensive_winner"]
                if pd.notna(d_asset):
                    target_weights.loc[dt_idx, str(d_asset)] += released
                else:
                    add_bucket_weights(target_weights, dt_idx, dict(preset["residual_bucket"]), released)
            else:
                add_bucket_weights(target_weights, dt_idx, dict(weak_cfg.get("bucket", {})), released)

    stress_cfg = dict(preset["stress"])
    stress_raw = (
        (proxy["market_amount_ratio_20_60"] < float(stress_cfg["ratio_cut"]))
        & (proxy["market_breadth_proxy"] < float(stress_cfg["breadth_cut"]))
    ).fillna(False)
    stress_mask = build_persistent_trigger_mask(
        stress_raw,
        enter_days=int(stress_cfg["enter_days"]),
        exit_days=int(stress_cfg["exit_days"]),
    )
    for dt_idx in prices.index[stress_mask]:
        risk_weight = float(target_weights.loc[dt_idx, risk_codes].sum())
        cap = float(stress_cfg["risk_cap"])
        if risk_weight > cap:
            scale = 0.0 if cap <= 0 else cap / risk_weight
            before = risk_weight
            target_weights.loc[dt_idx, risk_codes] = target_weights.loc[dt_idx, risk_codes] * scale
            released = before - cap
            add_bucket_weights(target_weights, dt_idx, dict(stress_cfg.get("bucket", {})), released)
        residual = max(0.0, 1.0 - float(target_weights.loc[dt_idx].sum()))
        add_bucket_weights(target_weights, dt_idx, dict(stress_cfg.get("bucket", {})), residual)

    base_result, _ = run_target_weights_strategy(
        prices,
        pd.DataFrame({"code": prices.columns, "theme": prices.columns, "name": prices.columns}),
        target_weights,
        frame["current_momentum"],
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
    )
    overheat_cfg = dict(preset["overheat"])
    for dt_idx in prices.index:
        risk_weight = float(target_weights.loc[dt_idx, risk_codes].sum())
        if risk_weight <= 0:
            continue
        current_mom = float(frame.loc[dt_idx, "current_momentum"]) if pd.notna(frame.loc[dt_idx, "current_momentum"]) else float("nan")
        drawdown = float(base_result.loc[dt_idx, "drawdown"]) if pd.notna(base_result.loc[dt_idx, "drawdown"]) else float("nan")
        if pd.isna(current_mom) or pd.isna(drawdown) or drawdown < float(overheat_cfg["drawdown_cut"]):
            continue

        cap = None
        if current_mom >= float(overheat_cfg["high_cut"]):
            cap = float(overheat_cfg["high_cap"])
        elif current_mom >= float(overheat_cfg["overheat_cut"]):
            cap = float(overheat_cfg["overheat_cap"])
        elif current_mom >= float(overheat_cfg["pre_start_cut"]):
            start = float(overheat_cfg["pre_start_cut"])
            end = float(overheat_cfg["pre_end_cut"])
            end_exposure = float(overheat_cfg["pre_end_exposure"])
            if end > start:
                progress = min(max((current_mom - start) / (end - start), 0.0), 1.0)
                cap = 1.0 + (end_exposure - 1.0) * progress
            else:
                cap = end_exposure

        if cap is not None and risk_weight > cap:
            scale = cap / risk_weight if cap > 0 else 0.0
            before = risk_weight
            target_weights.loc[dt_idx, risk_codes] = target_weights.loc[dt_idx, risk_codes] * scale
            add_bucket_weights(target_weights, dt_idx, dict(preset["residual_bucket"]), before - cap)

    target_weights = target_weights.clip(lower=0.0)
    row_sum = target_weights.sum(axis=1)
    over_alloc_mask = row_sum > 1.0 + 1e-12
    if over_alloc_mask.any():
        target_weights.loc[over_alloc_mask] = target_weights.loc[over_alloc_mask].div(row_sum.loc[over_alloc_mask], axis=0)

    frame["top2_close_risk_cap_triggered"] = close_flag.reindex(frame.index).fillna(False)
    frame["stress_bond_trigger"] = stress_mask.reindex(frame.index).fillna(False)
    frame["weak_market_trigger"] = weak_mask.reindex(frame.index).fillna(False)
    return target_weights, frame



def build_drawdown_episode_report(result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame, benchmark_nav: pd.Series) -> pd.DataFrame:
    drawdown = result["nav"] / result["nav"].cummax() - 1
    underwater = drawdown < 0
    segments = []
    in_segment = False
    start = None
    for dt_idx, is_underwater in underwater.items():
        if is_underwater and not in_segment:
            in_segment = True
            start = dt_idx
        elif not is_underwater and in_segment and start is not None:
            segments.append((start, dt_idx))
            in_segment = False
            start = None
    if in_segment and start is not None:
        segments.append((start, result.index[-1]))

    rows = []
    trade_dates = pd.to_datetime(trades["date"], errors="coerce") if not trades.empty and "date" in trades.columns else pd.Series(dtype="datetime64[ns]")
    for start_date, end_date in segments:
        segment_dd = drawdown.loc[start_date:end_date]
        trough_date = segment_dd.idxmin()
        peak_date = result.loc[:start_date, "nav"].idxmax()
        peak_nav = float(result.loc[peak_date, "nav"])
        trough_nav = float(result.loc[trough_date, "nav"])
        recovered = abs(float(drawdown.loc[end_date])) < 1e-12
        episode_trades = int(((trade_dates >= peak_date) & (trade_dates <= end_date)).sum()) if not trade_dates.empty else 0
        rows.append({
            "peak_date": peak_date.date().isoformat(),
            "start_date": start_date.date().isoformat(),
            "trough_date": trough_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "peak_to_trough_days": int(result.loc[peak_date:trough_date].shape[0] - 1),
            "recovery_days": int(result.loc[trough_date:end_date].shape[0] - 1) if recovered else "",
            "total_days": int(result.loc[peak_date:end_date].shape[0] - 1),
            "max_drawdown": float(segment_dd.min()),
            "strategy_peak_to_trough_return": float(trough_nav / peak_nav - 1),
            "hs300_peak_to_trough_return": float(benchmark_nav.loc[trough_date] / benchmark_nav.loc[peak_date] - 1),
            "peak_holding": str(result.loc[peak_date, "holding"]) if pd.notna(result.loc[peak_date, "holding"]) else "",
            "trough_holding": str(result.loc[trough_date, "holding"]) if pd.notna(result.loc[trough_date, "holding"]) else "",
            "recovery_holding": str(result.loc[end_date, "holding"]) if pd.notna(result.loc[end_date, "holding"]) else "",
            "trade_actions_in_episode": episode_trades,
            "recovered": recovered,
        })
    return pd.DataFrame(rows).sort_values("max_drawdown").reset_index(drop=True) if rows else pd.DataFrame()


def save_core_charts(output_dir: Path, result: pd.DataFrame, trades: pd.DataFrame, benchmark_nav: pd.Series) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result.index, result["nav"], color="#2b6cb0", linewidth=2.0, label="策略净值")
    if not trades.empty and "date" in trades.columns:
        trade_dates = pd.to_datetime(trades["date"], errors="coerce").dropna()
        trade_dates = trade_dates[trade_dates.isin(result.index)]
        if not trade_dates.empty:
            ax.scatter(trade_dates, result.loc[trade_dates, "nav"], s=12, color="#d64545", alpha=0.7, label="交易点")
    ax.set_title("策略净值与交易点", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure_atomic(fig, output_dir / "nav_with_trades.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result.index, result["current_momentum"], color="#805ad5", linewidth=1.8)
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("动量序列", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("动量")
    fig.tight_layout()
    save_figure_atomic(fig, output_dir / "momentum_series.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    drawdown = result["nav"] / result["nav"].cummax() - 1
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result.index, drawdown, color="#d64545", linewidth=1.8)
    ax.fill_between(result.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("最大回撤", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤")
    fig.tight_layout()
    save_figure_atomic(fig, output_dir / "max_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result.index, result["nav"], color="#2b6cb0", linewidth=2.0, label="策略")
    ax.plot(benchmark_nav.index, benchmark_nav, color="#718096", linewidth=1.8, label="沪深300ETF")
    ax.set_title("策略 vs 沪深300", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure_atomic(fig, output_dir / "strategy_vs_hs300.png", dpi=180, bbox_inches="tight")
    plt.close(fig)



def save_analysis_charts(analysis_dir: Path, result: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
    drawdown = result["nav"] / result["nav"].cummax() - 1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result.index, drawdown, color="#d64545", linewidth=1.8)
    ax.fill_between(result.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("历史回撤相对净值高点", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤")
    fig.tight_layout()
    save_figure_atomic(fig, analysis_dir / "historical_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(result.index, result["nav"], color="#2b6cb0", linewidth=2.0)
    ax1.set_title("策略净值与回撤", loc="left", fontsize=16, fontweight="bold")
    ax1.set_ylabel("净值")
    ax2.plot(result.index, drawdown, color="#d64545", linewidth=1.8)
    ax2.fill_between(result.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax2.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax2.set_xlabel("日期")
    ax2.set_ylabel("回撤")
    fig.tight_layout()
    save_figure_atomic(fig, analysis_dir / "nav_and_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)



def cleanup_legacy_flat_outputs(output_dir: Path) -> None:
    legacy_names = {
        "backtest_nav.csv",
        "drawdown_episodes.csv",
        "historical_nav.csv",
        "max_drawdown.png",
        "momentum_series.png",
        "nav_with_trades.png",
        "prices.csv",
        "selected_etfs.csv",
        "strategy_vs_hs300.csv",
        "strategy_vs_hs300.png",
        "summary.csv",
        "summary.json",
        "trades.csv",
        "yearly_returns.csv",
        "contribution_summary.csv",
        "contribution_overview.csv",
        "contribution_rate.png",
        "contribution_rate_cn.png",
    }
    for name in legacy_names:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def write_preset_outputs(
    output_dir: Path,
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    core_dir = output_dir / "core"
    analysis_dir = output_dir / "analysis"
    core_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    prices_to_write = prices.reset_index().rename(columns={"index": "date"})
    write_dataframe_csv_atomic(prices_to_write, core_dir / "prices.csv", index=False)
    write_dataframe_csv_atomic(result.reset_index().rename(columns={"index": "date"}), core_dir / "backtest_nav.csv", index=False)
    write_dataframe_csv_atomic(trades, core_dir / "trades.csv", index=False)
    write_dataframe_csv_atomic(selected, core_dir / "selected_etfs.csv", index=False)
    historical_nav = result[["nav"]].rename(columns={"nav": "historical_nav"}).reset_index().rename(columns={"index": "date"})
    write_dataframe_csv_atomic(historical_nav, core_dir / "historical_nav.csv", index=False)

    contribution_df = build_contribution_summary(prices, selected, result)
    contribution_overview = build_contribution_overview(result, contribution_df)
    write_dataframe_csv_atomic(contribution_df, core_dir / "contribution_summary.csv", index=False)
    write_dataframe_csv_atomic(contribution_overview, analysis_dir / "contribution_overview.csv", index=False)
    save_contribution_chart(contribution_df, core_dir / "contribution_rate.png")
    save_contribution_chart(contribution_df, core_dir / "contribution_rate_cn.png")

    benchmark_nav = build_benchmark_nav(prices)
    save_core_charts(core_dir, result, trades, benchmark_nav)
    save_analysis_charts(analysis_dir, result)
    drawdown_report = build_drawdown_episode_report(result, trades, selected, benchmark_nav)
    write_dataframe_csv_atomic(drawdown_report, analysis_dir / "drawdown_episodes.csv", index=False)
    compare = pd.DataFrame({"nav": result["nav"], "benchmark_nav": benchmark_nav})
    compare.index.name = "date"
    write_dataframe_csv_atomic(compare.reset_index(), core_dir / "strategy_vs_hs300.csv", index=False)
    compare_for_year = compare.copy()
    compare_for_year["benchmark_return"] = compare_for_year["benchmark_nav"].pct_change().fillna(0.0)
    yearly_rows = build_yearly_return_rows(
        compare_for_year,
        strategy_col="nav",
        benchmark_col="benchmark_nav",
        benchmark_return_col="benchmark_return",
    )
    write_dataframe_csv_atomic(pd.DataFrame(yearly_rows), core_dir / "yearly_returns.csv", index=False)
    write_dataframe_csv_atomic(pd.DataFrame([summary]), core_dir / "summary.csv", index=False)
    with open(core_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    cleanup_legacy_flat_outputs(output_dir)


def run_preset(
    preset_name: str,
    preset: dict[str, object],
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    target_weights, frame = build_target_weights(prices, proxy, preset)
    target_weights = apply_rebalance_mode(target_weights, str(preset["rebalance_mode"]))
    result, trades = run_target_weights_strategy(
        prices,
        selected,
        target_weights,
        frame["current_momentum"],
        fee_rate,
        slippage_rate,
    )
    result["holding_for_return"] = result["holding"]
    result["exposure_for_return"] = result["exposure"]
    result["effective_momentum"] = frame["risk_momentum"].reindex(result.index)
    result["signal_risk_winner"] = frame["risk_winner"].reindex(result.index)
    result["signal_defensive_winner"] = frame["defensive_winner"].reindex(result.index)
    result["signal_asset_momentum"] = compute_signal_asset_momentum(prices, result["signal"], lookback=int(preset["lookback"]))
    result["base_target_exposure"] = frame["target_exposure"].reindex(result.index)
    result["target_exposure"] = target_weights.sum(axis=1).reindex(result.index)
    result["top2_close_risk_cap_triggered"] = frame["top2_close_risk_cap_triggered"].reindex(result.index).fillna(False)
    result["top2_close_gap"] = float(preset["top2_close_gap"])
    result["top2_close_risk_cap"] = float(preset["top2_risk_cap"])
    result["stress_bond_trigger"] = frame["stress_bond_trigger"].reindex(result.index).fillna(False)
    result["weak_market_trigger"] = frame["weak_market_trigger"].reindex(result.index).fillna(False)
    weight_cols = [col for col in result.columns if col.startswith("weight_")]
    for col in weight_cols:
        result[f"return_{col}"] = result[col]
    result = pd.concat([result, target_weights.add_prefix("target_weight_")], axis=1)

    summary = build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)
    summary["preset"] = preset_name
    summary["description"] = str(preset["description"])
    summary["rebalance_mode"] = str(preset["rebalance_mode"])
    summary["absolute_threshold"] = float(preset["absolute_threshold"])
    summary["weak_trend_defensive_weight"] = float(preset["weak_trend_defensive_weight"])
    summary["top2_close_gap"] = float(preset["top2_close_gap"])
    summary["top2_risk_cap"] = float(preset["top2_risk_cap"])
    summary["defensive_full_exposure_cap"] = float(preset.get("defensive_full_exposure_cap", 1.0))
    summary["drawdown_integral_definition"] = "full_history"
    summary["drawdown_integral_recomputed"] = True
    return result, trades, summary


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    presets = build_presets()
    preset_names = [args.preset] if args.preset != "all" else ["base", "conservative"]
    rows = build_rows()
    selected_all = make_selected(rows, preset_names, presets)
    prices = load_prices(selected_all, years=args.years, refresh=args.refresh)

    risk_codes_union = []
    for name in preset_names:
        for code in presets[name]["risk_codes"]:
            if code not in risk_codes_union:
                risk_codes_union.append(code)
    proxy = load_proxy(prices, years=args.years, refresh=args.refresh, risk_codes=risk_codes_union)

    summary_rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame(index=prices.index)

    for name in preset_names:
        preset = presets[name]
        needed_codes = []
        for code in list(preset["risk_codes"]) + list(preset["defensive_codes"]):
            if str(code) not in needed_codes:
                needed_codes.append(str(code))
        for code in preset["stress"].get("bucket", {}).keys():
            if str(code) not in needed_codes:
                needed_codes.append(str(code))
        for code in preset["residual_bucket"].keys():
            if str(code) not in needed_codes:
                needed_codes.append(str(code))
        for code in preset["weak_market"].get("bucket", {}).keys():
            if str(code) not in needed_codes:
                needed_codes.append(str(code))

        selected = selected_all[selected_all["code"].astype(str).isin(needed_codes)].reset_index(drop=True)
        local_prices = prices[[code for code in needed_codes if code in prices.columns]].copy()
        result, trades, summary = run_preset(
            preset_name=name,
            preset=preset,
            selected=selected,
            prices=local_prices,
            proxy=proxy,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        summary_rows.append(summary)
        nav_compare[f"{name}_nav"] = result["nav"]
        write_preset_outputs(OUTPUT_DIR / name, selected, local_prices, result, trades, summary)

    write_dataframe_csv_atomic(pd.DataFrame(summary_rows), OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(nav_compare.reset_index().rename(columns={"index": "date"}), OUTPUT_DIR / "nav_compare.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
