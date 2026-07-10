#!/usr/bin/env python3
"""研究多周期与多周期斜率信号在国内 ETF 防守框架下的表现。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from domestic_etf_defensive_backtest.run_backtest import (
    OUTPUT_DIR,
    add_bucket_weights,
    apply_rebalance_mode,
    build_drawdown_episode_report,
    build_presets,
    build_rows,
    cleanup_legacy_flat_outputs,
    load_prices,
    load_proxy,
    save_analysis_charts,
    save_core_charts,
)
from momentum_backtest.compare_hs300_regime_fixes import run_target_weights_strategy
from momentum_backtest.run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    build_benchmark_nav,
    build_contribution_overview,
    build_contribution_summary,
    build_persistent_trigger_mask,
    build_strategy_summary,
    choose_signal_winner_with_margin,
    ensure_output_dirs,
    save_contribution_chart,
    write_dataframe_csv_atomic,
)

RESEARCH_DIR = OUTPUT_DIR / "research" / "multihorizon_slope"


VARIANT_SPECS: list[dict[str, object]] = [
    {
        "name": "single_25_slope085",
        "description": "单周期 25 日动量 + slope_085，作为当前基线。",
        "windows": [25],
        "weights": [1.0],
        "mode": "slope",
        "slope_penalty": 0.85,
    },
    {
        "name": "mh_raw_15_25_60",
        "description": "多周期原始动量：15/25/60 加权组合。",
        "windows": [15, 25, 60],
        "weights": [0.25, 0.50, 0.25],
        "mode": "raw",
        "slope_penalty": 0.0,
    },
    {
        "name": "mh_slope_15_25_60_sp085",
        "description": "多周期斜率：15/25/60 各自做 slope 惩罚后加权。",
        "windows": [15, 25, 60],
        "weights": [0.25, 0.50, 0.25],
        "mode": "slope",
        "slope_penalty": 0.85,
    },
    {
        "name": "mh_raw_20_25_60",
        "description": "多周期原始动量：20/25/60 加权组合。",
        "windows": [20, 25, 60],
        "weights": [0.25, 0.40, 0.35],
        "mode": "raw",
        "slope_penalty": 0.0,
    },
    {
        "name": "mh_slope_20_25_60_sp085",
        "description": "多周期斜率：20/25/60 各自做 slope 惩罚后加权。",
        "windows": [20, 25, 60],
        "weights": [0.25, 0.40, 0.35],
        "mode": "slope",
        "slope_penalty": 0.85,
    },
    {
        "name": "mh_slope_25_60_120_sp085",
        "description": "多周期斜率：25/60/120，偏中长趋势。",
        "windows": [25, 60, 120],
        "weights": [0.50, 0.30, 0.20],
        "mode": "slope",
        "slope_penalty": 0.85,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="研究多周期与多周期斜率信号。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取缺失历史数据。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def short_window_for(window: int) -> int:
    if window <= 25:
        return 5
    if window <= 60:
        return 10
    return 20


def build_multihorizon_frames(
    prices: pd.DataFrame,
    windows: list[int],
    weights: list[float],
    *,
    mode: str,
    slope_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_sum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    score_sum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    valid_sum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for window, weight in zip(windows, weights):
        raw = prices / prices.shift(window) - 1
        if mode == "slope":
            short_window = short_window_for(window)
            short_ret = prices / prices.shift(short_window) - 1
            excess_slope = (short_ret - raw / max(window / short_window, 1)).clip(lower=0.0)
            score = raw - slope_penalty * excess_slope
        else:
            score = raw.copy()
        usable = raw.notna().astype(float)
        raw_sum = raw_sum.add(raw.fillna(0.0) * weight, fill_value=0.0)
        score_sum = score_sum.add(score.fillna(0.0) * weight, fill_value=0.0)
        valid_sum = valid_sum.add(usable * weight, fill_value=0.0)

    raw_combo = raw_sum.divide(valid_sum.where(valid_sum > 0))
    score_combo = score_sum.divide(valid_sum.where(valid_sum > 0))
    return raw_combo, score_combo


def build_variant_frame(prices: pd.DataFrame, preset: dict[str, object], variant: dict[str, object]) -> pd.DataFrame:
    raw_combo, score_combo = build_multihorizon_frames(
        prices,
        windows=[int(x) for x in variant["windows"]],
        weights=[float(x) for x in variant["weights"]],
        mode=str(variant["mode"]),
        slope_penalty=float(variant["slope_penalty"]),
    )
    risk_codes = [str(code) for code in preset["risk_codes"] if str(code) in prices.columns]
    defensive_codes = [str(code) for code in preset["defensive_codes"] if str(code) in prices.columns]
    risk_score = score_combo[risk_codes]
    defensive_score = score_combo[defensive_codes]

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    risk_winner = pd.Series(index=prices.index, dtype="object", name="risk_winner")
    defensive_winner = pd.Series(index=prices.index, dtype="object", name="defensive_winner")
    risk_momentum = pd.Series(index=prices.index, dtype="float64", name="risk_momentum")
    defensive_momentum = pd.Series(index=prices.index, dtype="float64", name="defensive_momentum")

    prev_risk: str | None = None
    prev_def: str | None = None
    leader_margin = float(preset["leader_margin"])
    absolute_threshold = float(preset["absolute_threshold"])
    weak_weight = float(preset["weak_trend_defensive_weight"])

    for dt_idx in prices.index:
        r_asset = choose_signal_winner_with_margin(risk_score.loc[dt_idx], prev_risk, leader_margin)
        d_asset = choose_signal_winner_with_margin(defensive_score.loc[dt_idx], prev_def, 0.0)
        r_mom = float(raw_combo.loc[dt_idx, r_asset]) if r_asset and pd.notna(raw_combo.loc[dt_idx, r_asset]) else float("nan")
        d_mom = float(raw_combo.loc[dt_idx, d_asset]) if d_asset and pd.notna(raw_combo.loc[dt_idx, d_asset]) else float("nan")
        risk_winner.loc[dt_idx] = r_asset if r_asset else pd.NA
        defensive_winner.loc[dt_idx] = d_asset if d_asset else pd.NA
        risk_momentum.loc[dt_idx] = r_mom
        defensive_momentum.loc[dt_idx] = d_mom

        if pd.notna(r_mom) and r_mom > absolute_threshold:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = r_mom
        elif pd.notna(r_mom) and r_mom > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = weak_weight
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


def build_top2_close_flag_from_frame(frame: pd.DataFrame, score_combo: pd.DataFrame, preset: dict[str, object]) -> pd.Series:
    risk_codes = [str(code) for code in preset["risk_codes"] if str(code) in score_combo.columns]
    close_gap = float(preset["top2_close_gap"])
    absolute_threshold = float(preset["absolute_threshold"])
    close_flag = pd.Series(False, index=score_combo.index, dtype=bool)
    for dt_idx in score_combo.index:
        valid = score_combo.loc[dt_idx, risk_codes].dropna().sort_values(ascending=False)
        if len(valid) < 2:
            continue
        top_asset = str(valid.index[0])
        second_asset = str(valid.index[1])
        top_mom = frame.loc[dt_idx, "risk_momentum"] if frame.loc[dt_idx, "risk_winner"] == top_asset else score_combo.loc[dt_idx, top_asset]
        second_mom = score_combo.loc[dt_idx, second_asset]
        if pd.isna(top_mom) or pd.isna(second_mom):
            continue
        if float(top_mom) <= absolute_threshold or float(second_mom) <= absolute_threshold:
            continue
        if float(valid.iloc[0] - valid.iloc[1]) <= close_gap:
            close_flag.loc[dt_idx] = True
    return close_flag


def build_target_weights_for_variant(prices: pd.DataFrame, proxy: pd.DataFrame, preset: dict[str, object], variant: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_combo, score_combo = build_multihorizon_frames(
        prices,
        windows=[int(x) for x in variant["windows"]],
        weights=[float(x) for x in variant["weights"]],
        mode=str(variant["mode"]),
        slope_penalty=float(variant["slope_penalty"]),
    )
    frame = build_variant_frame(prices, preset, variant)
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

    close_flag = build_top2_close_flag_from_frame(frame, score_combo, preset)
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


def write_variant_outputs(output_dir: Path, selected: pd.DataFrame, prices: pd.DataFrame, result: pd.DataFrame, trades: pd.DataFrame, summary: dict[str, object]) -> None:
    core_dir = output_dir / "core"
    analysis_dir = output_dir / "analysis"
    core_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(prices.reset_index().rename(columns={"index": "date"}), core_dir / "prices.csv", index=False)
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
    from momentum_backtest.run_backtest import build_yearly_return_rows
    yearly_rows = build_yearly_return_rows(compare_for_year, strategy_col="nav", benchmark_col="benchmark_nav", benchmark_return_col="benchmark_return")
    write_dataframe_csv_atomic(pd.DataFrame(yearly_rows), core_dir / "yearly_returns.csv", index=False)
    write_dataframe_csv_atomic(pd.DataFrame([summary]), core_dir / "summary.csv", index=False)
    with open(core_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    cleanup_legacy_flat_outputs(output_dir)


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    presets = build_presets()
    base_preset = dict(presets["base"])
    selected = pd.DataFrame([rows[code] for code in list(dict.fromkeys(base_preset["risk_codes"] + base_preset["defensive_codes"] + ["511260"]))])
    prices = load_prices(selected, years=args.years, refresh=args.refresh)
    proxy = load_proxy(prices, years=args.years, refresh=args.refresh, risk_codes=list(base_preset["risk_codes"]))

    nav_compare = pd.DataFrame(index=prices.index)
    summary_rows: list[dict[str, object]] = []

    for variant in VARIANT_SPECS:
        target_weights, frame = build_target_weights_for_variant(prices, proxy, base_preset, variant)
        target_weights = apply_rebalance_mode(target_weights, str(base_preset["rebalance_mode"]))
        result, trades = run_target_weights_strategy(
            prices,
            selected,
            target_weights,
            frame["current_momentum"],
            args.fee_rate,
            args.slippage_rate,
        )
        result["holding_for_return"] = result["holding"]
        result["exposure_for_return"] = result["exposure"]
        result["effective_momentum"] = frame["risk_momentum"].reindex(result.index)
        result["signal_risk_winner"] = frame["risk_winner"].reindex(result.index)
        result["signal_defensive_winner"] = frame["defensive_winner"].reindex(result.index)
        result["signal_asset_momentum"] = frame["current_momentum"].reindex(result.index)
        result["base_target_exposure"] = frame["target_exposure"].reindex(result.index)
        result["target_exposure"] = target_weights.sum(axis=1).reindex(result.index)
        result["top2_close_risk_cap_triggered"] = frame["top2_close_risk_cap_triggered"].reindex(result.index).fillna(False)
        result["top2_close_gap"] = float(base_preset["top2_close_gap"])
        result["top2_close_risk_cap"] = float(base_preset["top2_risk_cap"])
        result["stress_bond_trigger"] = frame["stress_bond_trigger"].reindex(result.index).fillna(False)
        result["weak_market_trigger"] = frame["weak_market_trigger"].reindex(result.index).fillna(False)
        for col in [c for c in result.columns if c.startswith("weight_")]:
            result[f"return_{col}"] = result[col]
        result = pd.concat([result, target_weights.add_prefix("target_weight_")], axis=1)

        summary = build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)
        summary["strategy"] = str(variant["name"])
        summary["description"] = str(variant["description"])
        summary["windows"] = json.dumps(variant["windows"], ensure_ascii=False)
        summary["weights"] = json.dumps(variant["weights"], ensure_ascii=False)
        summary["mode"] = str(variant["mode"])
        summary["slope_penalty"] = float(variant["slope_penalty"])
        summary["drawdown_integral_definition"] = "full_history"
        summary["drawdown_integral_recomputed"] = True
        summary_rows.append(summary)
        nav_compare[f"{variant['name']}_nav"] = result["nav"]
        write_variant_outputs(RESEARCH_DIR / str(variant["name"]), selected, prices, result, trades, summary)

    summary_df = pd.DataFrame(summary_rows)
    baseline = summary_df[summary_df["strategy"] == "single_25_slope085"].iloc[0]
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(baseline["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(baseline["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(baseline["max_drawdown_integral"])
    summary_df["max_drawdown_diff"] = summary_df["max_drawdown"] - float(baseline["max_drawdown"])
    write_dataframe_csv_atomic(summary_df.sort_values(["annualized_return", "sharpe_rf0"], ascending=False), RESEARCH_DIR / "summary.csv", index=False)
    nav_compare.index.name = "date"
    write_dataframe_csv_atomic(nav_compare.reset_index(), RESEARCH_DIR / "nav_compare.csv", index=False)

    top = summary_df.sort_values(["annualized_return", "sharpe_rf0"], ascending=False).head(3)
    lines = [
        "# 多周期 / 多周期斜率研究结论",
        "",
        "基线：single_25_slope085",
        "",
        "Top 3（按年化收益优先，Sharpe 次优先）：",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"- {row.strategy}: 年化 {row.annualized_return:.2%} | Sharpe {row.sharpe_rf0:.3f} | 最大回撤 {row.max_drawdown:.2%} | 全历史回撤积分 {row.max_drawdown_integral:.2f}"
        )
    (RESEARCH_DIR / "conclusion.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
