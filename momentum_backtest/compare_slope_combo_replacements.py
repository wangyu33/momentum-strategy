#!/usr/bin/env python3
"""对比多周期斜率主导信号，作为当前正式基线的激进替换候选。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import pandas as pd

from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    choose_signal_winner_with_margin,
    resolve_strategy_universe,
    write_dataframe_csv_atomic,
)
from official_candidate_runner import load_official_research_context, summarize_candidate
from compare_goal_optimizations import build_parametrized_hs300_trend_filter
from compare_hs300_regime_fixes import run_target_weights_strategy

OUTPUT_DIR = Path('momentum_backtest/output/research/slope_combo_replacements')


VARIANTS = [
    {
        'name': 'slope_combo_25_10',
        'description': '0.7×25日斜率质量 + 0.3×10日斜率质量',
        'windows': [25, 10],
        'weights': [0.7, 0.3],
        'mode': 'slope',
        'slope_penalty': 0.85,
        'extra_hot_penalty': 0.0,
    },
    {
        'name': 'slope_combo_25_10_60',
        'description': '0.6×25日斜率质量 + 0.25×10日斜率质量 + 0.15×60日斜率质量',
        'windows': [25, 10, 60],
        'weights': [0.6, 0.25, 0.15],
        'mode': 'slope',
        'slope_penalty': 0.85,
        'extra_hot_penalty': 0.0,
    },
    {
        'name': 'slope_combo_25_10_hotpenalty',
        'description': '0.7×25日斜率质量 + 0.3×10日斜率质量，再叠加额外短期过热惩罚',
        'windows': [25, 10],
        'weights': [0.7, 0.3],
        'mode': 'slope',
        'slope_penalty': 0.85,
        'extra_hot_penalty': 0.50,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='对比多周期斜率主导替换候选。')
    parser.add_argument('--years', type=int, default=15)
    return parser.parse_args()


def short_window_for(window: int) -> int:
    if window <= 25:
        return 5
    if window <= 60:
        return 10
    return 20


def build_multihorizon_scores(
    prices: pd.DataFrame,
    windows: list[int],
    weights: list[float],
    *,
    mode: str,
    slope_penalty: float,
    extra_hot_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_sum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    score_sum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    valid_sum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for window, weight in zip(windows, weights):
        raw = prices / prices.shift(window) - 1
        score = raw.copy()
        if mode == 'slope':
            short_window = short_window_for(window)
            short_ret = prices / prices.shift(short_window) - 1
            excess_slope = (short_ret - raw / max(window / short_window, 1)).clip(lower=0.0)
            score = raw - slope_penalty * excess_slope
        usable = raw.notna().astype(float)
        raw_sum = raw_sum.add(raw.fillna(0.0) * weight, fill_value=0.0)
        score_sum = score_sum.add(score.fillna(0.0) * weight, fill_value=0.0)
        valid_sum = valid_sum.add(usable * weight, fill_value=0.0)

    raw_combo = raw_sum.divide(valid_sum.where(valid_sum > 0))
    score_combo = score_sum.divide(valid_sum.where(valid_sum > 0))
    if extra_hot_penalty > 0:
        raw25 = prices / prices.shift(25) - 1
        ret5 = prices / prices.shift(5) - 1
        excess_hot = (ret5 - raw25 / 5).clip(lower=0.0)
        score_combo = score_combo - extra_hot_penalty * excess_hot
    return raw_combo, score_combo


def build_variant_signal_frame(prices: pd.DataFrame, params: dict[str, object], variant: dict[str, object]) -> pd.DataFrame:
    raw_combo, score_combo = build_multihorizon_scores(
        prices,
        windows=[int(x) for x in variant['windows']],
        weights=[float(x) for x in variant['weights']],
        mode=str(variant['mode']),
        slope_penalty=float(variant['slope_penalty']),
        extra_hot_penalty=float(variant['extra_hot_penalty']),
    )
    risk_codes, defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params['risk_codes']],
        defensive_codes=[str(code) for code in params['defensive_codes']],
    )
    risk_score = score_combo[risk_codes]
    defensive_score = score_combo[defensive_codes]

    signal = pd.Series(index=prices.index, dtype='object', name='signal')
    target_exposure = pd.Series(index=prices.index, dtype='float64', name='target_exposure')
    current_momentum = pd.Series(index=prices.index, dtype='float64', name='current_momentum')
    risk_winner = pd.Series(index=prices.index, dtype='object', name='risk_winner')
    defensive_winner = pd.Series(index=prices.index, dtype='object', name='defensive_winner')
    prev_risk = None
    prev_def = None

    for dt_idx in prices.index:
        r_asset = choose_signal_winner_with_margin(risk_score.loc[dt_idx], prev_risk, float(params.get('signal_leader_margin', 0.0)))
        d_asset = choose_signal_winner_with_margin(defensive_score.loc[dt_idx], prev_def, 0.0)
        r_mom = float(raw_combo.loc[dt_idx, r_asset]) if r_asset and pd.notna(raw_combo.loc[dt_idx, r_asset]) else float('nan')
        d_mom = float(raw_combo.loc[dt_idx, d_asset]) if d_asset and pd.notna(raw_combo.loc[dt_idx, d_asset]) else float('nan')
        risk_winner.loc[dt_idx] = r_asset if r_asset else pd.NA
        defensive_winner.loc[dt_idx] = d_asset if d_asset else pd.NA
        if pd.notna(r_mom) and r_mom > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = r_mom
        elif pd.notna(r_mom) and r_mom > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT
            current_momentum.loc[dt_idx] = r_mom
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_mom) and d_mom > 0 else 0.0
            current_momentum.loc[dt_idx] = d_mom if pd.notna(d_mom) else float('nan')
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float('nan')
        prev_risk = str(r_asset) if r_asset else None
        prev_def = str(d_asset) if d_asset else None

    frame = pd.DataFrame({
        'signal': signal,
        'target_exposure': target_exposure,
        'current_momentum': current_momentum,
        'risk_winner': risk_winner,
        'defensive_winner': defensive_winner,
    })
    frame.attrs['raw_combo'] = raw_combo
    return frame


def run_variant(prices: pd.DataFrame, selected: pd.DataFrame, proxy: pd.DataFrame, params: dict[str, object], variant: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = build_variant_signal_frame(prices, params, variant)
    active_risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params['risk_codes']],
        defensive_codes=[str(code) for code in params['defensive_codes']],
    )
    variants_data = {}
    for key, core_weight, mom60_cut, mom120_cut, ma_window in [
        ('aggressive', float(params['aggressive_core_weight']), 0.05, 0.10, 90),
        ('conservative', float(params['conservative_core_weight']), 0.04, 0.10, 120),
    ]:
        local_frame = build_variant_signal_frame(prices, params, variant)
        trend_filter = build_parametrized_hs300_trend_filter(prices, mom60_cut=mom60_cut, mom120_cut=mom120_cut, ma_window=ma_window)
        hs300_mom60 = prices['510300'] / prices['510300'].shift(60) - 1
        dynamic_core_weight = trend_filter.astype(float) * core_weight
        satellite_scale = 1.0 - dynamic_core_weight
        tw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        tw['510300'] += dynamic_core_weight
        for code in prices.columns:
            mask = local_frame['signal'] == code
            if mask.any():
                tw.loc[mask, code] += satellite_scale.loc[mask] * local_frame['target_exposure'].loc[mask]
        blended_mom = (satellite_scale * local_frame['current_momentum'] + dynamic_core_weight * hs300_mom60).rename('current_momentum')
        variants_data[key] = (tw, blended_mom, local_frame, trend_filter)

    ag_tw, ag_mom, ag_frame, ag_trend = variants_data['aggressive']
    co_tw, co_mom, co_frame, _ = variants_data['conservative']
    aggressive_mask = ag_trend & (ag_mom >= float(params['regime_momentum_cut']))
    target_weights = co_tw.copy()
    target_weights.loc[aggressive_mask] = ag_tw.loc[aggressive_mask]
    mixed_momentum = co_mom.copy()
    mixed_momentum.loc[aggressive_mask] = ag_mom.loc[aggressive_mask]

    row_risk_weight = target_weights[active_risk_codes].sum(axis=1)
    volume_weak_mask = (
        (proxy['market_amount_ratio_20_60'] < float(params['volume_ratio_cut']))
        & (proxy['market_amount_ratio_5_20'] < float(params['volume_short_ratio_cut']))
        & (proxy['market_breadth_proxy'] < float(params['volume_breadth_cut']))
        & (mixed_momentum <= float(params['volume_guard_momentum_ceiling']))
    ).fillna(False)
    weak_cap = float(params['volume_guard_cap'])
    weak_scale_mask = volume_weak_mask & (row_risk_weight > weak_cap)
    if weak_scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype='float64')
        scale.loc[weak_scale_mask] = weak_cap / row_risk_weight.loc[weak_scale_mask]
        target_weights.loc[weak_scale_mask, active_risk_codes] = target_weights.loc[weak_scale_mask, active_risk_codes].mul(scale.loc[weak_scale_mask], axis=0)

    # reuse current stress-bond threshold style
    risk_budget_codes = list(active_risk_codes)
    if '510300' in prices.columns and '510300' not in risk_budget_codes:
        risk_budget_codes.append('510300')
    row_risk_weight = target_weights[risk_budget_codes].sum(axis=1)
    raw_trigger = ((proxy['market_amount_ratio_20_60'] < 0.90) & (proxy['market_breadth_proxy'] < -0.031)).fillna(False)
    stress_mask = raw_trigger  # keep close to official stress threshold without persistence coupling changes
    if stress_mask.any():
        treasury_code = '511260'
        moved = row_risk_weight.loc[stress_mask]
        target_weights.loc[stress_mask, risk_budget_codes] = 0.0
        target_weights.loc[stress_mask, treasury_code] = target_weights.loc[stress_mask, treasury_code].add(moved, fill_value=0.0)

    result, trades = run_target_weights_strategy(prices, selected, target_weights, mixed_momentum, 0.0003, 0.0002)
    result['signal'] = frame['signal']
    result['current_momentum'] = frame['current_momentum']
    result['effective_momentum'] = mixed_momentum
    result['target_exposure'] = target_weights.sum(axis=1)
    result['stress_bond_trigger'] = stress_mask.reindex(result.index).fillna(False)
    result = pd.concat([result, target_weights.add_prefix('target_weight_')], axis=1)
    return result, trades


def main() -> int:
    _ = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = load_official_research_context()
    selected = context['selected']
    prices = context['prices']
    proxy = context['effective_proxy']
    params = context['params']
    official_result = context['official_result']
    official_trades = context['official_trades']

    rows = []
    rows.append(summarize_candidate(official_result, official_trades, selected, variant='official_baseline', extra_fields={'description': '当前正式基线'}))

    for variant in VARIANTS:
        result, trades = run_variant(prices, selected, proxy, params, variant)
        summary = summarize_candidate(
            result,
            trades,
            selected,
            variant=variant['name'],
            extra_fields={
                'description': variant['description'],
                'windows': ','.join(str(x) for x in variant['windows']),
                'weights': ','.join(f"{float(x):.2f}" for x in variant['weights']),
                'mode': variant['mode'],
                'slope_penalty': variant['slope_penalty'],
                'extra_hot_penalty': variant['extra_hot_penalty'],
            },
        )
        rows.append(summary)
        out_dir = OUTPUT_DIR / variant['name']
        out_dir.mkdir(parents=True, exist_ok=True)
        write_dataframe_csv_atomic(result.reset_index(), out_dir / 'backtest_nav.csv', index=False)
        write_dataframe_csv_atomic(trades, out_dir / 'trades.csv', index=False)
        write_dataframe_csv_atomic(pd.DataFrame([summary]), out_dir / 'summary.csv', index=False)

    out = pd.DataFrame(rows)
    base_row = out.iloc[0]
    for col in ['annualized_return', 'sharpe_rf0', 'max_drawdown', 'max_drawdown_integral', 'trade_count']:
        out[f'{col}_diff_vs_base'] = out[col] - base_row[col]
    write_dataframe_csv_atomic(out, OUTPUT_DIR / 'summary.csv', index=False)
    keep = ['variant', 'annualized_return', 'sharpe_rf0', 'max_drawdown', 'max_drawdown_integral', 'trade_count', 'annualized_return_diff_vs_base', 'sharpe_rf0_diff_vs_base', 'max_drawdown_diff_vs_base', 'max_drawdown_integral_diff_vs_base', 'trade_count_diff_vs_base']
    print(out[keep].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
