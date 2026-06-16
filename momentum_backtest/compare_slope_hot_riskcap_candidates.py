#!/usr/bin/env python3
"""比较短期斜率过热风控候选。"""

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
    build_signal_quality_score,
    resolve_strategy_universe,
    write_dataframe_csv_atomic,
)
from official_candidate_runner import (
    extract_target_weights,
    load_official_research_context,
    run_candidate_from_weights,
    summarize_candidate,
)

OUTPUT_DIR = Path('momentum_backtest/output/research/slope_hot_riskcap_candidates')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='比较短期斜率过热风控候选。')
    parser.add_argument('--slope-hot-threshold', type=float, default=0.03, help='excess_slope 触发阈值，默认 3%。')
    return parser.parse_args()


def build_excess_slope_for_signal(prices: pd.DataFrame, signal: pd.Series, lookback: int, slope_penalty: float) -> pd.Series:
    raw_mom, _ = build_signal_quality_score(prices, lookback=lookback, method='slope', slope_penalty=slope_penalty)
    ret5 = prices / prices.shift(5) - 1
    excess = (ret5 - raw_mom / max(lookback / 5, 1)).clip(lower=0.0)
    vals = []
    for dt in signal.index:
        sig = signal.loc[dt]
        if pd.notna(sig) and sig in excess.columns and pd.notna(excess.loc[dt, sig]):
            vals.append(excess.loc[dt, sig])
        else:
            vals.append(float('nan'))
    return pd.Series(vals, index=signal.index, name='signal_excess_slope')


def apply_slope_hot_riskcap(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    official_result: pd.DataFrame,
    risk_codes: list[str],
    threshold: float,
    risk_cap: float,
    mom_gate: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    target_weights = extract_target_weights(official_result)
    effective_momentum = official_result['effective_momentum'] if 'effective_momentum' in official_result.columns else official_result['current_momentum']
    signal_excess_slope = build_excess_slope_for_signal(
        prices,
        official_result['signal'],
        lookback=25,
        slope_penalty=0.85,
    )
    risk_budget_codes = [c for c in risk_codes if c in target_weights.columns]
    if '510300' in target_weights.columns and '510300' not in risk_budget_codes:
        risk_budget_codes.append('510300')
    row_risk_weight = target_weights[risk_budget_codes].sum(axis=1)
    trigger = signal_excess_slope >= threshold
    if mom_gate is not None:
        trigger = trigger & (official_result['current_momentum'] > mom_gate)
    trigger = trigger.fillna(False)
    scale_mask = trigger & (row_risk_weight > risk_cap)
    adjusted = target_weights.copy()
    if scale_mask.any():
        scale = pd.Series(1.0, index=adjusted.index, dtype='float64')
        scale.loc[scale_mask] = risk_cap / row_risk_weight.loc[scale_mask]
        adjusted.loc[scale_mask, risk_budget_codes] = adjusted.loc[scale_mask, risk_budget_codes].mul(scale.loc[scale_mask], axis=0)

    result, trades = run_candidate_from_weights(prices, selected, adjusted, official_result)
    result['signal_excess_slope'] = signal_excess_slope
    result['slope_hot_triggered'] = trigger
    result['slope_hot_threshold'] = threshold
    result['slope_hot_risk_cap'] = risk_cap
    return result, trades, trigger


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    context = load_official_research_context()
    selected = context["selected"]
    prices = context["prices"]
    params = context["params"]
    official_result = context["official_result"]
    official_trades = context["official_trades"]
    active_risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(c) for c in params['risk_codes']],
        defensive_codes=[str(c) for c in params['defensive_codes']],
    )

    rows = []
    baseline = summarize_candidate(official_result, official_trades, selected, variant='baseline', extra_fields={'slope_hot_threshold': args.slope_hot_threshold, 'risk_cap': 1.0, 'mom_gate': None, 'trigger_days': 0})
    rows.append(baseline)

    variants = [
        ('slope_hot_cap80', 0.80, None),
        ('slope_hot_cap70', 0.70, None),
        ('slope_hot_cap70_only_mom15', 0.70, 0.15),
    ]
    for name, risk_cap, mom_gate in variants:
        result, trades, trigger = apply_slope_hot_riskcap(prices, selected, official_result, active_risk_codes, args.slope_hot_threshold, risk_cap, mom_gate)
        summary = summarize_candidate(result, trades, selected, variant=name, extra_fields={'slope_hot_threshold': args.slope_hot_threshold, 'risk_cap': risk_cap, 'mom_gate': mom_gate, 'trigger_days': int(trigger.sum())})
        rows.append(summary)
        out_dir = OUTPUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        write_dataframe_csv_atomic(result.reset_index(), out_dir / 'backtest_nav.csv', index=False)
        write_dataframe_csv_atomic(trades, out_dir / 'trades.csv', index=False)
        write_dataframe_csv_atomic(pd.DataFrame([summary]), out_dir / 'summary.csv', index=False)

    out = pd.DataFrame(rows)
    base_row = out.iloc[0]
    for col in ['annualized_return','sharpe_rf0','max_drawdown','max_drawdown_integral','trade_count','trigger_days']:
        out[f'{col}_diff_vs_base'] = out[col] - base_row[col]
    write_dataframe_csv_atomic(out, OUTPUT_DIR / 'summary.csv', index=False)
    keep = ['variant','slope_hot_threshold','risk_cap','mom_gate','trigger_days','annualized_return','sharpe_rf0','max_drawdown','max_drawdown_integral','trade_count','annualized_return_diff_vs_base','sharpe_rf0_diff_vs_base','max_drawdown_diff_vs_base','max_drawdown_integral_diff_vs_base','trade_count_diff_vs_base']
    print(out[keep].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
