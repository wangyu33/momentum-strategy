#!/usr/bin/env python3
"""比较从 domestic_etf_defensive_backtest 吸收过来的结构性改进。"""

from __future__ import annotations

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
    choose_signal_winner_with_margin,
    resolve_strategy_universe,
    write_dataframe_csv_atomic,
)
from official_candidate_runner import (
    extract_target_weights,
    load_official_research_context,
    run_candidate_from_weights,
    summarize_candidate,
)

OUTPUT_DIR = Path('momentum_backtest/output/research/domestic_idea_adoptions')


def compute_winners(prices: pd.DataFrame, params: dict[str, object]) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    raw_mom, score = build_signal_quality_score(
        prices,
        lookback=25,
        method=str(params.get('signal_quality_method', 'raw')),
        slope_penalty=float(params.get('signal_slope_penalty', 0.0)),
    )
    risk_codes, defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=[str(c) for c in params['risk_codes']],
        defensive_codes=[str(c) for c in params['defensive_codes']],
    )
    risk_score = score[risk_codes]
    defensive_score = score[defensive_codes]
    risk_winner = pd.Series(index=prices.index, dtype='object')
    defensive_winner = pd.Series(index=prices.index, dtype='object')
    prev_risk = None
    prev_def = None
    for dt in prices.index:
        r = choose_signal_winner_with_margin(risk_score.loc[dt], prev_risk, float(params.get('signal_leader_margin', 0.0)))
        d = choose_signal_winner_with_margin(defensive_score.loc[dt], prev_def, 0.0)
        risk_winner.loc[dt] = r if r else pd.NA
        defensive_winner.loc[dt] = d if d else pd.NA
        prev_risk = str(r) if r else None
        prev_def = str(d) if d else None
    return risk_winner, defensive_winner, raw_mom, score


def apply_weak_guard_to_defensive_winner(
    target_weights: pd.DataFrame,
    defensive_winner: pd.Series,
    weak_mask: pd.Series,
    stress_mask: pd.Series,
    max_total_fill: float | None,
) -> pd.DataFrame:
    adjusted = target_weights.copy()
    effective_mask = weak_mask & (~stress_mask.fillna(False))
    for dt in adjusted.index[effective_mask.fillna(False)]:
        winner = defensive_winner.loc[dt]
        if pd.isna(winner) or winner not in adjusted.columns:
            continue
        row_sum = float(adjusted.loc[dt].sum())
        residual = max(0.0, 1.0 - row_sum)
        if residual <= 1e-12:
            continue
        if max_total_fill is None:
            fill = residual
        else:
            fill = max(0.0, min(residual, max_total_fill - row_sum))
        if fill > 1e-12:
            adjusted.loc[dt, str(winner)] += fill
    return adjusted


def apply_reduce_core_when_leader_strong(
    target_weights: pd.DataFrame,
    official_result: pd.DataFrame,
    risk_winner: pd.Series,
    score: pd.DataFrame,
    raw_mom: pd.DataFrame,
    params: dict[str, object],
) -> tuple[pd.DataFrame, pd.Series]:
    adjusted = target_weights.copy()
    trigger = pd.Series(False, index=adjusted.index, dtype=bool)
    risk_codes = [str(c) for c in params['risk_codes'] if str(c) in score.columns]
    for dt in adjusted.index:
        lead = risk_winner.loc[dt]
        if pd.isna(lead) or str(lead) == '510300' or str(lead) not in risk_codes:
            continue
        valid = score.loc[dt, risk_codes].dropna().sort_values(ascending=False)
        if len(valid) < 2:
            continue
        gap = float(valid.iloc[0] - valid.iloc[1])
        lead_mom = float(raw_mom.loc[dt, str(lead)]) if pd.notna(raw_mom.loc[dt, str(lead)]) else float('nan')
        if pd.isna(lead_mom):
            continue
        if gap >= 0.01 and lead_mom >= 0.10 and '510300' in adjusted.columns:
            hs300_weight = float(adjusted.loc[dt, '510300'])
            if hs300_weight > 1e-12:
                released = hs300_weight * 0.5
                adjusted.loc[dt, '510300'] = hs300_weight - released
                adjusted.loc[dt, str(lead)] += released
                trigger.loc[dt] = True
    return adjusted, trigger


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = load_official_research_context()
    selected = context["selected"]
    prices = context["prices"]
    proxy = context["effective_proxy"]
    params = context["params"]
    official_result = context["official_result"]
    official_trades = context["official_trades"]
    target_weights = extract_target_weights(official_result)
    risk_winner, defensive_winner, raw_mom, score = compute_winners(prices, params)

    weak_mask = official_result['weak_market_trigger'].fillna(False) if 'weak_market_trigger' in official_result.columns else pd.Series(False, index=official_result.index)
    stress_mask = official_result['stress_bond_trigger'].fillna(False) if 'stress_bond_trigger' in official_result.columns else pd.Series(False, index=official_result.index)

    rows = []
    base = summarize_candidate(official_result, official_trades, selected, variant="official_baseline", extra_fields={"trigger_days": 0})
    rows.append(base)

    variants = [
        ('weak_guard_to_defensive_winner', apply_weak_guard_to_defensive_winner(target_weights, defensive_winner, weak_mask, stress_mask, None), int((weak_mask & ~stress_mask).sum())),
        ('weak_guard_to_defensive_winner_cap70', apply_weak_guard_to_defensive_winner(target_weights, defensive_winner, weak_mask, stress_mask, 0.70), int((weak_mask & ~stress_mask).sum())),
    ]
    reduced_weights, leader_trigger = apply_reduce_core_when_leader_strong(target_weights, official_result, risk_winner, score, raw_mom, params)
    variants.append(('reduce_core_when_leader_strong', reduced_weights, int(leader_trigger.sum())))

    for name, weights, trigger_days in variants:
        result, trades = run_candidate_from_weights(prices, selected, weights, official_result)
        summary = summarize_candidate(result, trades, selected, variant=name, extra_fields={"trigger_days": trigger_days})
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
    keep = ['variant','trigger_days','annualized_return','sharpe_rf0','max_drawdown','max_drawdown_integral','trade_count','annualized_return_diff_vs_base','sharpe_rf0_diff_vs_base','max_drawdown_diff_vs_base','max_drawdown_integral_diff_vs_base','trade_count_diff_vs_base']
    print(out[keep].to_string(index=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
