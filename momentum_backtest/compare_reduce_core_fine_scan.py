#!/usr/bin/env python3
"""围绕 reduce_core_when_leader_strong 做二轮精扫。"""

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

from run_backtest import build_signal_quality_score, choose_signal_winner_with_margin, resolve_strategy_universe, write_dataframe_csv_atomic
from official_candidate_runner import load_official_research_context, extract_target_weights, run_candidate_from_weights, summarize_candidate

OUTPUT_DIR = Path('momentum_backtest/output/research/reduce_core_fine_scan')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='围绕 reduce_core_when_leader_strong 做二轮精扫。')
    parser.add_argument('--release-ratios', type=str, default='0.3,0.5,0.7', help='释放 HS300 核心仓比例列表。')
    parser.add_argument('--leader-gap', type=float, default=0.01, help='主信号领先第二名的最小质量差。默认 1%。')
    parser.add_argument('--mom-threshold', type=float, default=0.10, help='主信号动量下限。默认 10%。')
    return parser.parse_args()


def parse_ratio_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(',') if x.strip()]
    if not vals:
        raise ValueError('empty release ratios')
    for v in vals:
        if not (0 < v < 1):
            raise ValueError(f'invalid release ratio {v}')
    return vals


def compute_risk_winner_and_scores(prices: pd.DataFrame, params: dict[str, object]) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    raw_mom, score = build_signal_quality_score(
        prices,
        lookback=25,
        method=str(params.get('signal_quality_method', 'raw')),
        slope_penalty=float(params.get('signal_slope_penalty', 0.0)),
    )
    risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(c) for c in params['risk_codes']],
        defensive_codes=[str(c) for c in params['defensive_codes']],
    )
    risk_score = score[risk_codes]
    risk_winner = pd.Series(index=prices.index, dtype='object')
    prev_risk = None
    for dt in prices.index:
        r = choose_signal_winner_with_margin(risk_score.loc[dt], prev_risk, float(params.get('signal_leader_margin', 0.0)))
        risk_winner.loc[dt] = r if r else pd.NA
        prev_risk = str(r) if r else None
    return risk_winner, risk_score, raw_mom


def apply_reduce_core_candidate(
    target_weights: pd.DataFrame,
    risk_winner: pd.Series,
    risk_score: pd.DataFrame,
    raw_mom: pd.DataFrame,
    params: dict[str, object],
    *,
    leader_gap: float,
    mom_threshold: float,
    release_ratio: float,
) -> tuple[pd.DataFrame, pd.Series]:
    adjusted = target_weights.copy()
    trigger = pd.Series(False, index=adjusted.index, dtype=bool)
    risk_codes = [str(c) for c in params['risk_codes'] if str(c) in risk_score.columns]

    for dt in adjusted.index:
        lead = risk_winner.loc[dt]
        if pd.isna(lead):
            continue
        lead = str(lead)
        if lead == '510300' or lead not in risk_codes or '510300' not in adjusted.columns:
            continue
        valid = risk_score.loc[dt, risk_codes].dropna().sort_values(ascending=False)
        if len(valid) < 2:
            continue
        gap = float(valid.iloc[0] - valid.iloc[1])
        lead_mom = float(raw_mom.loc[dt, lead]) if pd.notna(raw_mom.loc[dt, lead]) else float('nan')
        if pd.isna(lead_mom):
            continue
        if gap >= leader_gap and lead_mom >= mom_threshold:
            hs300_weight = float(adjusted.loc[dt, '510300'])
            if hs300_weight > 1e-12:
                released = hs300_weight * release_ratio
                adjusted.loc[dt, '510300'] = hs300_weight - released
                adjusted.loc[dt, lead] += released
                trigger.loc[dt] = True
    return adjusted, trigger


def main() -> int:
    args = parse_args()
    release_ratios = parse_ratio_list(args.release_ratios)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    context = load_official_research_context()
    selected = context['selected']
    prices = context['prices']
    params = context['params']
    official_result = context['official_result']
    official_trades = context['official_trades']
    target_weights = extract_target_weights(official_result)
    risk_winner, risk_score, raw_mom = compute_risk_winner_and_scores(prices, params)

    rows = []
    baseline = summarize_candidate(
        official_result,
        official_trades,
        selected,
        variant='official_baseline',
        extra_fields={'leader_gap': args.leader_gap, 'mom_threshold': args.mom_threshold, 'release_ratio': 0.0, 'trigger_days': 0},
    )
    rows.append(baseline)

    for release_ratio in release_ratios:
        weights, trigger = apply_reduce_core_candidate(
            target_weights,
            risk_winner,
            risk_score,
            raw_mom,
            params,
            leader_gap=float(args.leader_gap),
            mom_threshold=float(args.mom_threshold),
            release_ratio=float(release_ratio),
        )
        result, trades = run_candidate_from_weights(prices, selected, weights, official_result)
        summary = summarize_candidate(
            result,
            trades,
            selected,
            variant=f'reduce_core_r{int(round(release_ratio*100)):02d}_g{int(round(args.leader_gap*1000)):03d}_m{int(round(args.mom_threshold*100)):02d}',
            extra_fields={
                'leader_gap': float(args.leader_gap),
                'mom_threshold': float(args.mom_threshold),
                'release_ratio': float(release_ratio),
                'trigger_days': int(trigger.sum()),
            },
        )
        rows.append(summary)
        out_dir = OUTPUT_DIR / str(summary['variant'])
        out_dir.mkdir(parents=True, exist_ok=True)
        write_dataframe_csv_atomic(result.reset_index(), out_dir / 'backtest_nav.csv', index=False)
        write_dataframe_csv_atomic(trades, out_dir / 'trades.csv', index=False)
        write_dataframe_csv_atomic(pd.DataFrame([summary]), out_dir / 'summary.csv', index=False)

    out = pd.DataFrame(rows)
    base_row = out.iloc[0]
    for col in ['annualized_return','sharpe_rf0','max_drawdown','max_drawdown_integral','trade_count','trigger_days']:
        out[f'{col}_diff_vs_base'] = out[col] - base_row[col]
    write_dataframe_csv_atomic(out, OUTPUT_DIR / 'summary.csv', index=False)
    keep = ['variant','leader_gap','mom_threshold','release_ratio','trigger_days','annualized_return','sharpe_rf0','max_drawdown','max_drawdown_integral','trade_count','annualized_return_diff_vs_base','sharpe_rf0_diff_vs_base','max_drawdown_diff_vs_base','max_drawdown_integral_diff_vs_base','trade_count_diff_vs_base']
    print(out[keep].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
