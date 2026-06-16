#!/usr/bin/env python3
"""生成两个 reduce_core 候选与正式基线的并排审阅材料。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from .compare_reduce_core_fine_scan import compute_risk_winner_and_scores, apply_reduce_core_candidate
    from .official_candidate_runner import load_official_research_context, extract_target_weights, run_candidate_from_weights, summarize_candidate
    from .monitor_snapshot import build_market_volume_proxy_for_monitor
    from .monitor_snapshot_builders import build_signal_snapshot
    from .monitor_render import format_market_message, format_trade_message
    from .daily_monitor import SignalSnapshot
    from .run_backtest import DEFAULT_STRATEGY_NAME
except ImportError:
    from compare_reduce_core_fine_scan import compute_risk_winner_and_scores, apply_reduce_core_candidate
    from official_candidate_runner import load_official_research_context, extract_target_weights, run_candidate_from_weights, summarize_candidate
    from monitor_snapshot import build_market_volume_proxy_for_monitor
    from monitor_snapshot_builders import build_signal_snapshot
    from monitor_render import format_market_message, format_trade_message
    from daily_monitor import SignalSnapshot
    from run_backtest import DEFAULT_STRATEGY_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='生成两个 reduce_core 候选与正式基线的并排审阅材料。')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--dates', default='2026-06-03,2026-06-04,2026-06-05')
    parser.add_argument('--candidate-a', default='rc_r95_g003_m10')
    parser.add_argument('--a-gap', type=float, default=0.003)
    parser.add_argument('--a-mom', type=float, default=0.10)
    parser.add_argument('--a-release', type=float, default=0.95)
    parser.add_argument('--candidate-b', default='rc_r80_g003_m10')
    parser.add_argument('--b-gap', type=float, default=0.003)
    parser.add_argument('--b-mom', type=float, default=0.10)
    parser.add_argument('--b-release', type=float, default=0.80)
    return parser.parse_args()


def build_config(label: str, out_file: Path, selected: pd.DataFrame) -> dict[str, object]:
    return {
        'strategy_id': 'default',
        'strategy_label': label,
        'selected_pool': selected.to_dict('records'),
        'backtest_nav_file': str(out_file),
        'state_file': str(out_file.with_suffix('.state.json')),
        'extra_cap_label': '命中额外降仓',
    }


def build_candidate(context: dict[str, object], *, leader_gap: float, mom_threshold: float, release_ratio: float):
    selected = context['selected']
    prices = context['prices']
    params = context['params']
    official_result = context['official_result']
    target_weights = extract_target_weights(official_result)
    risk_winner, risk_score, raw_mom = compute_risk_winner_and_scores(prices, params)
    weights, trigger = apply_reduce_core_candidate(
        target_weights,
        risk_winner,
        risk_score,
        raw_mom,
        params,
        leader_gap=leader_gap,
        mom_threshold=mom_threshold,
        release_ratio=release_ratio,
    )
    result, trades = run_candidate_from_weights(prices, selected, weights, official_result)
    return result, trades, trigger


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = [d.strip() for d in args.dates.split(',') if d.strip()]

    context = load_official_research_context()
    selected = context['selected']
    prices = context['prices']
    official_result = context['official_result']
    official_trades = context['official_trades']

    cand_a_result, cand_a_trades, trig_a = build_candidate(context, leader_gap=args.a_gap, mom_threshold=args.a_mom, release_ratio=args.a_release)
    cand_b_result, cand_b_trades, trig_b = build_candidate(context, leader_gap=args.b_gap, mom_threshold=args.b_mom, release_ratio=args.b_release)

    # summaries
    summary_rows = [
        summarize_candidate(official_result, official_trades, selected, variant='official_baseline'),
        summarize_candidate(cand_a_result, cand_a_trades, selected, variant=args.candidate_a, extra_fields={'leader_gap': args.a_gap, 'mom_threshold': args.a_mom, 'release_ratio': args.a_release, 'trigger_days': int(trig_a.sum())}),
        summarize_candidate(cand_b_result, cand_b_trades, selected, variant=args.candidate_b, extra_fields={'leader_gap': args.b_gap, 'mom_threshold': args.b_mom, 'release_ratio': args.b_release, 'trigger_days': int(trig_b.sum())}),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / 'summary_compare.csv', index=False)

    # path compare
    compare_dates = sorted(set(dates + ['2026-05-21','2026-05-22','2026-05-25','2026-05-26','2026-05-27','2026-05-28','2026-05-29','2026-06-01','2026-06-02']))
    merged = pd.DataFrame({'date': pd.to_datetime(compare_dates)})
    for prefix, res, trig in [
        ('official', official_result, pd.Series(False, index=official_result.index)),
        ('a', cand_a_result, trig_a),
        ('b', cand_b_result, trig_b),
    ]:
        df = res.reset_index()
        df = df[df['date'].astype(str).isin(compare_dates)][['date','signal','holding','current_momentum','effective_momentum','exposure','target_exposure','drawdown']].copy()
        trig_dates = set(pd.Index(trig[trig].index).strftime('%Y-%m-%d').tolist())
        df[f'{prefix}_reduce_core_triggered'] = df['date'].astype(str).isin(trig_dates)
        df.columns = ['date'] + [f'{prefix}_{c}' for c in df.columns[1:]]
        merged = merged.merge(df, on='date', how='left')
    merged.to_csv(out_dir / 'latest_path_compare.csv', index=False)

    # messages
    rows = []
    for variant_name, result_df, trades_df in [
        ('official', official_result, official_trades),
        (args.candidate_a, cand_a_result, cand_a_trades),
        (args.candidate_b, cand_b_result, cand_b_trades),
    ]:
        out_file = out_dir / f'{variant_name}_backtest_nav.csv'
        for d in dates:
            ts = pd.Timestamp(d)
            prices_slice = prices.loc[prices.index <= ts].copy()
            result_slice = result_df.loc[result_df.index <= ts].copy()
            if result_slice.empty or prices_slice.empty:
                continue
            trades_slice = trades_df.loc[pd.to_datetime(trades_df['date']) <= ts].copy() if not trades_df.empty else trades_df.copy()
            result_slice.reset_index().to_csv(out_file, index=False)
            _, market_proxy_context = build_market_volume_proxy_for_monitor(prices_slice)
            now = datetime(ts.year, ts.month, ts.day, 16, 0, 0)
            snapshot = build_signal_snapshot(
                SignalSnapshot,
                prices=prices_slice,
                strategy_config=build_config(DEFAULT_STRATEGY_NAME + '__' + variant_name, out_file, selected),
                result=result_slice,
                close_result=result_slice,
                close_trades=trades_slice,
                base_target_exposure=result_slice['base_target_exposure'] if 'base_target_exposure' in result_slice.columns else result_slice['target_exposure'],
                market_proxy_context=market_proxy_context,
                spot_prices={},
                raw_closes={},
                now=now,
            )
            rows.append({'variant': variant_name, 'date': d, 'market_message': format_market_message(snapshot), 'trade_message': format_trade_message(snapshot)})
    msg_df = pd.DataFrame(rows)
    msg_df.to_json(out_dir / 'message_compare.json', orient='records', force_ascii=False, indent=2)

    md = ['# 候选并排审阅：official vs ' + args.candidate_a + ' vs ' + args.candidate_b, '']
    for d in dates:
        md.append(f'## {d}')
        for variant in ['official', args.candidate_a, args.candidate_b]:
            sub = msg_df[(msg_df['date'] == d) & (msg_df['variant'] == variant)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            md.append(f'### {variant}')
            md.append('```text')
            md.append(row['market_message'])
            md.append('')
            md.append(row['trade_message'])
            md.append('```')
            md.append('')
    (out_dir / 'message_compare.md').write_text('\n'.join(md), encoding='utf-8')

    print('written', out_dir)
    print(summary_df.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
