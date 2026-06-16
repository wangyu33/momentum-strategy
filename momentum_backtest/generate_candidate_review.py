#!/usr/bin/env python3
"""为候选策略生成最近持仓/通知审阅材料。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from .official_candidate_runner import load_official_research_context
    from .monitor_snapshot import build_market_volume_proxy_for_monitor
    from .monitor_snapshot_builders import build_signal_snapshot
    from .monitor_render import format_market_message, format_trade_message
    from .daily_monitor import SignalSnapshot
    from .run_backtest import DEFAULT_STRATEGY_NAME
except ImportError:
    from official_candidate_runner import load_official_research_context
    from monitor_snapshot import build_market_volume_proxy_for_monitor
    from monitor_snapshot_builders import build_signal_snapshot
    from monitor_render import format_market_message, format_trade_message
    from daily_monitor import SignalSnapshot
    from run_backtest import DEFAULT_STRATEGY_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='生成候选策略最近持仓/通知审阅材料。')
    parser.add_argument('--candidate-name', required=True, help='候选名称，例如 rc_r95_g003_m10')
    parser.add_argument('--candidate-backtest', required=True, help='候选 backtest_nav.csv 路径')
    parser.add_argument('--candidate-trades', required=True, help='候选 trades.csv 路径')
    parser.add_argument('--out-dir', required=True, help='输出目录')
    parser.add_argument('--dates', default='2026-06-03,2026-06-04,2026-06-05', help='要模拟通知的日期列表')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_official_research_context()
    selected = ctx['selected']
    prices = ctx['prices']
    official_result = ctx['official_result']
    official_trades = ctx['official_trades']

    candidate_result = pd.read_csv(args.candidate_backtest, parse_dates=['date']).set_index('date')
    candidate_trades = pd.read_csv(args.candidate_trades)

    keep_dates = [d.strip() for d in args.dates.split(',') if d.strip()]
    compare_dates = sorted(set(keep_dates + ['2026-05-21','2026-05-22','2026-05-25','2026-05-26','2026-05-27','2026-05-28','2026-05-29','2026-06-01','2026-06-02']))
    base = official_result.reset_index()
    base = base[base['date'].astype(str).isin(compare_dates)][['date','signal','holding','current_momentum','effective_momentum','exposure','target_exposure','drawdown']].copy()
    base.columns = ['date'] + [f'official_{c}' for c in base.columns[1:]]

    cand = candidate_result.reset_index()
    cand = cand[cand['date'].astype(str).isin(compare_dates)][['date','signal','holding','current_momentum','effective_momentum','exposure','target_exposure','drawdown']].copy()
    cand.columns = ['date'] + [f'cand_{c}' for c in cand.columns[1:]]
    path_compare = base.merge(cand, on='date', how='outer').sort_values('date')
    path_compare.to_csv(out_dir / 'latest_path_compare.csv', index=False)

    def build_config(label: str, out_file: Path) -> dict[str, object]:
        return {
            'strategy_id': 'default',
            'strategy_label': label,
            'selected_pool': selected.to_dict('records'),
            'backtest_nav_file': str(out_file),
            'state_file': str(out_file.with_suffix('.state.json')),
            'extra_cap_label': '命中额外降仓',
        }

    rows = []
    markdown_sections = []
    for variant_name, result_df, trades_df, label_suffix in [
        ('official', official_result, official_trades, '__official'),
        (args.candidate_name, candidate_result, candidate_trades, f'__{args.candidate_name}'),
    ]:
        out_file = out_dir / f'{variant_name}_backtest_nav.csv'
        for d in keep_dates:
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
                strategy_config=build_config(DEFAULT_STRATEGY_NAME + label_suffix, out_file),
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
    msg_df.to_json(out_dir / 'official_vs_candidate_messages.json', orient='records', force_ascii=False, indent=2)

    markdown_sections.append(f'# 最近通知对照：official vs {args.candidate_name}\n')
    for d in keep_dates:
        markdown_sections.append(f'## {d}')
        for variant in ['official', args.candidate_name]:
            sub = msg_df[(msg_df['date'] == d) & (msg_df['variant'] == variant)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            markdown_sections.append(f'### {variant}')
            markdown_sections.append('```text')
            markdown_sections.append(row['market_message'])
            markdown_sections.append('')
            markdown_sections.append(row['trade_message'])
            markdown_sections.append('```\n')
    (out_dir / 'official_vs_candidate_messages.md').write_text('\n'.join(markdown_sections), encoding='utf-8')
    print('written', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
