#!/usr/bin/env python3
"""根据候选对比摘要生成决策矩阵。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='根据候选对比摘要生成决策矩阵。')
    parser.add_argument('--summary-csv', required=True)
    parser.add_argument('--out-md', required=True)
    return parser.parse_args()


def pct(v: float) -> str:
    return f'{float(v):.2%}'


def num(v: float, d: int = 3) -> str:
    sign = '+' if float(v) >= 0 else ''
    return f'{sign}{float(v):.{d}f}'


def classify(ann: float, shp: float, ddi: float) -> str:
    if ann > 0 and shp > 0 and ddi <= 0:
        return '严格占优'
    if ann > 0 and shp > 0 and ddi <= 2.0:
        return '高潜力主候选'
    if ann > 0 and shp > 0:
        return '收益优先候选'
    if ann > 0 or shp > 0:
        return '平衡备选'
    return '不优先'


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    if 'variant' not in df.columns:
        raise RuntimeError('summary csv missing variant column')
    base = df[df['variant'].str.contains('official_baseline|baseline', regex=True, na=False)].iloc[0]
    cand = df[df['variant'] != base['variant']].copy()
    rows = []
    for _, row in cand.iterrows():
        rows.append({
            'variant': row['variant'],
            'annualized_return_diff_vs_base': row.get('annualized_return_diff_vs_base', row['annualized_return'] - base['annualized_return']),
            'sharpe_rf0_diff_vs_base': row.get('sharpe_rf0_diff_vs_base', row['sharpe_rf0'] - base['sharpe_rf0']),
            'max_drawdown_diff_vs_base': row.get('max_drawdown_diff_vs_base', row['max_drawdown'] - base['max_drawdown']),
            'max_drawdown_integral_diff_vs_base': row.get('max_drawdown_integral_diff_vs_base', row['max_drawdown_integral'] - base['max_drawdown_integral']),
            'trade_count_diff_vs_base': row.get('trade_count_diff_vs_base', row['trade_count'] - base['trade_count']),
            'label': classify(
                float(row.get('annualized_return_diff_vs_base', row['annualized_return'] - base['annualized_return'])),
                float(row.get('sharpe_rf0_diff_vs_base', row['sharpe_rf0'] - base['sharpe_rf0'])),
                float(row.get('max_drawdown_integral_diff_vs_base', row['max_drawdown_integral'] - base['max_drawdown_integral'])),
            ),
        })
    out = pd.DataFrame(rows).sort_values(['label','annualized_return_diff_vs_base','sharpe_rf0_diff_vs_base'], ascending=[True,False,False])
    md = ['# 候选决策矩阵', '', '## 基线', '']
    md.append(f"- {base['variant']}: 年化 {pct(base['annualized_return'])} | Sharpe {float(base['sharpe_rf0']):.3f} | 最大回撤 {pct(base['max_drawdown'])} | 回撤积分 {float(base['max_drawdown_integral']):.3f}")
    md.append('')
    md.append('## 候选分层')
    md.append('')
    for label in ['严格占优','高潜力主候选','收益优先候选','平衡备选','不优先']:
        sub = out[out['label'] == label]
        if sub.empty:
            continue
        md.append(f'### {label}')
        md.append('')
        md.append(sub.to_markdown(index=False))
        md.append('')
    Path(args.out_md).write_text('\n'.join(md), encoding='utf-8')
    print('written', args.out_md)
    print(out.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
