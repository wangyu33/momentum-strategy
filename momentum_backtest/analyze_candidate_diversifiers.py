#!/usr/bin/env python3
"""扫描哪些候选和当前高联动资产组的相关性更低。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import pandas as pd

try:
    from .candidate_pool_common import (
        ETF_159930,
        ETF_159985,
        ETF_164824,
        ETF_508000,
        ETF_510050,
        ETF_510230,
        ETF_510410,
        ETF_510500,
        ETF_510880,
        ETF_510900,
        ETF_511090,
        ETF_511260,
        ETF_511380,
        ETF_512100,
        ETF_512480,
        ETF_513030,
        ETF_513080,
        ETF_513180,
        ETF_513300,
        ETF_513400,
        ETF_513660,
        ETF_515220,
        ETF_515790,
        ETF_588000,
    )
except ImportError:
    from candidate_pool_common import (
        ETF_159930,
        ETF_159985,
        ETF_164824,
        ETF_508000,
        ETF_510050,
        ETF_510230,
        ETF_510410,
        ETF_510500,
        ETF_510880,
        ETF_510900,
        ETF_511090,
        ETF_511260,
        ETF_511380,
        ETF_512100,
        ETF_512480,
        ETF_513030,
        ETF_513080,
        ETF_513180,
        ETF_513300,
        ETF_513400,
        ETF_513660,
        ETF_515220,
        ETF_515790,
        ETF_588000,
    )
from run_backtest import fetch_histories


BASE_GROUP = [
    {"theme": "创业板50", "code": "159949", "name": "创业板50ETF华安", "sina_symbol": "sz159949"},
    {"theme": "纳指ETF", "code": "159941", "name": "纳指ETF广发", "sina_symbol": "sz159941"},
    {"theme": "日经ETF", "code": "513880", "name": "日经225ETF华安", "sina_symbol": "sh513880"},
]

CANDIDATES = [
    ETF_510410,
    ETF_159930,
    ETF_515220,
    ETF_159985,
    ETF_513030,
    ETF_513080,
    ETF_510900,
    ETF_513180,
    ETF_588000,
    ETF_512480,
    ETF_515790,
    ETF_510500,
    ETF_512100,
    ETF_513300,
    ETF_513660,
    ETF_508000,
    ETF_510880,
    ETF_511260,
    ETF_511090,
    ETF_511380,
    ETF_510050,
    ETF_510230,
    ETF_164824,
    ETF_513400,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描候选分散器。")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = pd.DataFrame(BASE_GROUP + CANDIDATES).drop_duplicates(subset=["code"]).reset_index(drop=True)
    prices = fetch_histories(selected, years=args.years).dropna(how="any")
    returns = prices.pct_change().dropna(how="any")

    base_codes = [item["code"] for item in BASE_GROUP]
    rows: list[dict[str, object]] = []
    for item in CANDIDATES:
        code = str(item["code"])
        pair_corrs = {}
        for base_code in base_codes:
            pair_corrs[base_code] = float(returns[code].corr(returns[base_code]))
        rows.append(
            {
                "code": code,
                "theme": item["theme"],
                "name": item["name"],
                "corr_cyb50": pair_corrs["159949"],
                "corr_nasdaq": pair_corrs["159941"],
                "corr_nikkei": pair_corrs["513880"],
                "avg_corr": sum(pair_corrs.values()) / len(pair_corrs),
                "max_corr": max(pair_corrs.values()),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["avg_corr", "max_corr", "code"], ascending=[True, True, True])
    print(frame.head(args.top).to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
