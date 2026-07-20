#!/usr/bin/env python3
"""Focused comparison for China Internet and Germany ETF pool variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

try:
    from ...runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib

from archive_data_loaders import build_archive_flat_output_dir
from candidate_compare_helpers import (
    run_candidate_compare_entrypoint,
)
from candidate_pool_common import (
    BASE_DEFENSIVE_CODES,
    BASE_RISK_CODES,
    ETF_513030,
    ETF_513050,
    summarize,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    load_fixed_etf_pool,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("compare_china_internet_germany_variants")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare China Internet and Germany ETF variants.")
    parser.add_argument("--years", type=int, default=6, help="History years to request before overlap alignment.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Force full history fetch instead of preferring local caches.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    candidates = [
        ("base_pool", [], BASE_RISK_CODES, BASE_DEFENSIVE_CODES),
        ("china_internet_risk", [ETF_513050], BASE_RISK_CODES + ["513050"], BASE_DEFENSIVE_CODES),
        ("germany_risk", [ETF_513030], BASE_RISK_CODES + ["513030"], BASE_DEFENSIVE_CODES),
        ("germany_defensive", [ETF_513030], BASE_RISK_CODES, BASE_DEFENSIVE_CODES + ["513030"]),
        (
            "china_internet_plus_germany_risk",
            [ETF_513050, ETF_513030],
            BASE_RISK_CODES + ["513050", "513030"],
            BASE_DEFENSIVE_CODES,
        ),
        (
            "china_internet_plus_germany_defensive",
            [ETF_513050, ETF_513030],
            BASE_RISK_CODES + ["513050"],
            BASE_DEFENSIVE_CODES + ["513030"],
        ),
    ]

    return run_candidate_compare_entrypoint(
        args,
        base_pool=base_pool,
        candidates=candidates,
        output_dir=OUTPUT_DIR,
        name_column="variant",
        title="China Internet / Germany ETF Focused Variants",
        lines=[
            ("base_pool", "Base Pool", 2.3),
            ("china_internet_risk", "+ China Internet", 1.8),
            ("germany_risk", "+ Germany Risk", 1.8),
            ("germany_defensive", "+ Germany Defensive", 1.8),
            ("china_internet_plus_germany_risk", "+ Both as Risk", 1.8),
            ("china_internet_plus_germany_defensive", "+ CI Risk + Germany Def", 1.8),
        ],
        summarize_fn=summarize,
        use_official_baseline_anchor=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
