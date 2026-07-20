#!/usr/bin/env python3
"""对比能源、资源和海外收益类 ETF 候选。"""

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
    ETF_159930,
    ETF_510410,
    ETF_513300,
    ETF_513660,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    load_fixed_etf_pool,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("compare_energy_income_candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比能源、资源和海外收益类 ETF 候选。")
    parser.add_argument("--years", type=int, default=6, help="向前抓取多少年历史数据，再对齐公共区间。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="强制重新抓取整池历史，而不是优先复用本地缓存。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    candidates = [
        ("base_pool", [], BASE_RISK_CODES, BASE_DEFENSIVE_CODES),
        ("plus_energy", [ETF_159930], BASE_RISK_CODES + ["159930"], BASE_DEFENSIVE_CODES),
        ("plus_resources", [ETF_510410], BASE_RISK_CODES + ["510410"], BASE_DEFENSIVE_CODES),
        ("plus_hk_div_lowvol", [ETF_513660], BASE_RISK_CODES, BASE_DEFENSIVE_CODES + ["513660"]),
        ("plus_global_dividend", [ETF_513300], BASE_RISK_CODES, BASE_DEFENSIVE_CODES + ["513300"]),
        (
            "plus_energy_hk_div",
            [ETF_159930, ETF_513660],
            BASE_RISK_CODES + ["159930"],
            BASE_DEFENSIVE_CODES + ["513660"],
        ),
    ]

    return run_candidate_compare_entrypoint(
        args,
        base_pool=base_pool,
        candidates=candidates,
        output_dir=OUTPUT_DIR,
        name_column="pool",
        title="Energy / Income ETF Candidate Comparison",
        lines=[
            ("base_pool", "Base Pool", 2.2),
            ("plus_energy", "+ Energy", 1.8),
            ("plus_resources", "+ Resources", 1.8),
            ("plus_hk_div_lowvol", "+ HK Div LowVol", 1.8),
            ("plus_global_dividend", "+ Global Dividend", 1.8),
            ("plus_energy_hk_div", "+ Energy + HK Div", 1.8),
        ],
        use_official_baseline_anchor=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
