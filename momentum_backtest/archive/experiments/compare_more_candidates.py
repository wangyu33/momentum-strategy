#!/usr/bin/env python3
"""对比与当前池子重合度更低的一批额外 ETF 候选。"""

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
    ETF_159985,
    ETF_508000,
    ETF_510880_SH_DIVIDEND,
    ETF_515220,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    load_fixed_etf_pool,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("compare_more_candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比与当前池子重合度更低的一批额外 ETF 候选。")
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
        ("plus_sh_dividend", [ETF_510880_SH_DIVIDEND], BASE_RISK_CODES, BASE_DEFENSIVE_CODES + ["510880"]),
        ("plus_soymeal", [ETF_159985], BASE_RISK_CODES + ["159985"], BASE_DEFENSIVE_CODES),
        ("plus_coal", [ETF_515220], BASE_RISK_CODES + ["515220"], BASE_DEFENSIVE_CODES),
        ("plus_reits", [ETF_508000], BASE_RISK_CODES, BASE_DEFENSIVE_CODES + ["508000"]),
    ]

    return run_candidate_compare_entrypoint(
        args,
        base_pool=base_pool,
        candidates=candidates,
        output_dir=OUTPUT_DIR,
        name_column="pool",
        title="Additional ETF Candidate Comparison",
        lines=[
            ("base_pool", "Base Pool", 2.2),
            ("plus_sh_dividend", "+ SH Dividend", 1.8),
            ("plus_soymeal", "+ Soymeal", 1.8),
            ("plus_coal", "+ Coal", 1.8),
            ("plus_reits", "+ REITs", 1.8),
        ],
        use_official_baseline_anchor=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
