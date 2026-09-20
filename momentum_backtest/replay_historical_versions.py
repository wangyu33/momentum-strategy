#!/usr/bin/env python3
"""用当前重构引擎复刻历史策略版本并排名。"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
    from .core.config import DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE, OUTPUT_DIR
    from .core.data import load_core_selected_and_prices
    from .core.proxy import MARKET_VOLUME_CACHE_PATH
    from .core.version_replay import build_historical_version_ranking, save_historical_version_ranking
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports
    from core.config import DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE, OUTPUT_DIR
    from core.data import load_core_selected_and_prices
    from core.proxy import MARKET_VOLUME_CACHE_PATH
    from core.version_replay import build_historical_version_ranking, save_historical_version_ranking


prepare_local_imports(__file__)
configure_matplotlib_env()
warnings.filterwarnings("ignore")


DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "analysis" / "historical_version_replay.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复刻历史策略版本到当前重构引擎并输出收益排名。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument(
        "--market-proxy-file",
        type=Path,
        default=MARKET_VOLUME_CACHE_PATH,
        help="市场量能代理 CSV，默认读取 output/monitor/market_volume_proxy.csv。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="排名输出路径。",
    )
    return parser.parse_args()


def load_market_proxy_from_file(path: Path) -> pd.DataFrame:
    proxy = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    required_cols = {"market_amount_ratio_20_60", "market_amount_ratio_5_20", "market_breadth_proxy"}
    missing_cols = required_cols - set(proxy.columns)
    if missing_cols:
        raise ValueError(f"market proxy missing columns: {sorted(missing_cols)}")
    return proxy


def main() -> int:
    args = parse_args()
    selected, prices = load_core_selected_and_prices()
    market_proxy = load_market_proxy_from_file(args.market_proxy_file)
    ranking = build_historical_version_ranking(
        prices,
        selected,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        market_proxy=market_proxy,
    )
    save_historical_version_ranking(ranking, args.output)
    print(ranking.to_string(index=False))
    print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
