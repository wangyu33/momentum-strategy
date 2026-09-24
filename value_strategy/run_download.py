"""下载原始数据到本地缓存。

用法：
    python run_download.py                 # 全量下载中证800
    python run_download.py --limit 20      # 先跑小样本验证链路
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config as cfg  # noqa: E402
from core import data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只下载前 N 只（调试用）")
    ap.add_argument("--force", action="store_true", help="忽略已有缓存重新下载")
    args = ap.parse_args()

    pool = data.fetch_pool_codes()
    codes = pool["code"].tolist()
    if args.limit:
        codes = codes[: args.limit]
    print(f"目标股票池：{cfg.POOL_NAME}，本次 {len(codes)} 只")

    data.download_prices(codes, force=args.force)
    data.download_shares(codes, force=args.force)
    data.download_financials(codes, force=args.force)
    data.download_dividends(codes, force=args.force)
    bench = data.download_benchmark(force=args.force)
    print(f"基准 {cfg.BENCHMARK_NAME}：{len(bench)} 行，{bench.index[0].date()} ~ {bench.index[-1].date()}")
    print("下载完成。")


if __name__ == "__main__":
    main()
