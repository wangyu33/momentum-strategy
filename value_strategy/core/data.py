"""数据下载与本地缓存。

所有原始数据只下载一次，之后走本地缓存。缓存存在 `output/data/`。
"""

from __future__ import annotations

import socket
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import config as cfg

socket.setdefaulttimeout(40)
warnings.filterwarnings("ignore")

import akshare as ak  # noqa: E402


# ---------------------------------------------------------------- 工具

def to_sina_symbol(code: str) -> str:
    """600519 -> sh600519；000001 -> sz000001。"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("0", "2", "3")):
        return f"sz{code}"
    return f"bj{code}"


def _read(path: Path):
    if not path.exists():
        return None
    try:
        if path.suffix == ".pkl":
            return pd.read_pickle(path)
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None


def _write(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pkl":
        obj.to_pickle(path)
    else:
        obj.to_csv(path)


# ---------------------------------------------------------------- 股票池

def fetch_pool_codes() -> pd.DataFrame:
    """取中证800（沪深300 + 中证500）成分股。返回 [code, name, index_code]。"""
    frames = []
    for idx in cfg.POOL_INDEX_CODES:
        df = ak.index_stock_cons(symbol=idx)
        df = df.rename(columns={"品种代码": "code", "品种名称": "name", "纳入日期": "join_date"})
        df["index_code"] = idx
        frames.append(df[["code", "name", "index_code", "join_date"]])

    pool = pd.concat(frames, ignore_index=True)
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    pool = pool.drop_duplicates(subset="code").reset_index(drop=True)

    mask = ~pool["code"].str.startswith(tuple(cfg.EXCLUDE_PREFIXES))
    pool = pool[mask].reset_index(drop=True)
    return pool.sort_values("code").reset_index(drop=True)


# ---------------------------------------------------------------- 行情

def _one_price(code: str):
    sym = to_sina_symbol(code)
    raw = ak.stock_zh_a_daily(symbol=sym, adjust="")
    hfq = ak.stock_zh_a_daily(symbol=sym, adjust="hfq")
    if raw is None or raw.empty or hfq is None or hfq.empty:
        return code, None, None, None

    def pick(df, col):
        s = df.set_index("date")[col]
        s.index = pd.to_datetime(s.index)
        s.name = code
        return s

    return code, pick(raw, "close"), pick(hfq, "close"), pick(raw, "volume")


def download_prices(codes: list[str], force: bool = False) -> None:
    """下载不复权收盘价、后复权收盘价、成交量三个宽表并缓存。"""
    raw_p, hfq_p, vol_p = (cfg.DATA_DIR / "close_raw.pkl",
                           cfg.DATA_DIR / "close_hfq.pkl",
                           cfg.DATA_DIR / "volume.pkl")

    cached = {p: (None if force else _read(p)) for p in (raw_p, hfq_p, vol_p)}
    done: set[str] = set()
    if cached[raw_p] is not None and cached[hfq_p] is not None:
        done = set(cached[raw_p].columns) & set(cached[hfq_p].columns)
    todo = [c for c in codes if c not in done]
    if not todo:
        print(f"  价格缓存已完整（{len(done)} 只），跳过下载")
        return
    print(f"  待下载价格：{len(todo)} 只（已缓存 {len(done)} 只）")

    raw_cols, hfq_cols, vol_cols = [], [], []
    if cached[raw_p] is not None:
        raw_cols.append(cached[raw_p])
        hfq_cols.append(cached[hfq_p])
        vol_cols.append(cached[vol_p])

    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as ex:
        futures = {ex.submit(_one_price, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            code = futures[fut]
            try:
                _, r, h, v = fut.result()
            except Exception:
                continue
            if r is not None:
                raw_cols.append(r)
                hfq_cols.append(h)
                vol_cols.append(v)
            if i % 50 == 0:
                print(f"    进度 {i}/{len(todo)}", flush=True)

    for cols, path in ((raw_cols, raw_p), (hfq_cols, hfq_p), (vol_cols, vol_p)):
        df = pd.concat(cols, axis=1).sort_index()
        df.index.name = "date"
        _write(df, path)
    print(f"  价格缓存完成：{len(raw_cols)} 只")


def _one_share(code: str):
    """流通股本（新浪日线自带 outstanding_share，单位：股）。"""
    sym = to_sina_symbol(code)
    df = ak.stock_zh_a_daily(symbol=sym, adjust="")
    if df is None or df.empty:
        return code, None
    s = df.set_index("date")["outstanding_share"]
    s.index = pd.to_datetime(s.index)
    s.name = code
    return code, pd.to_numeric(s, errors="coerce")


def download_shares(codes: list[str], force: bool = False) -> None:
    """单独下载流通股本（算市值必需）。"""
    path = cfg.DATA_DIR / "share.pkl"
    cached = None if force else _read(path)
    done = set(cached.columns) if cached is not None else set()
    todo = [c for c in codes if c not in done]
    if not todo:
        print(f"  股本缓存已完整（{len(done)} 只），跳过")
        return
    print(f"  待下载股本：{len(todo)} 只")

    cols = [cached] if cached is not None else []
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as ex:
        futures = {ex.submit(_one_share, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                _, s = fut.result()
            except Exception:
                s = None
            if s is not None:
                cols.append(s)
            if i % 100 == 0:
                print(f"    进度 {i}/{len(todo)}", flush=True)

    df = pd.concat(cols, axis=1).sort_index()
    df.index.name = "date"
    _write(df, path)
    print(f"  股本缓存完成：{df.shape[1]} 只")


# ---------------------------------------------------------------- 财务

def _one_finance(code: str, start_years: tuple[int, ...]):
    """新浪财务接口有个坑：若起始年份早于该股数据起点，会**整体返回空表**
    （例如 002410 用 2006 返回 0 行，用 2008 返回 71 行）。因此按年份逐个回退重试。"""
    import time
    for i, sy in enumerate(start_years):
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year=str(sy))
            if df is not None and not df.empty:
                df = df.copy()
                df["code"] = code
                return code, df
        except Exception:
            pass
        if i < len(start_years) - 1:
            time.sleep(0.3)
    return code, None


def download_financials(codes: list[str], force: bool = False) -> None:
    path = cfg.DATA_DIR / "financials.csv.gz"

    existing = None
    done: set[str] = set()
    if path.exists() and not force:
        existing = pd.read_csv(path)
        done = set(existing["code"].astype(str).str.zfill(6).unique())
    todo = [c for c in codes if c not in done]
    if not todo:
        print(f"  财务缓存已完整（{len(done)} 只），跳过下载")
        return
    print(f"  待下载财务：{len(todo)} 只（已缓存 {len(done)} 只）")

    out = []
    years = tuple(cfg.FINANCE_START_YEARS)
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as ex:
        futures = {ex.submit(_one_finance, c, years): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                _, df = fut.result()
            except Exception:
                df = None
            if df is not None:
                out.append(df)
            if i % 50 == 0:
                print(f"    进度 {i}/{len(todo)}", flush=True)

    if not out:
        print("  本次无新增财务数据")
        return
    allf = pd.concat(([existing] if existing is not None else []) + out, ignore_index=True)
    allf = allf.drop_duplicates(subset=["code", "日期"], keep="last")
    allf.to_csv(path, index=False, compression="gzip")
    print(f"  财务缓存完成：{len(allf)} 行 / {allf['code'].nunique()} 只")


# ---------------------------------------------------------------- 分红

def _one_dividend(code: str):
    try:
        df = ak.stock_history_dividend_detail(symbol=code)
        if df is None or df.empty:
            return code, None
        df = df.copy()
        df["code"] = code
        return code, df
    except Exception:
        return code, None


def download_dividends(codes: list[str], force: bool = False) -> None:
    path = cfg.DATA_DIR / "dividends.csv.gz"

    existing = None
    done: set[str] = set()
    if path.exists() and not force:
        existing = pd.read_csv(path)
        done = set(existing["code"].astype(str).str.zfill(6).unique())
    todo = [c for c in codes if c not in done]
    if not todo:
        print(f"  分红缓存已完整（{len(done)} 只），跳过下载")
        return
    print(f"  待下载分红：{len(todo)} 只（已缓存 {len(done)} 只）")

    out = []
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as ex:
        futures = {ex.submit(_one_dividend, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                _, df = fut.result()
            except Exception:
                df = None
            if df is not None:
                out.append(df)
            if i % 100 == 0:
                print(f"    进度 {i}/{len(todo)}", flush=True)

    frames = ([existing] if existing is not None else []) + out
    if not frames:
        print("  分红数据全部失败（依赖分红的策略将不可用）")
        return
    alld = pd.concat(frames, ignore_index=True)
    alld.to_csv(path, index=False, compression="gzip")
    print(f"  分红缓存完成：{len(alld)} 行 / {alld['code'].nunique()} 只")


# ---------------------------------------------------------------- 基准

def download_benchmark(force: bool = False) -> pd.Series:
    path = cfg.DATA_DIR / "benchmark.csv"
    if not force and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]

    df = ak.stock_zh_index_daily(symbol=cfg.BENCHMARK_SYMBOL)
    s = df.set_index("date")["close"]
    s.index = pd.to_datetime(s.index)
    s.name = cfg.BENCHMARK_NAME
    s.to_frame().to_csv(path)
    return s


# ---------------------------------------------------------------- 统一入口

def load_all(codes: list[str] | None = None, force: bool = False) -> dict:
    """下载（若需要）并加载全部原始数据。"""
    print("取股票池 ...", flush=True)
    pool = fetch_pool_codes()
    if codes is None:
        codes = pool["code"].tolist()
    print(f"  股票池 {len(codes)} 只", flush=True)

    print("下载价格 ...", flush=True)
    download_prices(codes, force=force)
    print("下载财务 ...", flush=True)
    download_financials(codes, force=force)
    print("下载分红 ...", flush=True)
    download_dividends(codes, force=force)
    print("下载基准 ...", flush=True)
    bench = download_benchmark(force=force)

    data = {
        "pool": pool,
        "close_raw": pd.read_pickle(cfg.DATA_DIR / "close_raw.pkl"),
        "close_hfq": pd.read_pickle(cfg.DATA_DIR / "close_hfq.pkl"),
        "volume": pd.read_pickle(cfg.DATA_DIR / "volume.pkl"),
        "benchmark": bench,
    }
    fp = cfg.DATA_DIR / "financials.csv.gz"
    if fp.exists():
        data["financials"] = pd.read_csv(fp)
    dp = cfg.DATA_DIR / "dividends.csv.gz"
    if dp.exists():
        data["dividends"] = pd.read_csv(dp)
    return data
