#!/usr/bin/env python3
"""A 股细分行业龙头股多队列动量策略原型。"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from momentum_backtest.runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import akshare as ak
import numpy as np
import pandas as pd

from momentum_backtest.run_backtest import (
    annualized_return,
    build_benchmark_nav,
    build_position_series_from_weight_frame,
    build_strategy_summary,
    build_trades_from_weight_frame,
    configure_matplotlib,
    fetch_histories,
    max_drawdown,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("stock_multi_queue_backtest/output")
CORE_OUTPUT_DIR = OUTPUT_DIR / "core"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"
STOCK_CACHE_DIR = CORE_OUTPUT_DIR / "stock_cache"
FUNDAMENTAL_CACHE_DIR = CORE_OUTPUT_DIR / "fundamental_cache"
STOCK_PRICE_ADJUST = "hfq"
STOCK_CACHE_VERSION = f"stock_{STOCK_PRICE_ADJUST}_v1"

DEFAULT_YEARS = 8
DEFAULT_COST_RATE = 0.0015
QUEUE_COUNT = 5
WARMUP_DAYS = 180
DEFAULT_MIN_HISTORY_DAYS = 250
DEFAULT_MIN_MEDIAN_AMOUNT_20 = 5e8
QUALITY_REQUIRED_METRICS = [
    "index_full_diluted_roe",
    "parent_holder_net_profit",
    "deduct_net_profit_yoy_growth_ratio",
    "index_per_operating_cash_flow_net",
]

LEADER_STOCKS = [
    {"role": "equity", "industry": "白酒", "theme": "白酒", "code": "600519", "name": "贵州茅台"},
    {"role": "equity", "industry": "白酒", "theme": "白酒", "code": "000858", "name": "五粮液"},
    {"role": "equity", "industry": "白酒", "theme": "白酒", "code": "000568", "name": "泸州老窖"},
    {"role": "equity", "industry": "创新药", "theme": "创新药", "code": "600276", "name": "恒瑞医药"},
    {"role": "equity", "industry": "创新药", "theme": "创新药", "code": "688235", "name": "百济神州"},
    {"role": "equity", "industry": "创新药", "theme": "创新药", "code": "300558", "name": "贝达药业"},
    {"role": "equity", "industry": "CRO", "theme": "CRO", "code": "603259", "name": "药明康德"},
    {"role": "equity", "industry": "CRO", "theme": "CRO", "code": "300347", "name": "泰格医药"},
    {"role": "equity", "industry": "CRO", "theme": "CRO", "code": "300759", "name": "康龙化成"},
    {"role": "equity", "industry": "中药", "theme": "中药", "code": "600436", "name": "片仔癀"},
    {"role": "equity", "industry": "中药", "theme": "中药", "code": "000999", "name": "华润三九"},
    {"role": "equity", "industry": "中药", "theme": "中药", "code": "600332", "name": "白云山"},
    {"role": "equity", "industry": "医疗器械", "theme": "医疗器械", "code": "300760", "name": "迈瑞医疗"},
    {"role": "equity", "industry": "医疗器械", "theme": "医疗器械", "code": "688271", "name": "联影医疗"},
    {"role": "equity", "industry": "医疗器械", "theme": "医疗器械", "code": "300832", "name": "新产业"},
    {"role": "equity", "industry": "连锁医疗", "theme": "连锁医疗", "code": "300015", "name": "爱尔眼科"},
    {"role": "equity", "industry": "连锁医疗", "theme": "连锁医疗", "code": "600763", "name": "通策医疗"},
    {"role": "equity", "industry": "芯片设备", "theme": "芯片设备", "code": "002371", "name": "北方华创"},
    {"role": "equity", "industry": "芯片设备", "theme": "芯片设备", "code": "688012", "name": "中微公司"},
    {"role": "equity", "industry": "芯片设备", "theme": "芯片设备", "code": "300604", "name": "长川科技"},
    {"role": "equity", "industry": "晶圆制造", "theme": "晶圆制造", "code": "688981", "name": "中芯国际"},
    {"role": "equity", "industry": "晶圆制造", "theme": "晶圆制造", "code": "688396", "name": "华润微"},
    {"role": "equity", "industry": "晶圆制造", "theme": "晶圆制造", "code": "600460", "name": "士兰微"},
    {"role": "equity", "industry": "光模块", "theme": "光模块", "code": "300308", "name": "中际旭创"},
    {"role": "equity", "industry": "光模块", "theme": "光模块", "code": "300502", "name": "新易盛"},
    {"role": "equity", "industry": "光模块", "theme": "光模块", "code": "300394", "name": "天孚通信"},
    {"role": "equity", "industry": "通信设备", "theme": "通信设备", "code": "000063", "name": "中兴通讯"},
    {"role": "equity", "industry": "通信设备", "theme": "通信设备", "code": "002281", "name": "光迅科技"},
    {"role": "equity", "industry": "动力电池", "theme": "动力电池", "code": "300750", "name": "宁德时代"},
    {"role": "equity", "industry": "动力电池", "theme": "动力电池", "code": "300014", "name": "亿纬锂能"},
    {"role": "equity", "industry": "整车", "theme": "整车", "code": "002594", "name": "比亚迪"},
    {"role": "equity", "industry": "整车", "theme": "整车", "code": "601127", "name": "赛力斯"},
    {"role": "equity", "industry": "工控自动化", "theme": "工控自动化", "code": "300124", "name": "汇川技术"},
    {"role": "equity", "industry": "工控自动化", "theme": "工控自动化", "code": "300450", "name": "先导智能"},
    {"role": "equity", "industry": "家电", "theme": "家电", "code": "000333", "name": "美的集团"},
    {"role": "equity", "industry": "家电", "theme": "家电", "code": "000651", "name": "格力电器"},
    {"role": "equity", "industry": "家电", "theme": "家电", "code": "600690", "name": "海尔智家"},
    {"role": "equity", "industry": "券商", "theme": "券商", "code": "300059", "name": "东方财富"},
    {"role": "equity", "industry": "券商", "theme": "券商", "code": "600030", "name": "中信证券"},
    {"role": "equity", "industry": "券商", "theme": "券商", "code": "601688", "name": "华泰证券"},
    {"role": "equity", "industry": "银行", "theme": "银行", "code": "600036", "name": "招商银行"},
    {"role": "equity", "industry": "银行", "theme": "银行", "code": "601166", "name": "兴业银行"},
    {"role": "equity", "industry": "银行", "theme": "银行", "code": "002142", "name": "宁波银行"},
    {"role": "equity", "industry": "工程机械", "theme": "工程机械", "code": "600031", "name": "三一重工"},
    {"role": "equity", "industry": "工程机械", "theme": "工程机械", "code": "000425", "name": "徐工机械"},
    {"role": "equity", "industry": "工程机械", "theme": "工程机械", "code": "000157", "name": "中联重科"},
    {"role": "equity", "industry": "煤炭", "theme": "煤炭", "code": "601088", "name": "中国神华"},
    {"role": "equity", "industry": "煤炭", "theme": "煤炭", "code": "601225", "name": "陕西煤业"},
    {"role": "equity", "industry": "煤炭", "theme": "煤炭", "code": "601898", "name": "中煤能源"},
    {"role": "equity", "industry": "有色贵金属", "theme": "有色贵金属", "code": "601899", "name": "紫金矿业"},
    {"role": "equity", "industry": "有色贵金属", "theme": "有色贵金属", "code": "603993", "name": "洛阳钼业"},
    {"role": "equity", "industry": "有色贵金属", "theme": "有色贵金属", "code": "600547", "name": "山东黄金"},
    {"role": "equity", "industry": "农牧", "theme": "农牧", "code": "002714", "name": "牧原股份"},
    {"role": "equity", "industry": "农牧", "theme": "农牧", "code": "300498", "name": "温氏股份"},
    {"role": "equity", "industry": "农牧", "theme": "农牧", "code": "002311", "name": "海大集团"},
    {"role": "equity", "industry": "免税旅游", "theme": "免税旅游", "code": "601888", "name": "中国中免"},
    {"role": "equity", "industry": "免税旅游", "theme": "免税旅游", "code": "600754", "name": "锦江酒店"},
    {"role": "equity", "industry": "乳制品", "theme": "乳制品", "code": "600887", "name": "伊利股份"},
    {"role": "equity", "industry": "乳制品", "theme": "乳制品", "code": "600882", "name": "妙可蓝多"},
    {"role": "equity", "industry": "美妆个护", "theme": "美妆个护", "code": "603605", "name": "珀莱雅"},
    {"role": "equity", "industry": "美妆个护", "theme": "美妆个护", "code": "603983", "name": "丸美生物"},
    {"role": "equity", "industry": "办公软件", "theme": "办公软件", "code": "688111", "name": "金山办公"},
    {"role": "equity", "industry": "办公软件", "theme": "办公软件", "code": "688369", "name": "致远互联"},
    {"role": "equity", "industry": "AI应用", "theme": "AI应用", "code": "002230", "name": "科大讯飞"},
    {"role": "equity", "industry": "AI应用", "theme": "AI应用", "code": "688787", "name": "海天瑞声"},
    {"role": "equity", "industry": "金融IT", "theme": "金融IT", "code": "600570", "name": "恒生电子"},
    {"role": "equity", "industry": "金融IT", "theme": "金融IT", "code": "300033", "name": "同花顺"},
]

DEFENSE_ASSETS = [
    {"role": "defense", "industry": "防守", "theme": "十年国债", "code": "511260", "name": "十年国债ETF", "sina_symbol": "sh511260"},
]

BENCHMARK_ASSET = {
    "role": "benchmark",
    "industry": "基准",
    "theme": "沪深300",
    "code": "510300",
    "name": "沪深300ETF华泰柏瑞",
    "sina_symbol": "sh510300",
}

STRATEGY_VARIANTS = {
    "q1_top8_bond": {
        "label": "只持有第一队列前8只 + 国债防守",
        "max_queue": 1,
        "max_positions": 8,
        "max_per_industry": 1,
        "min_score": 0.12,
        "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
        "min_median_amount_20": DEFAULT_MIN_MEDIAN_AMOUNT_20,
        "trend_filter": True,
        "fallback": "511260",
        "score_mode": "base",
        "weight_mode": "equal",
        "market_filter": "none",
    },
    "q1q2_top12_bond": {
        "label": "持有第一二队列前12只 + 国债防守",
        "max_queue": 2,
        "max_positions": 12,
        "max_per_industry": 1,
        "min_score": 0.08,
        "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
        "min_median_amount_20": DEFAULT_MIN_MEDIAN_AMOUNT_20,
        "trend_filter": True,
        "fallback": "511260",
        "score_mode": "base",
        "weight_mode": "equal",
        "market_filter": "none",
    },
    "top8_guard": {
        "label": "前8只 + 60日波动惩罚 + 沪深300年线过滤",
        "max_queue": 1,
        "max_positions": 8,
        "max_per_industry": 1,
        "min_score": 0.05,
        "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
        "min_median_amount_20": DEFAULT_MIN_MEDIAN_AMOUNT_20,
        "trend_filter": True,
        "fallback": "511260",
        "score_mode": "vol60_penalty",
        "weight_mode": "equal",
        "market_filter": "hs300_sma200",
    },
    "top8_guard_rv": {
        "label": "前8只 + 60日波动惩罚 + 年线过滤 + 逆波动权重",
        "max_queue": 1,
        "max_positions": 8,
        "max_per_industry": 1,
        "min_score": 0.05,
        "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
        "min_median_amount_20": DEFAULT_MIN_MEDIAN_AMOUNT_20,
        "trend_filter": True,
        "fallback": "511260",
        "score_mode": "vol60_penalty",
        "weight_mode": "inverse_vol_60",
        "market_filter": "hs300_sma200",
        "market_scale_mode": "binary",
    },
    "top8_guard_soft_rv": {
        "label": "前8只 + 波动惩罚 + 年线连续缩放 + 逆波动权重",
        "max_queue": 1,
        "max_positions": 8,
        "max_per_industry": 1,
        "min_score": 0.05,
        "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
        "min_median_amount_20": DEFAULT_MIN_MEDIAN_AMOUNT_20,
        "trend_filter": True,
        "fallback": "511260",
        "score_mode": "vol60_penalty",
        "weight_mode": "inverse_vol_60",
        "market_filter": "hs300_sma200",
        "market_scale_mode": "ratio_linear_035_30",
    },
    "top8_guard_soft_liq2_rv": {
        "label": "前8只 + 波动惩罚 + 年线连续缩放 + 逆波动权重 + 2亿流动性门槛",
        "max_queue": 1,
        "max_positions": 8,
        "max_per_industry": 1,
        "min_score": 0.05,
        "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
        "min_median_amount_20": 2e8,
        "trend_filter": True,
        "fallback": "511260",
        "score_mode": "vol60_penalty",
        "weight_mode": "inverse_vol_60",
        "market_filter": "hs300_sma200",
        "market_scale_mode": "ratio_linear_035_30",
    },
    "top8_stable_rv": {
        "label": "前8只 + 稳定性增强 + 年线过滤 + 逆波动权重",
        "max_queue": 1,
        "max_positions": 8,
        "max_per_industry": 1,
        "min_score": 0.05,
        "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
        "min_median_amount_20": DEFAULT_MIN_MEDIAN_AMOUNT_20,
        "trend_filter": True,
        "fallback": "511260",
        "score_mode": "stable_r2_down20",
        "weight_mode": "inverse_vol_60",
        "market_filter": "hs300_sma200",
        "market_scale_mode": "binary",
        "quality_filter_mode": "none",
    },
    "top8_stable_liq2_rv": {
        "label": "前8只 + 稳定性增强 + 年线过滤 + 逆波动权重 + 2亿流动性门槛",
        "max_queue": 1,
        "max_positions": 8,
        "max_per_industry": 1,
        "min_score": 0.05,
        "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
        "min_median_amount_20": 2e8,
        "trend_filter": True,
        "fallback": "511260",
        "score_mode": "stable_r2_down20",
        "weight_mode": "inverse_vol_60",
        "market_filter": "hs300_sma200",
        "market_scale_mode": "binary",
        "quality_filter_mode": "none",
    },
    "top8_quality_rv": {
        "label": "前8只 + 稳定性增强 + 简单质量过滤 + 年线过滤",
        "max_queue": 1,
        "max_positions": 8,
        "max_per_industry": 1,
        "min_score": 0.05,
        "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
        "min_median_amount_20": DEFAULT_MIN_MEDIAN_AMOUNT_20,
        "trend_filter": True,
        "fallback": "511260",
        "score_mode": "stable_r2_down20",
        "weight_mode": "inverse_vol_60",
        "market_filter": "hs300_sma200",
        "market_scale_mode": "binary",
        "quality_filter_mode": "basic_quality",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 A 股细分行业龙头股多队列动量策略原型。")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="分析最近多少年。")
    parser.add_argument("--cost-rate", type=float, default=DEFAULT_COST_RATE, help="单边换仓总成本率。")
    parser.add_argument(
        "--strategy",
        choices=["all", *STRATEGY_VARIANTS.keys()],
        default="all",
        help="运行单个策略或全部策略。",
    )
    return parser.parse_args()


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, CORE_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR, STOCK_CACHE_DIR, FUNDAMENTAL_CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_selected() -> pd.DataFrame:
    return pd.DataFrame([*LEADER_STOCKS, *DEFENSE_ASSETS, BENCHMARK_ASSET])


def load_cached_stock_history(code: str) -> pd.DataFrame:
    cache_path = STOCK_CACHE_DIR / f"{code}_{STOCK_CACHE_VERSION}.csv"
    if not cache_path.exists():
        return pd.DataFrame()
    hist = pd.read_csv(cache_path, parse_dates=["date"])
    if hist.empty:
        return pd.DataFrame()
    return hist


def cache_covers_range(hist: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> bool:
    if hist.empty:
        return False
    valid_dates = hist["date"].dropna()
    if valid_dates.empty:
        return False
    return valid_dates.min() <= pd.Timestamp(start_date) and valid_dates.max() >= pd.Timestamp(end_date) - pd.Timedelta(days=3)


def save_cached_stock_history(code: str, hist: pd.DataFrame) -> None:
    cache_path = STOCK_CACHE_DIR / f"{code}_{STOCK_CACHE_VERSION}.csv"
    write_dataframe_csv_atomic(hist, cache_path, index=False)


def price_panel_is_sane(prices: pd.DataFrame) -> bool:
    if prices.empty:
        return False
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all().all():
        return False
    if (numeric <= 0).any().any():
        return False
    daily_returns = numeric.pct_change(fill_method=None)
    extreme_moves = daily_returns.abs() > 5.0
    return not extreme_moves.any().any()


def load_cached_fundamental_abstract(code: str) -> pd.DataFrame:
    cache_path = FUNDAMENTAL_CACHE_DIR / f"{code}.csv"
    if not cache_path.exists():
        return pd.DataFrame()
    return pd.read_csv(cache_path, parse_dates=["report_date"])


def save_cached_fundamental_abstract(code: str, df: pd.DataFrame) -> None:
    cache_path = FUNDAMENTAL_CACHE_DIR / f"{code}.csv"
    write_dataframe_csv_atomic(df, cache_path, index=False)


def normalize_fundamental_abstract(raw_df: pd.DataFrame, *, code: str) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    required_columns = {"report_date", "metric_name", "value"}
    if not required_columns.issubset(raw_df.columns):
        raise RuntimeError(f"unexpected fundamental schema for {code}: {sorted(raw_df.columns.tolist())}")
    df = raw_df[["report_date", "metric_name", "value"]].copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["metric_name"] = df["metric_name"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["report_date", "metric_name"]) 
    df = df[df["metric_name"].isin(QUALITY_REQUIRED_METRICS)].copy()
    return df.sort_values(["report_date", "metric_name"]).reset_index(drop=True)


def fetch_fundamental_abstract(code: str) -> pd.DataFrame:
    raw_df = ak.stock_financial_abstract_new_ths(symbol=code, indicator="按报告期")
    return normalize_fundamental_abstract(raw_df, code=code)


def report_effective_date(report_date: pd.Timestamp) -> pd.Timestamp:
    month_day = report_date.strftime("%m-%d")
    if month_day == "03-31":
        return report_date + pd.Timedelta(days=45)
    if month_day == "06-30":
        return report_date + pd.Timedelta(days=60)
    if month_day == "09-30":
        return report_date + pd.Timedelta(days=45)
    if month_day == "12-31":
        return report_date + pd.Timedelta(days=120)
    return report_date + pd.Timedelta(days=60)


def build_quality_filter_panel(stock_selected: pd.DataFrame, price_index: pd.Index) -> pd.DataFrame:
    quality_frames: list[pd.Series] = []
    for row in stock_selected.itertuples(index=False):
        code = str(row.code)
        abstract_df = load_cached_fundamental_abstract(code)
        if abstract_df.empty:
            abstract_df = fetch_fundamental_abstract(code)
            save_cached_fundamental_abstract(code, abstract_df)
        abstract_df = normalize_fundamental_abstract(abstract_df, code=code)
        if abstract_df.empty:
            quality_frames.append(pd.Series(index=price_index, name=code, dtype="boolean"))
            continue
        metric_df = abstract_df.pivot_table(index="report_date", columns="metric_name", values="value", aggfunc="last").sort_index()
        metric_df["effective_date"] = metric_df.index.to_series().map(report_effective_date)
        metric_df = metric_df.drop_duplicates(subset=["effective_date"], keep="last").set_index("effective_date").sort_index()
        metric_df = metric_df.reindex(price_index).ffill()
        quality_ok = (
            (metric_df["index_full_diluted_roe"] > 0)
            & (metric_df["parent_holder_net_profit"] > 0)
            & (metric_df["index_per_operating_cash_flow_net"] > 0)
            & (metric_df["deduct_net_profit_yoy_growth_ratio"] > -30)
        )
        quality_frames.append(quality_ok.rename(code))
    if not quality_frames:
        return pd.DataFrame(index=price_index)
    return pd.concat(quality_frames, axis=1).reindex(price_index)


def load_cached_price_snapshot(selected: pd.DataFrame, years: int) -> pd.DataFrame:
    prices_path = CORE_OUTPUT_DIR / "prices.csv"
    if not prices_path.exists():
        return pd.DataFrame()
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    if prices.empty:
        return pd.DataFrame()
    end_date = pd.Timestamp.today().normalize()
    requested_start = end_date - pd.DateOffset(years=years)
    required_codes = [str(code) for code in selected["code"].astype(str)]
    if not set(required_codes).issubset(prices.columns.astype(str)):
        return pd.DataFrame()
    if prices.index.min() > pd.Timestamp(requested_start) or prices.index.max() < end_date - pd.Timedelta(days=3):
        return pd.DataFrame()
    prices = prices.sort_index()
    prices.columns = prices.columns.map(str)
    prices.index.name = "date"
    prices = prices.reindex(columns=required_codes).copy()
    if not price_panel_is_sane(prices):
        return pd.DataFrame()
    return prices


def normalize_stock_history(raw_hist: pd.DataFrame, *, code: str, name: str) -> pd.DataFrame:
    if raw_hist.empty:
        raise RuntimeError(f"empty stock history for {code} {name}")
    hist = raw_hist.rename(columns={"日期": "date", "收盘": "close", "成交额": "amount"}).copy()
    required_columns = {"date", "close"}
    if not required_columns.issubset(hist.columns):
        raise RuntimeError(f"unexpected stock history schema for {code} {name}: {sorted(hist.columns.tolist())}")
    if "amount" not in hist.columns:
        hist["amount"] = pd.NA
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
    hist["amount"] = pd.to_numeric(hist["amount"], errors="coerce")
    hist = hist[["date", "close", "amount"]].dropna(subset=["date", "close"])
    hist = hist[hist["close"] > 0].copy()
    hist = hist.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if hist.empty:
        raise RuntimeError(f"no valid stock history for {code} {name}")
    return hist


def infer_tx_symbol(code: str) -> str:
    code = str(code).strip()
    if code[:1] in {"5", "6", "9"}:
        return f"sh{code}"
    if code[:1] in {"0", "1", "2", "3"}:
        return f"sz{code}"
    if code[:1] in {"4", "8"}:
        return f"bj{code}"
    raise ValueError(f"unsupported stock code for tx source: {code}")


def fetch_stock_history_eastmoney(code: str, start_text: str, end_text: str) -> pd.DataFrame:
    return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_text, end_date=end_text, adjust=STOCK_PRICE_ADJUST)


def fetch_stock_history_tx(code: str, start_text: str, end_text: str) -> pd.DataFrame:
    return ak.stock_zh_a_hist_tx(symbol=infer_tx_symbol(code), start_date=start_text, end_date=end_text, adjust=STOCK_PRICE_ADJUST, timeout=15)


def fetch_stock_histories(selected: pd.DataFrame, years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=max(years + 2, 5))
    start_text = start_date.strftime("%Y%m%d")
    end_text = end_date.strftime("%Y%m%d")
    close_frames: list[pd.Series] = []
    amount_frames: list[pd.Series] = []

    for row in selected.itertuples(index=False):
        code = str(row.code)
        hist = load_cached_stock_history(code)
        if not cache_covers_range(hist, start_date, end_date):
            hist = pd.DataFrame()
            last_error: Exception | None = None
            fetchers = [
                ("eastmoney", fetch_stock_history_eastmoney, 4),
                ("tencent", fetch_stock_history_tx, 3),
            ]
            for source_name, fetcher, retry_count in fetchers:
                for attempt in range(retry_count):
                    try:
                        candidate = fetcher(code, start_text, end_text)
                        hist = normalize_stock_history(candidate, code=code, name=str(row.name))
                    except Exception as exc:
                        last_error = exc
                        time.sleep(1.0 + attempt)
                        continue
                    if not hist.empty:
                        break
                if not hist.empty:
                    break
            if hist.empty and last_error is not None:
                raise RuntimeError(f"failed stock history for {code} {row.name}: {last_error}") from last_error
            if hist.empty:
                raise RuntimeError(f"empty stock history for {code} {row.name}")
            save_cached_stock_history(code, hist)
        else:
            hist = normalize_stock_history(hist, code=code, name=str(row.name))
        hist = hist[(hist["date"] >= pd.Timestamp(start_date)) & (hist["date"] <= pd.Timestamp(end_date))].copy()
        if hist.empty:
            raise RuntimeError(f"no in-range stock history for {code} {row.name}")
        indexed = hist.set_index("date")
        close_frames.append(indexed["close"].rename(code))
        amount_frames.append(indexed["amount"].rename(code))

    prices = pd.concat(close_frames, axis=1).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.index.name = "date"
    amounts = pd.concat(amount_frames, axis=1).sort_index()
    amounts = amounts[~amounts.index.duplicated(keep="last")]
    amounts.index.name = "date"
    return prices, amounts


def load_amount_panel_from_cache(stock_selected: pd.DataFrame, price_index: pd.Index) -> pd.DataFrame:
    amount_frames: list[pd.Series] = []
    for row in stock_selected.itertuples(index=False):
        code = str(row.code)
        hist = load_cached_stock_history(code)
        if hist.empty:
            continue
        hist = normalize_stock_history(hist, code=code, name=str(row.name))
        indexed = hist.set_index("date")
        amount_frames.append(indexed["amount"].rename(code))
    if not amount_frames:
        return pd.DataFrame(index=price_index, columns=stock_selected["code"].astype(str).tolist(), dtype="float64")
    amounts = pd.concat(amount_frames, axis=1).sort_index()
    amounts = amounts[~amounts.index.duplicated(keep="last")]
    amounts.index.name = "date"
    return amounts.reindex(price_index).reindex(columns=stock_selected["code"].astype(str).tolist())


def load_prices(years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = build_selected()
    stock_selected = selected[selected["role"] == "equity"].copy()
    etf_selected = selected[selected["role"].isin(["defense", "benchmark"])].copy()
    cached_prices = load_cached_price_snapshot(selected, years)
    if not cached_prices.empty:
        amount_panel = load_amount_panel_from_cache(stock_selected, cached_prices.index)
        return cached_prices, amount_panel
    stock_prices, stock_amounts = fetch_stock_histories(stock_selected, years=years)
    etf_prices = fetch_histories(etf_selected, years=max(years + 2, 8))
    prices = pd.concat([stock_prices, etf_prices], axis=1).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.ffill(limit=2)
    ordered_codes = [str(code) for code in selected["code"] if str(code) in prices.columns]
    amounts = stock_amounts.reindex(prices.index).sort_index()
    amounts = amounts[~amounts.index.duplicated(keep="last")]
    return prices[ordered_codes].copy(), amounts.reindex(columns=stock_selected["code"].astype(str).tolist())


def compute_analysis_start(prices: pd.DataFrame, years: int) -> pd.Timestamp:
    end_date = prices.index.max()
    requested_start = end_date - pd.DateOffset(years=years)
    min_available_start = min(series.index.min() for _, series in prices.items() if not series.dropna().empty)
    warmup_start = pd.Timestamp(min_available_start) + pd.Timedelta(days=WARMUP_DAYS)
    return max(requested_start, warmup_start)


def compute_composite_score(prices: pd.DataFrame, equity_codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ret20 = prices[equity_codes] / prices[equity_codes].shift(20) - 1.0
    ret60 = prices[equity_codes] / prices[equity_codes].shift(60) - 1.0
    ret120 = prices[equity_codes] / prices[equity_codes].shift(120) - 1.0
    sma120 = prices[equity_codes].rolling(120).mean()
    score = 0.5 * ret20 + 0.3 * ret60 + 0.2 * ret120
    trend_ok = prices[equity_codes] > sma120
    return score, trend_ok, sma120


def compute_volatility_frames(prices: pd.DataFrame, equity_codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_returns = prices[equity_codes].pct_change(fill_method=None)
    vol20 = daily_returns.rolling(20).std() * (252**0.5)
    vol60 = daily_returns.rolling(60).std() * (252**0.5)
    return vol20, vol60


def compute_downside_volatility_frame(prices: pd.DataFrame, equity_codes: list[str], window: int = 20) -> pd.DataFrame:
    daily_returns = prices[equity_codes].pct_change(fill_method=None)
    downside_returns = daily_returns.clip(upper=0.0)
    return downside_returns.pow(2).rolling(window).mean().pow(0.5) * (252**0.5)


def compute_price_r2_frame(prices: pd.DataFrame, equity_codes: list[str], window: int = 20) -> pd.DataFrame:
    log_price = np.log(prices[equity_codes])
    x = np.arange(window, dtype=float)
    x_center = x - x.mean()
    x_var = float((x_center**2).sum())

    def calc_r2(window_values: np.ndarray) -> float:
        if np.isnan(window_values).any():
            return float("nan")
        y_center = window_values - window_values.mean()
        y_var = float((y_center**2).sum())
        if y_var <= 0:
            return float("nan")
        cov = float((x_center * y_center).sum())
        corr = cov / math.sqrt(x_var * y_var)
        return corr * corr

    return log_price.rolling(window).apply(calc_r2, raw=True)


def build_score_frame(
    prices: pd.DataFrame,
    equity_codes: list[str],
    *,
    score_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ret20 = prices[equity_codes] / prices[equity_codes].shift(20) - 1.0
    ret60 = prices[equity_codes] / prices[equity_codes].shift(60) - 1.0
    ret120 = prices[equity_codes] / prices[equity_codes].shift(120) - 1.0
    base_score = 0.5 * ret20 + 0.3 * ret60 + 0.2 * ret120
    sma120 = prices[equity_codes].rolling(120).mean()
    trend_ok = prices[equity_codes] > sma120
    vol20, vol60 = compute_volatility_frames(prices, equity_codes)
    down20 = compute_downside_volatility_frame(prices, equity_codes, window=20)
    price_r2_20 = compute_price_r2_frame(prices, equity_codes, window=20)

    if score_mode == "base":
        score = base_score
    elif score_mode == "vol60_penalty":
        score = base_score - 0.20 * vol60
    elif score_mode == "stable_r2_down20":
        score = base_score - 0.20 * vol60 - 0.10 * down20 + 0.06 * (price_r2_20 - 0.5)
    else:
        raise ValueError(f"unsupported score_mode: {score_mode}")
    return score, trend_ok, vol20, vol60


def evaluate_market_filter(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    *,
    market_filter: str,
    benchmark_code: str,
) -> bool:
    if market_filter == "none":
        return True

    benchmark = prices[benchmark_code]
    if market_filter == "hs300_sma200":
        bench_sma200 = benchmark.rolling(200).mean()
        return bool(pd.notna(bench_sma200.loc[signal_date]) and benchmark.loc[signal_date] > bench_sma200.loc[signal_date])

    raise ValueError(f"unsupported market_filter: {market_filter}")


def compute_market_exposure_scale(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    *,
    market_filter: str,
    market_scale_mode: str,
    benchmark_code: str,
) -> float:
    if market_scale_mode == "binary":
        return 1.0 if evaluate_market_filter(
            prices,
            signal_date,
            market_filter=market_filter,
            benchmark_code=benchmark_code,
        ) else 0.0

    benchmark = prices[benchmark_code]
    if market_scale_mode == "ratio_linear_035_30":
        bench_sma200 = benchmark.rolling(200).mean()
        if pd.isna(bench_sma200.loc[signal_date]) or bench_sma200.loc[signal_date] == 0:
            return 0.0
        ratio = float(benchmark.loc[signal_date] / bench_sma200.loc[signal_date] - 1.0)
        return float(min(max(0.35 + 30.0 * ratio, 0.0), 1.0))

    raise ValueError(f"unsupported market_scale_mode: {market_scale_mode}")


def allocate_portfolio_weights(
    selected_codes: list[str],
    signal_date: pd.Timestamp,
    *,
    weight_mode: str,
    vol60: pd.DataFrame,
) -> dict[str, float]:
    if not selected_codes:
        return {}
    if weight_mode == "equal":
        equal_weight = 1.0 / len(selected_codes)
        return {code: equal_weight for code in selected_codes}
    if weight_mode == "inverse_vol_60":
        vols = vol60.loc[signal_date].reindex(selected_codes).clip(lower=0.18).fillna(0.6)
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return {str(code): float(weight) for code, weight in weights.items()}
    raise ValueError(f"unsupported weight_mode: {weight_mode}")


def assign_queue_labels(score_row: pd.Series, queue_count: int = QUEUE_COUNT) -> dict[str, int]:
    valid = score_row.dropna().sort_values(ascending=False)
    if valid.empty:
        return {}
    total = len(valid)
    queue_map: dict[str, int] = {}
    for idx, code in enumerate(valid.index.tolist()):
        queue_map[str(code)] = min((idx * queue_count) // total + 1, queue_count)
    return queue_map


def format_queue_members(codes: list[str], name_map: dict[str, str]) -> str:
    if not codes:
        return ""
    return "；".join(f"{name_map.get(code, code)}({code})" for code in codes)


def select_with_industry_cap(
    ranked_codes: list[str],
    industry_map: dict[str, str],
    *,
    max_positions: int,
    max_per_industry: int,
) -> list[str]:
    selected_codes: list[str] = []
    industry_counts: dict[str, int] = {}
    for code in ranked_codes:
        industry = industry_map.get(str(code), "")
        if industry_counts.get(industry, 0) >= max_per_industry:
            continue
        selected_codes.append(str(code))
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected_codes) >= max_positions:
            break
    return selected_codes


def build_monthly_targets(
    prices: pd.DataFrame,
    amount_panel: pd.DataFrame,
    equity_codes: list[str],
    selected: pd.DataFrame,
    *,
    max_queue: int,
    max_positions: int,
    max_per_industry: int,
    min_score: float,
    min_history_days: int,
    min_median_amount_20: float,
    trend_filter: bool,
    fallback: str,
    score_mode: str,
    weight_mode: str,
    market_filter: str,
    market_scale_mode: str,
    benchmark_code: str,
    quality_filter_mode: str,
    quality_ok_panel: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    score, trend_ok, _, vol60 = build_score_frame(prices, equity_codes, score_mode=score_mode)
    listing_ok = prices[equity_codes].notna().rolling(min_history_days).sum() >= min_history_days
    amount_median_20 = amount_panel[equity_codes].rolling(20).median()
    liquidity_ok = amount_median_20 >= min_median_amount_20
    month_end_dates = prices.groupby(prices.index.to_period("M")).tail(1).index.tolist()
    investable_codes = [*equity_codes, fallback]
    signal_weight_rows: list[dict[str, float]] = []
    signal_weight_index: list[pd.Timestamp] = []
    score_signal = pd.Series(index=prices.index, dtype="float64", name="signal_score")

    name_map = selected.set_index("code")["name"].astype(str).to_dict()
    industry_map = selected.set_index("code")["industry"].astype(str).to_dict()
    signal_rows: list[dict[str, object]] = []

    for signal_pos, signal_date in enumerate(month_end_dates):
        if signal_date not in score.index:
            continue
        score_row = score.loc[signal_date].copy()
        score_row = score_row.where(listing_ok.loc[signal_date], other=pd.NA)
        if signal_date in liquidity_ok.index:
            score_row = score_row.where(liquidity_ok.loc[signal_date], other=pd.NA)
        if quality_filter_mode != "none" and quality_ok_panel is not None and signal_date in quality_ok_panel.index:
            score_row = score_row.where(quality_ok_panel.loc[signal_date], other=pd.NA)
        if trend_filter:
            score_row = score_row.where(trend_ok.loc[signal_date], other=pd.NA)
        score_row = score_row.dropna()
        score_row = score_row[score_row >= min_score].sort_values(ascending=False)
        queue_map = assign_queue_labels(score_row, queue_count=QUEUE_COUNT)

        ranked_codes = [str(code) for code in score_row.index if queue_map.get(str(code), QUEUE_COUNT + 1) <= max_queue]
        selected_codes = select_with_industry_cap(
            ranked_codes,
            industry_map,
            max_positions=max_positions,
            max_per_industry=max_per_industry,
        )
        market_scale = compute_market_exposure_scale(
            prices,
            signal_date,
            market_filter=market_filter,
            market_scale_mode=market_scale_mode,
            benchmark_code=benchmark_code,
        )

        weight_row = {code: 0.0 for code in investable_codes}
        if selected_codes and market_scale > 0:
            portfolio_weights = allocate_portfolio_weights(
                selected_codes,
                signal_date,
                weight_mode=weight_mode,
                vol60=vol60,
            )
            for code, weight in portfolio_weights.items():
                weight_row[code] = weight * market_scale
            weight_row[fallback] = max(0.0, 1.0 - sum(weight_row[code] for code in selected_codes))
            score_signal.loc[signal_date] = float(score_row.reindex(selected_codes).mean())
        else:
            weight_row[fallback] = 1.0
            score_signal.loc[signal_date] = 0.0

        signal_weight_rows.append(weight_row)
        signal_weight_index.append(signal_date)

        queue_members: dict[int, list[str]] = {idx: [] for idx in range(1, QUEUE_COUNT + 1)}
        for code, queue_idx in queue_map.items():
            queue_members[queue_idx].append(code)
        signal_loc = prices.index.get_loc(signal_date)
        execute_date = prices.index[signal_loc + 1] if signal_loc + 1 < len(prices.index) else pd.NaT
        signal_rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "execute_date": execute_date.date().isoformat() if pd.notna(execute_date) else "",
                "selected_count": len(selected_codes),
                "max_per_industry": max_per_industry,
                "min_history_days": min_history_days,
                "min_median_amount_20": min_median_amount_20,
                "market_scale": market_scale,
                "score_mode": score_mode,
                "weight_mode": weight_mode,
                "market_filter": market_filter,
                "market_scale_mode": market_scale_mode,
                "quality_filter_mode": quality_filter_mode,
                "selected_portfolio": format_queue_members(selected_codes, name_map),
                "queue_1": format_queue_members(queue_members[1], name_map),
                "queue_2": format_queue_members(queue_members[2], name_map),
                "queue_3": format_queue_members(queue_members[3], name_map),
                "queue_4": format_queue_members(queue_members[4], name_map),
                "queue_5": format_queue_members(queue_members[5], name_map),
            }
        )

    signal_weights = pd.DataFrame(signal_weight_rows, index=signal_weight_index, columns=investable_codes, dtype="float64")
    daily_target = signal_weights.reindex(prices.index).ffill()
    if fallback in daily_target.columns:
        daily_target = daily_target.fillna(0.0)
        initial_mask = daily_target.sum(axis=1) < 1e-12
        daily_target.loc[initial_mask, fallback] = 1.0
    return daily_target, pd.DataFrame(signal_rows), score_signal.ffill().fillna(0.0)


def run_target_weight_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    target_weights: pd.DataFrame,
    signal_score: pd.Series,
    *,
    cost_rate: float,
    equity_codes: list[str],
    fallback: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices[target_weights.columns].pct_change(fill_method=None).fillna(0.0)
    confirmed_weights = target_weights.reindex(prices.index).fillna(0.0)
    return_weights = confirmed_weights.shift(1).fillna(0.0)
    prev_confirmed_weights = confirmed_weights.shift(1).fillna(0.0)
    turnover = (confirmed_weights - prev_confirmed_weights).abs().sum(axis=1)
    gross_return = (return_weights * returns).sum(axis=1)
    strategy_return = (1.0 + gross_return) * (1.0 - turnover * cost_rate) - 1.0
    nav = (1.0 + strategy_return.fillna(0.0)).cumprod()
    if not nav.empty:
        nav.iloc[0] = 1.0
    holding, _ = build_position_series_from_weight_frame(confirmed_weights)
    exposure = confirmed_weights[equity_codes].sum(axis=1).rename("exposure")
    drawdown = nav / nav.cummax() - 1.0
    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_return,
            "turnover": turnover,
            "holding": holding,
            "exposure": exposure,
            "current_momentum": signal_score.reindex(prices.index),
            "drawdown": drawdown,
        },
        index=prices.index,
    )
    for code in confirmed_weights.columns:
        result[f"weight_{code}"] = confirmed_weights[code]
    trades = build_trades_from_weight_frame(confirmed_weights, result["nav"], selected[selected["code"].isin(target_weights.columns)].copy())
    return result, trades


def build_drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def save_analysis_outputs(compare_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(14, 7))
    for column in compare_df.columns:
        ax.plot(compare_df.index, compare_df[column], linewidth=2.0 if column == "hs300_benchmark" else 1.8, label=column)
    ax.set_title("Stock Multi-Queue NAV vs Benchmark", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, ANALYSIS_OUTPUT_DIR / "nav_vs_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    for column in compare_df.columns:
        ax.plot(compare_df.index, build_drawdown_series(compare_df[column]), linewidth=2.0 if column == "hs300_benchmark" else 1.8, label=column)
    ax.set_title("Stock Multi-Queue Drawdown Compare", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, ANALYSIS_OUTPUT_DIR / "drawdown_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    write_dataframe_csv_atomic(summary_df, CORE_OUTPUT_DIR / "summary.csv", index=False)


def main() -> int:
    args = parse_args()
    ensure_output_dirs()

    selected = build_selected()
    full_prices, full_amount_panel = load_prices(args.years)
    analysis_start = compute_analysis_start(full_prices, args.years)
    prices = full_prices.loc[full_prices.index >= analysis_start].copy()
    amount_panel = full_amount_panel.reindex(prices.index)
    selected_to_save = selected.copy()
    selected_to_save["code"] = selected_to_save["code"].astype(str)

    equity_codes = selected[selected["role"] == "equity"]["code"].astype(str).tolist()
    benchmark_code = str(BENCHMARK_ASSET["code"])
    compare_df = pd.DataFrame(index=prices.index)
    compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code=benchmark_code)
    summary_rows: list[dict[str, object]] = []

    strategy_names = list(STRATEGY_VARIANTS) if args.strategy == "all" else [args.strategy]
    investable_selected = selected[selected["role"] != "benchmark"].copy()
    equity_selected = selected[selected["role"] == "equity"].copy()
    required_quality_modes = {str(STRATEGY_VARIANTS[name].get("quality_filter_mode", "none")) for name in strategy_names}
    quality_ok_panel = None
    if any(mode != "none" for mode in required_quality_modes):
        quality_ok_panel = build_quality_filter_panel(equity_selected, prices.index)

    write_dataframe_csv_atomic(selected_to_save, CORE_OUTPUT_DIR / "selected_candidates.csv", index=False)
    write_dataframe_csv_atomic(full_prices, CORE_OUTPUT_DIR / "prices.csv")

    for strategy_name in strategy_names:
        config = STRATEGY_VARIANTS[strategy_name]
        fallback = str(config["fallback"])
        target_weights, monthly_signals, signal_score = build_monthly_targets(
            prices,
            amount_panel,
            equity_codes,
            investable_selected,
            max_queue=int(config["max_queue"]),
            max_positions=int(config["max_positions"]),
            max_per_industry=int(config.get("max_per_industry", 1)),
            min_score=float(config["min_score"]),
            min_history_days=int(config.get("min_history_days", DEFAULT_MIN_HISTORY_DAYS)),
            min_median_amount_20=float(config.get("min_median_amount_20", DEFAULT_MIN_MEDIAN_AMOUNT_20)),
            trend_filter=bool(config["trend_filter"]),
            fallback=fallback,
            score_mode=str(config.get("score_mode", "base")),
            weight_mode=str(config.get("weight_mode", "equal")),
            market_filter=str(config.get("market_filter", "none")),
            market_scale_mode=str(config.get("market_scale_mode", "binary")),
            benchmark_code=benchmark_code,
            quality_filter_mode=str(config.get("quality_filter_mode", "none")),
            quality_ok_panel=quality_ok_panel,
        )
        result, trades = run_target_weight_strategy(
            prices,
            investable_selected,
            target_weights,
            signal_score,
            cost_rate=float(args.cost_rate),
            equity_codes=equity_codes,
            fallback=fallback,
        )
        summary = build_strategy_summary(
            result,
            trades,
            selected=investable_selected,
            exposure_series=result["exposure"],
            include_max_drawdown_integral=True,
        )
        summary["strategy"] = strategy_name
        summary["label"] = str(config["label"])
        summary_rows.append(summary)
        compare_df[strategy_name] = result["nav"]

        write_dataframe_csv_atomic(result, CORE_OUTPUT_DIR / f"backtest_nav_{strategy_name}.csv")
        write_dataframe_csv_atomic(trades, CORE_OUTPUT_DIR / f"trades_{strategy_name}.csv", index=False)
        write_dataframe_csv_atomic(monthly_signals, CORE_OUTPUT_DIR / f"monthly_signals_{strategy_name}.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    save_analysis_outputs(compare_df, summary_df)
    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
