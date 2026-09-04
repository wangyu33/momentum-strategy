#!/usr/bin/env python3
"""运行 ETF 动量轮动正式回测，并刷新 output/core 结果。"""

from __future__ import annotations

import argparse
import math
import numpy as np
import os
import tempfile
import warnings
from datetime import date
from pathlib import Path

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
    from .official_baseline import OFFICIAL_BASELINE_NAV_FILE, REFERENCE_OUTPUT_DIR, apply_official_baseline_nav_anchor
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports
    from official_baseline import OFFICIAL_BASELINE_NAV_FILE, REFERENCE_OUTPUT_DIR, apply_official_baseline_nav_anchor

prepare_local_imports(__file__)
configure_matplotlib_env()

warnings.filterwarnings("ignore")

import akshare as ak
import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import font_manager


OUTPUT_DIR = Path("momentum_backtest/output")
CORE_OUTPUT_DIR = OUTPUT_DIR / "core"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"
RESEARCH_OUTPUT_DIR = OUTPUT_DIR / "research"
MONITOR_OUTPUT_DIR = OUTPUT_DIR / "monitor"
TRADE_CALENDAR_CACHE = MONITOR_OUTPUT_DIR / "trade_calendar_cache.csv"
OPTIONAL_REALTIME_FALLBACK_PREFIXES = ("511",)
DEFAULT_LOOKBACK = 25
DEFAULT_YEARS = 15
DEFAULT_HISTORY_START = pd.Timestamp("2012-01-01")
DEFAULT_FEE_RATE = 0.0003
DEFAULT_SLIPPAGE_RATE = 0.0002
DEFAULT_CASH_THRESHOLD = None
DEFAULT_STRATEGY_NAME = "baseline_cf60top2"
DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD = 0.05
DEFAULT_RESOURCE_ABS_THRESHOLD = 0.08
DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT = 0.80
DEFAULT_OVERHEAT_DRAWDOWN_CUT = -0.03
DEFAULT_OVERHEAT_MOMENTUM_CUT = 0.25
DEFAULT_OVERHEAT_MAX_EXPOSURE = 0.40
DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT = 0.35
DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE = 0.20
FIXED_ETF_POOL = [
    {"theme": "沪深300", "code": "510300", "name": "沪深300ETF华泰柏瑞", "sina_symbol": "sh510300"},
    {"theme": "创业板50", "code": "159949", "name": "创业板50ETF华安", "sina_symbol": "sz159949"},
    {"theme": "H股", "code": "159954", "name": "H股ETF", "sina_symbol": "sz159954"},
    {"theme": "纳指ETF", "code": "159941", "name": "纳指ETF广发", "sina_symbol": "sz159941"},
    {"theme": "标普500ETF", "code": "513650", "name": "标普500ETF南方", "sina_symbol": "sh513650"},
    {"theme": "政金ETF", "code": "511580", "name": "国债政金债ETF招商", "sina_symbol": "sh511580"},
    {"theme": "黄金ETF", "code": "518880", "name": "黄金ETF华安", "sina_symbol": "sh518880"},
    {"theme": "日经ETF", "code": "513880", "name": "日经225ETF华安", "sina_symbol": "sh513880"},
    {"theme": "红利低波ETF", "code": "512890", "name": "红利低波ETF华泰柏瑞", "sina_symbol": "sh512890"},
    {"theme": "豆粕", "code": "159985", "name": "豆粕ETF", "sina_symbol": "sz159985"},
]
STRESS_BOND_ETF = {
    "theme": "十年国债ETF",
    "code": "511260",
    "name": "十年国债ETF",
    "sina_symbol": "sh511260",
}
RESOURCE_ETF = {"theme": "资源ETF", "code": "510410", "name": "资源ETF", "sina_symbol": "sh510410"}
RESOURCE_CODE = "510410"
RISK_CODES = ["510300", "159949", "159954", "159941", "513650", "513880"]
DEFENSIVE_CODES = ["511580", "518880", "512890", "159985"]
DEFAULT_BASELINE_DROP_CODES = ["511580", "513650"]
DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT = 0.28
DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT = 0.27
DEFAULT_REGIME_MIX_MOMENTUM_CUT = 0.06
DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT = 0.91
DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT = 0.90
DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT = -0.04
DEFAULT_REGIME_MIX_PROXY_KIND = "hybrid_breadth_blend"
DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP = 0.50
DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING = 0.16
DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT = -0.02
DEFAULT_REGIME_MIX_PRE_OVERHEAT_START_CUT = 0.15
DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_CUT = 0.20
DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_EXPOSURE = 0.90
DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT = 0.25
DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE = 0.30
DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT = 0.30
DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE = 0.10
DEFAULT_REGIME_MIX_OVERHEAT_CAP_MODE = "continuous"
DEFAULT_SIGNAL_QUALITY_METHOD = "slope"
DEFAULT_SIGNAL_SLOPE_PENALTY = 0.85
DEFAULT_SIGNAL_LEADER_MARGIN = 0.0
DEFAULT_CLOSE_TOP2_GAP = 0.005
DEFAULT_CLOSE_TOP2_RISK_CAP = 0.70
DEFAULT_STRESS_BOND_CODE = "511260"
DEFAULT_STRESS_BOND_RISK_CAP = 0.0
DEFAULT_STRESS_BOND_RATIO_CUT = 0.90
DEFAULT_STRESS_BOND_BREADTH_CUT = -0.031
DEFAULT_STRESS_BOND_ENTER_DAYS = 1
DEFAULT_STRESS_BOND_EXIT_DAYS = 1
DEFAULT_STRESS_BOND_FILL_RESIDUAL_CASH = False
STRUCTURAL_BREAK_RETURN_THRESHOLD = 0.45

PREFERRED_CJK_FONT_FILES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
]


def build_signal_quality_score(
    prices: pd.DataFrame,
    lookback: int,
    method: str = "raw",
    slope_penalty: float = 0.0,
    volatility_penalty: float = 0.0,
    downside_volatility_penalty: float = 0.0,
    r2_penalty: float = 0.0,
    volatility_state_lookback: int = 252,
    volatility_percentile_penalty: float = 0.0,
    downside_volatility_percentile_penalty: float = 0.0,
    volatility_percentile_divisor: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_momentum = prices / prices.shift(lookback) - 1
    if method not in {"raw", "slope"}:
        raise ValueError(f"unsupported signal quality method: {method}")

    quality_score = raw_momentum.copy()
    ret5 = prices / prices.shift(5) - 1
    if method == "slope":
        excess_slope = (ret5 - raw_momentum / max(lookback / 5, 1)).clip(lower=0.0)
        quality_score = quality_score - slope_penalty * excess_slope

    if (
        volatility_penalty > 0
        or downside_volatility_penalty > 0
        or volatility_percentile_penalty > 0
        or downside_volatility_percentile_penalty > 0
        or volatility_percentile_divisor > 0
    ):
        returns = prices.pct_change()
        total_vol20 = returns.rolling(20).std(ddof=0) * math.sqrt(252)
        if volatility_penalty > 0:
            quality_score = quality_score - volatility_penalty * total_vol20
        if downside_volatility_penalty > 0:
            downside_vol20 = returns.where(returns < 0, 0.0).rolling(20).std(ddof=0) * math.sqrt(252)
            quality_score = quality_score - downside_volatility_penalty * downside_vol20

        if volatility_percentile_penalty > 0 or volatility_percentile_divisor > 0:
            total_vol20_pct = build_asset_relative_state_percentile(total_vol20, lookback=volatility_state_lookback)
            if volatility_percentile_penalty > 0:
                quality_score = quality_score - volatility_percentile_penalty * total_vol20_pct
            if volatility_percentile_divisor > 0:
                quality_score = quality_score / (1.0 + volatility_percentile_divisor * total_vol20_pct)

        if downside_volatility_percentile_penalty > 0:
            downside_vol20 = returns.where(returns < 0, 0.0).rolling(20).std(ddof=0) * math.sqrt(252)
            downside_vol20_pct = build_asset_relative_state_percentile(downside_vol20, lookback=volatility_state_lookback)
            quality_score = quality_score - downside_volatility_percentile_penalty * downside_vol20_pct

    if r2_penalty > 0:
        log_prices = np.log(prices.replace(0.0, np.nan))
        time_index = pd.Series(np.arange(len(prices), dtype=float), index=prices.index)
        trend_r2 = pd.DataFrame(
            {
                code: log_prices[code].rolling(lookback).corr(time_index).pow(2)
                for code in prices.columns
            },
            index=prices.index,
        ).clip(lower=0.0, upper=1.0)
        quality_score = quality_score - r2_penalty * (1.0 - trend_r2)

    return raw_momentum, quality_score


def build_asset_relative_state_percentile(
    metric: pd.DataFrame,
    lookback: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1")
    if min_periods is None:
        min_periods = max(20, min(lookback, lookback // 4))

    def compute_last_percentile(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return float("nan")
        last = float(arr[-1])
        less = float(np.sum(arr < last))
        equal = float(np.sum(arr == last))
        return (less + 0.5 * equal) / float(arr.size)

    return metric.rolling(lookback, min_periods=min_periods).apply(compute_last_percentile, raw=True)


def build_asset_relative_volatility_percentile(
    prices: pd.DataFrame,
    vol_window: int = 20,
    state_lookback: int = 252,
    downside: bool = False,
) -> pd.DataFrame:
    returns = prices.pct_change()
    if downside:
        vol = returns.where(returns < 0, 0.0).rolling(vol_window).std(ddof=0) * math.sqrt(252)
    else:
        vol = returns.rolling(vol_window).std(ddof=0) * math.sqrt(252)
    return build_asset_relative_state_percentile(vol, lookback=state_lookback)


def build_asset_own_momentum_percentile(
    prices: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    state_lookback: int = 756,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """计算每个标的当前动量在其自身历史窗口中的分位。"""
    momentum = prices / prices.shift(lookback) - 1
    return build_asset_relative_state_percentile(
        momentum,
        lookback=state_lookback,
        min_periods=min_periods,
    )


def build_current_etf_momentum_percentile_table(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    state_lookback: int = 756,
    min_periods: int | None = 120,
) -> pd.DataFrame:
    """生成候选池最新一日的动量与自身历史分位快照。"""
    columns = [
        "date",
        "price_date",
        "code",
        "theme",
        "name",
        "price",
        f"mom_{lookback}",
        f"mom_{lookback}_percentile",
    ]
    if prices.empty or selected.empty:
        return pd.DataFrame(columns=columns)

    momentum = prices / prices.shift(lookback) - 1
    momentum_percentile = build_asset_own_momentum_percentile(
        prices,
        lookback=lookback,
        state_lookback=state_lookback,
        min_periods=min_periods,
    )

    latest_date = prices.index.max()
    rows: list[dict[str, object]] = []
    for _, item in selected.iterrows():
        code = str(item.get("code", "")).strip()
        if not code or code not in prices.columns:
            continue

        price_series = pd.to_numeric(prices[code], errors="coerce").dropna()
        if price_series.empty:
            continue

        latest_price_date = price_series.index.max()
        latest_momentum = float("nan")
        if code in momentum.columns and latest_price_date in momentum.index:
            latest_momentum = float(momentum.loc[latest_price_date, code])

        latest_percentile = float("nan")
        if code in momentum_percentile.columns and latest_price_date in momentum_percentile.index:
            latest_percentile = float(momentum_percentile.loc[latest_price_date, code])

        rows.append(
            {
                "date": latest_date.date().isoformat(),
                "price_date": latest_price_date.date().isoformat(),
                "code": code,
                "theme": str(item.get("theme", "")).strip(),
                "name": str(item.get("name", "")).strip(),
                "price": float(price_series.loc[latest_price_date]),
                f"mom_{lookback}": latest_momentum,
                f"mom_{lookback}_percentile": latest_percentile,
            }
        )

    snapshot = pd.DataFrame(rows, columns=columns)
    if snapshot.empty:
        return snapshot
    return snapshot.sort_values([f"mom_{lookback}", "code"], ascending=[False, True]).reset_index(drop=True)


def build_signal_stability_score(
    prices: pd.DataFrame,
    lookback: int,
    method: str = "none",
) -> pd.DataFrame | None:
    if method == "none":
        return None

    returns = prices.pct_change()
    if method == "total_vol":
        return -(returns.rolling(20).std(ddof=0) * math.sqrt(252))
    if method == "downside_vol":
        downside_vol20 = returns.where(returns < 0, 0.0).rolling(20).std(ddof=0) * math.sqrt(252)
        return -downside_vol20
    if method == "r2":
        log_prices = np.log(prices.replace(0.0, np.nan))
        time_index = pd.Series(np.arange(len(prices), dtype=float), index=prices.index)
        return pd.DataFrame(
            {code: log_prices[code].rolling(lookback).corr(time_index).pow(2) for code in prices.columns},
            index=prices.index,
        ).clip(lower=0.0, upper=1.0)
    raise ValueError(f"unsupported stability method: {method}")


def choose_signal_winner_with_margin(
    score_row: pd.Series,
    prev_asset: str | None,
    leader_margin: float,
    secondary_score_row: pd.Series | None = None,
    secondary_score_gap: float = 0.0,
) -> str | None:
    valid = score_row.dropna().sort_values(ascending=False)
    if valid.empty:
        return None
    top_asset = str(valid.index[0])
    top_score = float(valid.iloc[0])
    if secondary_score_gap > 0 and len(valid) >= 2 and float(valid.iloc[0] - valid.iloc[1]) <= secondary_score_gap:
        second_asset = str(valid.index[1])
        if secondary_score_row is not None:
            secondary_valid = secondary_score_row.reindex([top_asset, second_asset]).dropna().sort_values(ascending=False)
            if not secondary_valid.empty:
                top_asset = str(secondary_valid.index[0])
    if leader_margin > 0 and prev_asset and prev_asset in valid.index:
        prev_score = float(valid.loc[prev_asset])
        if top_score - prev_score < leader_margin:
            return str(prev_asset)
    return top_asset


def filter_score_row_by_confirmation(
    score_row: pd.Series,
    confirmation_row: pd.Series | None,
    top_n: int,
) -> pd.Series:
    if confirmation_row is None or top_n <= 0:
        return score_row
    confirm_valid = confirmation_row.dropna().sort_values(ascending=False)
    if confirm_valid.empty:
        return score_row
    allowed = set(confirm_valid.head(top_n).index)
    filtered = score_row.where(score_row.index.isin(allowed))
    return filtered if filtered.notna().any() else score_row


def build_top2_close_risk_flag(
    prices: pd.DataFrame,
    lookback: int,
    absolute_threshold: float = DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    signal_quality_method: str = "raw",
    slope_penalty: float = 0.0,
    volatility_penalty: float = 0.0,
    downside_volatility_penalty: float = 0.0,
    r2_penalty: float = 0.0,
    volatility_state_lookback: int = 252,
    volatility_percentile_penalty: float = 0.0,
    downside_volatility_percentile_penalty: float = 0.0,
    volatility_percentile_divisor: float = 0.0,
    confirmation_lookback: int = 0,
    confirmation_top_n: int = 0,
    risk_codes: list[str] | None = None,
    close_gap: float = 0.0,
) -> pd.Series:
    if close_gap <= 0:
        return pd.Series(False, index=prices.index, dtype=bool)

    momentum, score = build_signal_quality_score(
        prices,
        lookback=lookback,
        method=signal_quality_method,
        slope_penalty=slope_penalty,
        volatility_penalty=volatility_penalty,
        downside_volatility_penalty=downside_volatility_penalty,
        r2_penalty=r2_penalty,
        volatility_state_lookback=volatility_state_lookback,
        volatility_percentile_penalty=volatility_percentile_penalty,
        downside_volatility_percentile_penalty=downside_volatility_percentile_penalty,
        volatility_percentile_divisor=volatility_percentile_divisor,
    )
    active_risk_codes, _ = resolve_strategy_universe(prices, risk_codes=risk_codes, defensive_codes=[])
    if len(active_risk_codes) < 2:
        return pd.Series(False, index=prices.index, dtype=bool)

    risk_score = score[active_risk_codes]
    risk_confirmation = None
    if confirmation_lookback > 0 and confirmation_top_n > 0:
        risk_confirmation = (prices[active_risk_codes] / prices[active_risk_codes].shift(confirmation_lookback) - 1)
    close_flag = pd.Series(False, index=prices.index, dtype=bool, name="top2_close_risk_cap_triggered")
    for dt_idx in prices.index:
        score_row = risk_score.loc[dt_idx]
        if risk_confirmation is not None:
            score_row = filter_score_row_by_confirmation(score_row, risk_confirmation.loc[dt_idx], confirmation_top_n)
        valid = score_row.dropna().sort_values(ascending=False)
        if len(valid) < 2:
            continue
        top_asset = str(valid.index[0])
        second_asset = str(valid.index[1])
        top_momentum = float(momentum.loc[dt_idx, top_asset]) if pd.notna(momentum.loc[dt_idx, top_asset]) else float("nan")
        second_momentum = float(momentum.loc[dt_idx, second_asset]) if pd.notna(momentum.loc[dt_idx, second_asset]) else float("nan")
        if pd.isna(top_momentum) or pd.isna(second_momentum):
            continue
        if top_momentum <= absolute_threshold or second_momentum <= absolute_threshold:
            continue
        if float(valid.iloc[0] - valid.iloc[1]) <= close_gap:
            close_flag.loc[dt_idx] = True
    return close_flag


def load_fixed_etf_pool() -> pd.DataFrame:
    return pd.DataFrame(FIXED_ETF_POOL)


def load_default_strategy_pool() -> pd.DataFrame:
    return pd.DataFrame([item for item in FIXED_ETF_POOL if item["code"] not in set(DEFAULT_BASELINE_DROP_CODES)])


def load_default_strategy_backtest_pool() -> pd.DataFrame:
    base_pool = load_default_strategy_pool()
    return pd.concat([base_pool, pd.DataFrame([STRESS_BOND_ETF])], ignore_index=True)


def default_strategy_param_text(pool_size: int) -> str:
    params = build_default_strategy_params()
    return (
        f"{DEFAULT_STRATEGY_NAME}, 固定{pool_size}只ETF, k={DEFAULT_LOOKBACK}, remove={','.join(DEFAULT_BASELINE_DROP_CODES)}, "
        f"信号口径={DEFAULT_SIGNAL_QUALITY_METHOD}{int(round(DEFAULT_SIGNAL_SLOPE_PENALTY * 100)):03d}+60日前2确认, "
        f"进攻/防守核心仓={float(params.get('aggressive_core_weight', DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT)):.0%}/"
        f"{float(params.get('conservative_core_weight', DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT)):.0%}, "
        f"regime阈值={float(params.get('regime_momentum_cut', DEFAULT_REGIME_MIX_MOMENTUM_CUT)):.0%}, "
        f"量能门槛=20日/60日<{float(params.get('volume_ratio_cut', DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT)):.0%}, "
        f"5日/20日<{float(params.get('volume_short_ratio_cut', DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT)):.0%}, "
        f"广度<{float(params.get('volume_breadth_cut', DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT)):.1%}, "
        f"弱量能上限={float(params.get('volume_guard_cap', DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP)):.1%}"
        f"(动量<={float(params.get('volume_guard_momentum_ceiling', DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING)):.0%}), "
        f"历史动量分位控仓="
        f"{float(params.get('signal_selected_momentum_pct_cap_floor', 1.0)):.0%}"
        f"(>{float(params.get('signal_selected_momentum_pct_cap_start', 1.0)):.0%}->"
        f"{float(params.get('signal_selected_momentum_pct_cap_end', 1.0)):.0%}, "
        f"scope={str(params.get('signal_selected_momentum_pct_cap_scope', 'all'))}), "
        f"最小调仓阈值={float(params.get('target_min_rebalance_threshold', 0.0)):.0%}, "
        f"弱市切债={STRESS_BOND_ETF['name']}({DEFAULT_STRESS_BOND_CODE}), 风险仓上限={DEFAULT_STRESS_BOND_RISK_CAP:.1%}"
        f"(20/60<{DEFAULT_STRESS_BOND_RATIO_CUT:.0%}, 广度<{DEFAULT_STRESS_BOND_BREADTH_CUT:.1%})"
    )


def build_persistent_trigger_mask(raw_trigger: pd.Series, enter_days: int, exit_days: int) -> pd.Series:
    active = False
    enter_streak = 0
    exit_streak = 0
    result: list[bool] = []

    for is_triggered in raw_trigger.fillna(False).astype(bool).tolist():
        if is_triggered:
            enter_streak += 1
            exit_streak = 0
        else:
            exit_streak += 1
            enter_streak = 0

        if not active and enter_streak >= enter_days:
            active = True
        elif active and exit_streak >= exit_days:
            active = False

        result.append(active)

    return pd.Series(result, index=raw_trigger.index, dtype=bool)


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, CORE_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR, RESEARCH_OUTPUT_DIR, MONITOR_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def fetch_trade_calendar() -> pd.DataFrame:
    try:
        cal = ak.tool_trade_date_hist_sina().copy()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"]).dt.date
        ensure_output_dirs()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                prefix="trade_calendar_",
                dir=str(TRADE_CALENDAR_CACHE.parent),
                delete=False,
                encoding="utf-8",
            ) as handle:
                temp_path = Path(handle.name)
            cal.to_csv(temp_path, index=False)
            os.replace(temp_path, TRADE_CALENDAR_CACHE)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return cal
    except Exception:
        if TRADE_CALENDAR_CACHE.exists():
            try:
                cached = pd.read_csv(TRADE_CALENDAR_CACHE)
                if "trade_date" not in cached.columns:
                    raise ValueError("trade calendar cache missing trade_date")
                cached["trade_date"] = pd.to_datetime(cached["trade_date"]).dt.date
                return cached
            except Exception:
                pass
        raise


def is_trading_day(today: date) -> bool:
    try:
        cal = fetch_trade_calendar()
        return today in set(cal["trade_date"].tolist())
    except Exception:
        return today.weekday() < 5


def configure_matplotlib() -> None:
    for font_path in PREFERRED_CJK_FONT_FILES:
        if not Path(font_path).exists():
            continue
        try:
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = [font_name]
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def write_dataframe_csv_atomic(df: pd.DataFrame, output_path: Path, *, index: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f"{output_path.stem}_",
            dir=str(output_path.parent),
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_path = Path(handle.name)
        df.to_csv(temp_path, index=index)
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def save_figure_atomic(fig: plt.Figure, output_path: Path, **savefig_kwargs: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=output_path.suffix or ".png",
            prefix=f"{output_path.stem}_",
            dir=str(output_path.parent),
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        fig.savefig(temp_path, **savefig_kwargs)
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def normalize_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def count_trade_days(trades: pd.DataFrame) -> int:
    """按发生交易的日期计数，避免一日多条动作把一次调仓放大成多次。"""
    if trades.empty or "date" not in trades.columns:
        return 0
    trade_dates = pd.to_datetime(trades["date"], errors="coerce").dropna()
    return int(trade_dates.dt.normalize().nunique())


def is_recovered(drawdown_value: float, tol: float = 1e-10) -> bool:
    """判断回撤是否已经回到高点，允许极小浮点误差。"""
    return abs(float(drawdown_value)) <= tol


def count_rebalance_days(
    trades: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> int:
    """统计区间内发生调仓的交易日数，避免同日多资产动作被重复计数。"""
    if trades.empty or "date" not in trades.columns:
        return 0
    trade_dates = pd.to_datetime(trades["date"], errors="coerce").dropna()
    mask = (trade_dates >= start_date) & (trade_dates <= end_date)
    return int(trade_dates.loc[mask].dt.normalize().nunique())


def resolve_strategy_universe(
    prices: pd.DataFrame,
    *,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """解析当前价格面板下实际生效的风险池与防守池。"""
    effective_risk_codes = [code for code in (RISK_CODES if risk_codes is None else risk_codes) if code in prices.columns]
    effective_defensive_codes = [
        code for code in (DEFENSIVE_CODES if defensive_codes is None else defensive_codes) if code in prices.columns
    ]
    return effective_risk_codes, effective_defensive_codes


def get_latest_portfolio_text(selected: pd.DataFrame, result: pd.DataFrame) -> str:
    """输出最新一日的组合摘要，避免把组合仓位误报成单一 holding。"""
    if result.empty:
        return ""
    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    latest_row = result.iloc[-1]
    weight_cols = [col for col in result.columns if col.startswith("weight_")]
    allocations: list[tuple[str, float]] = []
    if weight_cols:
        for col in weight_cols:
            code = col.removeprefix("weight_")
            raw_value = latest_row.get(col, 0.0)
            weight = float(raw_value) if pd.notna(raw_value) else 0.0
            if abs(weight) > 1e-12:
                allocations.append((code, weight))
    if not allocations:
        holding = normalize_code(latest_row.get("holding"))
        exposure_raw = latest_row.get("exposure", 1.0)
        exposure = float(exposure_raw) if pd.notna(exposure_raw) else 0.0
        if holding and abs(exposure) > 1e-12:
            allocations.append((holding, exposure))
    if not allocations:
        return "空仓"
    allocations.sort(key=lambda item: item[1], reverse=True)
    parts = []
    for code, weight in allocations:
        theme = code_to_theme.get(code, "")
        name = code_to_name.get(code, "")
        label = f"{theme} / {name} ({code})" if theme else (f"{name} ({code})" if name else code)
        parts.append(f"{label} {weight:.0%}")
    return "；".join(parts)


def build_weight_frame_from_holding_exposure(
    index: pd.Index,
    codes: list[str],
    holding: pd.Series,
    exposure: pd.Series,
) -> pd.DataFrame:
    """把单标的 holding/exposure 口径展开成逐标的权重表。"""
    weights = pd.DataFrame(0.0, index=index, columns=codes, dtype="float64")
    holding_codes = holding.reindex(index).map(normalize_code)
    exposure_series = exposure.reindex(index).fillna(0.0)
    for code in codes:
        mask = holding_codes == code
        if mask.any():
            weights.loc[mask, code] = exposure_series.loc[mask].astype(float)
    return weights


def extract_target_weights_from_result(result: pd.DataFrame) -> pd.DataFrame:
    """从结果表里还原目标权重表。"""
    target_cols = [col for col in result.columns if col.startswith("target_weight_")]
    if target_cols:
        weights = result[target_cols].copy()
        weights.columns = [col.removeprefix("target_weight_") for col in target_cols]
        return weights.fillna(0.0)
    weight_cols = [col for col in result.columns if col.startswith("weight_")]
    weights = result[weight_cols].copy()
    weights.columns = [col.removeprefix("weight_") for col in weight_cols]
    return weights.fillna(0.0)


def build_position_series_from_weight_frame(weights: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """根据逐标的权重表还原组合主持仓与总仓位。"""
    normalized = weights.fillna(0.0)
    exposure = normalized.sum(axis=1).rename("exposure")
    holding = normalized.idxmax(axis=1).where(exposure > 1e-12, pd.NA).rename("holding")
    return holding, exposure


def build_trades_from_weight_frame(weights: pd.DataFrame, nav: pd.Series, selected: pd.DataFrame) -> pd.DataFrame:
    """按收盘确认后的权重变化生成交易记录。"""
    normalized_selected = selected.copy()
    normalized_selected["code"] = normalized_selected["code"].map(normalize_code)
    code_to_theme = normalized_selected.set_index("code")["theme"].to_dict()
    code_to_name = normalized_selected.set_index("code")["name"].to_dict()
    normalized = weights.fillna(0.0)
    prev_weights = normalized.shift(1).fillna(0.0)
    trades: list[dict[str, object]] = []

    for dt_idx in normalized.index:
        current = normalized.loc[dt_idx]
        prev = prev_weights.loc[dt_idx]
        for code in normalized.columns:
            curr_w = float(current[code]) if pd.notna(current[code]) else 0.0
            prev_w = float(prev[code]) if pd.notna(prev[code]) else 0.0
            if abs(curr_w - prev_w) < 1e-12:
                continue
            if curr_w > prev_w:
                action = "BUY" if prev_w == 0 else "ADD"
            else:
                action = "SELL" if curr_w == 0 else "REDUCE"
            trades.append(
                {
                    "date": dt_idx,
                    "action": action,
                    "code": normalize_code(code),
                    "theme": code_to_theme.get(code, ""),
                    "name": code_to_name.get(code, ""),
                    "from_exposure": prev_w,
                    "to_exposure": curr_w,
                    "nav": float(nav.loc[dt_idx]),
                }
            )
    return pd.DataFrame(trades)


def finalize_position_columns(
    result: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    confirmed_weights: pd.DataFrame | None = None,
    confirmed_holding: pd.Series | None = None,
    confirmed_exposure: pd.Series | None = None,
    return_weights: pd.DataFrame | None = None,
    return_holding: pd.Series | None = None,
    return_exposure: pd.Series | None = None,
) -> pd.DataFrame:
    """同时保留收盘确认持仓口径与收益计算口径。"""
    codes = [str(code) for code in selected["code"]]
    finalized = result.copy()

    if confirmed_weights is None:
        if confirmed_holding is None or confirmed_exposure is None:
            raise ValueError("confirmed position context is required")
        confirmed_weights = build_weight_frame_from_holding_exposure(
            finalized.index,
            codes,
            confirmed_holding,
            confirmed_exposure,
        )
    else:
        confirmed_weights = confirmed_weights.reindex(index=finalized.index, columns=codes, fill_value=0.0).fillna(0.0)

    if confirmed_holding is None or confirmed_exposure is None:
        confirmed_holding, confirmed_exposure = build_position_series_from_weight_frame(confirmed_weights)
    else:
        confirmed_holding = confirmed_holding.reindex(finalized.index).rename("holding")
        confirmed_exposure = confirmed_exposure.reindex(finalized.index).fillna(0.0).rename("exposure")

    if return_weights is None:
        if return_holding is None or return_exposure is None:
            return_weights = confirmed_weights.shift(1).fillna(0.0)
        else:
            return_weights = build_weight_frame_from_holding_exposure(
                finalized.index,
                codes,
                return_holding,
                return_exposure,
            )
    else:
        return_weights = return_weights.reindex(index=finalized.index, columns=codes, fill_value=0.0).fillna(0.0)

    if return_holding is None or return_exposure is None:
        return_holding, return_exposure = build_position_series_from_weight_frame(return_weights)
    else:
        return_holding = return_holding.reindex(finalized.index).rename("holding_for_return")
        return_exposure = return_exposure.reindex(finalized.index).fillna(0.0).rename("exposure_for_return")

    finalized["holding"] = confirmed_holding
    finalized["exposure"] = confirmed_exposure
    finalized["holding_for_return"] = return_holding.rename("holding_for_return")
    finalized["exposure_for_return"] = return_exposure.rename("exposure_for_return")
    finalized = finalized.drop(
        columns=[col for col in finalized.columns if col.startswith("weight_") or col.startswith("return_weight_")],
        errors="ignore",
    )
    finalized = pd.concat(
        [
            finalized,
            confirmed_weights.add_prefix("weight_"),
            return_weights.add_prefix("return_weight_"),
        ],
        axis=1,
    )
    return finalized


def recompute_return_chain(
    result: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    fee_rate: float,
    slippage_rate: float,
) -> pd.DataFrame:
    """用盘前持仓算当日收益，用收盘确认仓位算当日换手和成本。"""
    refreshed = result.copy()
    weight_cols = [col for col in refreshed.columns if col.startswith("weight_")]
    return_weight_cols = [col for col in refreshed.columns if col.startswith("return_weight_")]
    if not weight_cols or not return_weight_cols:
        return refreshed

    confirmed_weights = refreshed[weight_cols].copy()
    confirmed_weights.columns = [col.removeprefix("weight_") for col in weight_cols]
    return_weights = refreshed[return_weight_cols].copy()
    return_weights.columns = [col.removeprefix("return_weight_") for col in return_weight_cols]
    return_weights = return_weights.reindex(columns=confirmed_weights.columns, fill_value=0.0).fillna(0.0)
    returns = prices.reindex(index=refreshed.index, columns=confirmed_weights.columns).pct_change().fillna(0.0)

    gross_ret = (return_weights * returns).sum(axis=1)
    turnover = (confirmed_weights - confirmed_weights.shift(1).fillna(0.0)).abs().sum(axis=1).rename("turnover")
    trade_cost_rate = (turnover * (fee_rate + slippage_rate)).rename("trade_cost_rate")
    strategy_ret = ((1 + gross_ret) * (1 - trade_cost_rate) - 1).rename("strategy_return")
    nav = (1 + strategy_ret.fillna(0.0)).cumprod().rename("nav")
    if not nav.empty:
        nav.iloc[0] = 1.0
    drawdown = (nav / nav.cummax() - 1).rename("drawdown") if not nav.empty else nav.rename("drawdown")

    refreshed["turnover"] = turnover
    refreshed["trade_cost_rate"] = trade_cost_rate
    refreshed["strategy_return"] = strategy_ret
    refreshed["nav"] = nav
    refreshed["drawdown"] = drawdown
    return refreshed


def apply_selected_signal_momentum_pct_cap_to_target_weights(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    cap_start: float,
    cap_end: float,
    cap_floor: float,
    lookback: int = DEFAULT_LOOKBACK,
    state_lookback: int = 756,
    min_periods: int = 120,
    scope: str = "all",
    risk_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """按当前目标主信号的自身历史动量分位对总仓位做上限裁切。"""
    if cap_floor <= 0 or cap_floor > 1.0:
        raise ValueError("cap_floor must be within (0, 1]")
    if cap_end < cap_start:
        raise ValueError("cap_end must be >= cap_start")

    adjusted = target_weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0).fillna(0.0).copy()
    signal = adjusted.idxmax(axis=1).where(adjusted.max(axis=1) > 1e-12, pd.NA)
    selected_pct = pd.Series(index=prices.index, dtype="float64", name="selected_momentum_percentile")
    triggered = pd.Series(False, index=prices.index, dtype=bool, name="selected_momentum_pct_cap_triggered")
    signal_pct = build_asset_own_momentum_percentile(
        prices,
        lookback=lookback,
        state_lookback=state_lookback,
        min_periods=min_periods,
    )
    risk_set = set(risk_codes or [])
    all_set = set(prices.columns)
    if scope == "all":
        eligible_codes = all_set
    elif scope == "risk":
        eligible_codes = risk_set
    elif scope == "defensive":
        eligible_codes = all_set - risk_set
    else:
        raise ValueError(f"unsupported momentum pct cap scope: {scope}")

    for dt_idx in adjusted.index:
        signal_code = normalize_code(signal.loc[dt_idx])
        if signal_code is None or signal_code not in eligible_codes or signal_code not in signal_pct.columns:
            continue
        pct_value = signal_pct.loc[dt_idx, signal_code]
        if pd.isna(pct_value):
            continue
        selected_pct.loc[dt_idx] = float(pct_value)
        if float(pct_value) < cap_start:
            continue
        if cap_end > cap_start:
            progress = min(max((float(pct_value) - cap_start) / (cap_end - cap_start), 0.0), 1.0)
            cap_value = 1.0 + (cap_floor - 1.0) * progress
        else:
            cap_value = cap_floor
        total_weight = float(adjusted.loc[dt_idx].sum())
        if total_weight > cap_value + 1e-12:
            adjusted.loc[dt_idx, :] = adjusted.loc[dt_idx, :] * (cap_value / total_weight)
            triggered.loc[dt_idx] = True

    return adjusted, selected_pct, triggered


def apply_min_rebalance_threshold_to_target_weights(
    target_weights: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """只有当目标权重变化足够大时，才确认新的目标组合。"""
    adjusted = target_weights.copy()
    blocked = pd.Series(False, index=adjusted.index, dtype=bool, name="rebalance_threshold_blocked")
    if threshold <= 0 or adjusted.empty:
        return adjusted, blocked

    previous = adjusted.iloc[0].copy()
    for dt_idx in adjusted.index[1:]:
        desired = adjusted.loc[dt_idx]
        turnover = float((desired - previous).abs().sum())
        if turnover < threshold:
            adjusted.loc[dt_idx] = previous
            blocked.loc[dt_idx] = True
        else:
            previous = desired.copy()
    return adjusted, blocked


def compute_signal_asset_momentum(
    prices: pd.DataFrame,
    signal: pd.Series,
    lookback: int,
) -> pd.Series:
    """计算用户可见信号标的自身的回看动量。"""
    momentum = prices / prices.shift(lookback) - 1
    signal_momentum = pd.Series(index=prices.index, dtype="float64", name="signal_asset_momentum")
    for dt_idx in prices.index:
        code = normalize_code(signal.loc[dt_idx]) if dt_idx in signal.index else None
        if code is None or code not in momentum.columns:
            signal_momentum.loc[dt_idx] = float("nan")
            continue
        value = momentum.loc[dt_idx, code]
        signal_momentum.loc[dt_idx] = float(value) if pd.notna(value) else float("nan")
    return signal_momentum


def apply_sina_cash_dividend_total_return(hist: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """基于原始收盘价和已实现现金分红构造点位总收益序列。"""
    try:
        dividend = ak.fund_etf_dividend_sina(symbol=symbol)
    except Exception:
        return hist
    if dividend.empty:
        return hist
    dividend = dividend.rename(columns={"日期": "date", "累计分红": "cum_div"})
    if "date" not in dividend.columns or "cum_div" not in dividend.columns:
        return hist
    dividend["date"] = pd.to_datetime(dividend["date"])
    dividend["cum_div"] = pd.to_numeric(dividend["cum_div"], errors="coerce")
    dividend = dividend.dropna(subset=["date", "cum_div"]).sort_values("date")
    if dividend.empty:
        return hist
    # 同一天如果接口返回多条累计分红记录，直接 merge 会把价格面板扩成多行。
    # 这里保留同日最后一条累计值，避免重复日期污染收益链。
    dividend = dividend.groupby("date", as_index=False)["cum_div"].max()

    dividend["cash_div"] = dividend["cum_div"].diff()
    # 首行累计分红无法区分是“历史累积值”还是“当日首次分红”，保守按 0 处理，
    # 避免因为截断历史窗口而把更早的分红一次性算进来。
    dividend["cash_div"] = dividend["cash_div"].fillna(0.0)
    # 若累计分红序列发生回退/重置，说明进入了新的累计周期；
    # 此时应把当前累计值视作本轮新增现金分红，而不是直接裁成 0。
    reset_mask = dividend["cash_div"] < 0
    dividend.loc[reset_mask, "cash_div"] = dividend.loc[reset_mask, "cum_div"]
    dividend["cash_div"] = dividend["cash_div"].clip(lower=0.0)

    adjusted = hist.copy()
    adjusted = adjusted.sort_values("date").reset_index(drop=True)
    adjusted = adjusted.merge(dividend[["date", "cash_div"]], on="date", how="left")
    adjusted["cash_div"] = adjusted["cash_div"].fillna(0.0)
    adjusted["prev_close"] = adjusted["close"].shift(1)
    adjusted["total_return"] = adjusted["close"] / adjusted["prev_close"] - 1

    dividend_mask = adjusted["cash_div"] > 0
    adjusted.loc[dividend_mask, "total_return"] = (
        (adjusted.loc[dividend_mask, "close"] + adjusted.loc[dividend_mask, "cash_div"])
        / adjusted.loc[dividend_mask, "prev_close"]
        - 1
    )

    adjusted["tri_close"] = adjusted["close"].iloc[0] * (1 + adjusted["total_return"].fillna(0.0)).cumprod()
    adjusted["close"] = adjusted["tri_close"]
    adjusted = adjusted[["date", "close"]]
    return adjusted


def apply_structural_break_back_adjustment(
    hist: pd.DataFrame,
    *,
    return_threshold: float = STRUCTURAL_BREAK_RETURN_THRESHOLD,
) -> pd.DataFrame:
    """对 ETF 份额拆分/折算类断点做本地前复权，避免把结构变动当成真实涨跌。"""
    adjusted = hist.copy()
    adjusted = adjusted.sort_values("date").reset_index(drop=True)
    adjusted = adjusted[["date", "close"]].dropna(subset=["close"])
    if len(adjusted) < 2:
        return adjusted

    for row_idx in range(1, len(adjusted)):
        prev_close = float(adjusted.at[row_idx - 1, "close"])
        close = float(adjusted.at[row_idx, "close"])
        if prev_close <= 0 or close <= 0:
            continue
        ratio = close / prev_close
        daily_return = ratio - 1.0
        # ETF 单日真实波动极少超过 45%，这里更可能是拆分、折算或份额合并导致的口径断点。
        if abs(daily_return) <= return_threshold:
            continue
        adjusted.loc[: row_idx - 1, "close"] = adjusted.loc[: row_idx - 1, "close"] * ratio

    return adjusted


def fetch_histories(selected: pd.DataFrame, years: int) -> pd.DataFrame:
    end_date = pd.Timestamp.today().normalize()
    today = end_date.date()
    start_date = max(end_date - pd.DateOffset(years=years), DEFAULT_HISTORY_START)
    frames: list[pd.Series] = []
    adjusted_last_close: dict[str, float] = {}
    raw_last_close: dict[str, float] = {}
    allow_same_day_close = should_accept_same_day_history(today)

    for row in selected.itertuples(index=False):
        hist = ak.fund_etf_hist_sina(symbol=row.sina_symbol)
        if hist.empty:
            raise RuntimeError(f"empty history for {row.code} {row.name}")
        hist["date"] = pd.to_datetime(hist["date"])
        hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
        hist = hist[hist["date"] >= start_date].copy()
        hist = filter_history_to_confirmed_closes(hist, today=today, allow_same_day_close=allow_same_day_close)
        hist = hist[["date", "close"]].dropna()
        if hist.empty:
            raise RuntimeError(f"no {years}y history for {row.code} {row.name}")
        raw_last_close[str(row.code)] = float(hist["close"].iloc[-1])
        hist = apply_sina_cash_dividend_total_return(hist, symbol=row.sina_symbol)
        if hist.empty:
            raise RuntimeError(f"no {years}y history for {row.code} {row.name}")
        series = hist.set_index("date")["close"].rename(row.code)
        adjusted_last_close[str(row.code)] = float(series.iloc[-1])
        frames.append(series)

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.index.name = "date"
    prices = append_temporary_today_bar(
        prices=prices,
        selected=selected,
        adjusted_last_close=adjusted_last_close,
        raw_last_close=raw_last_close,
        today=today,
    )
    return prices


def fetch_realtime_etf_prices() -> dict[str, float]:
    errors: list[str] = []
    try:
        spot = ak.fund_etf_spot_em().copy()
        spot["代码"] = spot["代码"].astype(str)
        spot["最新价"] = pd.to_numeric(spot["最新价"], errors="coerce")
        spot = spot[["代码", "最新价"]].dropna()
        if not spot.empty:
            return dict(zip(spot["代码"], spot["最新价"]))
    except Exception as exc:
        errors.append(f"em: {exc}")

    try:
        spot = ak.fund_etf_category_sina(symbol="ETF基金").copy()
        spot["代码"] = spot["代码"].astype(str).str.replace(r"^(sh|sz)", "", regex=True)
        spot["最新价"] = pd.to_numeric(spot["最新价"], errors="coerce")
        spot = spot[["代码", "最新价"]].dropna()
        if not spot.empty:
            return dict(zip(spot["代码"], spot["最新价"]))
    except Exception as exc:
        errors.append(f"sina: {exc}")

    raise RuntimeError("failed to load realtime ETF prices: " + " | ".join(errors))


def should_append_temporary_today_bar(prices: pd.DataFrame, today: date) -> bool:
    if prices.empty:
        return False
    if pd.Timestamp(today) <= prices.index.max():
        return False
    if pd.Timestamp.now().hour < 15:
        return False
    return is_trading_day(today)


def should_accept_same_day_history(today: date) -> bool:
    if not is_trading_day(today):
        return True
    return pd.Timestamp.now().hour >= 15


def filter_history_to_confirmed_closes(
    hist: pd.DataFrame,
    *,
    today: date,
    allow_same_day_close: bool,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(today).normalize()
    if allow_same_day_close:
        return hist.loc[hist["date"] <= cutoff].copy()
    return hist.loc[hist["date"] < cutoff].copy()


def filter_indexed_frame_to_confirmed_closes(
    frame: pd.DataFrame,
    *,
    today: date | None = None,
    allow_same_day_close: bool | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    effective_today = pd.Timestamp.today().normalize().date() if today is None else today
    same_day_allowed = should_accept_same_day_history(effective_today) if allow_same_day_close is None else allow_same_day_close
    cutoff = pd.Timestamp(effective_today).normalize()
    if same_day_allowed:
        return frame.loc[frame.index <= cutoff].copy()
    return frame.loc[frame.index < cutoff].copy()


def load_core_selected_and_prices(
    *,
    today: date | None = None,
    allow_same_day_close: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 output/core 中缓存的标的池与价格面板，并统一裁到确认收盘口径。"""
    selected = pd.read_csv(CORE_OUTPUT_DIR / "selected_etfs.csv", dtype={"code": str})
    prices = pd.read_csv(CORE_OUTPUT_DIR / "prices.csv", parse_dates=["date"]).set_index("date")
    prices = filter_indexed_frame_to_confirmed_closes(
        prices,
        today=today,
        allow_same_day_close=allow_same_day_close,
    )
    return selected, prices


def append_temporary_today_bar(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    adjusted_last_close: dict[str, float],
    raw_last_close: dict[str, float],
    today: date,
) -> pd.DataFrame:
    if not should_append_temporary_today_bar(prices, today):
        return prices

    try:
        spot_prices = fetch_realtime_etf_prices()
    except Exception:
        return prices

    temp_values: dict[str, float] = {}
    covered_codes = 0
    required_codes: list[str] = []
    missing_required_codes: list[str] = []
    for row in selected.itertuples(index=False):
        code = str(row.code)
        is_optional_fallback = code.startswith(OPTIONAL_REALTIME_FALLBACK_PREFIXES)
        if not is_optional_fallback:
            required_codes.append(code)
        prev_adjusted = adjusted_last_close.get(code)
        prev_raw = raw_last_close.get(code)
        if prev_adjusted is None or prev_raw in (None, 0.0):
            continue
        latest_spot = spot_prices.get(code)
        if latest_spot is None or pd.isna(latest_spot):
            if not is_optional_fallback:
                missing_required_codes.append(code)
            latest_spot = prev_raw
        temp_values[code] = float(prev_adjusted) * float(latest_spot) / float(prev_raw)
        if code in spot_prices and pd.notna(spot_prices[code]):
            covered_codes += 1

    required_coverage = len(required_codes)
    if not temp_values or missing_required_codes or covered_codes < required_coverage:
        return prices

    temp_row = pd.DataFrame([temp_values], index=[pd.Timestamp(today)])
    prices = pd.concat([prices, temp_row], axis=0).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.index.name = "date"
    return prices


def run_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    cash_threshold: float | None = DEFAULT_CASH_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change()
    momentum = prices / prices.shift(lookback) - 1

    signal = momentum.idxmax(axis=1, skipna=True)
    signal[momentum.isna().all(axis=1)] = pd.NA
    max_momentum = momentum.max(axis=1, skipna=True)
    if cash_threshold is not None:
        signal[max_momentum < cash_threshold] = pd.NA

    holding_for_return = signal.shift(1).rename("holding_for_return")
    prev_holding_for_return = holding_for_return.shift(1)
    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")
    per_side_cost = fee_rate + slippage_rate

    for dt_idx in prices.index:
        asset = holding_for_return.loc[dt_idx]
        prev_asset = prev_holding_for_return.loc[dt_idx]
        gross_ret = 0.0
        if pd.notna(asset) and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = float(returns.loc[dt_idx, asset])

        trade_sides = 0
        if pd.isna(asset) and pd.isna(prev_asset):
            trade_sides = 0
        elif pd.notna(asset) and pd.notna(prev_asset):
            trade_sides = 0 if asset == prev_asset else 2
        else:
            trade_sides = 1

        turnover.loc[dt_idx] = float(trade_sides)
        cost_rate = trade_sides * per_side_cost
        trade_cost_rate.loc[dt_idx] = cost_rate
        strategy_ret.loc[dt_idx] = (1 + gross_ret) * (1 - cost_rate) - 1
        winner = signal.loc[dt_idx]
        if pd.notna(winner) and pd.notna(momentum.loc[dt_idx, winner]):
            current_momentum.loc[dt_idx] = float(momentum.loc[dt_idx, winner])
        elif pd.notna(max_momentum.loc[dt_idx]):
            current_momentum.loc[dt_idx] = float(max_momentum.loc[dt_idx])

    nav = (1 + strategy_ret.fillna(0)).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1

    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": current_momentum,
            "max_momentum": max_momentum,
            "signal": signal,
            "target_exposure": signal.notna().astype(float),
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )
    result = finalize_position_columns(
        result,
        selected,
        confirmed_holding=signal.rename("holding"),
        confirmed_exposure=signal.notna().astype(float).rename("exposure"),
        return_holding=holding_for_return,
        return_exposure=holding_for_return.notna().astype(float).rename("exposure_for_return"),
    )
    result = recompute_return_chain(result, prices, fee_rate=fee_rate, slippage_rate=slippage_rate)

    trades_df = build_trades_from_weight_frame(
        result[[col for col in result.columns if col.startswith("weight_")]].rename(columns=lambda c: c.removeprefix("weight_")),
        result["nav"],
        selected,
    )
    return result, trades_df


def build_threshold_dual_signal(
    prices: pd.DataFrame,
    lookback: int,
    absolute_threshold: float = DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    weak_trend_defensive_weight: float = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    signal_quality_method: str = "raw",
    slope_penalty: float = 0.0,
    volatility_penalty: float = 0.0,
    downside_volatility_penalty: float = 0.0,
    r2_penalty: float = 0.0,
    volatility_state_lookback: int = 252,
    volatility_percentile_penalty: float = 0.0,
    downside_volatility_percentile_penalty: float = 0.0,
    volatility_percentile_divisor: float = 0.0,
    confirmation_lookback: int = 0,
    confirmation_top_n: int = 0,
    secondary_stability_method: str = "none",
    secondary_stability_gap: float = 0.0,
    leader_margin: float = 0.0,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    momentum, score = build_signal_quality_score(
        prices,
        lookback=lookback,
        method=signal_quality_method,
        slope_penalty=slope_penalty,
        volatility_penalty=volatility_penalty,
        downside_volatility_penalty=downside_volatility_penalty,
        r2_penalty=r2_penalty,
        volatility_state_lookback=volatility_state_lookback,
        volatility_percentile_penalty=volatility_percentile_penalty,
        downside_volatility_percentile_penalty=downside_volatility_percentile_penalty,
        volatility_percentile_divisor=volatility_percentile_divisor,
    )
    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    risk_score = score[active_risk_codes]
    defensive_score = score[active_defensive_codes]
    risk_confirmation = None
    if confirmation_lookback > 0 and confirmation_top_n > 0:
        risk_confirmation = prices[active_risk_codes] / prices[active_risk_codes].shift(confirmation_lookback) - 1
    secondary_score = build_signal_stability_score(prices, lookback=lookback, method=secondary_stability_method)
    risk_secondary_score = secondary_score[active_risk_codes] if secondary_score is not None else None
    defensive_secondary_score = secondary_score[active_defensive_codes] if secondary_score is not None else None

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    prev_signal: str | None = None

    for dt_idx in prices.index:
        prev_risk = prev_signal if prev_signal in active_risk_codes else None
        prev_def = prev_signal if prev_signal in active_defensive_codes else None
        risk_score_row = risk_score.loc[dt_idx]
        if risk_confirmation is not None:
            risk_score_row = filter_score_row_by_confirmation(risk_score_row, risk_confirmation.loc[dt_idx], confirmation_top_n)
        r_asset = choose_signal_winner_with_margin(
            risk_score_row,
            prev_risk,
            leader_margin,
            None if risk_secondary_score is None else risk_secondary_score.loc[dt_idx],
            secondary_stability_gap,
        )
        d_asset = choose_signal_winner_with_margin(
            defensive_score.loc[dt_idx],
            prev_def,
            leader_margin,
            None if defensive_secondary_score is None else defensive_secondary_score.loc[dt_idx],
            secondary_stability_gap,
        )
        r_score = float(momentum.loc[dt_idx, r_asset]) if r_asset and pd.notna(momentum.loc[dt_idx, r_asset]) else float("nan")
        d_score = float(momentum.loc[dt_idx, d_asset]) if d_asset and pd.notna(momentum.loc[dt_idx, d_asset]) else float("nan")

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
            prev_signal = None
        elif pd.notna(r_score) and r_score > absolute_threshold:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = float(r_score)
            prev_signal = str(r_asset) if r_asset else None
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = weak_trend_defensive_weight
            current_momentum.loc[dt_idx] = float(r_score)
            prev_signal = str(d_asset) if d_asset else None
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            current_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
            prev_signal = str(d_asset) if d_asset else None
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
            prev_signal = None

    return signal, target_exposure, current_momentum, momentum.max(axis=1, skipna=True).rename("max_momentum")


def build_resource_abs08_signal(
    prices: pd.DataFrame,
    lookback: int,
    absolute_threshold: float = DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    resource_abs_threshold: float = DEFAULT_RESOURCE_ABS_THRESHOLD,
    weak_trend_defensive_weight: float = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [code for code in RISK_CODES + [RESOURCE_CODE] if code in prices.columns]
    defensive_codes = [code for code in DEFENSIVE_CODES if code in prices.columns]
    risk_mom = momentum[risk_codes]
    defensive_mom = momentum[defensive_codes]

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        valid_risk = risk_mom.loc[dt_idx].dropna().sort_values(ascending=False)
        valid_def = defensive_mom.loc[dt_idx].dropna().sort_values(ascending=False)
        r_asset = valid_risk.index[0] if not valid_risk.empty else pd.NA
        r_score = float(valid_risk.iloc[0]) if not valid_risk.empty else float("nan")
        d_asset = valid_def.index[0] if not valid_def.empty else pd.NA
        d_score = float(valid_def.iloc[0]) if not valid_def.empty else float("nan")

        candidate_asset = r_asset
        candidate_score = r_score
        if pd.notna(r_asset) and str(r_asset) == RESOURCE_CODE and pd.notna(r_score) and r_score <= resource_abs_threshold:
            if len(valid_risk) >= 2:
                candidate_asset = valid_risk.index[1]
                candidate_score = float(valid_risk.iloc[1])
            else:
                candidate_asset = pd.NA
                candidate_score = float("nan")

        if pd.isna(candidate_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
        elif pd.notna(candidate_score) and candidate_score > absolute_threshold:
            signal.loc[dt_idx] = candidate_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = float(candidate_score)
        elif pd.notna(candidate_score) and candidate_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = weak_trend_defensive_weight
            current_momentum.loc[dt_idx] = float(candidate_score)
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            current_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")

    return signal, target_exposure, current_momentum, momentum.max(axis=1, skipna=True).rename("max_momentum")


def run_threshold_dual_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    absolute_threshold: float = DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    weak_trend_defensive_weight: float = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change()
    signal, target_exposure, current_momentum, max_momentum = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )

    holding_for_return = signal.shift(1).rename("holding_for_return")
    exposure_for_return = target_exposure.shift(1).fillna(0.0).rename("exposure_for_return")
    prev_holding_for_return = holding_for_return.shift(1)
    prev_exposure_for_return = exposure_for_return.shift(1).fillna(0.0)
    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")
    per_side_cost = fee_rate + slippage_rate

    for dt_idx in prices.index:
        asset = holding_for_return.loc[dt_idx]
        prev_asset = prev_holding_for_return.loc[dt_idx]
        weight = float(exposure_for_return.loc[dt_idx]) if pd.notna(exposure_for_return.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure_for_return.loc[dt_idx]) if pd.notna(prev_exposure_for_return.loc[dt_idx]) else 0.0

        gross_ret = 0.0
        if pd.notna(asset) and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = weight * float(returns.loc[dt_idx, asset])

        if pd.isna(asset) and pd.isna(prev_asset):
            day_turnover = abs(weight - prev_weight)
        elif pd.notna(asset) and pd.notna(prev_asset) and asset == prev_asset:
            day_turnover = abs(weight - prev_weight)
        else:
            day_turnover = prev_weight + weight

        turnover.loc[dt_idx] = day_turnover
        cost_rate = day_turnover * per_side_cost
        trade_cost_rate.loc[dt_idx] = cost_rate
        strategy_ret.loc[dt_idx] = (1 + gross_ret) * (1 - cost_rate) - 1

    nav = (1 + strategy_ret.fillna(0.0)).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1

    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": current_momentum,
            "max_momentum": max_momentum,
            "signal": signal,
            "target_exposure": target_exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )
    result = finalize_position_columns(
        result,
        selected,
        confirmed_holding=signal.rename("holding"),
        confirmed_exposure=target_exposure.rename("exposure"),
        return_holding=holding_for_return,
        return_exposure=exposure_for_return,
    )
    result = recompute_return_chain(result, prices, fee_rate=fee_rate, slippage_rate=slippage_rate)

    trades = build_trades_from_weight_frame(
        result[[col for col in result.columns if col.startswith("weight_")]].rename(columns=lambda c: c.removeprefix("weight_")),
        result["nav"],
        selected,
    )
    return result, trades


def run_default_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    market_proxy: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result, _ = run_default_strategy_with_params(
        prices,
        selected,
        params=build_default_strategy_params(),
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        market_proxy=market_proxy,
    )
    result = apply_official_baseline_nav_anchor(result)
    trades = build_trades_from_weight_frame(
        result[[col for col in result.columns if col.startswith("weight_")]].rename(columns=lambda c: c.removeprefix("weight_")),
        result["nav"],
        selected,
    )
    return result, trades


def build_default_strategy_params(
    drop_codes: list[str] | None = None,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> dict[str, object]:
    # 正式默认基线继续保持历史确认过的 28 净值链，对外统一标记为 baseline_cf60top2。
    return {
        "drop_codes": list(DEFAULT_BASELINE_DROP_CODES if drop_codes is None else drop_codes),
        "risk_codes": list(RISK_CODES if risk_codes is None else risk_codes),
        "defensive_codes": list(DEFENSIVE_CODES if defensive_codes is None else defensive_codes),
        "proxy_kind": DEFAULT_REGIME_MIX_PROXY_KIND,
        "signal_quality_method": DEFAULT_SIGNAL_QUALITY_METHOD,
        "signal_slope_penalty": DEFAULT_SIGNAL_SLOPE_PENALTY,
        "signal_volatility_penalty": 0.0,
        "signal_downside_volatility_penalty": 0.0,
        "signal_r2_penalty": 0.0,
        "signal_volatility_state_lookback": 252,
        "signal_volatility_percentile_penalty": 0.0,
        "signal_downside_volatility_percentile_penalty": 0.0,
        "signal_volatility_percentile_divisor": 0.0,
        "signal_confirmation_lookback": 60,
        "signal_confirmation_top_n": 2,
        "signal_secondary_stability_method": "none",
        "signal_secondary_stability_gap": 0.0,
        "signal_leader_margin": DEFAULT_SIGNAL_LEADER_MARGIN,
        "close_top2_gap": DEFAULT_CLOSE_TOP2_GAP,
        "close_top2_risk_cap": DEFAULT_CLOSE_TOP2_RISK_CAP,
        "aggressive_core_weight": DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT,
        "conservative_core_weight": DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT,
        "regime_momentum_cut": DEFAULT_REGIME_MIX_MOMENTUM_CUT,
        "regime_transition_mode": "step",
        "regime_transition_start_cut": 0.0,
        "regime_transition_end_cut": DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        "absolute_momentum_threshold": DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        "dynamic_threshold_mode": "fixed",
        "dynamic_threshold_amount_20_60_cut": 1.0,
        "dynamic_threshold_amount_5_20_cut": 1.0,
        "dynamic_threshold_breadth_cut": 0.0,
        "dynamic_threshold_weak_value": 0.06,
        "dynamic_threshold_tier_step": 0.005,
        "dynamic_threshold_max": 0.065,
        "dynamic_threshold_strong_amount_20_60_cut": 1.05,
        "dynamic_threshold_strong_breadth_cut": 0.02,
        "dynamic_threshold_strong_value": 0.045,
        "volume_ratio_cut": DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT,
        "volume_short_ratio_cut": DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT,
        "volume_breadth_cut": DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT,
        "volume_guard_cap": DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
        "volume_guard_momentum_ceiling": DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING,
        "overheat_drawdown_cut": DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
        "pre_overheat_start_cut": DEFAULT_REGIME_MIX_PRE_OVERHEAT_START_CUT,
        "pre_overheat_end_cut": DEFAULT_REGIME_MIX_PRE_OVERHEAT_END_CUT,
        "pre_overheat_end_exposure": 1.0,
        "overheat_momentum_cut": DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
        "overheat_max_exposure": 1.0,
        "overheat_high_momentum_cut": DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
        "overheat_high_max_exposure": 1.0,
        "overheat_cap_mode": "step",
        "overheat_stability_method": "none",
        "overheat_stability_min_rank": 0.0,
        "overheat_stability_cap": 1.0,
        "signal_selected_volatility_cap_start": 1.0,
        "signal_selected_volatility_cap_end": 1.0,
        "signal_selected_volatility_cap_floor": 1.0,
        "signal_selected_momentum_pct_cap_start": 0.90,
        "signal_selected_momentum_pct_cap_end": 0.95,
        "signal_selected_momentum_pct_cap_floor": 0.50,
        "signal_selected_momentum_pct_cap_scope": "all",
        "signal_selected_momentum_pct_cap_stage": "post_overlay",
        "signal_selected_momentum_pct_lookback": 756,
        "signal_selected_momentum_pct_min_periods": 120,
        "target_min_rebalance_threshold": 0.05,
    }


def run_default_strategy_with_params(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    params: dict[str, object],
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    market_proxy: pd.DataFrame | None = None,
    treasury_code: str = DEFAULT_STRESS_BOND_CODE,
    risk_cap: float = DEFAULT_STRESS_BOND_RISK_CAP,
    ratio_cut: float = DEFAULT_STRESS_BOND_RATIO_CUT,
    breadth_cut: float = DEFAULT_STRESS_BOND_BREADTH_CUT,
    enter_days: int = DEFAULT_STRESS_BOND_ENTER_DAYS,
    exit_days: int = DEFAULT_STRESS_BOND_EXIT_DAYS,
    fill_residual_cash_to_treasury: bool = DEFAULT_STRESS_BOND_FILL_RESIDUAL_CASH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 默认正式基线由多支研究脚本里的稳定模块拼装而成，这里统一收口成正式入口，
    # 让外部调用不需要感知底层实验模块的拆分。
    # 延迟导入是为了避免研究链 helper 反向依赖 run_backtest 时形成循环引用。
    from compare_defensive_persistence import apply_persistent_overlay
    from goal_optimization_common import load_market_volume_proxy
    from market_proxy_common import build_proxy_catalog
    from official_strategy_core import build_official_target_weights

    years = max(int(math.ceil((prices.index.max() - prices.index.min()).days / 365.25)) + 1, DEFAULT_YEARS)
    if market_proxy is None:
        market_proxy = load_market_volume_proxy(years=years, refresh=False)
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(
            market_proxy,
            prices,
            risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        )
    }
    proxy_kind = str(params.get("proxy_kind", DEFAULT_REGIME_MIX_PROXY_KIND))
    if proxy_kind not in proxy_catalog:
        raise ValueError(f"unsupported proxy_kind: {proxy_kind}")
    effective_proxy = proxy_catalog[proxy_kind]

    base_weights, _, base_result = build_official_target_weights(prices, effective_proxy, params)
    risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        defensive_codes=[str(code) for code in params.get("defensive_codes", DEFENSIVE_CODES)],
    )
    proxy = effective_proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_codes].sum(axis=1)
    raw_trigger = (
        (proxy["market_amount_ratio_20_60"] < ratio_cut)
        & (proxy["market_breadth_proxy"] < breadth_cut)
    ).fillna(False)
    stress_mask = build_persistent_trigger_mask(
        raw_trigger,
        enter_days=enter_days,
        exit_days=exit_days,
    )
    result, trades, target_weights = apply_persistent_overlay(
        prices,
        selected,
        effective_proxy,
        params,
        treasury_code=treasury_code,
        risk_cap=risk_cap,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
        enter_days=enter_days,
        exit_days=exit_days,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        fill_residual_cash_to_treasury=fill_residual_cash_to_treasury,
        return_target_weights=True,
    )
    internal_momentum = result["current_momentum"].copy().rename("current_momentum")
    post_overlay_selected_momentum_percentile = None
    post_overlay_selected_momentum_trigger = None
    post_overlay_threshold_blocked = None

    momentum_pct_stage = str(params.get("signal_selected_momentum_pct_cap_stage", "core"))
    momentum_pct_cap_start = float(params.get("signal_selected_momentum_pct_cap_start", 1.0))
    momentum_pct_cap_end = float(params.get("signal_selected_momentum_pct_cap_end", 1.0))
    momentum_pct_cap_floor = float(params.get("signal_selected_momentum_pct_cap_floor", 1.0))
    if momentum_pct_stage == "post_overlay" and (momentum_pct_cap_start < 1.0 or momentum_pct_cap_end < 1.0):
        target_weights, post_overlay_selected_momentum_percentile, post_overlay_selected_momentum_trigger = (
            apply_selected_signal_momentum_pct_cap_to_target_weights(
                prices,
                target_weights,
                cap_start=momentum_pct_cap_start,
                cap_end=momentum_pct_cap_end,
                cap_floor=momentum_pct_cap_floor,
                lookback=DEFAULT_LOOKBACK,
                state_lookback=int(params.get("signal_selected_momentum_pct_lookback", 756)),
                min_periods=int(params.get("signal_selected_momentum_pct_min_periods", 120)),
                scope=str(params.get("signal_selected_momentum_pct_cap_scope", "all")),
                risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
            )
        )

    rebalance_threshold = float(params.get("target_min_rebalance_threshold", 0.0))
    if rebalance_threshold > 0:
        target_weights, post_overlay_threshold_blocked = apply_min_rebalance_threshold_to_target_weights(
            target_weights,
            rebalance_threshold,
        )

    if (
        post_overlay_selected_momentum_percentile is not None
        or post_overlay_threshold_blocked is not None
    ):
        from compare_hs300_regime_fixes import run_target_weights_strategy

        result, trades = run_target_weights_strategy(
            prices,
            selected,
            target_weights,
            internal_momentum,
            fee_rate,
            slippage_rate,
        )
    return_weight_cols = [col for col in result.columns if col.startswith("weight_")]
    if return_weight_cols:
        return_weights = result[return_weight_cols].copy()
        return_weights.columns = [col.removeprefix("weight_") for col in return_weight_cols]
    else:
        return_weights = build_weight_frame_from_holding_exposure(
            result.index,
            [str(code) for code in selected["code"]],
            result["holding"],
            result["exposure"],
        )
    result = finalize_position_columns(
        result,
        selected,
        confirmed_weights=target_weights,
        return_weights=return_weights,
        return_holding=result["holding"] if "holding" in result.columns else None,
        return_exposure=result["exposure"] if "exposure" in result.columns else None,
    )
    result = recompute_return_chain(result, prices, fee_rate=fee_rate, slippage_rate=slippage_rate)
    trades = build_trades_from_weight_frame(target_weights, result["nav"], selected)
    # 默认正式基线内部会用混合后的 regime/core 动量做风控判断。
    # 这里保留这条内部口径供排查使用，同时把 current_momentum 改回信号 ETF 自身动量，
    # 让 CSV、图表和通知看到的数值与用户直觉一致。
    effective_momentum = result["current_momentum"].copy().rename("effective_momentum")
    signal_momentum = compute_signal_asset_momentum(prices, result["signal"], DEFAULT_LOOKBACK)
    result["effective_momentum"] = effective_momentum
    result["signal_asset_momentum"] = signal_momentum
    result["current_momentum"] = signal_momentum
    # base_target_exposure 用来描述“在额外过热降仓之前，基础风险目标仓位有多高”。
    # 这里应复用 build_base_target_weights 返回的 base_result.exposure（已含弱量能保护，但不含后续连续过热压仓），
    # 不能直接读取最终 target_weights，否则 daily_monitor 无法识别这是一次额外降仓。
    result["base_target_exposure"] = base_result["exposure"].reindex(result.index).fillna(0.0)
    if "weak_market_trigger" in base_result.columns:
        result["weak_market_trigger"] = base_result["weak_market_trigger"].reindex(result.index).fillna(False)
    if "top2_close_risk_cap_triggered" in base_result.columns:
        result["top2_close_risk_cap_triggered"] = base_result["top2_close_risk_cap_triggered"].reindex(result.index).fillna(False)
        result["top2_close_gap"] = base_result["top2_close_gap"].reindex(result.index).ffill()
        result["top2_close_risk_cap"] = base_result["top2_close_risk_cap"].reindex(result.index).ffill()
    if "selected_volatility_rank" in base_result.columns:
        result["selected_volatility_rank"] = base_result["selected_volatility_rank"].reindex(result.index)
    if "absolute_momentum_threshold" in base_result.columns:
        result["absolute_momentum_threshold"] = base_result["absolute_momentum_threshold"].reindex(result.index).ffill()
    if "selected_volatility_cap_triggered" in base_result.columns:
        result["selected_volatility_cap_triggered"] = base_result["selected_volatility_cap_triggered"].reindex(result.index).fillna(False)
    if "selected_momentum_percentile" in base_result.columns:
        result["selected_momentum_percentile"] = base_result["selected_momentum_percentile"].reindex(result.index)
    if "selected_momentum_pct_cap_triggered" in base_result.columns:
        result["selected_momentum_pct_cap_triggered"] = base_result["selected_momentum_pct_cap_triggered"].reindex(result.index).fillna(False)
    if post_overlay_selected_momentum_percentile is not None:
        result["selected_momentum_percentile"] = post_overlay_selected_momentum_percentile.reindex(result.index)
    if post_overlay_selected_momentum_trigger is not None:
        result["selected_momentum_pct_cap_triggered"] = post_overlay_selected_momentum_trigger.reindex(result.index).fillna(False)
    if "extra_cap_triggered" in base_result.columns:
        result["extra_cap_triggered"] = base_result["extra_cap_triggered"].reindex(result.index).fillna(False)
    if "extra_cap_reason" in base_result.columns:
        result["extra_cap_reason"] = base_result["extra_cap_reason"].reindex(result.index)
    if "overheat_cap_triggered" in base_result.columns:
        result["overheat_cap_triggered"] = base_result["overheat_cap_triggered"].reindex(result.index).fillna(False)
    if "overheat_high_cap_triggered" in base_result.columns:
        result["overheat_high_cap_triggered"] = base_result["overheat_high_cap_triggered"].reindex(result.index).fillna(False)
    if "overheat_stability_cap_triggered" in base_result.columns:
        result["overheat_stability_cap_triggered"] = base_result["overheat_stability_cap_triggered"].reindex(result.index).fillna(False)
    result["target_exposure"] = target_weights.sum(axis=1).rename("target_exposure")
    result["stress_bond_trigger"] = stress_mask
    if post_overlay_threshold_blocked is not None:
        result["rebalance_threshold_blocked"] = post_overlay_threshold_blocked.reindex(result.index).fillna(False)
    # 保留目标组合权重，供 daily_monitor 直接展示“今日目标组合”，
    # 避免把多资产目标误压缩成“单一主信号 + 总仓位”。
    result = pd.concat([result, target_weights.add_prefix("target_weight_")], axis=1)
    return result, trades


def run_threshold_dual_with_overheat_cap_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    absolute_threshold: float = DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    weak_trend_defensive_weight: float = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    overheat_drawdown_cut: float = -0.03,
    overheat_momentum_cut: float = 0.25,
    overheat_max_exposure: float = 0.50,
    overheat_high_momentum_cut: float | None = None,
    overheat_high_max_exposure: float | None = None,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )

    base_result, _ = run_threshold_dual_strategy(
        prices,
        selected,
        lookback=lookback,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    reduce_mask = (
        signal.isin(active_risk_codes)
        & (target_exposure > overheat_max_exposure)
        & (base_result["drawdown"] >= overheat_drawdown_cut)
        & (current_momentum >= overheat_momentum_cut)
    )
    capped_exposure = target_exposure.copy()
    capped_exposure.loc[reduce_mask] = overheat_max_exposure
    if overheat_high_momentum_cut is not None and overheat_high_max_exposure is not None:
        high_reduce_mask = (
            signal.isin(active_risk_codes)
            & (capped_exposure > overheat_high_max_exposure)
            & (base_result["drawdown"] >= overheat_drawdown_cut)
            & (current_momentum >= overheat_high_momentum_cut)
        )
        capped_exposure.loc[high_reduce_mask] = overheat_high_max_exposure

    returns = prices.pct_change()
    holding_for_return = signal.shift(1).rename("holding_for_return")
    exposure_for_return = capped_exposure.shift(1).fillna(0.0).rename("exposure_for_return")
    prev_holding_for_return = holding_for_return.shift(1)
    prev_exposure_for_return = exposure_for_return.shift(1).fillna(0.0)
    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")
    per_side_cost = fee_rate + slippage_rate

    for dt_idx in prices.index:
        asset = holding_for_return.loc[dt_idx]
        prev_asset = prev_holding_for_return.loc[dt_idx]
        weight = float(exposure_for_return.loc[dt_idx]) if pd.notna(exposure_for_return.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure_for_return.loc[dt_idx]) if pd.notna(prev_exposure_for_return.loc[dt_idx]) else 0.0

        gross_ret = 0.0
        if pd.notna(asset) and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = weight * float(returns.loc[dt_idx, asset])

        if pd.isna(asset) and pd.isna(prev_asset):
            day_turnover = abs(weight - prev_weight)
        elif pd.notna(asset) and pd.notna(prev_asset) and asset == prev_asset:
            day_turnover = abs(weight - prev_weight)
        else:
            day_turnover = prev_weight + weight

        turnover.loc[dt_idx] = day_turnover
        cost_rate = day_turnover * per_side_cost
        trade_cost_rate.loc[dt_idx] = cost_rate
        strategy_ret.loc[dt_idx] = (1 + gross_ret) * (1 - cost_rate) - 1

    nav = (1 + strategy_ret.fillna(0.0)).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1

    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": current_momentum,
            "max_momentum": base_result["max_momentum"],
            "signal": signal,
            "target_exposure": capped_exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )
    result = finalize_position_columns(
        result,
        selected,
        confirmed_holding=signal.rename("holding"),
        confirmed_exposure=capped_exposure.rename("exposure"),
        return_holding=holding_for_return,
        return_exposure=exposure_for_return,
    )
    result = recompute_return_chain(result, prices, fee_rate=fee_rate, slippage_rate=slippage_rate)
    trades = build_trades_from_weight_frame(
        result[[col for col in result.columns if col.startswith("weight_")]].rename(columns=lambda c: c.removeprefix("weight_")),
        result["nav"],
        selected,
    )
    return result, trades


def run_resource_abs08_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    absolute_threshold: float = DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    resource_abs_threshold: float = DEFAULT_RESOURCE_ABS_THRESHOLD,
    weak_trend_defensive_weight: float = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    overheat_drawdown_cut: float | None = DEFAULT_OVERHEAT_DRAWDOWN_CUT,
    overheat_momentum_cut: float | None = DEFAULT_OVERHEAT_MOMENTUM_CUT,
    overheat_max_exposure: float | None = DEFAULT_OVERHEAT_MAX_EXPOSURE,
    overheat_high_momentum_cut: float | None = DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
    overheat_high_max_exposure: float | None = DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal, target_exposure, current_momentum, max_momentum = build_resource_abs08_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        resource_abs_threshold=resource_abs_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
    )

    returns = prices.pct_change()
    final_target_exposure = target_exposure.copy()
    holding_for_return = signal.shift(1).rename("holding_for_return")
    exposure_for_return = target_exposure.shift(1).fillna(0.0).rename("exposure_for_return")
    prev_holding_for_return = holding_for_return.shift(1)
    prev_exposure_for_return = exposure_for_return.shift(1).fillna(0.0)
    strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
    trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
    turnover = pd.Series(0.0, index=prices.index, name="turnover")
    per_side_cost = fee_rate + slippage_rate

    for dt_idx in prices.index:
        asset = holding_for_return.loc[dt_idx]
        prev_asset = prev_holding_for_return.loc[dt_idx]
        weight = float(exposure_for_return.loc[dt_idx]) if pd.notna(exposure_for_return.loc[dt_idx]) else 0.0
        prev_weight = float(prev_exposure_for_return.loc[dt_idx]) if pd.notna(prev_exposure_for_return.loc[dt_idx]) else 0.0
        gross_ret = 0.0
        if pd.notna(asset) and pd.notna(returns.loc[dt_idx, asset]):
            gross_ret = weight * float(returns.loc[dt_idx, asset])
        if pd.isna(asset) and pd.isna(prev_asset):
            day_turnover = abs(weight - prev_weight)
        elif pd.notna(asset) and pd.notna(prev_asset) and asset == prev_asset:
            day_turnover = abs(weight - prev_weight)
        else:
            day_turnover = prev_weight + weight
        turnover.loc[dt_idx] = day_turnover
        cost_rate = day_turnover * per_side_cost
        trade_cost_rate.loc[dt_idx] = cost_rate
        strategy_ret.loc[dt_idx] = (1 + gross_ret) * (1 - cost_rate) - 1

    nav = (1 + strategy_ret.fillna(0.0)).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1
    if (
        overheat_drawdown_cut is not None
        and overheat_momentum_cut is not None
        and overheat_max_exposure is not None
    ):
        risk_codes = [code for code in RISK_CODES + [RESOURCE_CODE] if code in prices.columns]
        capped_exposure = target_exposure.copy()
        reduce_mask = (
            signal.isin(risk_codes)
            & (capped_exposure > overheat_max_exposure)
            & (drawdown >= overheat_drawdown_cut)
            & (current_momentum >= overheat_momentum_cut)
        )
        capped_exposure.loc[reduce_mask] = overheat_max_exposure
        if overheat_high_momentum_cut is not None and overheat_high_max_exposure is not None:
            high_reduce_mask = (
                signal.isin(risk_codes)
                & (capped_exposure > overheat_high_max_exposure)
                & (drawdown >= overheat_drawdown_cut)
                & (current_momentum >= overheat_high_momentum_cut)
            )
            capped_exposure.loc[high_reduce_mask] = overheat_high_max_exposure
        final_target_exposure = capped_exposure
        exposure_for_return = capped_exposure.shift(1).fillna(0.0).rename("exposure_for_return")
        prev_exposure_for_return = exposure_for_return.shift(1).fillna(0.0)
        strategy_ret = pd.Series(0.0, index=prices.index, name="strategy_return")
        trade_cost_rate = pd.Series(0.0, index=prices.index, name="trade_cost_rate")
        turnover = pd.Series(0.0, index=prices.index, name="turnover")
        for dt_idx in prices.index:
            asset = holding_for_return.loc[dt_idx]
            prev_asset = prev_holding_for_return.loc[dt_idx]
            weight = float(exposure_for_return.loc[dt_idx]) if pd.notna(exposure_for_return.loc[dt_idx]) else 0.0
            prev_weight = float(prev_exposure_for_return.loc[dt_idx]) if pd.notna(prev_exposure_for_return.loc[dt_idx]) else 0.0
            gross_ret = 0.0
            if pd.notna(asset) and pd.notna(returns.loc[dt_idx, asset]):
                gross_ret = weight * float(returns.loc[dt_idx, asset])
            if pd.isna(asset) and pd.isna(prev_asset):
                day_turnover = abs(weight - prev_weight)
            elif pd.notna(asset) and pd.notna(prev_asset) and asset == prev_asset:
                day_turnover = abs(weight - prev_weight)
            else:
                day_turnover = prev_weight + weight
            turnover.loc[dt_idx] = day_turnover
            cost_rate = day_turnover * per_side_cost
            trade_cost_rate.loc[dt_idx] = cost_rate
            strategy_ret.loc[dt_idx] = (1 + gross_ret) * (1 - cost_rate) - 1
        nav = (1 + strategy_ret.fillna(0.0)).cumprod()
        nav.iloc[0] = 1.0
        drawdown = nav / nav.cummax() - 1

    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": current_momentum,
            "max_momentum": max_momentum,
            "signal": signal,
            "target_exposure": final_target_exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )
    result = finalize_position_columns(
        result,
        selected,
        confirmed_holding=signal.rename("holding"),
        confirmed_exposure=final_target_exposure.rename("exposure"),
        return_holding=holding_for_return,
        return_exposure=exposure_for_return,
    )
    result = recompute_return_chain(result, prices, fee_rate=fee_rate, slippage_rate=slippage_rate)
    trades = build_trades_from_weight_frame(
        result[[col for col in result.columns if col.startswith("weight_")]].rename(columns=lambda c: c.removeprefix("weight_")),
        result["nav"],
        selected,
    )
    return result, trades


def max_drawdown(nav: pd.Series) -> float:
    dd = nav / nav.cummax() - 1
    return float(dd.min())


def max_drawdown_episode_integral(nav: pd.Series) -> float:
    """计算最大回撤所在区间的积分。"""
    drawdown = (nav / nav.cummax() - 1).fillna(0.0)
    if drawdown.empty:
        return 0.0

    trough_idx = drawdown.idxmin()
    trough_value = float(drawdown.loc[trough_idx])
    if trough_value >= 0:
        return 0.0

    trough_pos = drawdown.index.get_loc(trough_idx)
    peak_nav = float(nav.cummax().loc[trough_idx])

    start_pos = 0
    for pos in range(trough_pos, -1, -1):
        if abs(float(drawdown.iloc[pos])) < 1e-12:
            start_pos = pos
            break

    end_pos = len(drawdown) - 1
    for pos in range(trough_pos + 1, len(drawdown)):
        if float(nav.iloc[pos]) >= peak_nav - 1e-12:
            end_pos = pos
            break

    episode = drawdown.iloc[start_pos : end_pos + 1]
    return float((-episode.clip(upper=0.0)).sum())


def max_drawdown_integral(nav: pd.Series) -> float:
    """计算全历史回撤积分，即整个样本期所有负回撤的累计面积。"""
    drawdown = (nav / nav.cummax() - 1).fillna(0.0)
    if drawdown.empty:
        return 0.0
    return float((-drawdown.clip(upper=0.0)).sum())


def annualized_return(nav: pd.Series) -> float:
    days = (nav.index[-1] - nav.index[0]).days
    years = max(days / 365.25, 1e-9)
    return float(nav.iloc[-1] ** (1 / years) - 1)


def build_strategy_summary(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    selected: pd.DataFrame | None = None,
    exposure_series: pd.Series | None = None,
    include_max_drawdown_integral: bool = False,
    official_anchor: bool = False,
) -> dict[str, float | int | str]:
    """汇总研究脚本常用绩效指标，统一年化/波动率/夏普等基础口径。"""
    effective_result = apply_official_baseline_nav_anchor(result.copy()) if official_anchor else result
    daily_ret = effective_result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(effective_result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    effective_exposure = exposure_series if exposure_series is not None else effective_result["exposure"]

    summary: dict[str, float | int | str] = {
        "total_return": float(effective_result["nav"].iloc[-1] - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown(effective_result["nav"]),
        "trade_count": count_trade_days(trades),
        "trade_action_count": int(len(trades)),
        "avg_exposure": float(effective_exposure.mean()),
        "latest_momentum": float(effective_result["current_momentum"].dropna().iloc[-1]),
        "latest_exposure": float(effective_exposure.iloc[-1]),
    }

    if include_max_drawdown_integral:
        summary["max_drawdown_integral"] = max_drawdown_integral(effective_result["nav"])
        summary["max_drawdown_episode_integral"] = max_drawdown_episode_integral(effective_result["nav"])
        # 兼容旧字段名的同时，补充更直观的别名，避免把“全历史累计面积”和“单段最大回撤区间面积”混淆。
        summary["full_history_drawdown_integral"] = summary["max_drawdown_integral"]
        summary["worst_drawdown_episode_integral"] = summary["max_drawdown_episode_integral"]

    if selected is not None:
        holding_series = effective_result["holding"].dropna()
        latest_holding = normalize_code(holding_series.iloc[-1]) if not holding_series.empty else ""
        latest_theme = ""
        latest_name = ""
        if latest_holding:
            row = selected[selected["code"] == latest_holding]
            if not row.empty:
                latest_theme = str(row.iloc[0]["theme"])
                latest_name = str(row.iloc[0]["name"])
        summary["latest_holding_code"] = latest_holding
        summary["latest_holding_theme"] = latest_theme
        summary["latest_holding_name"] = latest_name
        summary["latest_portfolio"] = get_latest_portfolio_text(selected, effective_result)

    return summary


def build_official_baseline_summary(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    selected: pd.DataFrame | None = None,
    exposure_series: pd.Series | None = None,
    include_max_drawdown_integral: bool = False,
) -> dict[str, float | int | str]:
    """统一输出线上官方基线口径的摘要。"""
    return build_strategy_summary(
        result,
        trades,
        selected=selected,
        exposure_series=exposure_series,
        include_max_drawdown_integral=include_max_drawdown_integral,
        official_anchor=True,
    )


def build_benchmark_nav(prices: pd.DataFrame, benchmark_code: str = "510300") -> pd.Series:
    benchmark_ret = prices[benchmark_code].pct_change().fillna(0.0)
    benchmark_nav = (1 + benchmark_ret).cumprod()
    benchmark_nav.iloc[0] = 1.0
    benchmark_nav.name = "benchmark_nav"
    return benchmark_nav


def build_yearly_return_rows(
    compare: pd.DataFrame,
    *,
    strategy_col: str,
    benchmark_col: str,
    benchmark_return_col: str,
) -> list[dict[str, object]]:
    # 首个自然年若从年中起跑，也保留一条“首段自然年收益”记录，
    # 避免 yearly_returns 从下一年才开始，阅读时误以为首年缺失。
    if compare.empty:
        return []

    compare = compare.sort_index().copy()
    rows: list[dict[str, object]] = []
    first_year = int(compare.index[0].year)
    year_ends = compare.groupby(compare.index.year).tail(1).copy()
    available_years = sorted(int(year) for year in year_ends.index.year.unique())

    for year in available_years:
        end_row = year_ends[year_ends.index.year == year].iloc[-1]
        end_date = year_ends[year_ends.index.year == year].index[-1]
        prev_year = year - 1

        if prev_year in available_years:
            start_row = year_ends[year_ends.index.year == prev_year].iloc[-1]
            start_date = year_ends[year_ends.index.year == prev_year].index[-1]
        elif year == first_year:
            start_row = compare.iloc[0]
            start_date = compare.index[0]
        else:
            continue

        strategy_return = float(end_row[strategy_col] / start_row[strategy_col] - 1)
        benchmark_return = float(end_row[benchmark_col] / start_row[benchmark_col] - 1)
        rows.append(
            {
                "year": int(year),
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "strategy_return": strategy_return,
                benchmark_return_col: benchmark_return,
                "excess_return": strategy_return - benchmark_return,
            }
        )
    return rows


def build_contribution_summary(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    result: pd.DataFrame,
) -> pd.DataFrame:
    returns = prices.pct_change().reindex(result.index)
    prev_nav = result["nav"].shift(1).fillna(1.0)
    total_profit = float(result["nav"].iloc[-1] - 1.0)
    codes = [str(code) for code in selected["code"]]

    weight_cols = [col for col in result.columns if col.startswith("weight_")]
    if weight_cols:
        confirmed_weights = result[weight_cols].copy()
        confirmed_weights.columns = [col.removeprefix("weight_") for col in weight_cols]
    else:
        confirmed_weights = pd.DataFrame(0.0, index=result.index, columns=codes)
        holding_code = result["holding"].map(normalize_code)
        exposure = result["exposure"].fillna(0.0) if "exposure" in result.columns else holding_code.notna().astype(float)
        for dt_idx in result.index:
            asset = holding_code.loc[dt_idx]
            if asset in confirmed_weights.columns:
                confirmed_weights.loc[dt_idx, asset] = float(exposure.loc[dt_idx])

    return_weight_cols = [col for col in result.columns if col.startswith("return_weight_")]
    if return_weight_cols:
        return_weights = result[return_weight_cols].copy()
        return_weights.columns = [col.removeprefix("return_weight_") for col in return_weight_cols]
    else:
        return_weights = confirmed_weights.shift(1).fillna(0.0)

    confirmed_weights = confirmed_weights.reindex(columns=codes, fill_value=0.0).fillna(0.0)
    return_weights = return_weights.reindex(columns=codes, fill_value=0.0).fillna(0.0)
    prev_confirmed_weights = confirmed_weights.shift(1).fillna(0.0)
    gross_profit_by_code = (return_weights * returns.reindex(columns=return_weights.columns).fillna(0.0)).mul(prev_nav, axis=0)
    gross_profit_total = gross_profit_by_code.sum(axis=1)
    # strategy_return 是收益率，gross_profit_total 是金额，二者必须先统一到金额口径再相减；
    # 否则会把成本拖累放大几十倍，直接导致贡献汇总出现大量离谱负值。
    daily_net_profit_total = prev_nav * result["strategy_return"].fillna(0.0)
    daily_cost_drag_total = daily_net_profit_total - gross_profit_total
    turnover = result["turnover"].fillna(0.0) if "turnover" in result.columns else (confirmed_weights - prev_confirmed_weights).abs().sum(axis=1)
    turnover_by_code = (confirmed_weights - prev_confirmed_weights).abs()
    turnover_share = turnover_by_code.div(turnover.replace(0.0, pd.NA), axis=0).fillna(0.0)
    cost_drag_by_code = turnover_share.mul(daily_cost_drag_total, axis=0)
    net_profit_by_code = gross_profit_by_code + cost_drag_by_code
    holding_mask = confirmed_weights.abs() > 1e-12
    total_holding_days = int(holding_mask.any(axis=1).sum())

    code_to_theme = selected.set_index("code")["theme"].to_dict()
    code_to_name = selected.set_index("code")["name"].to_dict()
    rows: list[dict[str, object]] = []

    for code in selected["code"]:
        code = str(code)
        holding_days = int(holding_mask[code].sum())
        gross_profit = float(gross_profit_by_code[code].sum())
        cost_drag = float(cost_drag_by_code[code].sum())
        net_profit = float(net_profit_by_code[code].sum())
        rows.append(
            {
                "code": code,
                "theme": code_to_theme.get(code, ""),
                "name": code_to_name.get(code, ""),
                "holding_days": holding_days,
                "holding_day_share": holding_days / total_holding_days if total_holding_days else 0.0,
                "gross_profit_contribution": gross_profit,
                "cost_drag_contribution": cost_drag,
                "net_profit_contribution": net_profit,
                "gross_contribution_rate": gross_profit / total_profit if total_profit else 0.0,
                "net_contribution_rate": net_profit / total_profit if total_profit else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("net_profit_contribution", ascending=False).reset_index(drop=True)


def build_contribution_overview(result: pd.DataFrame, contribution_df: pd.DataFrame) -> pd.DataFrame:
    """汇总贡献归因总账，便于核对净收益、毛收益与成本拖累是否对齐。"""
    weight_cols = [col for col in result.columns if col.startswith("weight_")]
    if weight_cols:
        active_days = int((result[weight_cols].fillna(0.0).abs().sum(axis=1) > 1e-12).sum())
    else:
        active_days = int(result["holding"].map(normalize_code).notna().sum())
    overview = pd.DataFrame(
        [
            {
                "total_profit": float(result["nav"].iloc[-1] - 1.0),
                "total_gross_profit": float(contribution_df["gross_profit_contribution"].sum()),
                "total_cost_profit": float(contribution_df["cost_drag_contribution"].sum()),
                "total_net_profit": float(contribution_df["net_profit_contribution"].sum()),
                "holding_days": active_days,
            }
        ]
    )
    return overview


def save_contribution_chart(contribution_df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
    chart_df = contribution_df.sort_values("net_profit_contribution", ascending=True)
    labels = [f"{row.theme}({row.code})" for row in chart_df.itertuples(index=False)]
    colors = ["#1f9d55" if value >= 0 else "#d64545" for value in chart_df["net_contribution_rate"]]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(labels, chart_df["net_contribution_rate"], color=colors)
    ax.axvline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("Pool Contribution Rate to Total Return", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Contribution Rate")
    fig.tight_layout()
    save_figure_atomic(fig, output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_episode_holdings(
    nav: pd.DataFrame,
    theme_map: dict[str, str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    tol: float = 1e-10,
) -> str:
    """按区间内真实权重均值总结主要持仓，避免把组合仓位误压缩成单一 holding。"""
    weight_cols = [col for col in nav.columns if col.startswith("weight_")]
    if weight_cols:
        weight_frame = nav.loc[start_date:end_date, weight_cols].copy()
        weight_frame.columns = [col.removeprefix("weight_") for col in weight_cols]
        avg_weights = weight_frame.fillna(0.0).mean().sort_values(ascending=False)
        avg_weights = avg_weights[avg_weights > tol].head(3)
        if not avg_weights.empty:
            return "; ".join(
                f"{theme_map.get(code, code)}({code}) {share:.1%}" for code, share in avg_weights.items()
            )

    period_holdings = nav.loc[start_date:end_date, "holding_code"].dropna()
    if period_holdings.empty:
        return ""
    top_holdings = period_holdings.value_counts(normalize=True).head(3)
    return "; ".join(
        f"{theme_map.get(code, code)}({code}) {share:.1%}" for code, share in top_holdings.items()
    )


def build_drawdown_episode_report(
    nav: pd.DataFrame,
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    selected: pd.DataFrame,
    compare: pd.DataFrame,
) -> pd.DataFrame:
    """基于当前 core 结果构造回撤分段报告，保证 analysis 与正式回测口径一致。"""
    theme_map = dict(zip(selected["code"], selected["theme"]))
    nav_frame = nav.copy()
    nav_frame["holding_code"] = nav_frame["holding"].map(normalize_code)
    drawdown = nav_frame["nav"] / nav_frame["nav"].cummax() - 1
    underwater = drawdown < 0

    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_segment = False
    start: pd.Timestamp | None = None
    for dt_idx, is_underwater in underwater.items():
        if is_underwater and not in_segment:
            in_segment = True
            start = dt_idx
        elif not is_underwater and in_segment and start is not None:
            segments.append((start, dt_idx))
            in_segment = False
            start = None
    if in_segment and start is not None:
        segments.append((start, nav_frame.index[-1]))

    rows: list[dict[str, object]] = []
    for start_date, end_date in segments:
        segment_dd = drawdown.loc[start_date:end_date]
        trough_date = segment_dd.idxmin()
        peak_date = nav_frame.loc[:start_date, "nav"].idxmax()
        peak_nav = float(nav_frame.loc[peak_date, "nav"])
        trough_nav = float(nav_frame.loc[trough_date, "nav"])
        recovered = is_recovered(float(drawdown.loc[end_date]))

        top_holdings_text = summarize_episode_holdings(nav_frame, theme_map, peak_date, end_date)
        asset_rets = (prices.loc[trough_date] / prices.loc[peak_date] - 1).dropna().sort_values()
        worst_assets_text = "; ".join(
            f"{theme_map.get(code, code)}({code}) {ret:.1%}" for code, ret in asset_rets.head(3).items()
        )
        trade_action_count = int(
            (
                (pd.to_datetime(trades["date"], errors="coerce") >= peak_date)
                & (pd.to_datetime(trades["date"], errors="coerce") <= end_date)
            ).sum()
        ) if not trades.empty else 0
        switches = count_rebalance_days(trades, peak_date, end_date)
        peak_holding = normalize_code(nav_frame.loc[peak_date, "holding"])
        trough_holding = normalize_code(nav_frame.loc[trough_date, "holding"])
        recovery_holding = normalize_code(nav_frame.loc[end_date, "holding"])

        rows.append(
            {
                "peak_date": peak_date.date().isoformat(),
                "start_date": start_date.date().isoformat(),
                "trough_date": trough_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "peak_to_trough_days": int(nav_frame.loc[peak_date:trough_date].shape[0] - 1),
                "recovery_days": int(nav_frame.loc[trough_date:end_date].shape[0] - 1) if recovered else "",
                "total_days": int(nav_frame.loc[peak_date:end_date].shape[0] - 1),
                "max_drawdown": float(segment_dd.min()),
                "strategy_peak_to_trough_return": float(trough_nav / peak_nav - 1),
                "hs300_peak_to_trough_return": float(
                    compare.loc[trough_date, "benchmark_nav"] / compare.loc[peak_date, "benchmark_nav"] - 1
                ),
                "peak_holding": f"{theme_map.get(peak_holding, peak_holding)}({peak_holding})" if peak_holding else "",
                "trough_holding": f"{theme_map.get(trough_holding, trough_holding)}({trough_holding})" if trough_holding else "",
                "recovery_holding": (
                    f"{theme_map.get(recovery_holding, recovery_holding)}({recovery_holding})" if recovery_holding else ""
                ),
                "switches_in_episode": switches,
                "trade_actions_in_episode": trade_action_count,
                "top_holdings": top_holdings_text,
                "worst_assets_peak_to_trough": worst_assets_text,
                "recovered": recovered,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "peak_date",
                "start_date",
                "trough_date",
                "end_date",
                "peak_to_trough_days",
                "recovery_days",
                "total_days",
                "max_drawdown",
                "strategy_peak_to_trough_return",
                "hs300_peak_to_trough_return",
                "peak_holding",
                "trough_holding",
                "recovery_holding",
                "switches_in_episode",
                "trade_actions_in_episode",
                "top_holdings",
                "worst_assets_peak_to_trough",
                "recovered",
            ]
        )
    return pd.DataFrame(rows).sort_values("max_drawdown").reset_index(drop=True)


def save_analysis_drawdown_charts(result: pd.DataFrame) -> None:
    """同步输出 analysis 目录下的回撤图，避免图表仍停留在旧回测结果。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
    drawdown = result["nav"] / result["nav"].cummax() - 1

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(drawdown.index, drawdown, color="#d64545", linewidth=1.8)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("历史回撤相对净值高点", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    save_figure_atomic(fig, ANALYSIS_OUTPUT_DIR / "historical_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    ax1.plot(result.index, result["nav"], color="#2b6cb0", linewidth=2.0)
    ax1.set_title("策略净值与回撤", loc="left", fontsize=16, fontweight="bold")
    ax1.set_ylabel("净值")

    ax2.plot(drawdown.index, drawdown, color="#d64545", linewidth=1.8)
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax2.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax2.set_xlabel("日期")
    ax2.set_ylabel("回撤")
    ax2.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")

    fig.tight_layout()
    save_figure_atomic(fig, ANALYSIS_OUTPUT_DIR / "nav_and_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_analysis_outputs(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    compare_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
) -> None:
    """同步刷新 output/analysis，避免 core 已更新但分析文件仍停留在旧快照。"""
    contribution_overview = build_contribution_overview(result, contribution_df)
    write_dataframe_csv_atomic(contribution_df, ANALYSIS_OUTPUT_DIR / "contribution_summary.csv", index=False)
    write_dataframe_csv_atomic(contribution_overview, ANALYSIS_OUTPUT_DIR / "contribution_overview.csv", index=False)
    save_contribution_chart(contribution_df, ANALYSIS_OUTPUT_DIR / "contribution_rate.png")

    drawdown_report = build_drawdown_episode_report(result, prices, trades, selected, compare_df)
    write_dataframe_csv_atomic(drawdown_report, ANALYSIS_OUTPUT_DIR / "drawdown_episodes.csv", index=False)
    save_analysis_drawdown_charts(result)


def save_outputs(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    lookback: int,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
) -> None:
    ensure_output_dirs()
    prices_to_save = prices.copy()
    prices_to_save.index.name = "date"
    result_to_save = result.copy()
    result_to_save.index.name = "date"
    write_dataframe_csv_atomic(selected, CORE_OUTPUT_DIR / "selected_etfs.csv", index=False)
    write_dataframe_csv_atomic(prices_to_save, CORE_OUTPUT_DIR / "prices.csv")
    write_dataframe_csv_atomic(result_to_save, CORE_OUTPUT_DIR / "backtest_nav.csv")
    write_dataframe_csv_atomic(trades, CORE_OUTPUT_DIR / "trades.csv", index=False)
    momentum_percentile_snapshot = build_current_etf_momentum_percentile_table(
        selected,
        prices,
        lookback=lookback,
    )
    write_dataframe_csv_atomic(
        momentum_percentile_snapshot,
        CORE_OUTPUT_DIR / "etf_momentum_percentiles.csv",
        index=False,
    )
    # 与各分支回测输出保持一致，统一把净值列落成 historical_nav，
    # 避免外部按同一 schema 读取不同策略目录时因列名不一致出错。
    write_dataframe_csv_atomic(
        result_to_save[["nav"]].rename(columns={"nav": "historical_nav"}),
        CORE_OUTPUT_DIR / "historical_nav.csv",
    )

    buy_trades = trades[trades["action"] == "BUY"]
    sell_trades = trades[trades["action"] == "SELL"]

    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result.index, result["nav"], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    if not buy_trades.empty:
        ax.scatter(
            buy_trades["date"],
            buy_trades["nav"],
            color="#1f9d55",
            marker="^",
            s=55,
            label="Buy",
            zorder=3,
        )
    if not sell_trades.empty:
        ax.scatter(
            sell_trades["date"],
            sell_trades["nav"],
            color="#d64545",
            marker="v",
            s=55,
            label="Sell",
            zorder=3,
        )
    ax.set_title(f"ETF Strategy NAV ({strategy_name}, lookback={lookback})", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / "nav_with_trades.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result.index, result["current_momentum"], color="#dd6b20", linewidth=2.0)
    ax.axhline(0.0, color="#4a5568", linestyle="--", linewidth=1.0)
    ax.set_title(f"Signal Momentum (trailing {lookback} trading days)", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Momentum")
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / "momentum_series.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    drawdown = result["drawdown"]
    max_dd_date = drawdown.idxmin()
    max_dd_value = float(drawdown.loc[max_dd_date])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(drawdown.index, drawdown, color="#d64545", linewidth=1.8)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax.scatter([max_dd_date], [max_dd_value], color="#9b2c2c", s=70, zorder=3)
    ax.annotate(
        f"Max DD {max_dd_value:.2%}\n{max_dd_date.date()}",
        xy=(max_dd_date, max_dd_value),
        xytext=(12, -8),
        textcoords="offset points",
        fontsize=10,
        color="#742a2a",
        ha="left",
        va="top",
    )
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("Historical Drawdown vs Running Peak", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / "max_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")
    compare_df = pd.concat([result["nav"], benchmark_nav], axis=1)
    write_dataframe_csv_atomic(compare_df, CORE_OUTPUT_DIR / "strategy_vs_hs300.csv")
    rows = build_yearly_return_rows(
        compare_df,
        strategy_col="nav",
        benchmark_col="benchmark_nav",
        benchmark_return_col="hs300_etf_return",
    )
    write_dataframe_csv_atomic(pd.DataFrame(rows), CORE_OUTPUT_DIR / "yearly_returns.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["nav"], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    ax.plot(compare_df.index, compare_df["benchmark_nav"], color="#d64545", linewidth=2.0, label="HS300 Benchmark")
    ax.set_title("Strategy vs HS300 Benchmark", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, CORE_OUTPUT_DIR / "strategy_vs_hs300.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    contribution_df = build_contribution_summary(prices, selected, result)
    write_dataframe_csv_atomic(contribution_df, CORE_OUTPUT_DIR / "contribution_summary.csv", index=False)
    save_contribution_chart(contribution_df, CORE_OUTPUT_DIR / "contribution_rate.png")
    save_contribution_chart(contribution_df, CORE_OUTPUT_DIR / "contribution_rate_cn.png")
    save_analysis_outputs(selected, prices, result, trades, compare_df, contribution_df)


def print_summary(
    selected: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    lookback: int,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
) -> None:
    summary = {
        "strategy": strategy_name,
        "lookback_days": lookback,
        **build_strategy_summary(
            result,
            trades,
            selected=selected,
            include_max_drawdown_integral=True,
            official_anchor=(strategy_name == DEFAULT_STRATEGY_NAME),
        ),
        "avg_trade_cost_rate": float(result["trade_cost_rate"][result["trade_cost_rate"] > 0].mean() if (result["trade_cost_rate"] > 0).any() else 0.0),
    }

    print("Fixed ETF pool:")
    print(selected.to_string(index=False))
    print("\nBacktest summary:")
    for key, value in summary.items():
        if isinstance(value, float) and math.isfinite(value):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ETF 动量轮动正式回测。")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--cash-threshold", type=float, default=DEFAULT_CASH_THRESHOLD, help="若最强动量低于该阈值，则切到空仓。")
    parser.add_argument("--overheat-drawdown-cut", type=float, default=DEFAULT_OVERHEAT_DRAWDOWN_CUT, help="当回撤处于高位且动量过热时触发降仓。")
    parser.add_argument("--overheat-momentum-cut", type=float, default=DEFAULT_OVERHEAT_MOMENTUM_CUT, help="高位过热降仓对应的动量阈值。")
    parser.add_argument("--overheat-max-exposure", type=float, default=DEFAULT_OVERHEAT_MAX_EXPOSURE, help="命中过热规则后的风险仓上限。")
    parser.add_argument("--overheat-high-momentum-cut", type=float, default=DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT, help="极热状态对应的第二档动量阈值。")
    parser.add_argument("--overheat-high-max-exposure", type=float, default=DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE, help="命中极热规则后的更严格风险仓上限。")
    parser.add_argument("--disable-overheat-cap", action="store_true", help="关闭默认正式基线，改跑更简化的 plain dual momentum 对照策略。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = load_default_strategy_backtest_pool()
    prices = fetch_histories(selected, years=args.years)
    if args.disable_overheat_cap:
        result, trades = run_threshold_dual_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        strategy_name = "threshold_dual_momentum"
    else:
        result, trades = run_default_strategy(
            prices,
            selected,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        strategy_name = DEFAULT_STRATEGY_NAME
    save_outputs(selected, prices, result, trades, lookback=args.lookback, strategy_name=strategy_name)
    print_summary(selected, result, trades, lookback=args.lookback, strategy_name=strategy_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
