"""正式策略核心的市场代理与 HS300 状态模块。"""

from __future__ import annotations

import akshare as ak
import pandas as pd

from .config import DEFAULT_YEARS, MONITOR_OUTPUT_DIR, RISK_CODES
from .io import write_dataframe_csv_atomic
from .signals import build_threshold_dual_signal, resolve_strategy_universe


MARKET_VOLUME_CACHE_PATH = MONITOR_OUTPUT_DIR / "market_volume_proxy.csv"


def fetch_market_volume_proxy(years: int) -> pd.DataFrame:
    end_date = pd.Timestamp.today().normalize()
    start_date = (end_date - pd.DateOffset(years=years + 1)).strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")

    def load_one(symbol: str, prefix: str) -> pd.DataFrame:
        errors: list[str] = []
        loaders = [
            ("tx", lambda: ak.stock_zh_index_daily_tx(symbol=symbol, start_date=start_date, end_date=end_date_str)),
            ("em", lambda: ak.stock_zh_index_daily_em(symbol=symbol, start_date=start_date, end_date=end_date_str)),
        ]
        for source_name, loader in loaders:
            try:
                df = loader()
                if df is None or df.empty:
                    errors.append(f"{source_name}: empty")
                    continue
                df = df.rename(columns={"close": f"{prefix}_close", "amount": f"{prefix}_amount"})[
                    ["date", f"{prefix}_close", f"{prefix}_amount"]
                ].copy()
                df["date"] = pd.to_datetime(df["date"])
                df[f"{prefix}_close"] = pd.to_numeric(df[f"{prefix}_close"], errors="coerce")
                df[f"{prefix}_amount"] = pd.to_numeric(df[f"{prefix}_amount"], errors="coerce")
                return df
            except Exception as exc:
                errors.append(f"{source_name}: {exc}")
        raise RuntimeError(f"failed to load {symbol}: {' | '.join(errors)}")

    sh = load_one("sh000001", "sh")
    sz = load_one("sz399106", "sz")

    proxy = sh.merge(sz, on="date", how="inner").dropna().sort_values("date")
    proxy["market_amount"] = proxy["sh_amount"] + proxy["sz_amount"]
    proxy["market_amount_ma20"] = proxy["market_amount"].rolling(20).mean()
    proxy["market_amount_ma60"] = proxy["market_amount"].rolling(60).mean()
    proxy["market_amount_ratio_20_60"] = proxy["market_amount_ma20"] / proxy["market_amount_ma60"]
    proxy["market_amount_ratio_5_20"] = proxy["market_amount"].rolling(5).mean() / proxy["market_amount_ma20"]
    proxy["market_breadth_proxy"] = (
        proxy["sh_close"] / proxy["sh_close"].shift(20) - 1
        + proxy["sz_close"] / proxy["sz_close"].shift(20) - 1
    ) / 2
    return proxy.set_index("date")


def load_market_volume_proxy(years: int, refresh: bool) -> pd.DataFrame:
    required_cols = {
        "date",
        "market_amount",
        "market_amount_ma20",
        "market_amount_ma60",
        "market_amount_ratio_20_60",
        "market_amount_ratio_5_20",
        "market_breadth_proxy",
    }
    end_date = pd.Timestamp.today().normalize()
    required_start = end_date - pd.DateOffset(years=max(years, DEFAULT_YEARS) + 1)
    stale_cutoff = end_date - pd.Timedelta(days=7)

    if not refresh and MARKET_VOLUME_CACHE_PATH.exists():
        try:
            proxy = pd.read_csv(MARKET_VOLUME_CACHE_PATH, parse_dates=["date"])
            if not required_cols.issubset(proxy.columns):
                raise ValueError("market volume cache missing required columns")
            if proxy.empty:
                raise ValueError("market volume cache is empty")
            if pd.Timestamp(proxy["date"].min()).normalize() > required_start.normalize():
                raise ValueError("market volume cache coverage is too short")
            if pd.Timestamp(proxy["date"].max()).normalize() < stale_cutoff.normalize():
                raise ValueError("market volume cache is stale")
            return proxy.set_index("date")
        except Exception:
            pass

    proxy = fetch_market_volume_proxy(years=years)
    write_dataframe_csv_atomic(proxy.reset_index(), MARKET_VOLUME_CACHE_PATH, index=False)
    return proxy


def build_parametrized_hs300_trend_filter(
    prices: pd.DataFrame,
    mom60_cut: float,
    mom120_cut: float,
    ma_window: int,
) -> pd.Series:
    hs300_code = "510300"
    close = prices[hs300_code]
    mom60 = close / close.shift(60) - 1
    mom120 = close / close.shift(120) - 1
    ma = close.rolling(ma_window).mean()
    return ((mom60 > mom60_cut) & (mom120 > mom120_cut) & (close > ma)).rename("hs300_trend_filter")


def build_risk_proxy_features(
    prices: pd.DataFrame,
    risk_codes: list[str] | None = None,
) -> pd.DataFrame:
    active_risk_codes, _ = resolve_strategy_universe(prices, risk_codes=risk_codes, defensive_codes=[])
    if not active_risk_codes:
        raise RuntimeError("missing risk ETF prices for proxy construction")

    risk_prices = prices[active_risk_codes].copy()
    ret20 = risk_prices / risk_prices.shift(20) - 1
    above_ma20 = (risk_prices > risk_prices.rolling(20).mean()).astype(float)
    above_ma60 = (risk_prices > risk_prices.rolling(60).mean()).astype(float)
    pos20 = (ret20 > 0).astype(float)

    features = pd.DataFrame(index=prices.index)
    features["eq20"] = ret20.mean(axis=1)
    features["median20"] = ret20.median(axis=1)
    features["pos20_frac"] = pos20.mean(axis=1)
    features["above_ma20_frac"] = above_ma20.mean(axis=1)
    features["above_ma60_frac"] = above_ma60.mean(axis=1)
    features["pos20_ratio_20_60"] = features["pos20_frac"].rolling(20).mean() / features["pos20_frac"].rolling(60).mean()
    features["pos20_ratio_5_20"] = features["pos20_frac"].rolling(5).mean() / features["pos20_frac"].rolling(20).mean()
    features["above_ma20_ratio_20_60"] = (
        features["above_ma20_frac"].rolling(20).mean() / features["above_ma20_frac"].rolling(60).mean()
    )
    features["above_ma20_ratio_5_20"] = (
        features["above_ma20_frac"].rolling(5).mean() / features["above_ma20_frac"].rolling(20).mean()
    )
    features["breadth_blend"] = 0.5 * features["eq20"] + 0.5 * (features["above_ma20_frac"] - 0.5)
    return features


def build_proxy_catalog(
    base_proxy: pd.DataFrame,
    prices: pd.DataFrame,
    risk_codes: list[str] | None = None,
) -> list[dict[str, object]]:
    features = build_risk_proxy_features(prices, risk_codes=risk_codes)
    base_proxy = base_proxy.reindex(prices.index).ffill().copy()

    def make_proxy(
        name: str,
        ratio_20_60: pd.Series,
        ratio_5_20: pd.Series,
        breadth: pd.Series,
        description: str,
        ratio_cuts: tuple[float, ...],
        short_ratio_cuts: tuple[float, ...],
        breadth_cuts: tuple[float, ...],
    ) -> dict[str, object]:
        proxy = pd.DataFrame(index=prices.index)
        proxy["market_amount_ratio_20_60"] = ratio_20_60
        proxy["market_amount_ratio_5_20"] = ratio_5_20
        proxy["market_breadth_proxy"] = breadth
        return {
            "name": name,
            "proxy": proxy,
            "description": description,
            "ratio_cuts": ratio_cuts,
            "short_ratio_cuts": short_ratio_cuts,
            "breadth_cuts": breadth_cuts,
        }

    return [
        make_proxy(
            "baseline_sh_sz",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            base_proxy["market_breadth_proxy"],
            "维持当前上证+深证综合成交额代理与指数广度代理。",
            (0.91,),
            (0.90,),
            (-0.01,),
        ),
        make_proxy(
            "hybrid_eq20",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.5 * base_proxy["market_breadth_proxy"] + 0.5 * features["eq20"],
            "保留当前成交额代理，仅把广度替换为指数广度与风险 ETF 等权 20 日收益的混合。",
            (0.91,),
            (0.90,),
            (-0.03, -0.02, -0.01, 0.00, 0.01),
        ),
        make_proxy(
            "hybrid_median20",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.5 * base_proxy["market_breadth_proxy"] + 0.5 * features["median20"],
            "保留当前成交额代理，仅把广度替换为指数广度与风险 ETF 中位数 20 日收益的混合。",
            (0.91,),
            (0.90,),
            (-0.03, -0.02, -0.01, 0.00, 0.01),
        ),
        make_proxy(
            "hybrid_breadth_blend",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.5 * base_proxy["market_breadth_proxy"] + 0.5 * features["breadth_blend"],
            "保留当前成交额代理，仅把广度替换为指数广度与风险 ETF 站上 20 日均线比例/20 日收益混合。",
            (0.91,),
            (0.90,),
            (-0.04, -0.03, -0.02, -0.01, 0.00, 0.01),
        ),
        make_proxy(
            "hybrid_breadth_blend_25_75",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.25 * base_proxy["market_breadth_proxy"] + 0.75 * features["breadth_blend"],
            "保留当前成交额代理，仅把广度替换为 25% 指数广度 + 75% 风险 ETF 宽度混合。",
            (0.91,),
            (0.90,),
            (-0.04, -0.03, -0.02, -0.01, 0.00),
        ),
        make_proxy(
            "hybrid_breadth_blend_75_25",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.75 * base_proxy["market_breadth_proxy"] + 0.25 * features["breadth_blend"],
            "保留当前成交额代理，仅把广度替换为 75% 指数广度 + 25% 风险 ETF 宽度混合。",
            (0.91,),
            (0.90,),
            (-0.04, -0.03, -0.02, -0.01, 0.00),
        ),
        make_proxy(
            "etf_pos20_regime",
            features["pos20_ratio_20_60"],
            features["pos20_ratio_5_20"],
            features["eq20"],
            "量能代理改为风险 ETF 横截面 20 日正收益占比的强弱变化。",
            (0.92, 0.96, 1.00),
            (0.90, 0.95, 1.00),
            (-0.03, -0.02, -0.01, 0.00),
        ),
        make_proxy(
            "etf_above_ma20_regime",
            features["above_ma20_ratio_20_60"],
            features["above_ma20_ratio_5_20"],
            features["median20"],
            "量能代理改为风险 ETF 站上 20 日均线比例的强弱变化。",
            (0.92, 0.96, 1.00),
            (0.90, 0.95, 1.00),
            (-0.03, -0.02, -0.01, 0.00),
        ),
        make_proxy(
            "etf_above_ma20_blend",
            features["above_ma20_ratio_20_60"],
            features["above_ma20_ratio_5_20"],
            features["breadth_blend"],
            "量能代理改为风险 ETF 站上 20 日均线比例，广度使用收益与宽度混合。",
            (0.92, 0.96, 1.00),
            (0.90, 0.95, 1.00),
            (-0.03, -0.02, -0.01, 0.00),
        ),
    ]


def build_dynamic_core_target_weights(
    prices: pd.DataFrame,
    lookback: int,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
    core_weight: float,
    hs300_mom60_cut: float,
    hs300_mom120_cut: float,
    hs300_ma_window: int,
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
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    signal, target_exposure, current_momentum, _ = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        signal_quality_method=signal_quality_method,
        slope_penalty=slope_penalty,
        volatility_penalty=volatility_penalty,
        downside_volatility_penalty=downside_volatility_penalty,
        r2_penalty=r2_penalty,
        volatility_state_lookback=volatility_state_lookback,
        volatility_percentile_penalty=volatility_percentile_penalty,
        downside_volatility_percentile_penalty=downside_volatility_percentile_penalty,
        volatility_percentile_divisor=volatility_percentile_divisor,
        confirmation_lookback=confirmation_lookback,
        confirmation_top_n=confirmation_top_n,
        secondary_stability_method=secondary_stability_method,
        secondary_stability_gap=secondary_stability_gap,
        leader_margin=leader_margin,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
    )
    trend_filter = build_parametrized_hs300_trend_filter(
        prices,
        mom60_cut=hs300_mom60_cut,
        mom120_cut=hs300_mom120_cut,
        ma_window=hs300_ma_window,
    )
    hs300_code = "510300"
    hs300_mom60 = prices[hs300_code] / prices[hs300_code].shift(60) - 1
    dynamic_core_weight = trend_filter.astype(float) * core_weight
    satellite_scale = 1.0 - dynamic_core_weight

    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype="float64")
    target_weights[hs300_code] += dynamic_core_weight
    for code in prices.columns:
        code_mask = signal == code
        if code_mask.any():
            target_weights.loc[code_mask, code] += satellite_scale.loc[code_mask] * target_exposure.loc[code_mask]

    blended_momentum = (satellite_scale * current_momentum + dynamic_core_weight * hs300_mom60).rename("current_momentum")
    return target_weights, blended_momentum, signal, target_exposure, trend_filter
