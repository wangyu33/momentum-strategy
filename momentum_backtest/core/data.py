"""正式策略核心数据加载。"""

from __future__ import annotations

from datetime import date

import akshare as ak
import pandas as pd

from .config import CORE_OUTPUT_DIR, DEFAULT_HISTORY_START
from .io import is_trading_day
from .validation import persist_price_validation_report


OPTIONAL_REALTIME_FALLBACK_PREFIXES = ("511",)


def apply_sina_cash_dividend_total_return(hist: pd.DataFrame, symbol: str) -> pd.DataFrame:
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

    dividend = dividend.groupby("date", as_index=False)["cum_div"].max()
    dividend["cash_div"] = dividend["cum_div"].diff().fillna(0.0)
    reset_mask = dividend["cash_div"] < 0
    dividend.loc[reset_mask, "cash_div"] = dividend.loc[reset_mask, "cum_div"]
    dividend["cash_div"] = dividend["cash_div"].clip(lower=0.0)

    adjusted = hist.copy().sort_values("date").reset_index(drop=True)
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
    return adjusted[["date", "close"]]


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

    if not temp_values or missing_required_codes or covered_codes < len(required_codes):
        return prices

    temp_row = pd.DataFrame([temp_values], index=[pd.Timestamp(today)])
    prices = pd.concat([prices, temp_row], axis=0).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.index.name = "date"
    return prices


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
    persist_price_validation_report(prices)
    return prices


def load_core_selected_and_prices(
    *,
    today: date | None = None,
    allow_same_day_close: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(CORE_OUTPUT_DIR / "selected_etfs.csv", dtype={"code": str})
    prices = pd.read_csv(CORE_OUTPUT_DIR / "prices.csv", parse_dates=["date"]).set_index("date")
    prices = filter_indexed_frame_to_confirmed_closes(
        prices,
        today=today,
        allow_same_day_close=allow_same_day_close,
    )
    persist_price_validation_report(prices)
    return selected, prices
