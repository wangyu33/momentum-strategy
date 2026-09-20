"""正式策略核心信号层。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_DUAL_CASH_EXIT_THRESHOLD,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    DEFENSIVE_CODES,
    RISK_CODES,
)


def resolve_strategy_universe(
    prices: pd.DataFrame,
    *,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    effective_risk_codes = [code for code in (RISK_CODES if risk_codes is None else risk_codes) if code in prices.columns]
    effective_defensive_codes = [
        code for code in (DEFENSIVE_CODES if defensive_codes is None else defensive_codes) if code in prices.columns
    ]
    return effective_risk_codes, effective_defensive_codes


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
    returns = prices.pct_change(fill_method=None)
    if downside:
        vol = returns.where(returns < 0, 0.0).rolling(vol_window).std(ddof=0) * math.sqrt(252)
    else:
        vol = returns.rolling(vol_window).std(ddof=0) * math.sqrt(252)
    return build_asset_relative_state_percentile(vol, lookback=state_lookback)


def build_asset_own_momentum_percentile(
    prices: pd.DataFrame,
    lookback: int,
    state_lookback: int = 756,
    min_periods: int | None = None,
) -> pd.DataFrame:
    momentum = prices / prices.shift(lookback) - 1
    return build_asset_relative_state_percentile(
        momentum,
        lookback=state_lookback,
        min_periods=min_periods,
    )


def build_current_etf_momentum_percentile_table(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    lookback: int,
    state_lookback: int = 756,
    min_periods: int | None = 120,
) -> pd.DataFrame:
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
        returns = prices.pct_change(fill_method=None)
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
            {code: log_prices[code].rolling(lookback).corr(time_index).pow(2) for code in prices.columns},
            index=prices.index,
        ).clip(lower=0.0, upper=1.0)
        quality_score = quality_score - r2_penalty * (1.0 - trend_r2)

    return raw_momentum, quality_score


def build_signal_stability_score(
    prices: pd.DataFrame,
    lookback: int,
    method: str = "none",
) -> pd.DataFrame | None:
    if method == "none":
        return None

    returns = prices.pct_change(fill_method=None)
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
        risk_confirmation = prices[active_risk_codes] / prices[active_risk_codes].shift(confirmation_lookback) - 1
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


def build_threshold_dual_signal(
    prices: pd.DataFrame,
    lookback: int,
    absolute_threshold: float = DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    weak_trend_defensive_weight: float = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    cash_exit_threshold: float = DEFAULT_DUAL_CASH_EXIT_THRESHOLD,
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
        both_weak_for_cash = (
            pd.notna(r_score)
            and pd.notna(d_score)
            and r_score <= cash_exit_threshold
            and d_score <= cash_exit_threshold
        )

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
            prev_signal = None
        elif both_weak_for_cash:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = max(float(r_score), float(d_score))
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


def build_quality_momentum_single_asset_signal(
    prices: pd.DataFrame,
    lookback: int,
    *,
    signal_quality_method: str = "raw",
    slope_penalty: float = 0.0,
    volatility_penalty: float = 0.0,
    downside_volatility_penalty: float = 0.0,
    r2_penalty: float = 0.0,
    volatility_state_lookback: int = 252,
    volatility_percentile_penalty: float = 0.0,
    downside_volatility_percentile_penalty: float = 0.0,
    volatility_percentile_divisor: float = 0.0,
    leader_margin: float = 0.0,
    candidate_codes: list[str] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    active_codes = [code for code in (list(prices.columns) if candidate_codes is None else candidate_codes) if code in prices.columns]
    if not active_codes:
        empty_signal = pd.Series(pd.NA, index=prices.index, dtype="object", name="signal")
        zero_exposure = pd.Series(0.0, index=prices.index, dtype="float64", name="target_exposure")
        nan_momentum = pd.Series(float("nan"), index=prices.index, dtype="float64", name="current_momentum")
        return empty_signal, zero_exposure, nan_momentum, nan_momentum.rename("max_momentum")

    active_prices = prices[active_codes]
    momentum, score = build_signal_quality_score(
        active_prices,
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

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    prev_signal: str | None = None

    for dt_idx in prices.index:
        winner = choose_signal_winner_with_margin(score.loc[dt_idx], prev_signal, leader_margin)
        if winner is None:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
            prev_signal = None
            continue
        signal.loc[dt_idx] = winner
        target_exposure.loc[dt_idx] = 1.0
        winner_momentum = momentum.loc[dt_idx, winner]
        current_momentum.loc[dt_idx] = float(winner_momentum) if pd.notna(winner_momentum) else float("nan")
        prev_signal = str(winner)

    return signal, target_exposure, current_momentum, momentum.max(axis=1, skipna=True).rename("max_momentum")


def compute_signal_asset_momentum(
    prices: pd.DataFrame,
    signal: pd.Series,
    lookback: int,
) -> pd.Series:
    momentum = prices / prices.shift(lookback) - 1
    signal_momentum = pd.Series(index=prices.index, dtype="float64", name="signal_asset_momentum")
    for dt_idx in prices.index:
        code = signal.loc[dt_idx]
        if pd.isna(code) or code not in momentum.columns:
            signal_momentum.loc[dt_idx] = float("nan")
            continue
        value = momentum.loc[dt_idx, code]
        signal_momentum.loc[dt_idx] = float(value) if pd.notna(value) else float("nan")
    return signal_momentum
