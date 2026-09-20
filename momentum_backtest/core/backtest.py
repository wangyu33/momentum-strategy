"""正式策略核心回测层。"""

from __future__ import annotations

import pandas as pd

from .config import DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD, DEFAULT_CASH_THRESHOLD, DEFAULT_DUAL_CASH_EXIT_THRESHOLD, DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE, DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT
from .io import normalize_code
from .signals import build_threshold_dual_signal


def build_weight_frame_from_holding_exposure(
    index: pd.Index,
    codes: list[str],
    holding: pd.Series,
    exposure: pd.Series,
) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=index, columns=codes, dtype="float64")
    holding_codes = holding.reindex(index).map(normalize_code)
    exposure_series = exposure.reindex(index).fillna(0.0)
    for code in codes:
        mask = holding_codes == code
        if mask.any():
            weights.loc[mask, code] = exposure_series.loc[mask].astype(float)
    return weights


def build_position_series_from_weight_frame(weights: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    normalized = weights.fillna(0.0)
    exposure = normalized.sum(axis=1).rename("exposure")
    holding = normalized.idxmax(axis=1).where(exposure > 1e-12, pd.NA).rename("holding")
    return holding, exposure


def build_trades_from_weight_frame(weights: pd.DataFrame, nav: pd.Series, selected: pd.DataFrame) -> pd.DataFrame:
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
            action = "BUY" if curr_w > prev_w and prev_w == 0 else "ADD" if curr_w > prev_w else "SELL" if curr_w == 0 else "REDUCE"
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
    codes = [str(code) for code in selected["code"]]
    finalized = result.copy()

    if confirmed_weights is None:
        if confirmed_holding is None or confirmed_exposure is None:
            raise ValueError("confirmed position context is required")
        confirmed_weights = build_weight_frame_from_holding_exposure(finalized.index, codes, confirmed_holding, confirmed_exposure)
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
            return_weights = build_weight_frame_from_holding_exposure(finalized.index, codes, return_holding, return_exposure)
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
    finalized = pd.concat([finalized, confirmed_weights.add_prefix("weight_"), return_weights.add_prefix("return_weight_")], axis=1)
    return finalized


def recompute_return_chain(
    result: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    fee_rate: float,
    slippage_rate: float,
) -> pd.DataFrame:
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
    returns = prices.reindex(index=refreshed.index, columns=confirmed_weights.columns).pct_change(fill_method=None).fillna(0.0)

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


def compute_signal_asset_momentum(
    prices: pd.DataFrame,
    signal: pd.Series,
    lookback: int,
) -> pd.Series:
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


def run_target_weights_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    target_weights: pd.DataFrame,
    signal_momentum: pd.Series,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    target_weights = target_weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0)
    weights = target_weights.shift(1).fillna(0.0)
    prev_weights = weights.shift(1).fillna(0.0)
    gross_ret = (weights * returns).sum(axis=1)
    turnover = (weights - prev_weights).abs().sum(axis=1)
    trade_cost_rate = turnover * (fee_rate + slippage_rate)
    strategy_ret = (1 + gross_ret) * (1 - trade_cost_rate) - 1
    nav = (1 + strategy_ret.fillna(0.0)).cumprod()
    if not nav.empty:
        nav.iloc[0] = 1.0
    exposure = weights.sum(axis=1).rename("exposure")
    primary_weight = weights.max(axis=1)
    holding = weights.idxmax(axis=1).where(primary_weight > 0, pd.NA).rename("holding")
    signal = target_weights.idxmax(axis=1).where(target_weights.max(axis=1) > 0, pd.NA).rename("signal")
    drawdown = nav / nav.cummax() - 1

    result = pd.DataFrame(
        {
            "nav": nav,
            "strategy_return": strategy_ret,
            "current_momentum": signal_momentum,
            "signal": signal,
            "holding": holding,
            "exposure": exposure,
            "trade_cost_rate": trade_cost_rate,
            "turnover": turnover,
            "drawdown": drawdown,
        }
    )
    result = pd.concat([result, weights.add_prefix("weight_")], axis=1)
    trades = build_trades_from_weight_frame(target_weights, result["nav"], selected)
    return result, trades


def run_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    cash_threshold: float | None = DEFAULT_CASH_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change(fill_method=None)
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


def run_threshold_dual_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    absolute_threshold: float = DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    weak_trend_defensive_weight: float = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    cash_exit_threshold: float = DEFAULT_DUAL_CASH_EXIT_THRESHOLD,
    risk_codes: list[str] | None = None,
    defensive_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change(fill_method=None)
    signal, target_exposure, current_momentum, max_momentum = build_threshold_dual_signal(
        prices,
        lookback=lookback,
        absolute_threshold=absolute_threshold,
        weak_trend_defensive_weight=weak_trend_defensive_weight,
        cash_exit_threshold=cash_exit_threshold,
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


def count_trade_days(trades: pd.DataFrame) -> int:
    if trades.empty or "date" not in trades.columns:
        return 0
    trade_dates = pd.to_datetime(trades["date"], errors="coerce").dropna()
    return int(trade_dates.dt.normalize().nunique())


def max_drawdown(nav: pd.Series) -> float:
    dd = nav / nav.cummax() - 1
    return float(dd.min())


def max_drawdown_episode_integral(nav: pd.Series) -> float:
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
    get_latest_portfolio_text=None,
) -> dict[str, float | int | str]:
    daily_ret = result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    effective_exposure = exposure_series if exposure_series is not None else result["exposure"]

    summary: dict[str, float | int | str] = {
        "total_return": float(result["nav"].iloc[-1] - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown(result["nav"]),
        "trade_count": count_trade_days(trades),
        "trade_action_count": int(len(trades)),
        "avg_exposure": float(effective_exposure.mean()),
        "latest_momentum": float(result["current_momentum"].dropna().iloc[-1]),
        "latest_exposure": float(effective_exposure.iloc[-1]),
        "latest_nav": float(result["nav"].iloc[-1]),
    }

    if include_max_drawdown_integral:
        summary["max_drawdown_integral"] = max_drawdown_integral(result["nav"])
        summary["max_drawdown_episode_integral"] = max_drawdown_episode_integral(result["nav"])
        summary["full_history_drawdown_integral"] = summary["max_drawdown_integral"]
        summary["worst_drawdown_episode_integral"] = summary["max_drawdown_episode_integral"]

    if selected is not None:
        holding_series = result["holding"].dropna()
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
        if get_latest_portfolio_text is not None:
            summary["latest_portfolio"] = get_latest_portfolio_text(selected, result)

    return summary


def build_benchmark_nav(prices: pd.DataFrame, benchmark_code: str = "510300") -> pd.Series:
    benchmark_ret = prices[benchmark_code].pct_change(fill_method=None).fillna(0.0)
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
