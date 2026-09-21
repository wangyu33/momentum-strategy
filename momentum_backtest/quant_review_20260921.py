"""Read frozen local outputs; write review artifacts without changing production data.

Run: PYTHONPATH=.tools/python_packages python3 momentum_backtest/quant_review_20260921.py
All experiments are retrospective diagnostics, not out-of-sample evidence.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from core.config import RISK_CODES, DEFENSIVE_CODES
from core.signals import build_threshold_dual_signal
from core.strategy import run_default_strategy


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/analysis/quant_review_20260921"


def weights_from_signal(prices, lookback=25, slope=0.85):
    signal, exposure, _, _ = build_threshold_dual_signal(
        prices, lookback, signal_quality_method="slope", slope_penalty=slope,
        risk_codes=list(RISK_CODES), defensive_codes=list(DEFENSIVE_CODES),
    )
    return pd.DataFrame(
        {c: signal.eq(c).fillna(False).astype(float) * exposure for c in prices},
        index=prices.index,
    )


def mid_percentile(frame, window, minimum):
    rolling = frame.rolling(window, min_periods=minimum)
    return (rolling.rank(method="average") - 0.5) / rolling.count()


def overlays(prices, weights, asset=True, boll=True, ma=True, strict_boll=False):
    weights = weights.copy()
    # Match production: own momentum percentile always uses 25 days.
    pct = mid_percentile(prices / prices.shift(25) - 1, 756, 120)
    if asset:
        for c in ("159985", "512890"):
            v = pct[c]
            cap = pd.Series(1.0, index=prices.index)
            mask = v.ge(.90) & v.lt(.95)
            cap.loc[mask] = 1 - .70 * (v.loc[mask] - .90) / .05
            mask = v.ge(.95) & v.lt(.97)
            cap.loc[mask] = .30 - .10 * (v.loc[mask] - .95) / .02
            cap.loc[v.ge(.97)] = .20
            active = weights[c].gt(0)
            scale = (cap / weights.sum(axis=1).replace(0, np.nan)).clip(upper=1).fillna(1)
            weights.loc[active] = weights.loc[active].mul(scale.loc[active], axis=0)
    if boll:
        mean = prices.rolling(20).mean()
        std = prices.rolling(20).std(ddof=0)
        bw_pct = mid_percentile(4 * std / mean, 120, 30)
        hot = prices.ge(mean + 2 * std) & bw_pct.ge(.80)
        extreme = hot & bw_pct.ge(.90)
        cap = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        cap = cap.mask(hot, .80).mask(extreme, .4225)
        active = weights.gt(0)
        selected_cap = cap.where(active, 1.0).min(axis=1)
        total = weights.sum(axis=1)
        allowed = total.gt(selected_cap if strict_boll else .80)
        scale = (selected_cap / total.replace(0, np.nan)).clip(upper=1).fillna(1)
        weights.loc[allowed] = weights.loc[allowed].mul(scale.loc[allowed], axis=0)
    if ma:
        below = prices.lt(prices.rolling(250).mean())
        triggered = (below & weights.gt(0)).any(axis=1)
        weights.loc[triggered] *= .5
    return weights


def evaluate(prices, weights, bps=5, delay=0):
    target = weights.shift(delay).fillna(0)
    ret = prices.pct_change(fill_method=None).fillna(0)
    gross = (target.shift(1).fillna(0) * ret).sum(axis=1)
    turnover = target.diff().fillna(target).abs().sum(axis=1)
    net = (1 + gross) * (1 - turnover * bps / 10000) - 1
    return net, turnover


def metrics(net, turnover):
    nav = (1 + net).cumprod()
    years = (net.index[-1] - net.index[0]).days / 365.25
    return {
        "end_nav": float(nav.iloc[-1]),
        "cagr": float(nav.iloc[-1] ** (1 / years) - 1),
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "sharpe_arithmetic_rf0": float(net.mean() / net.std(ddof=1) * np.sqrt(252)),
        "annual_volatility": float(net.std(ddof=1) * np.sqrt(252)),
        "annual_sum_abs_turnover": float(turnover.sum() / years),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prices_path = ROOT / "output/core/prices.csv"
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    selected = pd.read_csv(ROOT / "output/core/selected_etfs.csv", dtype={"code": str})
    saved = pd.read_csv(ROOT / "output/core/backtest_nav.csv", index_col=0, parse_dates=True)
    weights = saved.filter(regex="^target_weight_").rename(columns=lambda c: c.removeprefix("target_weight_"))
    reproduced, _ = run_default_strategy(prices, selected)
    np.testing.assert_allclose(reproduced.nav, saved.nav, rtol=1e-10, atol=1e-10)
    base = weights_from_signal(prices)
    np.testing.assert_allclose(overlays(prices, base), weights, atol=1e-10)
    print("Official baseline and independent overlay reproduction: matched", flush=True)
    scenarios = []
    yearly = {}

    def add(name, p, w, bps=5, delay=0):
        net, turnover = evaluate(p, w, bps, delay)
        row = {"scenario": name, **metrics(net, turnover)}
        scenarios.append(row)
        yearly[name] = (1 + net).groupby(net.index.year).prod() - 1
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
        return net

    baseline_net = add("baseline", prices, weights)
    np.testing.assert_allclose(baseline_net, saved.strategy_return, atol=1e-12)
    for bps in (0, 10, 20):
        add(f"cost_{bps}bps", prices, weights, bps=bps)
    add("execute_next_close", prices, weights, delay=1)
    for name, kwargs in (
        ("no_asset_cap", {"asset": False}),
        ("no_boll", {"boll": False}),
        ("no_ma", {"ma": False}),
        ("strict_boll_min_caps", {"strict_boll": True}),
    ):
        add(name, prices, overlays(prices, base, **kwargs))
    add("raw_momentum", prices, overlays(prices, weights_from_signal(prices, slope=0)))
    for lookback in (20, 30):
        add(f"selection_lookback_{lookback}", prices, overlays(prices, weights_from_signal(prices, lookback=lookback)))

    # SSE announcement: 2021-10-22 1:2 split, suspended on that day.
    # Keep suspension NaN to isolate the split adjustment; this is not a full data repair.
    adjusted = prices.copy()
    adjusted.loc[adjusted.index < "2021-10-22", "512890"] /= 2
    adjusted_weights = overlays(adjusted, weights_from_signal(adjusted))
    add("verified_split_adjustment_only", adjusted, adjusted_weights)
    diff = (adjusted_weights - weights).abs().sum(axis=1).gt(1e-10)
    split_detail = {
        "changed_target_days": int(diff.sum()),
        "first_changed_day": str(diff[diff].index.min()),
        "last_changed_day": str(diff[diff].index.max()),
    }

    strict = overlays(prices, base, strict_boll=True)
    strict_diff = (strict - weights).abs().sum(axis=1).gt(1e-10)
    strict.loc[strict_diff].to_csv(OUT / "boll_conflict_corrected_weights.csv")

    segment_rows = []
    turnover = evaluate(prices, weights)[1]
    for label, start, end in (
        ("2012_2018", "2012", "2018-12-31"),
        ("2019_2023", "2019", "2023-12-31"),
        ("2024_2026", "2024", "2026-12-31"),
        ("2020_onward_full_pool", "2020", "2026-12-31"),
    ):
        segment_rows.append({"segment": label, **metrics(baseline_net.loc[start:end], turnover.loc[start:end])})
    pd.DataFrame(segment_rows).to_csv(OUT / "segments.csv", index=False)
    pd.DataFrame(scenarios).to_csv(OUT / "scenarios.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "yearly_scenarios.csv", index_label="year")

    # Compare arithmetic gross contribution; unlike terminal currency P&L it does
    # not automatically give later years larger capital weights.
    contribution = weights.shift(1).fillna(0) * prices.pct_change(fill_method=None).fillna(0)
    contribution.groupby(contribution.index.year).sum().to_csv(OUT / "yearly_gross_arithmetic_contribution.csv")
    details = {
        "prices_sha256": hashlib.sha256(prices_path.read_bytes()).hexdigest(),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "n_rows": len(prices),
        "avg_exposure": float(weights.sum(axis=1).mean()),
        "single_asset_invested_days": int(weights.gt(0).sum(axis=1).eq(1).sum()),
        "full_exposure_days": int(weights.sum(axis=1).ge(.999999).sum()),
        "cash_days": int(weights.sum(axis=1).lt(1e-12).sum()),
        "strict_boll_changed_days": int(strict_diff.sum()),
        "split_adjustment": split_detail,
        "warmup_note": "Selection lookback experiments change selection only; own percentile stays at production 25 days.",
        "execution_note": "next_close delays targets and their costs one trading day; it is not a next-open simulation.",
    }
    (OUT / "diagnostics.json").write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(details, indent=2, ensure_ascii=False))
    print(pd.DataFrame(segment_rows).to_string(index=False))


if __name__ == "__main__":
    main()
