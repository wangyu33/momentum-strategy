"""在当前重构引擎中复刻历史策略版本。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtest import build_strategy_summary
from .config import (
    DEFAULT_STRESS_BOND_BREADTH_CUT,
    DEFAULT_STRESS_BOND_CODE,
    DEFAULT_STRESS_BOND_ENTER_DAYS,
    DEFAULT_STRESS_BOND_EXIT_DAYS,
    DEFAULT_STRESS_BOND_FILL_RESIDUAL_CASH,
    DEFAULT_STRESS_BOND_RATIO_CUT,
)
from .reporting import get_latest_portfolio_text
from .strategy import build_default_strategy_params, run_default_strategy, run_default_strategy_with_params


@dataclass(frozen=True)
class HistoricalVersionSpec:
    key: str
    commit: str
    description: str
    params_overrides: dict[str, object]
    stress_bond_risk_cap: float
    stress_bond_ratio_cut: float = DEFAULT_STRESS_BOND_RATIO_CUT
    stress_bond_breadth_cut: float = DEFAULT_STRESS_BOND_BREADTH_CUT
    stress_bond_enter_days: int = DEFAULT_STRESS_BOND_ENTER_DAYS
    stress_bond_exit_days: int = DEFAULT_STRESS_BOND_EXIT_DAYS
    stress_bond_fill_residual_cash: bool = DEFAULT_STRESS_BOND_FILL_RESIDUAL_CASH


def _build_neutralized_refactor_params() -> dict[str, object]:
    params = build_default_strategy_params()
    params.update(
        {
            "signal_selected_volatility_cap_start": 1.0,
            "signal_selected_volatility_cap_end": 1.0,
            "signal_selected_volatility_cap_floor": 1.0,
            "signal_selected_momentum_pct_cap_stage": "post_overlay",
            "signal_selected_momentum_pct_cap_start": 1.0,
            "signal_selected_momentum_pct_cap_end": 1.0,
            "signal_selected_momentum_pct_cap_floor": 1.0,
            "signal_selected_momentum_pct_lookback": 756,
            "signal_selected_momentum_pct_min_periods": 120,
            "signal_selected_momentum_pct_cap_scope": "all",
            "defensive_signal_selected_momentum_pct_cap_start": 1.0,
            "defensive_signal_selected_momentum_pct_cap_floor": 1.0,
            "target_min_rebalance_threshold": 0.0,
        }
    )
    return params


def _build_historical_version_specs() -> list[HistoricalVersionSpec]:
    return [
        HistoricalVersionSpec(
            key="v896ca00",
            commit="896ca00",
            description="早期极简正式版：无弱量能压仓、无前二接近降仓、无分位减仓、弱市切债风险仓上限 0%。",
            params_overrides={
                "close_top2_gap": 0.0,
                "close_top2_risk_cap": 1.0,
                "volume_guard_cap": 1.0,
                "pre_overheat_end_exposure": 0.9,
                "overheat_max_exposure": 0.3,
                "overheat_high_max_exposure": 0.1,
            },
            stress_bond_risk_cap=0.0,
        ),
        HistoricalVersionSpec(
            key="v8acc7b6",
            commit="8acc7b6",
            description="baseline_cf60top2 第一代正式骨架：加入弱量能压仓、前二接近降仓、过热压仓，弱市切债风险仓上限 0%。",
            params_overrides={
                "close_top2_gap": 0.005,
                "close_top2_risk_cap": 0.7,
                "volume_guard_cap": 0.5,
                "pre_overheat_end_exposure": 0.9,
                "overheat_max_exposure": 0.3,
                "overheat_high_max_exposure": 0.1,
            },
            stress_bond_risk_cap=0.0,
        ),
        HistoricalVersionSpec(
            key="v516efa6_raw",
            commit="516efa6",
            description="516efa6 的裸算版：策略参数与 8acc7b6 同代，但不走 official baseline 锚点。",
            params_overrides={
                "close_top2_gap": 0.005,
                "close_top2_risk_cap": 0.7,
                "volume_guard_cap": 0.5,
                "pre_overheat_end_exposure": 0.9,
                "overheat_max_exposure": 0.3,
                "overheat_high_max_exposure": 0.1,
            },
            stress_bond_risk_cap=0.0,
        ),
        HistoricalVersionSpec(
            key="vb5e5606_raw",
            commit="b5e5606",
            description="首次引入动量分位参数，但默认全 1.0，不生效。",
            params_overrides={
                "close_top2_gap": 0.005,
                "close_top2_risk_cap": 0.7,
                "volume_guard_cap": 0.5,
                "pre_overheat_end_exposure": 0.9,
                "overheat_max_exposure": 0.3,
                "overheat_high_max_exposure": 0.1,
                "signal_selected_momentum_pct_cap_start": 1.0,
                "signal_selected_momentum_pct_cap_end": 1.0,
                "signal_selected_momentum_pct_cap_floor": 1.0,
            },
            stress_bond_risk_cap=0.0,
        ),
        HistoricalVersionSpec(
            key="v72f03cf_raw",
            commit="72f03cf",
            description="分位减仓与最小调仓门槛上线，过热压仓默认关闭，弱市切债风险仓上限仍为 0%。",
            params_overrides={
                "close_top2_gap": 0.005,
                "close_top2_risk_cap": 0.7,
                "volume_guard_cap": 0.5,
                "pre_overheat_end_exposure": 1.0,
                "overheat_max_exposure": 1.0,
                "overheat_high_max_exposure": 1.0,
                "signal_selected_momentum_pct_cap_stage": "post_overlay",
                "signal_selected_momentum_pct_cap_start": 0.90,
                "signal_selected_momentum_pct_cap_end": 0.95,
                "signal_selected_momentum_pct_cap_floor": 0.50,
                "target_min_rebalance_threshold": 0.05,
            },
            stress_bond_risk_cap=0.0,
        ),
        HistoricalVersionSpec(
            key="vef2aa41_raw",
            commit="ef2aa41",
            description="当前主线前一版：分位减仓、最小调仓门槛、防守信号过热压仓，弱市切债风险仓上限 100%。",
            params_overrides={
                "close_top2_gap": 0.005,
                "close_top2_risk_cap": 0.7,
                "volume_guard_cap": 0.5,
                "pre_overheat_end_exposure": 1.0,
                "overheat_max_exposure": 1.0,
                "overheat_high_max_exposure": 1.0,
                "signal_selected_momentum_pct_cap_stage": "post_overlay",
                "signal_selected_momentum_pct_cap_start": 0.90,
                "signal_selected_momentum_pct_cap_end": 0.95,
                "signal_selected_momentum_pct_cap_floor": 0.50,
                "defensive_signal_selected_momentum_pct_cap_start": 0.95,
                "defensive_signal_selected_momentum_pct_cap_floor": 0.30,
                "target_min_rebalance_threshold": 0.05,
            },
            stress_bond_risk_cap=1.0,
        ),
        HistoricalVersionSpec(
            key="v8acc7b6_pct_risk",
            commit="HEAD",
            description="在 v8acc7b6 基线之上，仅开启风险标的 ETF 历史动量百分位压仓。",
            params_overrides={
                "close_top2_gap": 0.005,
                "close_top2_risk_cap": 0.7,
                "volume_guard_cap": 0.5,
                "pre_overheat_end_exposure": 0.9,
                "overheat_max_exposure": 0.3,
                "overheat_high_max_exposure": 0.1,
                "signal_selected_momentum_pct_cap_stage": "post_overlay",
                "signal_selected_momentum_pct_cap_start": 0.90,
                "signal_selected_momentum_pct_cap_end": 0.95,
                "signal_selected_momentum_pct_cap_floor": 0.50,
                "defensive_signal_selected_momentum_pct_cap_start": 1.0,
                "defensive_signal_selected_momentum_pct_cap_floor": 1.0,
            },
            stress_bond_risk_cap=0.0,
        ),
        HistoricalVersionSpec(
            key="v8acc7b6_pct_risk_defensive",
            commit="HEAD",
            description="在 v8acc7b6 基线之上，同时开启风险和防守标的 ETF 历史动量百分位压仓。",
            params_overrides={
                "close_top2_gap": 0.005,
                "close_top2_risk_cap": 0.7,
                "volume_guard_cap": 0.5,
                "pre_overheat_end_exposure": 0.9,
                "overheat_max_exposure": 0.3,
                "overheat_high_max_exposure": 0.1,
                "signal_selected_momentum_pct_cap_stage": "post_overlay",
                "signal_selected_momentum_pct_cap_start": 0.90,
                "signal_selected_momentum_pct_cap_end": 0.95,
                "signal_selected_momentum_pct_cap_floor": 0.50,
                "defensive_signal_selected_momentum_pct_cap_start": 0.95,
                "defensive_signal_selected_momentum_pct_cap_floor": 0.30,
            },
            stress_bond_risk_cap=0.0,
        ),
        HistoricalVersionSpec(
            key="latest_refactor_default",
            commit="HEAD",
            description="当前重构代码默认基线：25 日质量动量 + 5% 攻防切换；布林过热压到 80%，极热进一步压到 42.25%。",
            params_overrides={},
            stress_bond_risk_cap=0.0,
        ),
    ]


HISTORICAL_VERSION_SPECS = _build_historical_version_specs()
HISTORICAL_VERSION_MAP = {spec.key: spec for spec in HISTORICAL_VERSION_SPECS}


def build_versioned_strategy_params(version_key: str) -> tuple[dict[str, object], HistoricalVersionSpec]:
    if version_key not in HISTORICAL_VERSION_MAP:
        raise KeyError(f"unknown historical version: {version_key}")
    spec = HISTORICAL_VERSION_MAP[version_key]
    params = _build_neutralized_refactor_params()
    params.update(spec.params_overrides)
    return params, spec


def run_historical_version_with_current_engine(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    version_key: str,
    *,
    fee_rate: float,
    slippage_rate: float,
    market_proxy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, HistoricalVersionSpec]:
    params, spec = build_versioned_strategy_params(version_key)
    if version_key == "latest_refactor_default":
        result, trades = run_default_strategy(
            prices,
            selected,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            market_proxy=market_proxy,
        )
        return result, trades, spec

    result, trades = run_default_strategy_with_params(
        prices,
        selected,
        params=params,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        market_proxy=market_proxy,
        treasury_code=DEFAULT_STRESS_BOND_CODE,
        risk_cap=spec.stress_bond_risk_cap,
        ratio_cut=spec.stress_bond_ratio_cut,
        breadth_cut=spec.stress_bond_breadth_cut,
        enter_days=spec.stress_bond_enter_days,
        exit_days=spec.stress_bond_exit_days,
        fill_residual_cash_to_treasury=spec.stress_bond_fill_residual_cash,
    )
    return result, trades, spec


def build_historical_version_ranking(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    fee_rate: float,
    slippage_rate: float,
    market_proxy: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in HISTORICAL_VERSION_SPECS:
        result, trades, runtime_spec = run_historical_version_with_current_engine(
            prices,
            selected,
            spec.key,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            market_proxy=market_proxy,
        )
        summary = build_strategy_summary(
            result,
            trades,
            selected=selected,
            include_max_drawdown_integral=True,
            get_latest_portfolio_text=get_latest_portfolio_text,
        )
        rows.append(
            {
                "version_key": runtime_spec.key,
                "commit": runtime_spec.commit,
                "description": runtime_spec.description,
                "end_date": result.index[-1].date().isoformat(),
                "end_nav": float(result["nav"].iloc[-1]),
                "annualized_return": float(summary["annualized_return"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "sharpe_rf0": float(summary["sharpe_rf0"]),
                "trade_count": int(summary["trade_count"]),
                "avg_exposure": float(summary["avg_exposure"]),
                "latest_portfolio": str(summary.get("latest_portfolio", "")),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["end_nav", "annualized_return"], ascending=False).reset_index(drop=True)
    return ranking


def save_historical_version_ranking(ranking: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_path, index=False)
