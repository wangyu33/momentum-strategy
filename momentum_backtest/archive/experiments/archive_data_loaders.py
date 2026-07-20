#!/usr/bin/env python3
"""archive 历史研究脚本共用的数据裁切与市场代理加载 helper。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

try:
    from ...goal_optimization_common import load_market_volume_proxy
    from ...hs300_regime_common import load_cached_data
    from ...market_proxy_common import build_proxy_catalog
    from ...search_utils import try_join_missing_candidate_histories
except ImportError:
    from goal_optimization_common import load_market_volume_proxy
    from hs300_regime_common import load_cached_data
    from market_proxy_common import build_proxy_catalog
    from search_utils import try_join_missing_candidate_histories


ARCHIVE_FLAT_OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat")


def filter_selected_price_columns(prices: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """按 selected 的 code 顺序裁出当前价格面板中可用的列。"""
    selected_codes = selected["code"].astype(str).tolist()
    return prices[[code for code in selected_codes if code in prices.columns]].copy()


def extract_result_target_weights(result: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """从策略结果中抽取 `target_weight_*` 列，并还原成与价格面板对齐的权重矩阵。"""
    weight_cols = [col for col in result.columns if col.startswith("target_weight_")]
    target_weights = result[weight_cols].copy()
    target_weights.columns = [col.removeprefix("target_weight_") for col in weight_cols]
    return target_weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0).fillna(0.0)


def build_risk_defensive_momentum_frames(
    prices: pd.DataFrame,
    *,
    risk_codes: Sequence[str],
    defensive_codes: Sequence[str],
    lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按给定风险/防守池与回看窗口构造动量面板。"""
    momentum = prices / prices.shift(lookback) - 1
    risk_momentum = momentum[[code for code in risk_codes if code in momentum.columns]].copy()
    defensive_momentum = momentum[[code for code in defensive_codes if code in momentum.columns]].copy()
    return risk_momentum, defensive_momentum


def build_cash_tail_mask(
    risk_momentum: pd.DataFrame,
    defensive_momentum: pd.DataFrame,
    *,
    risk_pos_limit: int = 1,
    risk_max_cut: float = 0.03,
    defensive_max_cut: float = -0.005,
) -> pd.Series:
    """统一生成“风险池整体走弱且防守也不强时退到现金”的触发掩码。"""
    valid_risk_count = risk_momentum.notna().sum(axis=1)
    risk_positive_count = (risk_momentum > 0).sum(axis=1)
    risk_max = risk_momentum.max(axis=1, skipna=True)
    defensive_max = defensive_momentum.max(axis=1, skipna=True)
    return (
        (valid_risk_count >= max(len(risk_momentum.columns) - 1, 1))
        & (risk_positive_count <= risk_pos_limit)
        & (risk_max <= risk_max_cut)
        & (defensive_max <= defensive_max_cut)
    ).fillna(False)


def build_archive_flat_output_dir(name: str) -> Path:
    """统一生成 archive_flat 下的脚本输出目录。"""
    return ARCHIVE_FLAT_OUTPUT_DIR / name


def build_archive_flat_strategy_paths(name: str) -> tuple[Path, Path, Path]:
    """统一生成 archive_flat 历史策略目录及其 best/notify_state 路径。"""
    output_dir = build_archive_flat_output_dir(name)
    return output_dir, output_dir / "best.json", output_dir / "notify_state.json"


def dedupe_selected_pool(selected: pd.DataFrame, *, keep: str = "first") -> pd.DataFrame:
    """按 code 去重候选池，避免 archive 追加已存在标的时重复列污染价格面板。"""
    return selected.drop_duplicates(subset=["code"], keep=keep).reset_index(drop=True)


def load_recent_selected_prices(selected: pd.DataFrame, years: int) -> pd.DataFrame:
    """从 cached core 价格读取最近 N 年，并按候选池裁列。"""
    _, prices = load_cached_data()
    start_ts = prices.index.max() - pd.DateOffset(years=years)
    recent_prices = prices.loc[prices.index >= start_ts].copy()
    return filter_selected_price_columns(recent_prices, selected)


def load_core_cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """统一读取 archive 共用的整份 core 候选池与价格缓存。"""
    selected = pd.read_csv(Path("momentum_backtest/output/core/selected_etfs.csv"), dtype={"code": str})
    selected = dedupe_selected_pool(selected)
    _, prices = load_cached_data()
    return selected, filter_selected_price_columns(prices, selected)


def load_selected_prices(
    selected: pd.DataFrame,
    *,
    years: int,
    refresh: bool,
    fetch_fn: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    """统一处理 archive 单池脚本的 refresh/缓存价格加载。"""
    selected = dedupe_selected_pool(selected)
    if refresh:
        return fetch_fn(selected, years=years)
    return load_recent_selected_prices(selected, years)


def load_workspace_selected_prices(
    selected: pd.DataFrame,
    *,
    refresh: bool,
    fetch_fn: Callable[..., pd.DataFrame],
    years: int | None = None,
    start_date: pd.Timestamp | None = None,
    extra_paths: list[Path] | None = None,
    missing_label: str = "required histories",
) -> pd.DataFrame:
    """统一处理 archive 脚本“refresh 抓数 / 否则读 workspace 合并缓存”的加载语义。"""
    selected = dedupe_selected_pool(selected)
    if refresh:
        if years is None:
            raise ValueError("years is required when refresh=True")
        prices = fetch_fn(selected, years=years)
    else:
        prices = load_workspace_price_cache(extra_paths=extra_paths)
        if prices.empty:
            raise RuntimeError(
                f"workspace price cache is empty; rerun with --refresh to fetch {missing_label}"
            )
        if years is not None:
            start_ts = prices.index.max() - pd.DateOffset(years=years)
            prices = prices.loc[prices.index >= start_ts].copy()
        prices = filter_selected_price_columns(prices, selected)
        missing_codes = [code for code in selected["code"].astype(str).tolist() if code not in prices.columns]
        if missing_codes:
            missing_text = ", ".join(missing_codes)
            raise RuntimeError(
                f"cached prices missing {missing_label}: {missing_text}; rerun with --refresh when network access is available"
            )

    prices = filter_selected_price_columns(prices, selected)
    if start_date is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start_date)].copy()
    return prices


def load_core_selected_and_prices(
    selected: pd.DataFrame,
    *,
    years: int,
    refresh: bool,
    fetch_fn: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """统一处理 archive 阈值双动量链“refresh 抓整池 / 否则复用整份 core 缓存”的加载语义。"""
    selected = dedupe_selected_pool(selected)
    if refresh:
        prices = fetch_fn(selected, years=years)
        return selected, filter_selected_price_columns(prices, selected)
    _, prices = load_cached_data()
    return selected, filter_selected_price_columns(prices, selected)


def load_core_selected_and_prices_with_fallback(
    selected: pd.DataFrame,
    *,
    years: int,
    fetch_fn: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """优先读取整份 core 价格缓存；若本地 core 不存在，则回退到整池抓数。"""
    selected = dedupe_selected_pool(selected)
    core_prices_path = Path("momentum_backtest/output/core/prices.csv")
    if core_prices_path.exists():
        _, prices = load_cached_data()
        return selected, filter_selected_price_columns(prices, selected)
    prices = fetch_fn(selected, years=years)
    return selected, filter_selected_price_columns(prices, selected)


def load_core_price_panel_with_fallback(
    selected: pd.DataFrame,
    *,
    years: int,
    fetch_fn: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    """优先读取完整 core 价格面板；若本地 core 不存在，则回退到按 selected 抓数。"""
    core_prices_path = Path("momentum_backtest/output/core/prices.csv")
    if core_prices_path.exists():
        _, prices = load_cached_data()
        return prices
    selected = dedupe_selected_pool(selected)
    return fetch_fn(selected, years=years)


def load_selected_prices_from_start(selected: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    """从 cached core 价格读取给定起始日之后的候选池价格。"""
    _, prices = load_cached_data()
    recent_prices = prices.loc[prices.index >= start_date].copy()
    return filter_selected_price_columns(recent_prices, selected)


def load_cached_price_panel(paths: list[Path]) -> pd.DataFrame:
    """合并多份历史价格缓存，按列去重后返回。"""
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        frame.columns = [str(col) for col in frame.columns]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    return merged.sort_index()


def load_workspace_price_cache(extra_paths: list[Path] | None = None) -> pd.DataFrame:
    """优先合并 workspace 内常用 core/archive/research 价格缓存。"""
    cache_paths = [Path("momentum_backtest/output/core/prices.csv")]
    cache_paths.extend(sorted(ARCHIVE_FLAT_OUTPUT_DIR.glob("*/prices.csv")))
    cache_paths.extend(
        [
            Path("momentum_backtest/output/research/resource_abs08/prices.csv"),
            Path("momentum_backtest/output/research/china_internet_cap70/prices.csv"),
        ]
    )
    if extra_paths:
        cache_paths.extend(extra_paths)
    deduped_paths: list[Path] = []
    seen: set[Path] = set()
    for path in cache_paths:
        if path in seen:
            continue
        deduped_paths.append(path)
        seen.add(path)
    return load_cached_price_panel(deduped_paths)


def load_prices_with_candidate_join(
    base_selected: pd.DataFrame,
    extra_candidates: pd.DataFrame | Sequence[dict[str, str]],
    *,
    years: int,
    refresh: bool,
    fetch_fn: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str | None]:
    """为 archive 覆盖层脚本统一准备 base+候选价格，并尽量复用本地缓存。"""
    extra_df = pd.DataFrame(extra_candidates)
    selected_for_prices = pd.concat([base_selected, extra_df], ignore_index=True)
    if refresh:
        prices = fetch_fn(selected_for_prices, years=years)
        prices = filter_selected_price_columns(prices, selected_for_prices)
        return selected_for_prices, prices, [], None

    _, prices = load_cached_data()
    start_ts = prices.index.max() - pd.DateOffset(years=years)
    prices = prices.loc[prices.index >= start_ts].copy()
    prices, skipped_codes, fetch_error = try_join_missing_candidate_histories(
        prices=prices,
        candidates=extra_df,
        years=years,
        fetch_fn=fetch_fn,
    )
    prices = filter_selected_price_columns(prices, selected_for_prices)
    return selected_for_prices, prices, skipped_codes, fetch_error


def load_candidate_prices_with_checks(
    base_selected: pd.DataFrame,
    extra_candidates: pd.DataFrame | Sequence[dict[str, str]],
    *,
    years: int,
    refresh: bool,
    fetch_fn: Callable[..., pd.DataFrame],
    label: str,
    required_candidates: pd.DataFrame | Sequence[dict[str, str]] | None = None,
    required_label: str = "required baseline history",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    """统一处理 archive 候选拼接、缺历史 warning 与必需标的校验。"""
    extra_df = pd.DataFrame(extra_candidates)
    selected_for_prices, prices, skipped_codes, fetch_error = load_prices_with_candidate_join(
        base_selected,
        extra_df,
        years=years,
        refresh=refresh,
        fetch_fn=fetch_fn,
    )
    emit_skipped_candidate_warning(
        skipped_codes,
        label=label,
        fetch_error=fetch_error,
    )

    available_candidates = [
        row
        for row in extra_df.to_dict("records")
        if str(row.get("code", "")) in prices.columns
    ]
    if required_candidates is None:
        return selected_for_prices, prices, available_candidates

    required_df = pd.DataFrame(required_candidates)
    missing_required = [
        row
        for row in required_df.to_dict("records")
        if str(row.get("code", "")) not in prices.columns
    ]
    if missing_required:
        missing_text = ", ".join(
            f"{row.get('code', '')} {row.get('name', '')}".strip()
            for row in missing_required
        )
        raise RuntimeError(f"missing {required_label}: {missing_text}")
    return selected_for_prices, prices, available_candidates


def load_required_candidate_prices(
    base_selected: pd.DataFrame,
    extra_candidates: pd.DataFrame | Sequence[dict[str, str]],
    *,
    years: int,
    refresh: bool,
    fetch_fn: Callable[..., pd.DataFrame],
    label: str,
    failure_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """严格要求 extra_candidates 全部具备可用历史，否则直接报错。"""
    extra_df = pd.DataFrame(extra_candidates)
    selected_for_prices, prices, skipped_codes, fetch_error = load_prices_with_candidate_join(
        base_selected,
        extra_df,
        years=years,
        refresh=refresh,
        fetch_fn=fetch_fn,
    )
    if skipped_codes:
        emit_skipped_candidate_warning(
            skipped_codes,
            label=label,
            fetch_error=fetch_error,
        )
        reason = build_missing_history_reason(skipped_codes, fetch_error=fetch_error)
        raise RuntimeError(f"{failure_prefix}; {reason}. rerun with network access")
    return selected_for_prices, prices


def emit_skipped_candidate_warning(
    skipped_codes: Sequence[str],
    *,
    label: str,
    fetch_error: str | None,
) -> None:
    """统一打印 archive 候选资产缺历史时的 warning。"""
    if not skipped_codes:
        return
    skipped_text = ", ".join(skipped_codes)
    if fetch_error:
        print(f"[warn] skip {label} without local history: {skipped_text}; fetch failed: {fetch_error}")
    else:
        print(f"[warn] skip {label} without usable history: {skipped_text}")


def build_missing_history_reason(
    skipped_codes: Sequence[str],
    *,
    fetch_error: str | None,
) -> str:
    """统一构造 archive 缺历史时的可读原因文本。"""
    missing_text = ", ".join(str(code) for code in skipped_codes)
    reason = f"missing histories: {missing_text}"
    if fetch_error:
        reason = f"{reason}; fetch failed: {fetch_error}"
    return reason


def load_named_market_proxy(
    prices: pd.DataFrame,
    *,
    years: int,
    proxy_kind: str,
    refresh: bool = False,
    risk_codes: list[str] | None = None,
) -> pd.DataFrame:
    """基于给定价格面板构造命名 market proxy。"""
    base_market_proxy = load_market_volume_proxy(years=years, refresh=refresh)
    return build_named_market_proxy(
        prices,
        base_market_proxy=base_market_proxy,
        proxy_kind=proxy_kind,
        risk_codes=risk_codes,
    )


def build_named_market_proxy(
    prices: pd.DataFrame,
    *,
    base_market_proxy: pd.DataFrame,
    proxy_kind: str,
    risk_codes: list[str] | None = None,
) -> pd.DataFrame:
    """基于已加载的 market proxy 与价格面板构造命名代理。"""
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(base_market_proxy, prices, risk_codes=risk_codes)
    }
    return proxy_catalog[proxy_kind]
