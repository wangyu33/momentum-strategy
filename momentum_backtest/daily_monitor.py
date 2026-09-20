#!/usr/bin/env python3
"""ETF 动量策略每日巡检与双通知脚本。"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

warnings.filterwarnings("ignore")

import akshare as ak
import pandas as pd

try:
    from .core.config import (
        CORE_OUTPUT_DIR,
        DEFAULT_STRATEGY_NAME,
        MONITOR_OUTPUT_DIR,
        load_default_strategy_backtest_pool,
    )
    from .core.data import fetch_realtime_etf_prices, filter_history_to_confirmed_closes, should_accept_same_day_history
    from .core.io import ensure_output_dirs, is_trading_day
    from .monitor_snapshot import (
        classify_market_session,
        should_include_realtime_snapshot,
    )
    from .monitor_snapshot_builders import build_signal_snapshot
    from .monitor_pipeline import (
        build_price_panel,
        persist_strategy_outputs,
        run_strategy_snapshot,
    )
    from .monitor_delivery import (
        build_channel_plan,
        build_webhook_url,
        build_empty_delivery_progress,
        is_duplicate_snapshot,
        load_delivery_progress_for_bundle,
        load_state,
        save_state,
        send_card_bundle,
        state_uses_message_cache,
    )
    from .monitor_render import (
        build_market_card,
        build_trade_card,
        format_market_message,
        format_trade_message,
    )
except ImportError:
    from core.config import (
        CORE_OUTPUT_DIR,
        DEFAULT_STRATEGY_NAME,
        MONITOR_OUTPUT_DIR,
        load_default_strategy_backtest_pool,
    )
    from core.data import fetch_realtime_etf_prices, filter_history_to_confirmed_closes, should_accept_same_day_history
    from core.io import ensure_output_dirs, is_trading_day
    from monitor_snapshot import (
        classify_market_session,
        should_include_realtime_snapshot,
    )
    from monitor_snapshot_builders import build_signal_snapshot
    from monitor_pipeline import (
        build_price_panel,
        persist_strategy_outputs,
        run_strategy_snapshot,
    )
    from monitor_delivery import (
        build_channel_plan,
        build_webhook_url,
        build_empty_delivery_progress,
        is_duplicate_snapshot,
        load_delivery_progress_for_bundle,
        load_state,
        save_state,
        send_card_bundle,
        state_uses_message_cache,
    )
    from monitor_render import (
        build_market_card,
        build_trade_card,
        format_market_message,
        format_trade_message,
    )


LOOKBACK = 25
MONITOR_HISTORY_START = pd.Timestamp("2012-01-01")
FEISHU_OPEN_ID = "ou_a6a198ce5e9f97430f257a04b502f49b"
DEFAULT_FEISHU_WEBHOOK = "https://open.larkoffice.com/open-apis/bot/v2/hook/688de167-8de9-4822-aa1c-4dd723a4ace6"
DEBUG_ENABLED = False
WEBHOOK_RETRY_SLEEP_SECONDS = 1.5
WEBHOOK_MAX_RETRIES = 3
REGIME_MIN_SAMPLE_COUNT = 20
ALLOCATION_DISPLAY_DIGITS = 1
MONITOR_RUN_LOG = MONITOR_OUTPUT_DIR / "daily_monitor.run.log"
FORMAL_DELIVERY_SLOTS: tuple[tuple[int, int, str], ...] = (
    (9, 40, "0940"),
    (12, 10, "1210"),
    (14, 50, "1450"),
)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [daily_monitor] {message}"
    MONITOR_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MONITOR_RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if DEBUG_ENABLED:
        print(line, file=sys.stderr)


def fetch_realtime_prices() -> dict[str, float]:
    return fetch_realtime_etf_prices()


def infer_delivery_slot(now: datetime) -> str | None:
    current_hhmm = now.hour * 100 + now.minute
    active_slot: str | None = None
    for hour, minute, slot_name in FORMAL_DELIVERY_SLOTS:
        slot_hhmm = hour * 100 + minute
        if current_hhmm >= slot_hhmm:
            active_slot = slot_name
    return active_slot


def fetch_latest_raw_closes(selected_pool: list[dict[str, object]], today: date) -> dict[str, float]:
    allow_same_day_close = should_accept_same_day_history(today)
    raw_closes: dict[str, float] = {}
    for item in selected_pool:
        code = str(item.get("code", "")).strip()
        symbol = str(item.get("sina_symbol", "")).strip()
        if not code or not symbol:
            continue
        try:
            hist = ak.fund_etf_hist_sina(symbol=symbol)
            if hist.empty:
                continue
            hist["date"] = pd.to_datetime(hist["date"])
            hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
            hist = filter_history_to_confirmed_closes(
                hist[["date", "close"]].dropna(),
                today=today,
                allow_same_day_close=allow_same_day_close,
            )
            if hist.empty:
                continue
            raw_closes[code] = float(hist["close"].iloc[-1])
        except Exception as exc:
            log(f"failed to fetch raw close for {code}: {exc}")
    return raw_closes


@dataclass
class SignalSnapshot:
    # current_momentum 是面向用户展示的信号 ETF 动量。
    # effective_momentum 保留默认正式基线内部使用的混合热度口径。
    strategy_id: str
    strategy_label: str
    trade_date: str
    previous_signal: str | None
    previous_signal_name: str | None
    previous_signal_theme: str | None
    current_signal: str | None
    current_signal_name: str | None
    current_signal_theme: str | None
    previous_holding_date: str | None
    current_holding_date: str | None
    previous_holding: str | None
    previous_name: str | None
    previous_theme: str | None
    previous_exposure: float | None
    previous_portfolio: str
    current_holding: str | None
    current_name: str | None
    current_theme: str | None
    current_portfolio: str
    desired_holding: str | None
    desired_name: str | None
    desired_theme: str | None
    desired_exposure: float | None
    desired_portfolio: str
    risk_leader: str | None
    risk_leader_name: str | None
    risk_leader_theme: str | None
    risk_leader_momentum: float | None
    defensive_leader: str | None
    defensive_leader_name: str | None
    defensive_leader_theme: str | None
    defensive_leader_momentum: float | None
    momentum_lookback_days: int | None
    momentum_start_date: str | None
    momentum_end_date: str | None
    current_momentum: float | None
    effective_momentum: float | None
    current_exposure: float | None
    current_price: float | None
    previous_close_price: float | None
    intraday_price_return: float | None
    market_session_label: str | None
    live_nav_coverage: float | None
    market_volume_mode: str | None
    market_volume_progress: float | None
    market_amount_ratio_20_60: float | None
    market_amount_ratio_5_20: float | None
    market_breadth_proxy: float | None
    max_drawdown: float | None
    nav_date: str | None
    current_nav: float | None
    live_nav_date: str | None
    live_nav: float | None
    current_drawdown: float | None
    peak_nav_date: str | None
    peak_nav_value: float | None
    entry_risk_score: float | None
    entry_risk_level: str | None
    historical_regime_label: str | None
    historical_regime_count: int | None
    historical_win_rate_60: float | None
    historical_win_rate_60_percentile: float | None
    historical_avg_ret_60: float | None
    historical_avg_mdd_60: float | None
    historical_avg_ret_60_percentile: float | None
    momentum_percentile: float | None
    selected_momentum_percentile: float | None
    drawdown_buffer_ratio: float | None
    entry_advice: str | None
    extra_cap_triggered: bool
    extra_cap_label: str
    extra_cap_reason: str | None
    selected_momentum_pct_cap_triggered: bool
    top2_close_cap_triggered: bool
    top2_close_gap: float | None
    top2_close_risk_cap: float | None
    base_exposure: float | None
    confirmed_trade_date: str | None
    confirmed_trade_details: str
    confirmed_trade_reason: str
    trade_details: str
    pending_trade_reason: str
    changed: bool


def compute_snapshot(
    prices: pd.DataFrame,
    strategy_config: dict[str, object],
    result: pd.DataFrame,
    close_result: pd.DataFrame,
    close_trades: pd.DataFrame,
    base_target_exposure: pd.Series,
    market_proxy_context: dict[str, float | str | None],
    spot_prices: dict[str, float],
    raw_closes: dict[str, float],
    now: datetime,
) -> SignalSnapshot:
    return build_signal_snapshot(
        SignalSnapshot,
        prices=prices,
        strategy_config=strategy_config,
        result=result,
        close_result=close_result,
        close_trades=close_trades,
        base_target_exposure=base_target_exposure,
        market_proxy_context=market_proxy_context,
        spot_prices=spot_prices,
        raw_closes=raw_closes,
        now=now,
    )


def _load_selected_pool_from_output(output_dir: Path, fallback_loader) -> list[dict[str, object]]:
    selected_path = output_dir / "selected_etfs.csv"
    if selected_path.exists():
        try:
            selected = pd.read_csv(selected_path, dtype={"code": str, "sina_symbol": str}).fillna("")
            if "code" in selected.columns:
                selected["code"] = selected["code"].map(lambda value: str(value).strip())
            if "sina_symbol" in selected.columns:
                selected["sina_symbol"] = selected["sina_symbol"].map(lambda value: str(value).strip())
            return selected.to_dict("records")
        except Exception as exc:
            log(f"failed to load selected_etfs from {selected_path}, fallback to loader: {exc}")
    fallback = fallback_loader().copy()
    if "code" in fallback.columns:
        fallback["code"] = fallback["code"].map(lambda value: str(value).strip())
    if "sina_symbol" in fallback.columns:
        fallback["sina_symbol"] = fallback["sina_symbol"].map(lambda value: str(value).strip())
    return fallback.to_dict("records")


def build_strategy_config(strategy_id: str) -> dict[str, object]:
    if strategy_id != "default":
        raise ValueError(f"unsupported strategy id: {strategy_id}")
    output_dir = CORE_OUTPUT_DIR
    strategy_label = DEFAULT_STRATEGY_NAME
    selected_pool = _load_selected_pool_from_output(output_dir, load_default_strategy_backtest_pool)

    return {
        "strategy_id": strategy_id,
        "strategy_label": strategy_label,
        "selected_pool": selected_pool,
        "backtest_nav_file": str(output_dir / "backtest_nav.csv"),
        "state_file": str(MONITOR_OUTPUT_DIR / "daily_monitor_state.json"),
        "extra_cap_label": "命中额外降仓",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF 动量策略每日巡检。")
    parser.add_argument(
        "--strategy",
        choices=["default"],
        default="default",
        help="选择要巡检的策略快照。",
    )
    parser.add_argument("--force", action="store_true", help="忽略状态缓存并写入新状态；适合正式补发，且允许非交易日手动补发。")
    parser.add_argument("--preview-send", action="store_true", help="发送当前快照但不读写去重状态；人工补发默认应使用这个参数，且允许非交易日手动补发。")
    parser.add_argument("--debug", action="store_true", help="输出调试日志到 stderr。")
    parser.add_argument("--webhook-url", type=str, default="", help="临时指定飞书机器人 webhook。")
    parser.set_defaults(disable_direct_feishu=False)
    parser.add_argument("--disable-direct-feishu", dest="disable_direct_feishu", action="store_true", help="关闭直连飞书私信，仅保留 webhook。")
    parser.add_argument("--enable-direct-feishu", dest="disable_direct_feishu", action="store_false", help="开启直连飞书私信，与 webhook 并行发送（默认）。")
    return parser.parse_args()


def main() -> int:
    global DEBUG_ENABLED
    args = parse_args()
    DEBUG_ENABLED = args.debug
    ensure_output_dirs()
    log(
        f"start strategy={args.strategy}, force={args.force}, preview_send={args.preview_send}, debug={args.debug}, "
        f"disable_direct_feishu={args.disable_direct_feishu}"
    )
    strategy_config = build_strategy_config(args.strategy)
    webhook_url = build_webhook_url(args)
    state_file = Path(strategy_config["state_file"])
    log(f"state_file={state_file}")
    now = datetime.now()
    today = now.date()
    market_session_label = classify_market_session(now)

    manual_override_non_trading_day = args.force or args.preview_send
    if not is_trading_day(today):
        if manual_override_non_trading_day:
            log(f"non-trading day override enabled: {today}")
        else:
            log(f"skip non-trading day: {today}")
            return 0

    allow_same_day_close = market_session_label == "收盘后"
    close_prices = build_price_panel(
        today,
        strategy_config,
        include_realtime=False,
        allow_same_day_close=allow_same_day_close,
    )
    close_result, close_trades, _, _ = run_strategy_snapshot(close_prices, strategy_config)
    persist_strategy_outputs(close_prices, strategy_config, close_result, close_trades)

    include_realtime_snapshot = should_include_realtime_snapshot(market_session_label)
    # 收盘后仍尝试拉一份场内原始价格，只用于价格展示；
    # 净值和回撤仍严格复用正式收盘口径，避免通知再次偏离 core。
    if include_realtime_snapshot or market_session_label == "收盘后":
        try:
            spot_prices = fetch_realtime_prices()
        except Exception as exc:
            log(f"failed to fetch realtime ETF prices in main, use cached closes: {exc}")
            spot_prices = {}
        raw_closes = fetch_latest_raw_closes(list(strategy_config["selected_pool"]), today)
    else:
        spot_prices = {}
        raw_closes = {}

    snapshot_prices = build_price_panel(
        today,
        strategy_config,
        include_realtime=include_realtime_snapshot,
        allow_same_day_close=allow_same_day_close,
        base_prices=close_prices,
        spot_prices=spot_prices,
        raw_closes=raw_closes,
    )
    result, trades, base_target_exposure, market_proxy_context = run_strategy_snapshot(snapshot_prices, strategy_config)
    snapshot = compute_snapshot(
        snapshot_prices,
        strategy_config,
        result,
        close_result,
        close_trades,
        base_target_exposure,
        market_proxy_context,
        spot_prices,
        raw_closes,
        now,
    )
    snapshot.delivery_slot = infer_delivery_slot(now)
    market_message = format_market_message(snapshot)
    trade_message = format_trade_message(snapshot)
    market_card = build_market_card(snapshot)
    trade_card = build_trade_card(snapshot)
    message_bundle = [market_message, trade_message]
    log(
        f"snapshot ready: trade_date={snapshot.trade_date}, session={market_session_label}, "
        f"holding={snapshot.current_holding}, desired={snapshot.desired_holding}, exposure={snapshot.current_exposure}, "
        f"slot={snapshot.delivery_slot or 'manual'}"
    )
    state = {} if args.preview_send else load_state(state_file)
    channels = build_channel_plan(disable_direct_feishu=args.disable_direct_feishu, webhook_url=webhook_url)
    if args.preview_send:
        log("preview_send mode enabled: skip state read/write and duplicate suppression")
    if not args.force and not args.preview_send and is_duplicate_snapshot(state, snapshot, market_message, trade_message, channels=channels):
        if not state_uses_message_cache(state):
            # 兼容旧状态文件：第一次命中重复时顺手升级到“按消息正文去重”，
            # 避免老 schema 长期停留在基于少数字段的粗粒度比较上。
            save_state(
                snapshot,
                state_file,
                market_message,
                trade_message,
                delivery_progress=[{channel: True for channel in channels} for _ in range(len(message_bundle))],
            )
        for message in message_bundle:
            print(message)
            print()
        log(
            "skip duplicate state: "
            f"strategy={snapshot.strategy_label}, trade_date={snapshot.trade_date}, "
            f"holding={snapshot.current_holding}, exposure={snapshot.current_exposure}"
        )
        return 0
    if args.force:
        log("force mode enabled: state cache will be ignored")

    delivery_progress = (
        build_empty_delivery_progress(len(message_bundle), channels)
        if args.force or args.preview_send
        else load_delivery_progress_for_bundle(state, snapshot, message_bundle, channels)
    )
    if args.preview_send:
        progress_callback = None
    else:
        progress_callback = lambda progress: save_state(
            snapshot,
            state_file,
            market_message,
            trade_message,
            delivery_progress=progress,
        )
    final_progress = send_card_bundle(
        [market_card, trade_card],
        disable_direct_feishu=args.disable_direct_feishu,
        webhook_url=webhook_url,
        delivery_progress=delivery_progress,
        progress_callback=progress_callback,
        log_fn=log,
    )
    if args.preview_send:
        log("preview_send mode complete: state file left unchanged")
    else:
        save_state(snapshot, state_file, market_message, trade_message, delivery_progress=final_progress)
        log(f"state saved to {state_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
