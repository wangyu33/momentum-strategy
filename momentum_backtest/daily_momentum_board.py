#!/usr/bin/env python3
"""收盘后发送 ETF 候选池多窗口动量榜。"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import unicodedata

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports, write_json_atomic, write_text_atomic
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports, write_json_atomic, write_text_atomic

prepare_local_imports(__file__)
configure_matplotlib_env()

warnings.filterwarnings("ignore")

import pandas as pd

try:
    from .monitor_delivery import (
        build_channel_plan,
        build_empty_delivery_progress,
        build_webhook_url,
        load_state,
        normalize_delivery_progress,
        progress_is_complete,
        send_card_bundle,
    )
    from .monitor_snapshot import classify_market_session, should_include_realtime_snapshot
    from .monitor_pipeline import build_price_panel
    from .core.config import (
        CORE_OUTPUT_DIR,
        MONITOR_OUTPUT_DIR,
        load_default_strategy_backtest_pool,
    )
    from .core.io import ensure_output_dirs, is_trading_day, write_dataframe_csv_atomic
    from .core.signals import build_current_etf_momentum_percentile_table
except ImportError:
    from monitor_delivery import (
        build_channel_plan,
        build_empty_delivery_progress,
        build_webhook_url,
        load_state,
        normalize_delivery_progress,
        progress_is_complete,
        send_card_bundle,
    )
    from monitor_snapshot import classify_market_session, should_include_realtime_snapshot
    from monitor_pipeline import build_price_panel
    from core.config import (
        CORE_OUTPUT_DIR,
        MONITOR_OUTPUT_DIR,
        load_default_strategy_backtest_pool,
    )
    from core.io import ensure_output_dirs, is_trading_day, write_dataframe_csv_atomic
    from core.signals import build_current_etf_momentum_percentile_table


DEBUG_ENABLED = False
BOARD_WINDOWS = [25]
BOARD_STRATEGY_LABEL = "candidate_momentum_board_default"
BOARD_STATE_FILE = MONITOR_OUTPUT_DIR / "daily_momentum_board_state.json"
BOARD_CSV_FILE = MONITOR_OUTPUT_DIR / "daily_momentum_board.csv"
BOARD_TEXT_FILE = MONITOR_OUTPUT_DIR / "daily_momentum_board.txt"
BOARD_RUN_LOG = MONITOR_OUTPUT_DIR / "daily_momentum_board.run.log"


@dataclass
class MomentumBoardSnapshot:
    strategy_label: str
    trade_date: str
    latest_price_date: str
    market_session_label: str
    price_mode_label: str


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [daily_momentum_board] {message}"
    BOARD_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with BOARD_RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if DEBUG_ENABLED:
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF 候选池收盘后动量榜。")
    parser.add_argument("--force", action="store_true", help="忽略状态缓存并写入新状态。")
    parser.add_argument("--preview-send", action="store_true", help="发送当前榜单但不读写去重状态。")
    parser.add_argument("--debug", action="store_true", help="输出调试日志。")
    parser.add_argument("--webhook-url", type=str, default="", help="临时指定飞书机器人 webhook。")
    parser.set_defaults(disable_direct_feishu=True)
    parser.add_argument("--disable-direct-feishu", dest="disable_direct_feishu", action="store_true", help="关闭直连飞书私信，仅保留 webhook（默认）。")
    parser.add_argument("--enable-direct-feishu", dest="disable_direct_feishu", action="store_false", help="开启直连飞书私信，与 webhook 并行发送。")
    return parser.parse_args()


def load_selected_pool() -> list[dict[str, object]]:
    selected_path = CORE_OUTPUT_DIR / "selected_etfs.csv"
    if selected_path.exists():
        try:
            selected = pd.read_csv(selected_path, dtype={"code": str, "sina_symbol": str}).fillna("")
            if "code" in selected.columns:
                selected["code"] = selected["code"].map(lambda value: str(value).strip())
            if "sina_symbol" in selected.columns:
                selected["sina_symbol"] = selected["sina_symbol"].map(lambda value: str(value).strip())
            return selected.to_dict("records")
        except Exception as exc:
            log(f"failed to load selected_etfs from {selected_path}, fallback to default loader: {exc}")
    fallback = load_default_strategy_backtest_pool().copy()
    fallback["code"] = fallback["code"].map(lambda value: str(value).strip())
    fallback["sina_symbol"] = fallback["sina_symbol"].map(lambda value: str(value).strip())
    return fallback.to_dict("records")


def build_board_dataframe(prices: pd.DataFrame, selected_pool: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in selected_pool:
        code = str(item.get("code", "")).strip()
        if not code or code not in prices.columns:
            continue
        series = pd.to_numeric(prices[code], errors="coerce").dropna()
        if series.empty:
            continue

        row = {
            "code": code,
            "theme": str(item.get("theme", "")).strip(),
            "name": str(item.get("name", "")).strip(),
            "price": float(series.iloc[-1]),
        }
        for window in BOARD_WINDOWS:
            row[f"mom_{window}"] = float(series.iloc[-1] / series.iloc[-1 - window] - 1) if len(series) > window else float("nan")
        rows.append(row)

    board = pd.DataFrame(rows)
    if board.empty:
        return board
    percentile_snapshot = build_current_etf_momentum_percentile_table(
        pd.DataFrame(selected_pool),
        prices,
        lookback=25,
    )
    if not percentile_snapshot.empty:
        board = board.merge(
            percentile_snapshot[["code", "price_date", "mom_25_percentile"]],
            on="code",
            how="left",
        )
    if "price_date" not in board.columns:
        board["price_date"] = pd.NA
    if "mom_25_percentile" not in board.columns:
        board["mom_25_percentile"] = float("nan")
    return board.sort_values(["mom_25"], ascending=False).reset_index(drop=True)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.2f}%"


def display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def pad_display(text: str, target_width: int) -> str:
    padding = max(target_width - display_width(text), 0)
    return text + (" " * padding)


def build_board_message(snapshot: MomentumBoardSnapshot, board: pd.DataFrame) -> str:
    labels = [f"{row['code']} {row['theme']}" for _, row in board.iterrows()]
    values = [
        f"{format_pct(row['mom_25'])} ({format_pct(row['mom_25_percentile'])})"
        for _, row in board.iterrows()
    ]
    rank_width = max(2, len(str(len(board))))
    label_width = max(max(display_width(label) for label in labels), 18) if labels else 18
    value_width = max(max(display_width(value) for value in values), 8) if values else 8
    header = (
        f"{'排名':>{rank_width}}  "
        f"{pad_display('标的', label_width)}  "
        f"{'25日':>{value_width}}"
    )
    divider = "-" * display_width(header)
    lines = [
        f"ETF候选池动量榜 {snapshot.trade_date}",
        f"交易时段: {snapshot.market_session_label} / 价格口径: {snapshot.price_mode_label}",
        f"价格日期: {snapshot.latest_price_date} / 排序: 25日动量降序",
        header,
        divider,
    ]
    for index, row in board.iterrows():
        label = pad_display(labels[index], label_width)
        value_text = values[index]
        lines.append(
            f"{index + 1:>{rank_width}}  "
            f"{label}  "
            f"{value_text:>{value_width}}"
        )
    return "\n".join(lines)


def build_board_card(snapshot: MomentumBoardSnapshot, board: pd.DataFrame) -> dict[str, object]:
    board_fields: list[dict[str, object]] = []
    for index, row in board.iterrows():
        board_fields.extend(
            [
                {
                    "is_short": True,
                    "text": {"tag": "lark_md", "content": f"{index + 1}. {row['theme']}"},
                },
                {
                    "is_short": True,
                    "text": {
                        "tag": "lark_md",
                        "content": f"**{format_pct(row['mom_25'])} ({format_pct(row['mom_25_percentile'])})**",
                    },
                },
            ]
        )
    elements: list[dict[str, object]] = [
        {
            "tag": "markdown",
            "content": (
                f"交易时段 {snapshot.market_session_label} | 价格口径 {snapshot.price_mode_label}  \n"
                f"价格日期 {snapshot.latest_price_date} | 排序 25日动量降序"
            ),
        },
        {
            "tag": "div",
            "fields": board_fields,
        },
    ]
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": f"ETF候选池动量榜 {snapshot.trade_date}"},
        },
        "elements": elements,
    }


def state_matches_board(state: dict[str, object], snapshot: MomentumBoardSnapshot, message: str) -> bool:
    return (
        state.get("strategy_label") == snapshot.strategy_label
        and state.get("trade_date") == snapshot.trade_date
        and state.get("_message") == message
    )


def load_board_delivery_progress(
    state: dict[str, object],
    snapshot: MomentumBoardSnapshot,
    message: str,
    channels: list[str],
) -> list[dict[str, bool]]:
    if not state_matches_board(state, snapshot, message):
        return build_empty_delivery_progress(1, channels)
    return normalize_delivery_progress(state.get("_delivery_progress"), message_count=1, channels=channels)


def save_board_state(
    snapshot: MomentumBoardSnapshot,
    message: str,
    delivery_progress: list[dict[str, bool]] | None,
) -> None:
    payload: dict[str, object] = {
        **snapshot.__dict__,
        "_message": message,
        "_delivery_progress_schema": "v2",
    }
    if delivery_progress is not None and any(item for item in delivery_progress):
        payload["_delivery_progress"] = delivery_progress
    write_json_atomic(BOARD_STATE_FILE, payload)


def main() -> int:
    global DEBUG_ENABLED
    args = parse_args()
    DEBUG_ENABLED = args.debug
    ensure_output_dirs()
    log(
        f"start force={args.force}, preview_send={args.preview_send}, debug={args.debug}, "
        f"disable_direct_feishu={args.disable_direct_feishu}"
    )

    now = datetime.now()
    today = now.date()
    if not is_trading_day(today):
        log(f"skip non-trading day: {today}")
        return 0
    market_session_label = classify_market_session(now)
    include_realtime_snapshot = should_include_realtime_snapshot(market_session_label)
    allow_same_day_close = market_session_label == "收盘后"

    selected_pool = load_selected_pool()
    strategy_config = {
        "strategy_id": "default",
        "strategy_label": BOARD_STRATEGY_LABEL,
        "selected_pool": selected_pool,
        "backtest_nav_file": str(CORE_OUTPUT_DIR / "backtest_nav.csv"),
    }
    if include_realtime_snapshot or market_session_label == "收盘后":
        try:
            from .daily_monitor import fetch_latest_raw_closes, fetch_realtime_prices
        except ImportError:
            from daily_monitor import fetch_latest_raw_closes, fetch_realtime_prices
        try:
            spot_prices = fetch_realtime_prices() if include_realtime_snapshot else {}
        except Exception as exc:
            log(f"failed to fetch realtime ETF prices for momentum board, use cached closes: {exc}")
            spot_prices = {}
        raw_closes = fetch_latest_raw_closes(selected_pool, today)
    else:
        spot_prices = {}
        raw_closes = {}

    prices = build_price_panel(
        today,
        strategy_config,
        include_realtime=include_realtime_snapshot,
        allow_same_day_close=allow_same_day_close,
        spot_prices=spot_prices,
        raw_closes=raw_closes,
        log_fn=log,
    )
    if prices.empty:
        raise RuntimeError("empty price panel for momentum board")

    board = build_board_dataframe(prices, selected_pool)
    if board.empty:
        raise RuntimeError("empty momentum board")

    latest_price_date = str(prices.index.max().date())
    snapshot = MomentumBoardSnapshot(
        strategy_label=BOARD_STRATEGY_LABEL,
        trade_date=str(today),
        latest_price_date=latest_price_date,
        market_session_label=market_session_label,
        price_mode_label="盘中实时价" if include_realtime_snapshot else "收盘价",
    )
    message = build_board_message(snapshot, board)
    export_board = board.copy()
    for column in ["mom_25"]:
        export_board[column] = export_board[column].map(format_pct)
    export_board["price"] = export_board["price"].map(lambda value: f"{value:.4f}")
    write_dataframe_csv_atomic(export_board, BOARD_CSV_FILE, index=False)
    write_text_atomic(BOARD_TEXT_FILE, message + "\n")
    log(f"board ready: trade_date={snapshot.trade_date}, latest_price_date={snapshot.latest_price_date}, rows={len(board)}")

    webhook_url = build_webhook_url(args)
    channels = build_channel_plan(disable_direct_feishu=args.disable_direct_feishu, webhook_url=webhook_url)
    state = {} if args.preview_send else load_state(BOARD_STATE_FILE)
    if args.preview_send:
        log("preview_send mode enabled: skip state read/write and duplicate suppression")
    if args.force:
        log("force mode enabled: state cache will be ignored")

    delivery_progress: list[dict[str, bool]] | None = None
    if not args.force and not args.preview_send:
        delivery_progress = load_board_delivery_progress(state, snapshot, message, channels)
        if progress_is_complete(delivery_progress, channels):
            log(f"skip duplicate board: trade_date={snapshot.trade_date}, latest_price_date={snapshot.latest_price_date}")
            return 0
    elif not args.preview_send:
        delivery_progress = build_empty_delivery_progress(1, channels)

    if not args.preview_send:
        save_board_state(snapshot, message, delivery_progress)

    card = build_board_card(snapshot, board)
    final_progress = send_card_bundle(
        [card],
        disable_direct_feishu=args.disable_direct_feishu,
        webhook_url=webhook_url,
        delivery_progress=delivery_progress,
        progress_callback=(lambda progress: save_board_state(snapshot, message, progress)) if not args.preview_send else None,
        log_fn=log,
    )
    if not args.preview_send:
        save_board_state(snapshot, message, final_progress)
        log(f"state saved to {BOARD_STATE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
