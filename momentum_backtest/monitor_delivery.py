#!/usr/bin/env python3
"""daily_monitor 的状态与发送层。"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

try:
    from .runtime_env import write_json_atomic
except ImportError:
    from runtime_env import write_json_atomic

FEISHU_OPEN_ID = "ou_a6a198ce5e9f97430f257a04b502f49b"
DEFAULT_FEISHU_WEBHOOK = "https://open.larkoffice.com/open-apis/bot/v2/hook/688de167-8de9-4822-aa1c-4dd723a4ace6"
WEBHOOK_RETRY_SLEEP_SECONDS = 1.5
WEBHOOK_MAX_RETRIES = 3


def build_lark_cli_env() -> dict[str, str]:
    env = os.environ.copy()
    path_parts = env.get("PATH", "").split(":") if env.get("PATH") else []
    preferred_bins = [
        "/usr/local/bin",
        "/opt/homebrew/bin",
        str((Path.home() / ".nvm" / "current" / "bin").resolve()),
    ]
    merged: list[str] = []
    for item in [*preferred_bins, *path_parts]:
        if item and item not in merged:
            merged.append(item)
    env["PATH"] = ":".join(merged)
    return env


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def build_state_payload(
    snapshot,
    market_message: str,
    trade_message: str,
    delivery_progress: list[dict[str, bool]] | None = None,
) -> dict[str, object]:
    payload = dict(snapshot.__dict__)
    payload["_market_message"] = market_message
    payload["_trade_message"] = trade_message
    if delivery_progress is not None:
        payload["_delivery_progress_schema"] = "v2"
    if delivery_progress is not None and any(item for item in delivery_progress):
        payload["_delivery_progress"] = delivery_progress
    return payload


def write_state_payload(state_file: Path, payload: dict[str, object]) -> None:
    write_json_atomic(state_file, payload)


def save_state(
    snapshot,
    state_file: Path,
    market_message: str,
    trade_message: str,
    delivery_progress: list[dict[str, bool]] | None = None,
) -> None:
    payload = build_state_payload(snapshot, market_message, trade_message, delivery_progress=delivery_progress)
    write_state_payload(state_file, payload)


def state_uses_message_cache(state: dict[str, object]) -> bool:
    return "_market_message" in state and "_trade_message" in state


def build_channel_plan(*, disable_direct_feishu: bool, webhook_url: str) -> list[str]:
    channels: list[str] = []
    if not disable_direct_feishu:
        channels.append("direct")
    if webhook_url:
        channels.append("webhook")
    return channels


def state_matches_message_bundle(
    state: dict[str, object],
    snapshot,
    market_message: str,
    trade_message: str,
) -> bool:
    return (
        state.get("strategy_label") == snapshot.strategy_label
        and state.get("trade_date") == snapshot.trade_date
        and state.get("_market_message") == market_message
        and state.get("_trade_message") == trade_message
    )


def build_empty_delivery_progress(message_count: int, channels: list[str]) -> list[dict[str, bool]]:
    return [{channel: False for channel in channels} for _ in range(message_count)]


def normalize_delivery_progress(
    raw_progress: object,
    *,
    message_count: int,
    channels: list[str],
) -> list[dict[str, bool]]:
    normalized = build_empty_delivery_progress(message_count, channels)
    if not isinstance(raw_progress, list):
        return normalized
    for idx in range(min(len(raw_progress), message_count)):
        item = raw_progress[idx]
        if not isinstance(item, dict):
            continue
        for channel in channels:
            normalized[idx][channel] = bool(item.get(channel, False))
    return normalized


def progress_is_complete(progress: list[dict[str, bool]], channels: list[str]) -> bool:
    if not channels:
        return True
    return all(all(item.get(channel, False) for channel in channels) for item in progress)


def load_delivery_progress_for_bundle(
    state: dict[str, object],
    snapshot,
    messages: list[str],
    channels: list[str],
) -> list[dict[str, bool]]:
    if not state_uses_message_cache(state) or not state_matches_message_bundle(state, snapshot, messages[0], messages[1]):
        return build_empty_delivery_progress(len(messages), channels)
    raw_progress = state.get("_delivery_progress")
    if raw_progress is None:
        if state.get("_delivery_progress_schema") == "v2":
            return build_empty_delivery_progress(len(messages), channels)
        return [{channel: True for channel in channels} for _ in range(len(messages))]
    return normalize_delivery_progress(raw_progress, message_count=len(messages), channels=channels)


def send_feishu_message(text: str) -> None:
    print(text)
    cmd = [
        "./scripts/lark-cli",
        "im",
        "+messages-send",
        "--as",
        "bot",
        "--user-id",
        FEISHU_OPEN_ID,
        "--text",
        text,
        "--idempotency-key",
        str(uuid.uuid4()),
    ]
    subprocess.run(cmd, check=True, env=build_lark_cli_env())


def build_webhook_url(args) -> str:
    return args.webhook_url or os.getenv("DAILY_MONITOR_WEBHOOK_URL", DEFAULT_FEISHU_WEBHOOK)


def send_webhook_message(text: str, webhook_url: str, log_fn: Callable[[str], None] | None = None) -> None:
    # 兼容本地测试与离线占位值：若 webhook 不是完整 URL，则视作 no-op 成功发送。
    if not str(webhook_url).startswith(("http://", "https://")):
        return
    payload = json.dumps({"msg_type": "text", "content": {"text": text}}, ensure_ascii=False).encode("utf-8")
    last_error: RuntimeError | None = None
    for attempt in range(1, WEBHOOK_MAX_RETRIES + 1):
        request = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"webhook send failed: HTTP {exc.code} {detail}")
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"webhook send failed: {exc}")
        else:
            try:
                result = json.loads(body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"webhook send failed: invalid response {body}") from exc
            if result.get("code", 0) == 0:
                return
            last_error = RuntimeError(f"webhook send failed: {result}")
            if result.get("code") != 11232:
                raise last_error

        if attempt < WEBHOOK_MAX_RETRIES:
            if log_fn is not None:
                log_fn(f"webhook retry {attempt}/{WEBHOOK_MAX_RETRIES} after throttling or transient failure")
            time.sleep(WEBHOOK_RETRY_SLEEP_SECONDS)

    if last_error is not None:
        raise last_error


def send_message_bundle(
    messages: list[str],
    *,
    disable_direct_feishu: bool,
    webhook_url: str,
    delivery_progress: list[dict[str, bool]] | None = None,
    progress_callback: Callable[[list[dict[str, bool]]], None] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> list[dict[str, bool]]:
    channels = build_channel_plan(disable_direct_feishu=disable_direct_feishu, webhook_url=webhook_url)
    progress = (
        normalize_delivery_progress(delivery_progress, message_count=len(messages), channels=channels)
        if delivery_progress is not None
        else build_empty_delivery_progress(len(messages), channels)
    )
    webhook_failures = 0
    direct_failures = 0
    for index, message in enumerate(messages):
        if log_fn is not None:
            pending_channels = [channel for channel in channels if not progress[index].get(channel, False)]
            log_fn(f"delivery start message {index + 1}/{len(messages)} pending={','.join(pending_channels) or 'none'}")
        if "direct" in channels and not progress[index].get("direct", False):
            try:
                send_feishu_message(message)
            except Exception as exc:
                direct_failures += 1
                if log_fn is not None:
                    log_fn(f"direct feishu send failed for message {index + 1}/{len(messages)}: {exc}")
            else:
                progress[index]["direct"] = True
                if log_fn is not None:
                    log_fn(f"direct feishu sent message {index + 1}/{len(messages)}")
                if progress_callback is not None:
                    progress_callback(progress)
        if "webhook" in channels and not progress[index].get("webhook", False):
            try:
                send_webhook_message(message, webhook_url, log_fn=log_fn)
            except Exception as exc:
                webhook_failures += 1
                if log_fn is not None:
                    log_fn(f"webhook send failed for message {index + 1}/{len(messages)}: {exc}")
            else:
                progress[index]["webhook"] = True
                if log_fn is not None:
                    log_fn(f"webhook sent message {index + 1}/{len(messages)}")
                if progress_callback is not None:
                    progress_callback(progress)
                if index < len(messages) - 1:
                    time.sleep(WEBHOOK_RETRY_SLEEP_SECONDS)
    if log_fn is not None:
        log_fn(
            "delivery summary: "
            f"messages={len(messages)}, direct_failures={direct_failures}, webhook_failures={webhook_failures}"
        )
    if channels and not any(any(item.get(channel, False) for channel in channels) for item in progress):
        raise RuntimeError("all delivery channels failed for every message")
    return progress


def is_duplicate_snapshot(
    state: dict[str, object],
    snapshot,
    market_message: str,
    trade_message: str,
    channels: list[str] | None = None,
) -> bool:
    if state_uses_message_cache(state):
        if not state_matches_message_bundle(state, snapshot, market_message, trade_message):
            return False
        effective_channels = [] if channels is None else channels
        progress = load_delivery_progress_for_bundle(state, snapshot, [market_message, trade_message], effective_channels)
        return progress_is_complete(progress, effective_channels)
    return (
        state.get("strategy_label") == snapshot.strategy_label
        and state.get("trade_date") == snapshot.trade_date
        and state.get("previous_portfolio") == snapshot.previous_portfolio
        and state.get("current_holding") == snapshot.current_holding
        and state.get("current_exposure") == snapshot.current_exposure
        and state.get("current_portfolio") == snapshot.current_portfolio
        and state.get("desired_holding") == snapshot.desired_holding
        and state.get("desired_exposure") == snapshot.desired_exposure
        and state.get("desired_portfolio") == snapshot.desired_portfolio
    )
