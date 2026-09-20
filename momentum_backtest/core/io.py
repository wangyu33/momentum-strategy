"""正式策略核心 I/O 与通用工具。"""

from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

import akshare as ak
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ANALYSIS_OUTPUT_DIR, CORE_OUTPUT_DIR, MONITOR_OUTPUT_DIR, OUTPUT_DIR, TRADE_CALENDAR_CACHE


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, CORE_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR, MONITOR_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_dataframe_csv_atomic(df: pd.DataFrame, output_path: Path, *, index: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f"{output_path.stem}_",
            dir=str(output_path.parent),
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_path = Path(handle.name)
        df.to_csv(temp_path, index=index)
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def save_figure_atomic(fig: plt.Figure, output_path: Path, **savefig_kwargs: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=output_path.suffix or ".png",
            prefix=f"{output_path.stem}_",
            dir=str(output_path.parent),
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        fig.savefig(temp_path, **savefig_kwargs)
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def normalize_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def fetch_trade_calendar() -> pd.DataFrame:
    try:
        cal = ak.tool_trade_date_hist_sina().copy()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"]).dt.date
        ensure_output_dirs()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                prefix="trade_calendar_",
                dir=str(TRADE_CALENDAR_CACHE.parent),
                delete=False,
                encoding="utf-8",
            ) as handle:
                temp_path = Path(handle.name)
            cal.to_csv(temp_path, index=False)
            os.replace(temp_path, TRADE_CALENDAR_CACHE)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return cal
    except Exception:
        if TRADE_CALENDAR_CACHE.exists():
            cached = pd.read_csv(TRADE_CALENDAR_CACHE)
            cached["trade_date"] = pd.to_datetime(cached["trade_date"]).dt.date
            return cached
        raise


def is_trading_day(today: date) -> bool:
    try:
        cal = fetch_trade_calendar()
        return today in set(cal["trade_date"].tolist())
    except Exception:
        return today.weekday() < 5
