#!/usr/bin/env python3
"""Shared helpers for frontier factor exploration."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from runtime_env import write_text_atomic
    from run_backtest import (
        RESEARCH_OUTPUT_DIR,
        build_asset_own_momentum_percentile,
        load_core_selected_and_prices,
        write_dataframe_csv_atomic,
    )
except ImportError:  # pragma: no cover - local fallback
    from momentum_backtest.runtime_env import write_text_atomic
    from momentum_backtest.run_backtest import (
        RESEARCH_OUTPUT_DIR,
        build_asset_own_momentum_percentile,
        load_core_selected_and_prices,
        write_dataframe_csv_atomic,
    )


FRONTIER_OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "frontier_factors"


def ensure_frontier_output_dir() -> Path:
    FRONTIER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return FRONTIER_OUTPUT_DIR


def load_selected_prices():
    selected, prices = load_core_selected_and_prices()
    selected = selected.copy()
    selected["code"] = selected["code"].astype(str).str.strip()
    prices = prices.copy()
    prices.columns = [str(col).strip() for col in prices.columns]
    return selected, prices


def code_to_theme_map(selected):
    return dict(zip(selected["code"].astype(str), selected["theme"].astype(str)))


def render_markdown_table(df) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(col) for col in df.columns]
    rows = [cols]
    for _, row in df.iterrows():
        rows.append([str(row[col]) for col in df.columns])
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(cols))]

    def fmt_row(values):
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    header = fmt_row(cols)
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(cols))) + " |"
    body = [fmt_row(row) for row in rows[1:]]
    return "\n".join([header, divider, *body])


def save_markdown(path: Path, text: str) -> None:
    write_text_atomic(path, text)


__all__ = [
    "FRONTIER_OUTPUT_DIR",
    "build_asset_own_momentum_percentile",
    "code_to_theme_map",
    "ensure_frontier_output_dir",
    "load_selected_prices",
    "render_markdown_table",
    "save_markdown",
    "write_dataframe_csv_atomic",
]
