#!/usr/bin/env python3
"""历史防守覆盖层实验共用的候选资产清单。"""

from __future__ import annotations


CASH_CANDIDATES = [
    {"theme": "货币ETF", "code": "511990", "name": "华宝添益", "sina_symbol": "sh511990"},
    {"theme": "银华日利", "code": "511880", "name": "银华日利ETF", "sina_symbol": "sh511880"},
    {"theme": "保证金", "code": "159001", "name": "保证金ETF", "sina_symbol": "sz159001"},
    {"theme": "货币ETF", "code": "511850", "name": "货币ETF", "sina_symbol": "sh511850"},
    {"theme": "理财金", "code": "511810", "name": "理财金ETF", "sina_symbol": "sh511810"},
]

TREASURY_10Y = {"theme": "10年国债", "code": "511260", "name": "十年国债ETF", "sina_symbol": "sh511260"}
TREASURY_30Y = {"theme": "30年国债", "code": "511090", "name": "30年国债ETF", "sina_symbol": "sh511090"}
TREASURY_CANDIDATES = [TREASURY_10Y, TREASURY_30Y]


def lookup_overlay_asset(code: str) -> dict[str, str]:
    for row in CASH_CANDIDATES + TREASURY_CANDIDATES:
        if row["code"] == code:
            return row
    raise KeyError(code)
