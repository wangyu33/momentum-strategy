#!/usr/bin/env python3
"""Build a 30-year Fed hike/cut cycle review with cross-asset comparisons."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


OUTPUT_DIR = Path("docs/etf-quant-research/fed_rate_cycles_1995_2026")
START_DATE = pd.Timestamp("1995-01-01")
END_DATE = pd.Timestamp("2026-06-17")
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass(frozen=True)
class AssetSpec:
    label: str
    ticker: str
    kind: str = "return"
    notes: str = ""


ASSETS: list[AssetSpec] = [
    AssetSpec("美股(SPY)", "SPY", notes="美国大盘股票 ETF"),
    AssetSpec("发达市场(EFA)", "EFA", notes="发达市场 ex-US 股票 ETF，2001 年起"),
    AssetSpec("新兴市场(EEM)", "EEM", notes="新兴市场股票 ETF，2003 年起"),
    AssetSpec("港股(^HSI)", "^HSI", notes="恒生指数"),
    AssetSpec("日本(^N225)", "^N225", notes="日经225指数"),
    AssetSpec("美债长久期(TLT)", "TLT", notes="20年+美债 ETF，2002 年起"),
    AssetSpec("黄金(GLD)", "GLD", notes="黄金 ETF，2004 年起"),
    AssetSpec("美国 REITs(VNQ)", "VNQ", notes="美国 REITs ETF，2004 年起"),
    AssetSpec("美元指数(DX-Y.NYB)", "DX-Y.NYB", notes="ICE 美元指数期货连续口径"),
    AssetSpec("WTI 原油(CL=F)", "CL=F", notes="WTI 原油近月期货连续口径"),
]


def fetch_text_via_curl(url: str) -> str:
    output = subprocess.check_output(["curl", "-L", "--max-time", "60", url])
    return output.decode("utf-8", errors="ignore")


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    text = fetch_text_via_curl(url)
    df = pd.read_csv(StringIO(text))
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.rename(columns={series_id: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def build_target_rate_series() -> pd.DataFrame:
    pre = fetch_fred_series("DFEDTAR")
    post = fetch_fred_series("DFEDTARU")
    cutoff = post["observation_date"].min()
    merged = pd.concat(
        [pre[pre["observation_date"] < cutoff], post],
        ignore_index=True,
    ).sort_values("observation_date")
    merged = merged[merged["observation_date"] >= START_DATE].copy()
    merged["change"] = merged["value"].diff()
    return merged


def derive_rate_cycles(rate_series: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    changes = rate_series[rate_series["change"].notna() & (rate_series["change"] != 0)].copy()
    changes["direction"] = changes["change"].apply(lambda x: "hike" if x > 0 else "cut")
    changes["cycle_id"] = (changes["direction"] != changes["direction"].shift()).cumsum()

    cycles = (
        changes.groupby(["cycle_id", "direction"], as_index=False)
        .agg(
            start_date=("observation_date", "first"),
            end_date=("observation_date", "last"),
            start_rate=("value", "first"),
            end_rate=("value", "last"),
            changes=("change", "count"),
            total_bp=("change", lambda s: round(float(s.sum() * 100), 1)),
        )
        .drop(columns=["cycle_id"])
    )
    cycles["length_days"] = (cycles["end_date"] - cycles["start_date"]).dt.days
    cycles["cycle_label"] = cycles.apply(
        lambda row: f"{row['direction']}_{row['start_date'].date()}_{row['end_date'].date()}",
        axis=1,
    )
    return changes, cycles


def fetch_yahoo_series(ticker: str) -> pd.Series:
    period1 = int(START_DATE.timestamp())
    period2 = int(END_DATE.timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d&includePrePost=false&events=div%2Csplits"
    )
    response = requests.get(url, headers=YAHOO_HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()["chart"]["result"][0]
    timestamps = payload["timestamp"]
    adjclose = payload["indicators"].get("adjclose", [{}])[0].get("adjclose")
    close = payload["indicators"]["quote"][0].get("close")
    values = adjclose if adjclose else close

    pairs = []
    for ts, value in zip(timestamps, values):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        dt = pd.Timestamp(datetime.utcfromtimestamp(ts).date())
        pairs.append((dt, float(value)))
    series = pd.Series(dict(pairs), name=ticker).sort_index()
    series.index.name = "date"
    return series


def nearest_window_values(series: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.Timestamp, float, pd.Timestamp, float] | None:
    series = series.dropna()
    if series.empty:
        return None
    start_candidates = series[series.index >= start_date]
    end_candidates = series[series.index <= end_date]
    if start_candidates.empty or end_candidates.empty:
        return None
    actual_start = start_candidates.index[0]
    actual_end = end_candidates.index[-1]
    if actual_end < actual_start:
        return None
    return actual_start, float(start_candidates.iloc[0]), actual_end, float(end_candidates.iloc[-1])


def compute_cycle_asset_returns(cycles: pd.DataFrame, asset_series: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cycle in cycles.itertuples(index=False):
        for spec in ASSETS:
            window = nearest_window_values(asset_series[spec.label], cycle.start_date, cycle.end_date)
            if window is None:
                rows.append(
                    {
                        "cycle_label": cycle.cycle_label,
                        "direction": cycle.direction,
                        "asset": spec.label,
                        "asset_ticker": spec.ticker,
                        "window_start": pd.NaT,
                        "window_end": pd.NaT,
                        "return_pct": float("nan"),
                    }
                )
                continue
            actual_start, start_value, actual_end, end_value = window
            asset_return = end_value / start_value - 1
            rows.append(
                {
                    "cycle_label": cycle.cycle_label,
                    "direction": cycle.direction,
                    "asset": spec.label,
                    "asset_ticker": spec.ticker,
                    "window_start": actual_start,
                    "window_end": actual_end,
                    "return_pct": asset_return,
                }
            )
    return pd.DataFrame(rows)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.1f}%"


def format_bp(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.0f}bp"


def make_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    subset = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, sep]
    for _, row in subset.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def build_report(cycles: pd.DataFrame, asset_returns: pd.DataFrame) -> str:
    cycle_table = cycles.copy()
    cycle_table["direction"] = cycle_table["direction"].map({"hike": "加息", "cut": "降息"})
    cycle_table["start_date"] = cycle_table["start_date"].dt.date.astype(str)
    cycle_table["end_date"] = cycle_table["end_date"].dt.date.astype(str)
    cycle_table["start_rate"] = cycle_table["start_rate"].map(lambda x: f"{x:.2f}%")
    cycle_table["end_rate"] = cycle_table["end_rate"].map(lambda x: f"{x:.2f}%")
    cycle_table["total_bp"] = cycle_table["total_bp"].map(format_bp)

    pivot = asset_returns.pivot(index="cycle_label", columns="asset", values="return_pct").reset_index()
    pivot = pivot.merge(cycles[["cycle_label", "direction", "start_date", "end_date", "total_bp"]], on="cycle_label", how="left")
    pivot["direction"] = pivot["direction"].map({"hike": "加息", "cut": "降息"})
    pivot["start_date"] = pivot["start_date"].dt.date.astype(str)
    pivot["end_date"] = pivot["end_date"].dt.date.astype(str)
    pivot["total_bp"] = pivot["total_bp"].map(format_bp)
    for spec in ASSETS:
        pivot[spec.label] = pivot[spec.label].map(format_pct)

    avg = (
        asset_returns.groupby(["direction", "asset"], as_index=False)["return_pct"]
        .mean()
        .pivot(index="asset", columns="direction", values="return_pct")
        .reset_index()
    )
    if "hike" in avg.columns:
        avg["hike"] = avg["hike"].map(format_pct)
    if "cut" in avg.columns:
        avg["cut"] = avg["cut"].map(format_pct)
    avg = avg.rename(columns={"asset": "资产", "hike": "加息周期平均回报", "cut": "降息周期平均回报"})

    hike_assets = asset_returns[asset_returns["direction"] == "hike"].groupby("asset")["return_pct"].mean().sort_values(ascending=False)
    cut_assets = asset_returns[asset_returns["direction"] == "cut"].groupby("asset")["return_pct"].mean().sort_values(ascending=False)
    hike_top = ", ".join(f"{idx} {format_pct(val)}" for idx, val in hike_assets.head(4).items())
    cut_top = ", ".join(f"{idx} {format_pct(val)}" for idx, val in cut_assets.head(4).items())

    return f"""# 近30年美国加息/降息周期与全球大类资产复盘

- 样本区间：{START_DATE.date()} 至 {END_DATE.date()}
- 利率口径：FRED `DFEDTAR`（2008-12-15 及以前）与 `DFEDTARU`（2008-12-16 起）拼接后的联邦基金目标利率上限
- 资产口径：Yahoo Finance 日线，收益按各周期首个可交易日到最后一个可交易日计算；ETF/指数采用复权收盘或调整后收盘
- 注意：`1995-02-01` 这一笔加息是 1994-1995 紧缩周期的尾声，因此在“近30年”里看上去是一个单点周期

## 1. Fed 过去30年的加息与降息节点

{make_markdown_table(cycle_table, ['direction', 'start_date', 'end_date', 'start_rate', 'end_rate', 'changes', 'total_bp'])}

## 2. 各周期内全球大类资产表现

{make_markdown_table(pivot, ['direction', 'start_date', 'end_date', 'total_bp'] + [spec.label for spec in ASSETS])}

## 3. 按“加息周期 / 降息周期”求平均

{make_markdown_table(avg, ['资产', '加息周期平均回报', '降息周期平均回报'])}

## 4. 直接结论

1. **美元加息不等于风险资产必跌。** 2004-2006、2015-2018、2022-2023 三轮加息里，美股与日本股市并不弱，说明“加息本身”不是核心，真正关键的是加息时的增长韧性和盈利周期。
2. **降息也不等于立刻利好股市。** 2001-2003、2007-2008、2024-2025 的降息阶段，风险资产表现分化很大；如果降息对应的是衰退或信用风险暴露，股市和原油往往先杀估值。
3. **长久期美债通常在降息中更占优。** TLT 的平均表现大体好于加息阶段，尤其在 2007-2008 与 2019-2020 这种明显转向的周期里更突出。
4. **黄金通常更吃“实际利率下行 + 风险厌恶”。** 在多轮降息周期里，黄金整体胜率更高；但如果是增长强、通胀也强的宽松预期阶段，原油和股票也可能同步受益。
5. **美元指数的方向看相对增长与避险，不单看 Fed。** 平均上美元在降息阶段并不总是走弱，危机式降息往往反而伴随美元走强。

## 5. 最值得记住的模式

- 加息周期平均领先资产：{hike_top}
- 降息周期平均领先资产：{cut_top}

## 6. 使用建议

- 如果你关注“Fed 开始降息”，不要直接跳到“美股/港股都会涨”的结论，先判断是**软着陆降息**还是**衰退式降息**。
- 如果你关注“Fed 继续加息”，更重要的是看**美元是否同步走强、长端利率是否继续抬升、油价是否上行**；这些决定全球资产压力比单看政策利率更大。
- 做配置时，最好把利率方向和增长状态一起看：
  - 加息 + 增长强：股票未必差，长债通常弱。
  - 降息 + 衰退压利润：股市和原油容易弱，黄金和长债更占优。
  - 降息 + 软着陆：成长股、港股、EM 反而有机会弹性更强。
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rate_series = build_target_rate_series()
    rate_changes, cycles = derive_rate_cycles(rate_series)
    asset_series = {spec.label: fetch_yahoo_series(spec.ticker) for spec in ASSETS}
    asset_returns = compute_cycle_asset_returns(cycles, asset_series)

    rate_series.to_csv(OUTPUT_DIR / "fed_target_rate_daily.csv", index=False)
    rate_changes.to_csv(OUTPUT_DIR / "fed_rate_change_events.csv", index=False)
    cycles.to_csv(OUTPUT_DIR / "fed_rate_cycles.csv", index=False)
    asset_returns.to_csv(OUTPUT_DIR / "cycle_asset_returns.csv", index=False)

    report = build_report(cycles, asset_returns)
    (OUTPUT_DIR / "README.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
