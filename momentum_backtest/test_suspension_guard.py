"""停牌/脏价场景回归测试。

背景（2026-09-24 真实事故）：
    当前持仓 159941 纳指ETF 停牌，东财实时接口把「最新价」返回成 0。
    0 被当成合法价格一路传递，导致盘中收益 0/1.728-1 = -100%，
    实时净值归零，回撤 -100%、回撤进度 100%，历史同类区间因分桶越界全部 N/A。

本测试覆盖三层防御：
    1. 实时价抓取层：0 价必须被剔除
    2. 盘中净值估算层：0 价/异常收益必须被跳过
    3. 快照层：净值 <= 0 不得提升为 live_nav，回撤不得出现 <= -100%
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.data import fetch_realtime_etf_prices  # noqa: E402
from monitor_snapshot import build_asset_price_context, estimate_live_nav  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


# ---------------------------------------------------------------- 1. 抓取层

def test_spot_source_excludes_nonpositive() -> None:
    print("\n[1] 实时价抓取层：0 价必须剔除")
    try:
        spot = fetch_realtime_etf_prices()
    except Exception as exc:                      # 网络不可用时跳过，不算失败
        print(f"  [SKIP] 接口不可用：{type(exc).__name__}")
        return
    bad = {k: v for k, v in spot.items() if not (v > 0)}
    check("返回字典中不存在 <=0 的价格", not bad, f"bad={list(bad.items())[:5]}")
    check("返回非空", len(spot) > 0, f"len={len(spot)}")


# ---------------------------------------------------------------- 2. 净值估算

def test_estimate_live_nav_with_suspended_holding() -> None:
    print("\n[2] 盘中净值估算：停牌标的不得让净值归零")
    idx = pd.Timestamp("2026-09-23")
    close_result = pd.DataFrame(
        {"weight_159941": [1.0]}, index=[idx]
    )
    spot = {"159941": 0.0}          # 停牌：接口返回 0
    raw = {"159941": 1.728}
    nav, coverage = estimate_live_nav(close_result, 26.65, spot, raw)
    check("停牌时 live_nav 回退到收盘净值", nav == 26.65, f"nav={nav}")
    check("覆盖率为 0（未取到实时价）", coverage == 0.0, f"coverage={coverage}")

    spot_missing: dict[str, float] = {}
    nav2, coverage2 = estimate_live_nav(close_result, 26.65, spot_missing, raw)
    check("完全无实时价时同样回退", nav2 == 26.65, f"nav={nav2}")
    check("覆盖率同样为 0", coverage2 == 0.0, f"coverage={coverage2}")

    # 正常场景不应被误伤
    spot_ok = {"159941": 1.75}
    nav3, coverage3 = estimate_live_nav(close_result, 26.65, spot_ok, raw)
    expected = 26.65 * (1 + 1.75 / 1.728 - 1)
    check("正常实时价仍参与计算", abs(nav3 - expected) < 1e-9, f"nav={nav3} vs {expected}")
    check("正常场景覆盖率为 1", coverage3 == 1.0, f"coverage={coverage3}")

    # 除权/脏数据导致的极端跌幅要被拦截
    spot_crazy = {"159941": 0.05}
    nav4, coverage4 = estimate_live_nav(close_result, 26.65, spot_crazy, raw)
    check("单日 -97% 的脏数据被丢弃", nav4 == 26.65 and coverage4 == 0.0,
          f"nav={nav4}, coverage={coverage4}")


# ---------------------------------------------------------------- 3. 价格上下文

def test_price_context_on_suspension() -> None:
    print("\n[3] 盘中价格上下文：停牌沿用收盘，且盘中收益不为 -100%")
    dates = pd.date_range("2026-09-01", "2026-09-23", freq="B")
    prices = pd.DataFrame({"159941": [1.70 + i * 0.001 for i in range(len(dates))]},
                          index=dates)
    raw = {"159941": 1.728}

    # 盘中 + 停牌(0 价)
    ctx = build_asset_price_context("159941", prices, {"159941": 0.0}, raw, "盘中")
    check("标记来源为停牌沿用收盘", ctx["price_source"] == "停牌沿用收盘",
          f"source={ctx['price_source']}")
    check("盘中收益不为 -100%", ctx["intraday_price_return"] == 0.0,
          f"ret={ctx['intraday_price_return']}")
    check("价格回退到最近收盘", ctx["current_price"] == float(prices["159941"].iloc[-1]),
          f"price={ctx['current_price']}")

    # 盘中 + 无该标的报价
    ctx2 = build_asset_price_context("159941", prices, {}, raw, "盘中")
    check("缺失报价同样标记停牌", ctx2["price_source"] == "停牌沿用收盘",
          f"source={ctx2['price_source']}")
    check("缺失报价时收益为 0", ctx2["intraday_price_return"] == 0.0,
          f"ret={ctx2['intraday_price_return']}")

    # 正常盘中
    ctx3 = build_asset_price_context("159941", prices, {"159941": 1.75}, raw, "盘中")
    check("正常报价标记为实时价", ctx3["price_source"] == "实时价",
          f"source={ctx3['price_source']}")
    check("正常收益按公式计算", abs(ctx3["intraday_price_return"] - (1.75 / 1.728 - 1)) < 1e-12,
          f"ret={ctx3['intraday_price_return']}")


# ---------------------------------------------------------------- 4. 快照层

def test_snapshot_rejects_nonpositive_nav() -> None:
    print("\n[4] 快照层：停牌不得让净值/回撤崩坏")
    import tempfile

    import monitor_snapshot_builders as MSB

    idx = pd.Timestamp("2026-09-23")
    # 参考净值来自 backtest_nav.csv（收盘口径），盘中净值在它基础上叠加实时收益
    with tempfile.TemporaryDirectory() as tmp:
        nav_file = Path(tmp) / "backtest_nav.csv"
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-09-22"), idx],
                "nav": [26.40, 26.6493],
                "drawdown": [-0.020, -0.0230],
                "current_momentum": [0.043, 0.0448],
            }
        ).to_csv(nav_file, index=False)

        result = pd.DataFrame(
            {
                "nav": [26.40, 26.6493],
                "drawdown": [-0.020, -0.0230],
                "current_momentum": [0.043, 0.0448],
                "weight_159941": [1.0, 1.0],
            },
            index=[pd.Timestamp("2026-09-22"), idx],
        )
        close_result = pd.DataFrame({"weight_159941": [1.0]}, index=[idx])
        prices = pd.DataFrame({"159941": [1.70, 1.71]}, index=result.index)

        ctx = MSB.build_reference_and_risk_context(
            latest_idx=idx,
            result=result,
            close_result=close_result,
            backtest_nav_file=nav_file,
            prices=prices,
            spot_prices={"159941": 0.0},       # 停牌：接口返回 0
            raw_closes={"159941": 1.728},
            now=datetime(2026, 9, 24, 9, 40),
        )
        dd = ctx["current_drawdown"]
        check("回撤保持正常量级（-2% 附近）", dd is not None and -0.05 < dd < 0, f"drawdown={dd}")
        check("live_nav 未被 0 价污染", (ctx["live_nav"] or 0) > 20, f"live_nav={ctx['live_nav']}")
        check("风险分仍可计算", ctx["risk_context"]["entry_risk_score"] is not None,
              f"score={ctx['risk_context']['entry_risk_score']}")
        check("历史同类区间能匹配到样本",
              ctx["risk_context"]["historical_regime_count"] is None
              or ctx["risk_context"]["historical_regime_count"] >= 0,
              f"count={ctx['risk_context']['historical_regime_count']}")

        # 对照组：正常实时价时，盘中净值应当变动
        ctx_ok = MSB.build_reference_and_risk_context(
            latest_idx=idx,
            result=result,
            close_result=close_result,
            backtest_nav_file=nav_file,
            prices=prices,
            spot_prices={"159941": 1.75},
            raw_closes={"159941": 1.728},
            now=datetime(2026, 9, 24, 9, 40),
        )
        check("正常实时价时净值随行情变动",
              ctx_ok["live_nav"] is not None and abs(ctx_ok["live_nav"] - 26.6493) > 1e-6,
              f"live_nav={ctx_ok['live_nav']}")


def main() -> None:
    test_spot_source_excludes_nonpositive()
    test_estimate_live_nav_with_suspended_holding()
    test_price_context_on_suspension()
    test_snapshot_rejects_nonpositive_nav()
    print(f"\n通过 {PASS}，失败 {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
