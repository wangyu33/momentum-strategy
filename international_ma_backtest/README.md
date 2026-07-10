# international_ma_backtest

这是一套独立于 `momentum_backtest/` 的国际均线策略回测目录。

当前实现的是一版经典国际战术资产配置模型：

- 风险资产：`SPY / EFA / EEM / VNQ / DBC`
- 现金替代：`BIL`
- 信号：月末 `10-month SMA`
- 调仓：下月首个交易日生效
- 规则：每个风险资产固定战略权重 `20%`；若月末收盘价低于 10 月均线，则该 `20%` 切到 `BIL`

目录说明：

- `run_backtest.py`
  - 回测入口，默认回测最近 10 年
- `research_notes.md`
  - 国际均线策略调研摘要与本地落地方案
- `output/core/`
  - 价格、净值、交易、年度收益、摘要
- `output/analysis/`
  - 净值对比图、回撤图

示例：

```bash
python3 international_ma_backtest/run_backtest.py --years 10
```

默认输出：

- `output/core/prices.csv`
- `output/core/monthly_signals.csv`
- `output/core/backtest_nav.csv`
- `output/core/trades.csv`
- `output/core/yearly_returns.csv`
- `output/core/summary.csv`
- `output/core/summary.json`
- `output/analysis/nav_vs_benchmark.png`
- `output/analysis/max_drawdown.png`

摘要文件里会同时包含：

- 策略本身的年化、波动、Sharpe、最大回撤、回撤积分
- 等权风险资产基准的年化、波动、Sharpe、最大回撤
