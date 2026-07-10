# international_risk_parity_backtest

这是一套独立于 `momentum_backtest/` 的国际非动量策略回测目录。

当前实现的是一版经典多资产风险平价简化模型：

- 资产池：`SPY / EFA / EEM / IEF / VNQ / GLD / DBC`
- 现金替代：`BIL`
- 信号：过去 `63` 个交易日年化波动率
- 权重：月末按 `1 / 波动率` 分配，下月首个交易日生效
- 调仓频率：月频

它和动量策略的核心区别是：

- 动量是在“谁涨得更强，就配谁”
- 风险平价是在“谁波动更大，就少配谁；谁波动更小，就多配谁”

目录说明：

- `run_backtest.py`
  - 回测入口，默认回测最近 `15` 年
- `research_notes.md`
  - 除动量外的常见有效策略摘要，以及本地为何选风险平价落地
- `output/core/`
  - 价格、月度目标权重、净值、交易、年度收益、摘要
- `output/analysis/`
  - 净值对比图、回撤图、权重堆叠图

示例：

```bash
python3 international_risk_parity_backtest/run_backtest.py --years 15
```

默认输出：

- `output/core/prices.csv`
- `output/core/monthly_weights.csv`
- `output/core/backtest_nav.csv`
- `output/core/trades.csv`
- `output/core/yearly_returns.csv`
- `output/core/summary.csv`
- `output/core/summary.json`
- `output/analysis/nav_vs_benchmark.png`
- `output/analysis/max_drawdown.png`
- `output/analysis/weights_stack.png`

摘要文件里会同时包含：

- 策略本身的年化、波动、Sharpe、最大回撤、回撤积分
- 等权多资产基准的年化、波动、Sharpe、最大回撤
- 当前最新前三大权重
