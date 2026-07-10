# international_all_weather_backtest

这是一套独立于 `momentum_backtest/` 的国际非动量策略回测目录。

当前实现的是一版极简 `All Weather Lite`：

- 资产池：`SPY / EFA / EEM / IEF / GLD / DBC / VNQ`
- 现金替代：`BIL`
- 权重：固定战略配置，不做择时、不做动量排序
- 调仓：月末检查，下月首个交易日恢复目标权重

本地采用的固定权重是：

- `IEF` `35%`
- `SPY` `20%`
- `GLD` `15%`
- `EFA` `10%`
- `DBC` `10%`
- `EEM` `5%`
- `VNQ` `5%`

这套东西和动量最大的区别是：

- 它不判断“谁更强”
- 它假设不同宏观环境下，总会有一部分资产扛得住
- 重点是跨资产分散和再平衡，而不是信号择时

目录说明：

- `run_backtest.py`
  - 回测入口，默认回测最近 `15` 年
- `research_notes.md`
  - 非动量候选策略筛选摘要，以及为何最终落全天候
- `output/core/`
  - 价格、月度目标权重、净值、交易、年度收益、摘要
- `output/analysis/`
  - 净值对比图、回撤图、权重堆叠图

示例：

```bash
python3 international_all_weather_backtest/run_backtest.py --years 15
```
