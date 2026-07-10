# a_share_industry_etf_momentum_backtest

这是一套独立于 `momentum_backtest/` 的 A 股行业 ETF 动量研究与回测目录。

当前版本聚焦一件事：

- 在一组更完整、且允许按上市时间动态加入的 A 股行业 ETF 池中做月频动量轮动；
- 当行业动量整体转弱时，切到 `511260` 十年国债 ETF 防守；
- 输出净值、交易、月度信号、年度收益、绩效摘要和回撤事件表。

当前固定行业池：

- `512480` 半导体ETF国联安
- `512660` 军工ETF国泰
- `512010` 医药ETF易方达
- `512170` 医疗ETF华宝
- `512690` 酒ETF鹏华
- `159928` 消费ETF汇添富
- `159996` 家电ETF国泰
- `512800` 银行ETF华宝
- `512880` 证券ETF国泰
- `515220` 煤炭ETF国泰
- `512400` 有色金属ETF南方
- `516150` 稀土ETF嘉实
- `159870` 化工ETF鹏华
- `512200` 房地产ETF南方
- `515880` 通信ETF国泰
- `159997` 电子ETF天弘
- `159852` 软件ETF嘉实
- `159869` 游戏ETF华夏
- `512980` 传媒ETF广发
- `516160` 新能源ETF南方
- `515790` 光伏ETF华泰柏瑞
- `515030` 新能源车ETF华夏
- `516810` 农业ETF华夏

防守资产：

- `511260` 十年国债 ETF

当前策略变体：

- `top2_bond`
  - 月末按 `20/60/120` 日复合动量排序；
  - 只接受复合动量高于 `15%` 的行业 ETF；
  - 下月首个交易日持有最强的前 `2` 只行业 ETF 等权；
  - 若没有足够候选，剩余权重切到 `511260`。
- `top3_bond`
  - 规则相同，但放宽到前 `3` 只行业 ETF 等权；
  - 剩余权重同样留在 `511260`。

样本口径：

- 默认回测最近 `10` 年；
- 不再要求所有行业 ETF 从起点就同时存在；
- 新上市 ETF 在有足够历史窗口后自动加入打分与排序。

示例：

```bash
python3 a_share_industry_etf_momentum_backtest/run_backtest.py --years 10 --strategy all
python3 a_share_industry_etf_momentum_backtest/run_backtest.py --years 10 --strategy top3_bond
```

默认输出：

- `output/core/selected_etfs.csv`
- `output/core/prices.csv`
- `output/core/backtest_nav_top2_bond.csv`
- `output/core/backtest_nav_top3_bond.csv`
- `output/core/trades_top2_bond.csv`
- `output/core/trades_top3_bond.csv`
- `output/core/monthly_signals_top2_bond.csv`
- `output/core/monthly_signals_top3_bond.csv`
- `output/core/yearly_returns_top2_bond.csv`
- `output/core/yearly_returns_top3_bond.csv`
- `output/core/drawdown_episodes_top2_bond.csv`
- `output/core/drawdown_episodes_top3_bond.csv`
- `output/core/summary.csv`
- `output/analysis/nav_vs_benchmarks.png`
- `output/analysis/drawdown_compare.png`

研究说明见：[`research_notes.md`](./research_notes.md)
