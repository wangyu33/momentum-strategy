# stock_multi_queue_backtest

这是一个独立于 `momentum_backtest/` 的 A 股个股多队列动量原型。

当前原型聚焦一件事：

- 把一组“细分行业龙头股 top2-3”组成固定候选池；
- 月末按复合动量打分；
- 横截面切成 `5` 个队列；
- 同一细分行业最多持有 `1` 只；
- 仅允许上市历史不少于 `250` 个交易日、且近 `20` 日成交额中位数不低于 `5` 亿元的股票参与排序；
- 下月持有最强 `Q1` 或 `Q1+Q2`，剩余仓位切到 `511260` 十年国债 ETF 防守。

当前版本已经补了两项实用增强：

- 优先复用 `output/core/prices.csv` 和 `output/core/stock_cache/` 的本地缓存，避免每次全量回测都重新拉历史行情；
- 新增“波动惩罚 + 市场过滤”变体，专门用于压个股轮动里的极端回撤。

最近又补了一项关键修正：

- 个股历史价格改为 `hfq` 后复权，并在复用 `prices.csv` 前做非正价格/极端日收益校验，避免长历史 `qfq` 前复权把高分红股票早期价格压到接近 `0` 甚至负数，从而污染动量和回撤统计。

## 当前候选池

按更细的行业子方向手工选股，当前每个细分方向先放 `2-3` 只样例龙头：

- 白酒：`贵州茅台`、`五粮液`、`泸州老窖`
- 创新药：`恒瑞医药`、`百济神州`、`贝达药业`
- CRO：`药明康德`、`泰格医药`、`康龙化成`
- 中药：`片仔癀`、`华润三九`、`白云山`
- 医疗器械：`迈瑞医疗`、`联影医疗`、`新产业`
- 连锁医疗：`爱尔眼科`、`通策医疗`
- 芯片设备：`北方华创`、`中微公司`、`长川科技`
- 晶圆制造：`中芯国际`、`华润微`、`士兰微`
- 光模块：`中际旭创`、`新易盛`、`天孚通信`
- 通信设备：`中兴通讯`、`光迅科技`
- 动力电池：`宁德时代`、`亿纬锂能`
- 整车：`比亚迪`、`赛力斯`
- 工控自动化：`汇川技术`、`先导智能`
- 家电：`美的集团`、`格力电器`、`海尔智家`
- 券商：`东方财富`、`中信证券`、`华泰证券`
- 银行：`招商银行`、`兴业银行`、`宁波银行`
- 工程机械：`三一重工`、`徐工机械`、`中联重科`
- 煤炭：`中国神华`、`陕西煤业`、`中煤能源`
- 有色贵金属：`紫金矿业`、`洛阳钼业`、`山东黄金`
- 农牧：`牧原股份`、`温氏股份`、`海大集团`
- 免税旅游：`中国中免`、`锦江酒店`
- 乳制品：`伊利股份`、`妙可蓝多`
- 美妆个护：`珀莱雅`、`丸美生物`
- 办公软件：`金山办公`、`致远互联`
- AI 应用：`科大讯飞`、`海天瑞声`
- 金融 IT：`恒生电子`、`同花顺`

防守资产：

- `511260` 十年国债 ETF

基准：

- `510300` 沪深300 ETF

## 当前策略变体

- `q1_top8_bond`
  - 月末按 `20/60/120` 日复合动量打分；
  - 只保留价格站上 `120` 日均线、上市历史不少于 `250` 个交易日、近 `20` 日成交额中位数不低于 `5` 亿元且分数高于 `12%` 的股票；
  - 横截面切成 `5` 队列；
  - 同一细分行业最多入选 `1` 只；
  - 下月持有 `Q1` 中最强的前 `8` 只，等权；
  - 如果没有合格股票，切到 `511260`。

- `q1q2_top12_bond`
  - 规则相同；
  - 但把可投资范围放宽到 `Q1 + Q2`；
  - 同一细分行业最多入选 `1` 只；
  - 最多持有 `12` 只，等权。

- `top8_guard`
  - 仍然只看 `Q1` 前 `8` 只；
  - 但在 `20/60/120` 日复合动量上，额外减去 `0.20 * 60日年化波动率`；
  - 同时要求 `510300` 收盘价站上 `200` 日均线，否则整仓切到 `511260`。

- `top8_guard_rv`
  - 信号和 `top8_guard` 相同；
  - 只是持仓权重不再等权，而是按 `1 / 60日年化波动率` 分配；
  - 用来降低单只高波动强趋势股对组合净值的冲击。

- `top8_guard_soft_rv`
  - 同样使用 `60` 日波动惩罚和逆波动权重；
  - 但不再用“站上/跌破年线”的二元开关；
  - 而是按 `clip(0.35 + 30 * (510300 / 200日均线 - 1), 0, 1)` 连续缩放权益仓位；
  - 剩余仓位自动补到 `511260`。

- `top8_guard_soft_liq2_rv`
  - 与 `top8_guard_soft_rv` 相同；
  - 但把近 `20` 日成交额中位数门槛从 `5` 亿元放宽到 `2` 亿元；
  - 目的是修正长历史个股样本里“固定 5 亿门槛过严、可投月太少”的问题。

- `top8_stable_rv`
  - 在 `top8_guard_rv` 的基础上继续增强信号层；
  - 额外加入 `20` 日下行波动惩罚，以及 `20` 日对数价格线性拟合 `R²` 奖励；
  - 当前系数为：`-0.20 * 60日波动率 - 0.10 * 20日下行波动率 + 0.06 * (20日R² - 0.5)`；
  - 目标是优先保留“上涨更顺、更少急跌”的个股；
  - 其余持仓、年线过滤和逆波动权重规则不变。

- `top8_stable_liq2_rv`
  - 与 `top8_stable_rv` 相同；
  - 但同样把近 `20` 日成交额中位数门槛放宽到 `2` 亿元；
  - 用来验证“稳定性打分分支”是否主要受流动性过滤过严拖累。

- `top8_quality_rv`
  - 在 `top8_stable_rv` 的基础上增加一层最简单的质量过滤；
  - 当前条件是：`ROE > 0`、`归母净利润 > 0`、`每股经营现金流 > 0`、`扣非净利润同比 > -30%`；
  - 财务摘要按保守生效日点时化后再参与过滤，避免把财报发布时间用成未来函数；
  - 当前这版更适合作为基本面实验对照，不是默认基线。

当前结论：

- 修正到 `hfq` 口径后，原先的 `top8_stable_rv` 已不再是最优分支；
- 当前风险收益比最好的活跃变体是 `top8_guard_soft_liq2_rv`；
- 固定 `5` 亿元流动性门槛对 14 年个股样本过严，放宽到 `2` 亿元后，`stable` 和 `guard_soft` 两条分支都明显改善；
- 简单硬质量过滤仍然明显跑输；
- 同行业机械扩池到 `top4-5` 仍然没有跑赢原始候选池；
- 所以下一轮优化重点更应转向“候选结构重组 + 信号升级”，而不是继续抠质量阈值或简单加深同一行业候选数量。

## 当前14年结果

`2026-07-03` 这轮修正里，个股历史价格口径已经从 `qfq` 切到 `hfq`。旧版 README 里那组 `14` 年高收益结果建立在失真的长历史 `qfq` 价格上，不再作为有效结论。

当前已重新验证的 `14` 年结果如下：

- `q1_top8_bond`：年化 `11.84%`，波动 `28.65%`，最大回撤 `-39.00%`，夏普 `0.41`
- `q1q2_top12_bond`：年化 `12.26%`，波动 `30.20%`，最大回撤 `-46.83%`，夏普 `0.41`
- `top8_guard`：年化 `11.50%`，波动 `25.92%`，最大回撤 `-42.40%`，夏普 `0.44`
- `top8_guard_rv`：年化 `11.57%`，波动 `25.90%`，最大回撤 `-42.40%`，夏普 `0.45`
- `top8_guard_soft_rv`：年化 `11.92%`，波动 `25.48%`，最大回撤 `-42.40%`，夏普 `0.47`
- `top8_guard_soft_liq2_rv`：年化 `14.53%`，波动 `26.52%`，最大回撤 `-42.40%`，夏普 `0.55`
- `top8_stable_rv`：年化 `7.97%`，波动 `24.00%`，最大回撤 `-44.91%`，夏普 `0.33`
- `top8_stable_liq2_rv`：年化 `9.61%`，波动 `25.37%`，最大回撤 `-42.40%`，夏普 `0.38`
- `top8_quality_rv`：年化 `3.83%`，波动 `23.28%`，最大回撤 `-46.36%`，夏普 `0.16`

如果要先定一个“当前默认研究基线”，现在优先看 `top8_guard_soft_liq2_rv`：

- 它在活跃变体里年化和夏普都最高；
- 最大回撤没有比 `guard_soft` 分支恶化；
- 平均持仓月数更多，`169` 个调仓月里有持仓的月份提高到 `84` 个；
- 市场允许开仓但一个票都选不出来的月份，从 `top8_guard_soft_rv` 的 `37` 个降到 `31` 个。

另外两条重要对照也要记住：

- “稳定性增强”分支并非无效，但在原始 `5` 亿流动性门槛下被压得太死，放宽到 `2` 亿后从年化 `7.97%` / 夏普 `0.33` 提升到年化 `9.61%` / 夏普 `0.38`；
- 同口径下做过一次“细分行业从 `top2-3` 扩到 `top4-5`”的扩池实验，没有改善，详见 `output/core/pool_expansion_compare_top8_stable_rv.csv`。

流动性门槛扫描结果见 `output/core/liquidity_floor_compare.csv`。从当前结果看，下一步更值得继续试的是“围绕 `guard_soft_liq2` 这条主线换候选结构、调信号”，而不是再简单扩池。

## 运行方式

```bash
python3 stock_multi_queue_backtest/run_backtest.py --years 8 --strategy all
python3 stock_multi_queue_backtest/run_backtest.py --years 8 --strategy q1_top8_bond
```

## 输出文件

- `output/core/selected_candidates.csv`
- `output/core/prices.csv`
- `output/core/backtest_nav_q1_top8_bond.csv`
- `output/core/backtest_nav_q1q2_top12_bond.csv`
- `output/core/backtest_nav_top8_guard.csv`
- `output/core/backtest_nav_top8_guard_rv.csv`
- `output/core/backtest_nav_top8_guard_soft_rv.csv`
- `output/core/backtest_nav_top8_guard_soft_liq2_rv.csv`
- `output/core/backtest_nav_top8_stable_rv.csv`
- `output/core/backtest_nav_top8_stable_liq2_rv.csv`
- `output/core/backtest_nav_top8_quality_rv.csv`
- `output/core/trades_q1_top8_bond.csv`
- `output/core/trades_q1q2_top12_bond.csv`
- `output/core/trades_top8_guard.csv`
- `output/core/trades_top8_guard_rv.csv`
- `output/core/trades_top8_guard_soft_rv.csv`
- `output/core/trades_top8_guard_soft_liq2_rv.csv`
- `output/core/trades_top8_stable_rv.csv`
- `output/core/trades_top8_stable_liq2_rv.csv`
- `output/core/trades_top8_quality_rv.csv`
- `output/core/monthly_signals_q1_top8_bond.csv`
- `output/core/monthly_signals_q1q2_top12_bond.csv`
- `output/core/monthly_signals_top8_guard.csv`
- `output/core/monthly_signals_top8_guard_rv.csv`
- `output/core/monthly_signals_top8_guard_soft_rv.csv`
- `output/core/monthly_signals_top8_guard_soft_liq2_rv.csv`
- `output/core/monthly_signals_top8_stable_rv.csv`
- `output/core/monthly_signals_top8_stable_liq2_rv.csv`
- `output/core/monthly_signals_top8_quality_rv.csv`
- `output/core/pool_expansion_compare_top8_stable_rv.csv`
- `output/core/liquidity_floor_compare.csv`
- `output/core/summary.csv`
- `output/analysis/nav_vs_benchmark.png`
- `output/analysis/drawdown_compare.png`

## 当前定位

这还是一个“研究原型”，不是正式基线，后面优先可以继续补三类东西：

- 候选池扩展：每个细分行业补成更完整的 `top5-10`
- 打分升级：加入更平滑的稳定性指标、盈利质量或基本面过滤
- 风控升级：单票止损、行业上限、市场弱势连续降仓而不是二元开关
