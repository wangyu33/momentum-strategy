# ETF Momentum Backtest

这个目录现在只保留正式链路：

- 正式回测
- 每日通知
- 动量榜单

如果只关心当前正式版本，只看下面 3 个入口：

- `run_backtest.py`
- `daily_monitor.py`
- `daily_momentum_board.py`

## 正式架构

正式实现已经收口到 `core/`，当前分成 8 层：

- `core/config.py`
  正式常量、资产池定义、输出目录定义

- `core/io.py`
  原子写文件、交易日历、代码标准化、通用 I/O

- `core/data.py`
  ETF 历史价格加载、现金分红总收益复权、实时价格拼接、正式价格缓存读取

- `core/validation.py`
  正式价格校验
  当前会输出 `output/core/price_validation_issues.csv`
  用来显式检查拆分/断点类异常；已知的拆分类断点会先本地回补复权，再把剩余异常写出来

- `core/signals.py`
  动量、质量分数、确认筛选、top2 接近压仓等信号层逻辑

- `core/proxy.py`
  大盘量能代理、市场广度代理、HS300 regime 过滤

- `core/backtest.py`
  权重回测、收益链重算、交易流水、绩效汇总

- `core/strategy.py`
  正式基线主链
  当前正式策略的 core-satellite、弱量能保护、过热压仓、弱市切债都在这一层

当前状态下：

- `run_backtest.py` 是正式最小 wrapper
- 正式实现只来自 `core/`
- 旧研究脚本已经移除，不再作为正式依赖
- 当前正式口径以仓库内这套实现和 `output/core/backtest_nav.csv` 为准
- 正式回测现在不再使用历史锚点
  - 固定起点：`2012-01-01`
  - 固定终点：最新确认交易日
  - 结果全部按当前代码、当前参数、当前价格口径全历史裸重算

## 当前正式指标

最近一次正式回测结果：

- 截至 `2026-09-18`
- 期末净值：`16.5629`
- 年化收益：`21.6765%`
- 最大回撤：`-23.7022%`
- 最新目标组合：`159985 豆粕 100%`

这个数值不是 README 手填口径，而是每次改正式链路后都要重新跑 `run_backtest.py` 校验的结果。

## 当前口径与已知限制

当前这套正式口径现在是“固定起止日期的全历史裸重算”版本。

- 回测起点固定为 `2012-01-01`
- 回测终点固定为最新确认交易日
- 参数、代码、价格处理方式只要变化，整条历史 `nav` 都会跟着重跑`

- `price_validation_issues.csv` 现在是“已知拆分型断点先修，再把剩余异常显式暴露”
  - 像 `159941` 这种接近整数拆分因子的跳变，会在正式价格链里先做本地 back-adjust
  - 不能识别成拆分的异常 gap，仍然会继续保留在报告里，避免静默污染回测

- 当前没有把 walk-forward / 样本外验证并入正式链路
  - 这次重构重点是先把正式实现收口、把历史研究残留清理掉
  - 参数稳健性验证仍然需要后续单独补建设计

所以现在的仓库状态可以概括成：

- 正式代码结构已经大幅收口
- 正式产物和通知链路已经只依赖核心模块
- 正式收益口径已经改成当前代码全历史裸重算
- 数据断点检测和已知拆分类自动修复都已经并入正式链路

## 正式命令

正式回测：

```bash
PYTHONPATH=.tools/python_packages python3 momentum_backtest/run_backtest.py
```

默认会从 `2012-01-01` 起，重算到最新确认交易日。

每日通知预览发送：

```bash
PYTHONPATH=.tools/python_packages python3 momentum_backtest/daily_monitor.py --preview-send
```

动量榜单预览发送：

```bash
PYTHONPATH=.tools/python_packages python3 momentum_backtest/daily_momentum_board.py --preview-send
```

## 正式输出

正式结果统一写到：

- `output/core/`
- `output/analysis/`
- `output/monitor/`

其中最重要的是：

- `output/core/selected_etfs.csv`
- `output/core/prices.csv`
- `output/core/backtest_nav.csv`
- `output/core/trades.csv`
- `output/core/price_validation_issues.csv`

## 当前正式基线

当前正式策略标识：

`threshold_boll_extreme_cap`

可以把它理解成 3 层：

1. 风险池 / 防守池主选
   - 25 日动量
   - 正式基线这里用的是 `raw` 25 日动量，不带 `slope_085`
   - 风险池和防守池各自选出 1 个主信号

2. dual momentum 仓位框架
   - 风险池最强主信号 `> 5%`：满仓风险资产
   - 风险池最强主信号在 `(0, 5%]`：切到防守池，仓位 `80%`
   - 风险池最强主信号 `<= 0`：切到防守池，仓位 `100%`

3. Boll 过热压仓
   - 如果当前持仓价格触及 20 日布林上轨
   - 且布林带宽处在历史高位（120 日百分位 `>= 80%`），仓位上限压到 `80%`
   - 如果带宽进一步进入极热区（120 日百分位 `>= 90%`），仓位上限进一步压到 `42.25%`

当前正式基线不包含：

- regime 双核
- 弱量能压仓
- top2 接近降仓
- stress bond 覆盖
- 过热分位减仓 / RSI / KDJ / 最小调仓门槛

## 数据完整性约束

这次重构明确把之前踩过的坑固化进正式链路：

- 现金分红不能忽略
  - 历史价格会走本地总收益复权

- 份额拆分/断点不能静默通过
  - 会额外生成 `price_validation_issues.csv`
  - 像 `159941` 这种接近整数拆分因子的大跳变，会先做本地 back-adjust，再把修复后仍残留的异常打出来

- 正式策略收益不能靠研究脚本“顺手拼出来”
  - 正式回测必须走 `core/strategy.py`

- 正式输出 schema 必须稳定
  - `daily_monitor.py` 和 `daily_momentum_board.py` 依赖的字段保持兼容

## 重构方法

这次重构不是简单“拆文件”，而是按可持续演进的思路收口：

1. 先保收益，再收结构
   - 正式策略先保持行为不变，再把实现迁到 `core/`
   - 每次只改一小段，再跑正式回测确认指标没有漂移

2. 用最小 wrapper 隔离入口
   - `run_backtest.py`
   - `daily_monitor.py`
   - `daily_momentum_board.py`
   这 3 个入口尽量只做参数解析和流程编排，不再承载策略细节

3. 用稳定接口替代跨脚本耦合
   - `core/strategy.py` 只负责正式策略
   - `core/data.py` 只负责价格链
   - `core/reporting.py` 只负责落盘和汇总
   这样后面扩展新变体时，可以新增旁路模块，而不是再去复制正式脚本

4. 先补观察点，再做大改
   - 价格链先加 `price_validation_issues.csv`
   - 正式结果先稳定输出 `selected_momentum_percentile`、`target_exposure`、`rebalance_threshold_blocked` 这些关键状态字段
   这样后续重构不只看终值，还能看内部状态是否漂移

## 外部参考

这次重构的做法主要参考下面几类方法：

- Martin Fowler 的 `Refactoring`
  - 强调用一系列小的、行为保持的变换来改结构，而不是一次性重写
  - 这里对应到本项目，就是每次收一个模块后都重新回测，而不是直接整体翻掉

- Martin Fowler / Trunk-Based Development 的 `Branch by Abstraction`
  - 大改动先在稳定入口后面引入新抽象层，再逐步把旧实现迁过去
  - 这里对应到本项目，就是先让 `run_backtest.py` / 通知入口继续稳定，再把正式逻辑收口到 `core/`

- Legacy code 的 `seam` / characterization 思路
  - 先把已有行为显式观察出来，再做结构调整
  - 这里对应到本项目，就是先把正式输出和关键状态字段稳定下来，再删旧逻辑

- 固定输入、全量重跑的可复现回测原则
  - 固定起点、固定终点、固定价格处理口径之后，策略结果应完全由当前代码和参数决定
  - 这里对应到本项目，就是不再引入历史净值锚点，而是每次从头跑到尾

参考材料：

- Martin Fowler, Refactoring
  - `https://martinfowler.com/books/refactoring.html`
- Martin Fowler, Branch by Abstraction
  - `https://martinfowler.com/bliki/BranchByAbstraction.html`
- Trunk-Based Development, Branch by Abstraction
  - `https://trunkbaseddevelopment.com/branch-by-abstraction/`
- Martin Fowler, Legacy Seam
  - `https://martinfowler.com/bliki/LegacySeam.html`

## 后续开发原则

后续如果继续开发正式策略，按这几个原则来：

- 先改 `core/`
- 新策略变体先做旁路，不直接改正式入口
- 正式字段新增可以，重命名或删字段要先检查通知链路
- 任何价格处理改动，都先看 `price_validation_issues.csv`
- 任何正式逻辑改动，都必须重新跑正式回测
- 每次改完至少核对：
  - 期末净值
  - 年化
  - 最大回撤
  - trade_count / trade_action_count
  - 最新目标组合
  - `target_exposure`
  - `selected_momentum_percentile`

## 最小验收流程

每次动正式链路，按下面顺序验收：

1. 编译检查

```bash
PYTHONPYCACHEPREFIX=/Users/bytedance/wy_test/momentum-strategy/.pycache python3 -m py_compile \
  momentum_backtest/core/config.py \
  momentum_backtest/core/strategy.py \
  momentum_backtest/monitor_snapshot_builders.py \
  momentum_backtest/monitor_render.py \
  momentum_backtest/daily_monitor.py \
  momentum_backtest/daily_momentum_board.py
```

2. 正式回测

```bash
MPLCONFIGDIR=/Users/bytedance/wy_test/momentum-strategy/.mpl \
PYTHONPATH=.tools/python_packages \
python3 momentum_backtest/run_backtest.py
```

3. 通知预览

```bash
PYTHONPATH=.tools/python_packages python3 momentum_backtest/daily_monitor.py --preview-send
PYTHONPATH=.tools/python_packages python3 momentum_backtest/daily_momentum_board.py --preview-send
```

4. 对比影响

- 如果是纯重构，`backtest_nav.csv` 的净值链应保持不变
- 如果是策略改动，至少要明确说明对年化、最大回撤、净值终值、最新组合的影响
- 如果是数据修正或参数变动，接受整条历史 `nav` 随当前代码全量重算

## 当前重构状态

已经完成的部分：

- 新建正式 `core/` 包
- 正式策略主链迁入 `core/strategy.py`
- 正式市场代理迁入 `core/proxy.py`
- 正式价格复权与价格校验迁入 `core/data.py` / `core/validation.py`
- 正式输出与汇总迁入 `core/reporting.py`
- `run_backtest.py` 已收成正式最小 wrapper
- 非正式研究脚本与归档文件已移除
