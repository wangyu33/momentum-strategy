# ETF 动量系统时点口径说明

这份文档用于说明当前 `momentum_backtest/` 系统里：

- 信号是在哪一天生成的
- 目标持仓是在哪一天记录的
- 确认持仓是在哪一天生效的
- `trades.csv` 是按哪一天记账的
- 为什么会出现“今日目标组合已变化，但今日交易仍显示无”

## 1. 先说结论

当前这套回测/监控系统的底层口径，不是“当日收盘信号、当日收盘立即成交并记为已持仓”，而是：

- `T` 日生成 `target_weights`
- `T+1` 日 `weights` 才生效
- `holding/exposure/nav/trades` 都跟随 `weights`

也就是说，当前实现更接近：

`T 日收盘决定目标组合，T+1 日持仓与交易结果落地。`

因此：

- `signal` / `target_weight_*` 表示“今天收盘后决定的目标”
- `holding` / `weight_*` / `trades.csv` 表示“已经确认生效的持仓结果”

## 2. 代码依据

最核心的实现位于：

- `compare_hs300_regime_fixes.py`（位于 `momentum_backtest/compare_hs300_regime_fixes.py`）

关键逻辑：

```python
weights = target_weights.shift(1).fillna(0.0)
```

以及：

```python
holding = weights.idxmax(axis=1)
signal = target_weights.idxmax(axis=1)
```

这意味着：

- `signal` 是今天目标指向谁
- `holding` 是上一天目标在今天真正生效后的持仓

`trades.csv` 也是基于同一套 `weights = target_weights.shift(1)` 逻辑构建的，因此它天然和 `holding` 同口径。

## 3. 各字段分别是什么意思

### 3.1 `signal`

含义：

- 当日根据价格面板和策略规则算出的目标信号

特点：

- 代表“今天收盘后策略想切到哪里”
- 不代表“今天已经持有哪里”

### 3.2 `target_weight_*`

含义：

- 当日收盘后生成的目标组合权重

特点：

- 是策略想要的目标仓位
- 还没有成为确认持仓

### 3.3 `holding`

含义：

- 当前已经确认生效的持仓主标的

特点：

- 对应的是 `weights`
- 不是 `target_weights`
- 因此天然比 `signal/target_weight_*` 慢一拍

### 3.4 `weight_*`

含义：

- 当前已经确认生效的组合权重

特点：

- 是 `target_weights.shift(1)` 后的结果
- 和 `holding`、`exposure`、`nav`、`trades.csv` 同口径

### 3.5 `trades.csv`

含义：

- 已经确认落地的权重变化记录

特点：

- 不是“当天目标变化记录”
- 而是“当天实际生效的持仓变化记录”

所以如果某天只出现了目标变化，但确认持仓尚未切换，那么：

- `target_weight_*` 会变
- `signal` 会变
- `trades.csv` 不一定会变

## 4. daily_monitor 当前如何取数

当前 `daily_monitor.py` 里有两条链：

### 4.1 确认链

用于展示：

- 当前确认持仓
- 确认仓位
- 确认交易
- 当前净值

来源：

- `close_result["holding"]`
- `close_result["weight_*"]`
- `close_trades`

也就是完全跟回测结果对齐。

### 4.2 目标链

用于展示：

- 今日目标组合
- 当前信号
- 风控压仓后的目标暴露

来源：

- `result["signal"]`
- `result["target_weight_*"]`
- `result["target_exposure"]`

也就是“今天收盘信号推出来的下一步目标”。

## 5. 为什么会出现“今日目标组合变了，但今日交易是无”

这是当前系统最容易引起误解的地方。

原因是：

- `今日目标组合` 看的是当日 `target_weight_*`
- `今日交易`（在收盘后口径下）看的是 `trades.csv`
- 而 `trades.csv` 跟 `holding/weight_*` 同步，使用的是 `shift(1)` 生效逻辑

所以当某一天发生如下情况时：

1. 当天收盘，策略信号改变
2. `signal/target_weight_*` 已经指向新组合
3. 但 `holding/weight_*` 仍是旧组合
4. `trades.csv` 也还没有记录这次变化

通知就会出现：

- 当前确认持仓：旧组合
- 今日目标组合：新组合
- 今日交易：无

这不是通知 bug，而是当前底层成交时点定义导致的结果。

## 6. 当前系统的通知口径

### 6.1 盘中

盘中允许展示：

- 当前确认持仓
- 今日目标组合
- 若目标与确认持仓不同，则提示“今天按当前信号需要执行的交易”

原因：

- 盘中本来就是给人工参考
- 可以展示“待执行动作”

### 6.2 收盘后

收盘后当前已经收紧为：

- 严格以 `backtest_nav.csv + trades.csv` 为准
- 不额外把“目标组合”推演成“已成交”

原因：

- 避免通知先行，和回测结果漂移
- 保证 `当前确认持仓`、`confirmed_trade_date`、`trades.csv` 三者始终同口径

## 7. 用一个例子说明

假设：

- `2026-05-29` 当天收盘信号从 `纳指73% + 沪深30027%`
  变成 `纳指100%`

在当前回测实现里，可能会出现：

- `2026-05-29`
  - `signal = 纳指ETF`
  - `target_weight = 纳指100%`
  - `holding = 纳指73% + 沪深30027%`
  - `trades.csv` 对 2026-05-29 仍然没有新交易

因此收盘后通知显示：

- 当前确认持仓：73/27
- 今日目标组合：100% 纳指
- 今日交易：无

这是当前实现下的自洽结果。

## 8. 如果未来想改成“当日收盘成交”

那就不能只改通知，必须改回测底层。

需要改的不是：

- `daily_monitor` 文案

而是：

- `run_target_weights_strategy(...)`
- `holding / weight / trade` 的定义
- `nav` 和手续费落账时点

否则会出现：

- 通知说今天成交了
- 但 `trades.csv` 没记
- `backtest_nav.csv` 也没同步
- 系统整体漂移

## 9. 当前推荐理解方式

在不改底层回测时点之前，建议统一这样理解：

- `signal / 今日目标组合`
  - 表示“今天收盘后策略想去哪里”

- `holding / 当前确认持仓 / trades.csv`
  - 表示“已经在回测结果里确认生效的仓位和交易”

只要按这个口径理解，当前系统就是一致的。

