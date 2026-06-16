# 动量策略模块职责总览

更新时间：2026-06-08

## 1. 正式运行主入口

### `momentum_backtest/run_backtest.py`
职责：
- 正式基线回测入口
- 读取默认参数与正式标的池
- 调用正式策略主链
- 写出 `output/core/*` 正式结果

适用场景：
- 正式回测
- 定时 backfill
- 需要刷新 core 输出时

### `momentum_backtest/daily_monitor.py`
职责：
- 每日通知主编排入口
- 调度 close snapshot / realtime snapshot
- 组装消息 bundle
- 调用 state / delivery / render 模块

适用场景：
- 每日 monitor 定时任务
- 手动预览通知

---

## 2. 正式策略主链

### `momentum_backtest/official_strategy_core.py`
职责：
- 正式 target weight 主链唯一实现
- 封装：
  - core-satellite / regime mix
  - weak market guard
  - top2 close risk cap
  - continuous overheat cap

核心函数：
- `build_official_target_weights(...)`

使用建议：
- 任何要复用正式基线 target weight 主链的研究脚本，优先从这里接入
- 不要再在候选脚本里复制同样逻辑

---

## 3. 候选研究共享层

### `momentum_backtest/official_candidate_runner.py`
职责：
- 研究脚本共享 runner
- 统一加载正式研究上下文
- 统一从正式结果抽取 target weights
- 统一从局部变换后的权重生成 candidate 结果与摘要

核心函数：
- `load_official_research_context()`
- `extract_target_weights()`
- `run_candidate_from_weights()`
- `summarize_candidate()`

适用场景：
- 在正式基线上做局部候选改动

---

## 4. 通知层分层结构

### `momentum_backtest/monitor_snapshot.py`
职责：
- snapshot helper
- 风险评分、价格上下文、市场时段、回测参考上下文等基础计算

### `momentum_backtest/monitor_snapshot_builders.py`
职责：
- snapshot 装配 helper
- 负责把多个上下文拼成最终 `SignalSnapshot`

核心函数：
- `build_reference_and_risk_context()`
- `build_position_and_trade_context()`
- `build_signal_snapshot()`

### `momentum_backtest/monitor_render.py`
职责：
- 纯格式化 / 纯渲染
- 持仓文案、交易原因、市场热度消息、交易消息渲染

### `momentum_backtest/monitor_delivery.py`
职责：
- state 读写
- 去重
- delivery progress
- channel 计划
- direct / webhook 发送

### `momentum_backtest/monitor_pipeline.py`
职责：
- 运行编排辅助
- build price panel
- run strategy snapshot
- persist outputs

---

## 5. 当前已迁移到统一主链 / 共享 runner 的研究脚本

### 统一主链 + 参数覆盖
- `momentum_backtest/compare_top2_riskcap_candidates.py`

### 统一主链 + 局部权重变换
- `momentum_backtest/compare_top2_split_candidates.py`
- `momentum_backtest/compare_domestic_idea_adoptions.py`
- `momentum_backtest/compare_slope_hot_riskcap_candidates.py`

### 共享研究上下文已接入
- `momentum_backtest/compare_slope_combo_replacements.py`

---

## 6. 仍建议后续继续收口的脚本

优先级从高到低：

1. `momentum_backtest/compare_domestic_idea_adoptions.py`
   - 还能继续减少内部局部逻辑重复
2. `momentum_backtest/compare_slope_combo_replacements.py`
   - 已接共享上下文，但仍有较多独立逻辑
3. `momentum_backtest/archive/experiments/*`
   - 若未来仍频繁使用，建议逐步迁到统一主链模式

---

## 7. 改动入口建议

### 如果要改“正式策略逻辑”
优先看：
- `official_strategy_core.py`
- `run_backtest.py`

### 如果要改“候选研究框架”
优先看：
- `official_candidate_runner.py`
- 对应 `compare_*.py`

### 如果要改“通知文案 / 展示”
优先看：
- `monitor_render.py`

### 如果要改“风险评分 / 快照字段”
优先看：
- `monitor_snapshot.py`
- `monitor_snapshot_builders.py`

### 如果要改“去重 / webhook / 发送恢复”
优先看：
- `monitor_delivery.py`

### 如果要改“价格面板构造 / 运行快照 / 输出持久化”
优先看：
- `monitor_pipeline.py`

---

## 8. 当前策略维护原则

1. 正式主链只保留一个实现
2. 候选研究尽量通过“参数覆盖”或“局部权重变换”完成
3. 展示层需求优先改 `monitor_render.py`
4. 不再在研究脚本中复制正式 target weight 主链
5. 重构后必须跑：
   - `python3 momentum_backtest/validate_strategy_outputs.py`



## 9. 候选审阅辅助脚本

### `momentum_backtest/generate_candidate_review.py`
职责：
- 为单个候选生成最近持仓路径对照
- 生成 official vs candidate 通知对照
- 生成候选审阅输出目录

### `momentum_backtest/generate_reduce_core_pair_review.py`
职责：
- 为主候选与平衡备选生成并排审阅包
- 输出 summary/path/message 对照
- 支持候选升级决策材料整理


### `momentum_backtest/generate_candidate_decision_matrix.py`
职责：
- 根据候选 summary compare 结果生成决策矩阵
- 用于把主候选 / 平衡候选 / 不优先候选做结构化分层
