# 动量策略结构重构设计稿

更新时间：2026-06-08

## 1. 背景与目标

当前 `momentum_backtest/` 已经从“单一回测脚本”演化成：

- 正式基线策略入口
- 每日通知链路
- 多套候选研究脚本
- 多种风控与仓位管理变体

策略逻辑本身仍可解释，但代码结构已经出现明显的**职责混杂、主链复制、研究脚本分叉实现**问题。继续在现结构上叠加功能，会越来越难：

1. 明确“正式基线”到底是哪条主链
2. 保证研究候选只改了自己该改的那一条规则
3. 让通知口径与正式回测口径保持完全一致
4. 降低新增候选时的复制/粘贴实现成本

本设计稿的目标不是先删规则，而是先把结构收口，让后续优化更便宜、更安全、更容易验证。

---

## 2. 当前现状（基于仓库实况）

### 2.1 关键文件体量

当前几个核心文件已经明显偏大：

- `momentum_backtest/run_backtest.py`: 2234 行
- `momentum_backtest/daily_monitor.py`: 1949 行
- `momentum_backtest/compare_goal_optimizations.py`: 2210 行
- `momentum_backtest/compare_tail_risk_bond_overlay.py`: 563 行
- `momentum_backtest/compare_top2_riskcap_candidates.py`: 396 行
- `momentum_backtest/compare_top2_split_candidates.py`: 412 行

### 2.2 当前正式基线的逻辑主链

当前正式策略可概括为：

1. `slope` 信号质量排序
2. core-satellite / regime mix
3. weak market guard
4. top2 close risk cap
5. continuous overheat cap
6. stress bond overlay
7. 日常通知快照与解释

这条逻辑本身仍然是可讲清楚的，不属于“必须删规则”的阶段。

### 2.3 主要结构问题

#### 问题 A：正式主链在多个研究脚本中被重复实现

当前仓库内存在多套“主链变体复制”：

- `compare_goal_optimizations.py` 中的 `build_dynamic_core_target_weights`
- `compare_tail_risk_bond_overlay.py` 中的 `build_base_target_weights`
- `compare_top2_riskcap_candidates.py` 中的 `build_dynamic_core_target_weights_closeflag` / `build_base_target_weights_closeflag`
- `compare_top2_split_candidates.py` 中的 `build_dynamic_core_target_weights_split` / `build_base_target_weights_split`
- 以及后续新增的若干候选脚本

这些函数本质上都在做：

> 先构建正式策略主链，再额外改一层局部规则。

但现在很多候选是通过**复制主链再微改**来实现的，容易出现口径漂移。

#### 问题 B：`daily_monitor.py` 职责过重

当前 `daily_monitor.py` 同时承担：

- 价格拉取
- 市场时段判断
- 历史上下文恢复
- 风险评分
- 历史 regime 统计
- 交易解释
- 通知渲染
- 去重与发送状态管理

结果是：
- 展示层需求要改核心文件
- 风险计算与文案逻辑耦合
- 定位 bug 成本高

#### 问题 C：研究脚本缺少统一候选 runner

当前大部分研究脚本重复：

- 读取 prices/proxy
- 构建 baseline
- 跑 candidate
- 写 summary/nav/trades

这导致每个脚本都在重复做“框架搭建”，而不是只关心“候选规则本身”。

#### 问题 D：正式回测输出与研究脚本输出共享部分概念但实现路径不统一

例如：

- `target_weight_*`
- `base_target_exposure`
- `effective_momentum`
- `top2_close_risk_cap_triggered`
- `stress_bond_trigger`

这些字段在不同研究脚本里是“手工补齐”的，而不是统一由正式内核产出，增加了一致性风险。

---

## 3. 重构原则（Karpathy 风格）

### 原则 1：先收口，不先炫技

优先解决：
- 代码结构重复
- 口径漂移
- 职责不清

不优先解决：
- 新功能扩张
- 配置泛化
- 大规模抽象框架化

### 原则 2：正式主链必须唯一

任何候选研究都应该满足：

> 正式策略主链只定义一次；候选仅通过“局部变换”改动。

### 原则 3：展示层与计算层分离

通知文案可以持续优化，但不能再和核心计算深度耦合。

### 原则 4：每次重构都要可验证

重构必须至少验证：

- `python3 momentum_backtest/validate_strategy_outputs.py`
- 关键输出字段一致
- 定时任务运行路径不变
- 通知口径不回退

---

## 4. 目标结构（建议）

建议将当前结构逐步收口为以下分层：

### 4.1 `core/`：正式策略内核层

建议新增逻辑模块（不一定一步到位创建目录，但结构上按此收口）：

#### A. `strategy_signal.py`
负责：
- `build_signal_quality_score`
- `choose_signal_winner_with_margin`
- `build_top2_close_risk_flag`
- 信号赢家、原始动量、质量分、近似并列判断等

#### B. `strategy_weights.py`
负责：
- core-satellite / regime mix 主体
- target weight 主框架构建
- 统一生成 `signal / target_exposure / effective_momentum / target_weight_*`

#### C. `risk_overlays.py`
负责：
- weak market guard
- top2 close risk cap
- continuous overheat cap
- stress bond overlay
- 未来可能加入的：strong leader reduce core、defensive winner fill 等

这些都应成为：

```python
weights = apply_xxx(weights, context)
```

形式的局部变换器。

#### D. `strategy_outputs.py`
负责：
- `finalize_position_columns`
- `recompute_return_chain`
- `build_trades_from_weight_frame`
- `build_strategy_summary`
- `compute_signal_asset_momentum`

### 4.2 `runner/`：正式策略运行层

保留 `run_backtest.py` 作为正式入口，但它只负责：

1. 读参数
2. 读价格/代理
3. 调用正式内核
4. 写 core 输出

而不再承载大量底层逻辑实现。

### 4.3 `monitor/`：通知与快照层

把 `daily_monitor.py` 逻辑拆成三层：

#### A. snapshot 层
负责结构化数据：
- 当前确认持仓
- 今日目标组合
- 风险评分
- 同类60日统计
- 期望收益/百分位

#### B. render 层
负责把 snapshot 渲染成：
- 市场热度消息
- 交易与回撤消息

#### C. delivery 层
负责：
- 去重
- 落盘
- direct/webhook 发送

### 4.4 `research/`：候选研究层

所有候选研究，建议统一为：

```python
baseline_context = load_official_strategy_context(...)
candidate_result = run_candidate(baseline_context, candidate_transform)
```

候选脚本只定义：
- 这条候选到底改什么
- 如何验证它优于 baseline

而不是自己再复制完整正式主链。

---

## 5. 最值得优先收口的重复逻辑

### 第一优先级：正式 target weight 主链

建议首先收口：

- `compare_goal_optimizations.build_dynamic_core_target_weights`
- `compare_tail_risk_bond_overlay.build_base_target_weights`
- `run_default_strategy_with_params` 中正式主链构建部分

目标：

> 形成唯一一个“正式 target weight 构建器”。

例如：

```python
def build_official_target_weights(prices, proxy, params) -> OfficialTargetContext:
    ...
```

统一产出：
- target weights
- signal / winners
- raw / effective momentum
- top2 trigger
- weak market trigger
- overheat trigger
- stress trigger
- pre-cap / post-cap exposure

### 第二优先级：通知层渲染分离

优先把：
- `build_entry_risk_context`
- `compute_snapshot`
- `format_market_message`
- `format_trade_message`

之间的边界整理清楚。

目标：
- 以后加一个通知字段，不再碰底层计算链。

### 第三优先级：研究脚本统一 runner

把：
- top2 risk cap 候选
- top2 split 候选
- domestic 优点吸收候选
- slope hot 候选

统一成共享 runner 流程。

---

## 6. 推荐分阶段迁移方案

## Phase 1：收口正式权重主链（最小高价值）

### 目标
建立唯一正式主链构建器，减少复制逻辑。

### 动作
1. 新增正式主链构建函数（可先放在 `run_backtest.py` 内，后续再迁文件）
2. `run_default_strategy_with_params` 改为只调用这一个主链
3. `compare_tail_risk_bond_overlay.py` 改为复用它
4. 新候选研究脚本优先也走这条主链

### 验证
- `validate_strategy_outputs.py` 全绿
- 当前正式基线 summary 不漂移
- 通知关键字段不漂移

## Phase 2：收口通知 snapshot / render / delivery

### 目标
减少 `daily_monitor.py` 的展示耦合。

### 动作
1. 抽出 snapshot 数据模型构建
2. 抽出 market/trade message 渲染器
3. 保留 main 作为 orchestration

### 验证
- 新旧通知正文字段等价
- 定时任务输出不回退

## Phase 3：建立统一 candidate runner

### 目标
后续候选不再复制主链。

### 动作
1. 抽一个 `run_candidate_from_official_context(...)`
2. 把现有 2~3 个研究脚本迁移成示范版本
3. 逐步淘汰复制型候选脚本

### 验证
- 候选 summary 与原脚本结果一致
- 新脚本代码量显著减少

---

## 7. 不建议现在做的事

### 不建议 1：先删规则
当前正式逻辑虽然长，但仍可解释；问题主要在实现结构，不在规则数量本身。

### 不建议 2：先做高度配置化框架
目前更需要“收口”，不是“抽象成通用平台”。

### 不建议 3：一次性大拆目录
先在现有文件中收口唯一主链，再迁移目录会更稳。

---

## 8. 当前最值得继续优化的策略方向（基于最近研究结果）

结合近期回测，最值得继续研究的并不是再叠加新模型，而是：

### 第一优先级
- `reduce_core_when_leader_strong`
- 含义：主信号明显强时，减少 HS300 核心仓稀释

### 第二优先级
- `top2_close_riskcap_70 @ gap=0.5%`
- 已进入正式基线，可继续观察通知与实盘观感

### 第三优先级
- `weak_guard_to_defensive_winner_cap70`
- 可作为次优候选保留

### 不建议继续的线
- 激进斜率主导替换
- 过于激进的 slope_hot 压仓
- top2 split 分仓
- leader margin 胜者保留

---

## 9. 下一步实施建议

### 立即建议执行
1. 先做 **Phase 1**：收口正式 target weight 主链
2. 然后把 `compare_domestic_idea_adoptions.py` 改成复用统一主链的第一个示范脚本
3. 最后再继续围绕 `reduce_core_when_leader_strong` 做二轮精扫

### 验证清单
- [ ] `python3 momentum_backtest/validate_strategy_outputs.py`
- [ ] 正式基线回测 summary 不异常漂移
- [ ] monitor 通知仍能输出当前字段
- [ ] LaunchAgent 路径与稳定运行版不变

---

## 10. 一句话总结

当前最值得做的，不是继续往策略里堆新规则，而是：

> **把正式基线主链收口成唯一实现，把候选研究改成“在正式主链上做局部变换”。**

这样后续你无论继续优化：
- core 稀释问题
- 弱市资金承接
- top2 压仓
- 通知解释

都会更快、更稳、更不容易出错。


---

## 11. 已落地进展（Phase 1 进行中）

截至 2026-06-08，已完成一个最小但高价值的结构收口：

### 已新增
- `momentum_backtest/official_strategy_core.py`

### 已收口的正式主链函数
- `build_official_target_weights(prices, proxy, params)`

该函数统一承载了当前正式基线中的：
- core-satellite / regime mix 主体
- weak market guard
- top2 close risk cap
- continuous overheat cap

### 已切换为直接复用统一主链的调用方
- `momentum_backtest/run_backtest.py`
- `momentum_backtest/compare_defensive_persistence.py`

### 已保留兼容包装的旧入口
- `momentum_backtest/compare_tail_risk_bond_overlay.py::build_base_target_weights`

该旧入口现在只是一个薄包装，内部直接委托给：
- `official_strategy_core.build_official_target_weights(...)`

### 已完成验证
- `python3 momentum_backtest/validate_strategy_outputs.py`
- 结果：`validation_failed=0`
- 额外 AST 解析校验：
  - `momentum_backtest/official_strategy_core.py`
  - `momentum_backtest/compare_tail_risk_bond_overlay.py`
  - `momentum_backtest/compare_defensive_persistence.py`
  - `momentum_backtest/run_backtest.py`

### 当前状态判断
这一步已经证明：
- 正式主链可以开始从研究脚本中抽离
- 不需要一次性大拆目录
- 兼容包装策略可行

### 下一步建议
继续沿着相同方式，把：
- 其他复制正式主链的候选脚本
- 逐步改成“调用统一主链 + 局部变换”的模式


### 第二个已迁移的示范脚本

截至 2026-06-08，`momentum_backtest/compare_top2_riskcap_candidates.py` 已迁移为：

- 直接调用 `run_default_strategy_with_params(...)`
- 通过覆盖 `close_top2_gap / close_top2_risk_cap` 来评估候选
- 不再复制正式 target weight 主链实现

这意味着：

- 第二个复制型候选脚本已完成“统一主链 + 参数覆盖”迁移示范
- 后续同类候选（例如 top2 split / slope hot / domestic idea adoptions）可继续按同一模式迁移


### 第三个已迁移的示范脚本

截至 2026-06-08，`momentum_backtest/compare_top2_split_candidates.py` 已迁移为：

- 直接调用 `build_official_target_weights(...)` 获取正式主链输出
- 只额外补充一个轻量级 `top2 context` 辅助函数，用于定位第一/第二风险赢家
- 分仓逻辑只作为局部权重变换，不再复制正式主链

这说明：

- “统一主链 + 少量候选上下文补充 + 局部权重变换” 这套模式已可覆盖更复杂的候选脚本
- 后续 `slope_hot` / `domestic_idea` 继续迁移的阻力会进一步降低


### 已新增共享 candidate runner

截至 2026-06-08，已新增：

- `momentum_backtest/official_candidate_runner.py`

该模块负责候选研究的共享动作：

- 加载正式研究上下文 `load_official_research_context()`
- 从正式结果中抽取目标权重 `extract_target_weights()`
- 基于局部变换后的权重运行候选 `run_candidate_from_weights()`
- 统一汇总候选摘要 `summarize_candidate()`

### 已接入共享 candidate runner 的脚本

- `compare_domestic_idea_adoptions.py`
- `compare_slope_hot_riskcap_candidates.py`

### 已迁移为统一主链 + 参数覆盖的脚本

- `compare_top2_riskcap_candidates.py`

### 已迁移为统一主链 + 轻量上下文补充 + 局部变换的脚本

- `compare_top2_split_candidates.py`

这意味着当前重构已形成三种可复用迁移模板：

1. 统一主链 + 参数覆盖
2. 统一主链 + 正式上下文复用
3. 统一主链 + 轻量候选上下文 + 局部权重变换


### 通知层第一刀已落地

截至 2026-06-08，已新增：

- `momentum_backtest/monitor_render.py`

该模块已承接 `daily_monitor.py` 中的纯渲染/格式化职责，包括：

- 持仓/权重展示格式化
- 交易描述与确认原因渲染
- 同类60日与风险评分卡渲染
- 市场热度消息渲染
- 交易与回撤消息渲染

`daily_monitor.py` 现已改为直接复用该渲染模块，核心计算逻辑（快照构建、风险计算、状态落盘、发送编排）仍保留在原文件中。

这意味着通知层拆分已进入：

- 第一步：render 层抽离（已完成）
- 第二步：snapshot / delivery 分离（待继续）


### 通知层第二刀已落地（snapshot helper 抽离）

截至 2026-06-08，已新增：

- `momentum_backtest/monitor_snapshot.py`

该模块已承接 `daily_monitor.py` 中一批 snapshot 计算辅助函数，包括：

- 回测参考上下文恢复 `load_backtest_reference_context()`
- 建仓风险上下文 `build_entry_risk_context()`
- 资产价格上下文 `build_asset_price_context()`
- 市场时段分类 `classify_market_session()`
- 是否盘中快照 `should_include_realtime_snapshot()`
- 市场量能代理上下文 `build_market_volume_proxy_for_monitor()`
- forward regime 统计 `build_forward_regime_stats()`
- 盘中净值估计 `estimate_live_nav()`

目前 `daily_monitor.py` 仍保留：

- `compute_snapshot(...)` 主快照装配逻辑
- 状态落盘与发送编排逻辑

这意味着通知层重构已进入：

1. render 抽离（已完成）
2. snapshot helper 抽离（已完成）
3. compute_snapshot / delivery 继续拆分（待完成）


### 通知层第三刀已落地（compute_snapshot 装配 helper 抽离）

截至 2026-06-08，已新增：

- `momentum_backtest/monitor_snapshot_builders.py`

该模块已承接 `compute_snapshot()` 中两类装配逻辑：

1. 回测/风险上下文装配 `build_reference_and_risk_context()`
2. 持仓/交易上下文装配 `build_position_and_trade_context()`

目前 `compute_snapshot()` 已改为复用这些 helper，函数本体不再直接堆叠所有中间装配细节。

这意味着通知层拆分已进入：

1. render 抽离（已完成）
2. snapshot helper 抽离（已完成）
3. compute_snapshot 装配逻辑抽离（已完成第一步）
4. delivery 层拆分（待完成）


### 通知层第四刀已落地（delivery/state 抽离）

截至 2026-06-08，已新增：

- `momentum_backtest/monitor_delivery.py`

该模块已承接 `daily_monitor.py` 中的 delivery / state 管理职责，包括：

- state 读取与落盘
- 去重判定
- delivery_progress 归一化与恢复
- channel plan 构建
- direct/webhook 发送
- message bundle 分发

这意味着通知层当前已经形成：

1. `monitor_snapshot.py`：snapshot helper
2. `monitor_snapshot_builders.py`：snapshot 装配 helper
3. `monitor_render.py`：render 层
4. `monitor_delivery.py`：delivery/state 层
5. `daily_monitor.py`：主编排入口

至此，通知层的三层拆分（snapshot / render / delivery）已全部有独立模块承接。


### 通知层当前结构已基本成型

截至 2026-06-08，通知链路已形成以下职责划分：

- `monitor_snapshot.py`：snapshot helper
- `monitor_snapshot_builders.py`：snapshot 装配 helper 与 `build_signal_snapshot()`
- `monitor_render.py`：render 层
- `monitor_delivery.py`：delivery / state 层
- `daily_monitor.py`：薄包装入口与 orchestration

其中：

- `compute_snapshot()` 已改为仅作为薄包装入口，直接委托给 `build_signal_snapshot()`
- `daily_monitor.py` 的大块展示、状态、发送逻辑已基本迁出

这意味着通知层目标结构已基本实现，后续主要工作将转向：

1. 接口收紧与清理
2. 旧 helper 进一步去重
3. 稳定运行版同步与运行回归检查


### 研究脚本进一步收口进展（2026-06-08）

截至 2026-06-08，以下研究脚本已纳入更统一的研究结构：

- `compare_domestic_idea_adoptions.py`：复用 shared candidate runner 与正式上下文
- `compare_slope_hot_riskcap_candidates.py`：复用 shared candidate runner 与正式上下文
- `compare_top2_riskcap_candidates.py`：统一主链 + 参数覆盖
- `compare_top2_split_candidates.py`：统一主链 + 轻量 top2 context + 局部权重变换
- `compare_slope_combo_replacements.py`：复用 shared research context，减少独立加载/汇总逻辑

这意味着研究层已经基本形成两条稳定模式：

1. 参数覆盖型候选
2. 局部权重变换型候选

后续若新增候选，默认应优先复用这两类模式，而不是再复制正式主链实现。
