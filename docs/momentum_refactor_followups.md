# 动量策略重构后续清单

更新时间：2026-06-08

## 1. 已完成的核心收口

### 正式主链
- `official_strategy_core.py`
- 正式基线已切到统一主链

### 研究层
已形成三类稳定模式：
1. 参数覆盖型候选
2. 局部权重变换型候选
3. shared runner 驱动型候选

### 通知层
已形成：
- `monitor_snapshot.py`
- `monitor_snapshot_builders.py`
- `monitor_render.py`
- `monitor_delivery.py`
- `monitor_pipeline.py`
- `daily_monitor.py`（orchestration）

### 运行态
- 已同步到稳定运行版 `/private/tmp/wy_test_runtime`
- LaunchAgent 重载成功
- runtime 验证通过

---

## 2. 剩余建议继续收口的脚本

### 优先级 A：仍值得继续清理

#### `momentum_backtest/compare_domestic_idea_adoptions.py`
当前状态：
- 已复用正式上下文
- 仍有一部分候选局部逻辑可继续压缩

建议：
- 把局部变换逻辑进一步抽成更小 helper
- 使其更像“纯候选定义文件”

#### `momentum_backtest/compare_slope_combo_replacements.py`
当前状态：
- 已接入 shared context
- 仍保留较多独立 variant-specific 逻辑

建议：
- 若后续不再深挖斜率替换，可保持现状
- 若后续继续研究，应进一步接入统一候选 runner 模式

### 优先级 B：按需处理

#### `momentum_backtest/archive/experiments/*`
当前状态：
- 大量历史实验脚本仍沿用旧模式

建议：
- 如果未来仍频繁使用，再择优迁移
- 如果只是历史留档，不建议现在为它们投入大量重构成本

---

## 3. 后续优化建议

### 方向 1：接口收紧
- 清理 `daily_monitor.py` 中剩余 bridge 风格薄包装
- 统一模块 import 风格
- 进一步压缩 orchestration 文件中的中间变量

### 方向 2：候选研究接口标准化
建议以后新增候选时默认遵守：

1. 优先参数覆盖
2. 其次局部权重变换
3. 最后才允许独立构造主链

### 方向 3：维护入口文档保持同步
每次结构性调整后，优先同步：
- `docs/momentum_refactor_design.md`
- `docs/momentum_module_map.md`
- 本清单

---

## 4. 当前建议的工作重点

如果下一阶段继续做“结构优化”，建议优先级：

1. 收紧 `compare_domestic_idea_adoptions.py`
2. 视研究需要决定是否继续清理 `compare_slope_combo_replacements.py`
3. 不急着迁 archive，先保证正式链与常用研究链可维护

如果下一阶段回到“策略优化”，建议优先级：

1. 围绕 `reduce_core_when_leader_strong` 继续做候选精扫
2. 保持所有新候选走统一 runner / 主链模式
3. 任何上线前改动都继续走：
   - 回测对比
   - 你审核
   - 同步稳定运行版

---

## 5. 最终维护原则

1. 正式主链只保留一个实现
2. 通知层尽量只改 render，不动 snapshot/delivery
3. 候选研究默认不复制主链
4. 稳定运行版始终通过 `sync_runtime_and_reload.sh` 同步
5. 所有结构性改动必须跑：
   - `python3 momentum_backtest/validate_strategy_outputs.py`



## 6. 当前最优结构化候选（截至 2026-06-08）

基于 `reduce_core_when_leader_strong` 的多轮精扫（见 `momentum_backtest/output/research/reduce_core_fine_scan/grid_summary_round3.csv`），当前表现最强的一组为：

- `rc_r95_g003_m10`
  - leader_gap = 0.3%
  - mom_threshold = 10%
  - release_ratio = 95%
  - trigger_days = 207
  - annualized_return_diff_vs_base = +0.8682%
  - sharpe_rf0_diff_vs_base = +0.0181
  - max_drawdown_diff_vs_base = +0.0038%（几乎持平）
  - max_drawdown_integral_diff_vs_base = +1.9997（略差）

含义：
- 当主信号相对第二名明显领先，且自身动量不低时，强主信号阶段更激进地释放 HS300 核心仓，让赢家拿到更多权重。

当前判断：
- 这是当前最值得继续研究的结构化候选方向。
- 但它的全历史回撤积分略有变差，因此不建议直接当作正式基线替换，仍应继续做：
  1. 最近阶段持仓/通知观感评估；
  2. 候选版解释优化；
  3. 若准备上线，先走候选审批流程。


## 7. `reduce_core_when_leader_strong` 第三轮精扫结果（2026-06-08）

结果文件：
- `momentum_backtest/output/research/reduce_core_fine_scan/grid_summary_round3.csv`

当前最优组：
- `rc_r95_g003_m10`
  - leader_gap = 0.3%
  - mom_threshold = 10%
  - release_ratio = 95%
  - trigger_days = 207
  - annualized_return_diff_vs_base = +0.8682%
  - sharpe_rf0_diff_vs_base = +0.0181
  - max_drawdown_diff_vs_base = +0.0038%（几乎持平）
  - max_drawdown_integral_diff_vs_base = +1.9997（略差）

判断：
- 收益与 Sharpe 提升进一步扩大；
- 最大回撤没有显著恶化；
- 但回撤积分仍然略差，说明这条线仍应作为“高潜力候选”，而非直接替代正式基线。

下一步若继续优化，建议：
1. 补做最近阶段持仓/通知观感对比；
2. 评估是否需要限制 release_ratio 的上限，换取更低回撤积分；
3. 若准备上线，必须继续走候选审批流程，而不是直接替换正式基线。

## 8. `rc_r95_g003_m10` 最近持仓与通知观感

相关产物：
- `momentum_backtest/output/research/reduce_core_fine_scan/latest_path_compare.csv`
- `momentum_backtest/output/research/reduce_core_fine_scan/candidate_review_rc_r95_g003_m10/recent_close_messages.json`

### 关键观察

1. 候选版不会频繁改主信号方向，主要影响的是：
   - 在强主信号阶段更早、更充分释放 HS300 核心仓
   - 因而最近路径中最明显的差异体现在仓位节奏，而不是标的切换

2. 最近样本里，候选版最明显的变化点出现在：
   - 2026-05-27 ~ 2026-06-01 附近
   - 候选版在正式版仍保留更多核心仓时，更快回到主信号高仓位

3. 通知观感上：
   - 文案主体仍然容易理解，没有出现难以解释的多标的复杂描述
   - 但若未来准备上线，建议补一条更明确的说明：
     - “主信号显著领先时，减少 HS300 核心仓稀释”

### 当前判断

- 该候选在最近持仓与通知观感上没有出现明显不可接受的问题；
- 若后续继续推进，可进入下一阶段候选审阅，但仍不建议直接替换正式基线；
- 若要进一步提高可解释性，建议在候选通知版中补充“强主信号释放核心仓”的原因说明。


### 候选审阅说明文档

- `momentum_backtest/output/research/reduce_core_fine_scan/candidate_review_rc_r95_g003_m10/README.md`

该文档已汇总：
- 回测摘要
- 相对正式基线差异
- 最近持仓路径对照
- 最近通知模拟
- 是否值得进入下一阶段候选审阅的判断

### 候选审阅生成脚本

- `momentum_backtest/generate_candidate_review.py`

用途：
- 基于候选 backtest/trades 结果，自动生成：
  - 最近持仓路径对照
  - official vs candidate 通知对照
  - 候选审阅材料输出目录

建议：
- 后续若继续推进新的候选线，优先复用该脚本生成审阅材料，避免再次手工拼装通知对照。

## 9. `reduce_core_when_leader_strong` 候选分层建议

基于 `momentum_backtest/output/research/reduce_core_fine_scan/grid_summary_round3.csv` 的多目标筛选：

### 激进最优
- `rc_r95_g003_m10`
- 特征：
  - 收益提升最大
  - Sharpe 提升最大
  - 回撤积分代价也相对更高

适用定位：
- 若后续希望优先追求年化与 Sharpe 提升，可优先继续审阅这一组。

### 平衡最优
- `rc_r80_g003_m10`（及同档位近邻组）
- 特征：
  - 收益与 Sharpe 仍明显优于基线
  - 回撤积分代价比 90%/95% 释放版本更轻

适用定位：
- 若后续更看重“收益提升 + 风险代价可接受”的平衡性，可优先把 80% 释放档作为备选。

### 当前建议
- 继续保留 `rc_r95_g003_m10` 作为**高潜力主候选**；
- 同时把 `rc_r80_g003_m10` 记为**平衡备选**，方便后续做正式候选审阅时两组并排比较。

### 双候选并排审阅材料

- `momentum_backtest/output/research/reduce_core_fine_scan/pair_review_rc95_vs_rc80/summary_compare.csv`
- `momentum_backtest/output/research/reduce_core_fine_scan/pair_review_rc95_vs_rc80/latest_path_compare.csv`
- `momentum_backtest/output/research/reduce_core_fine_scan/pair_review_rc95_vs_rc80/message_compare.md`
- `momentum_backtest/output/research/reduce_core_fine_scan/pair_review_rc95_vs_rc80/message_compare.json`

用途：
- 并排比较当前主候选 `rc_r95_g003_m10` 与平衡备选 `rc_r80_g003_m10` 的：
  - 回测摘要
  - 最近持仓路径
  - 最近通知观感

- `momentum_backtest/output/research/reduce_core_fine_scan/pair_review_rc95_vs_rc80/README.md`

用途：
- 为主候选 `rc_r95_g003_m10` 与平衡备选 `rc_r80_g003_m10` 提供可直接阅读的并排审阅说明，便于后续做候选升级判断。

### 候选升级审阅说明与模板

- `docs/momentum_candidate_upgrade_review_rc95_vs_rc80.md`
- `docs/momentum_candidate_upgrade_review_template.md`

用途：
- 第一份用于当前主候选 / 平衡备选的正式升级审阅说明；
- 第二份用于后续任意候选线进入升级审阅时复用。

### 候选决策矩阵

- `momentum_backtest/generate_candidate_decision_matrix.py`
- `momentum_backtest/output/research/reduce_core_fine_scan/pair_review_rc95_vs_rc80/decision_matrix.md`

用途：
- 基于候选对比摘要，自动生成“主候选 / 平衡候选 / 不优先”这类决策分层材料。

### 候选升级建议书

- `docs/momentum_candidate_upgrade_recommendation.md`

用途：
- 给当前主候选 / 平衡备选形成明确推荐结论，便于后续是否进入正式候选升级池的判断。
