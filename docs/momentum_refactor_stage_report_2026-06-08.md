# 动量策略结构重构阶段报告（2026-06-08）

## 1. 目标

本阶段围绕以下目标推进：

- 设计结构重构设计稿
- 基于设计稿收口正式主链、候选研究层与通知层
- 在新结构基础上持续优化动量策略候选
- 为候选升级审阅提供可复用的材料生成能力

---

## 2. 已完成的结构重构成果

### 2.1 正式主链
- `momentum_backtest/official_strategy_core.py`
  - 收口正式 target weight 主链
  - 正式基线已切换为复用该主链

### 2.2 候选研究共享层
- `momentum_backtest/official_candidate_runner.py`
  - 统一加载正式研究上下文
  - 统一运行基于权重变换的候选
  - 统一汇总 candidate summary

### 2.3 通知层分层
- `monitor_snapshot.py`
- `monitor_snapshot_builders.py`
- `monitor_render.py`
- `monitor_delivery.py`
- `monitor_pipeline.py`
- `daily_monitor.py`（已基本收敛为 orchestration 入口）

### 2.4 候选审阅辅助脚本
- `generate_candidate_review.py`
- `generate_reduce_core_pair_review.py`

---

## 3. 已迁移到统一结构的研究脚本

### 参数覆盖型
- `compare_top2_riskcap_candidates.py`

### 局部权重变换型
- `compare_top2_split_candidates.py`
- `compare_domestic_idea_adoptions.py`
- `compare_slope_hot_riskcap_candidates.py`

### 共享研究上下文型
- `compare_slope_combo_replacements.py`
- `compare_reduce_core_fine_scan.py`

---

## 4. 运行态与验证证据

### 4.1 本地验证
已多轮执行：
- `python3 momentum_backtest/validate_strategy_outputs.py`

当前状态：
- `validation_failed=0`

### 4.2 运行版验证
已执行：
- `./scripts/sync_runtime_and_reload.sh`

当前状态：
- 稳定运行版路径：`/private/tmp/wy_test_runtime`
- LaunchAgents 仍正常指向稳定运行版
- runtime 校验通过

---

## 5. 基于新结构的策略优化成果

### 5.1 当前最有潜力的候选线
- `reduce_core_when_leader_strong`

### 5.2 已完成的精扫
- `momentum_backtest/compare_reduce_core_fine_scan.py`
- 输出：
  - `grid_summary_round3.csv`
  - `latest_path_compare.csv`

### 5.3 当前主候选与平衡备选
- 主候选：`rc_r95_g003_m10`
- 平衡备选：`rc_r80_g003_m10`

### 5.4 已完成的审阅材料
- `candidate_review_rc_r95_g003_m10/README.md`
- `pair_review_rc95_vs_rc80/README.md`
- `official_vs_candidate_messages.*`
- `message_compare.*`

---

## 6. 阶段判断

本阶段已经形成完整闭环：

1. 设计稿完成并持续同步
2. 结构重构落地并验证通过
3. 候选研究切换到统一结构
4. 新结构上继续做策略优化
5. 候选审阅材料可自动生成

这说明当前工作已从“重构基础设施搭建”进入“可持续优化与候选升级审阅”阶段。

---

## 7. 后续建议

### 方向 A：继续候选审阅推进
- 围绕 `rc_r95_g003_m10` 与 `rc_r80_g003_m10` 进入更正式候选审阅
- 重点观察：
  - 收益 / Sharpe 提升
  - 回撤积分代价
  - 最近通知与持仓观感

### 方向 B：继续做收尾清理
- 继续收口少量 remaining research 脚本
- 清理少量 orchestration 中的 bridge 逻辑

### 当前建议
优先做方向 A，让结构优化真正转化为候选升级决策能力。
