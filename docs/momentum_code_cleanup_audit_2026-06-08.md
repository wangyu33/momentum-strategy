# 动量策略无效代码清理审计（2026-06-08）

## 1. 清理目标

围绕 `momentum_backtest/` 识别并处理无效代码：

- 根目录中不再适合作为活跃入口的一次性脚本 -> 归档
- 已被主链完全替代的一次性工具脚本 -> 删除
- 根目录与 archive 中的死 import / 死残留 -> 删除

---

## 2. 当前保留在根目录的 Python 文件

共 34 个：

- `analyze_contribution.py`：活跃/支撑保留
- `analyze_drawdowns.py`：活跃/支撑保留
- `compare_candidate_pool_additions.py`：活跃/支撑保留
- `compare_china_internet_guards.py`：活跃/支撑保留
- `compare_current_best_fine_tune.py`：活跃/支撑保留
- `compare_current_best_pool_additions.py`：活跃/支撑保留
- `compare_defensive_persistence.py`：活跃/支撑保留
- `compare_domestic_idea_adoptions.py`：活跃/支撑保留
- `compare_goal_optimizations.py`：活跃/支撑保留
- `compare_hs300_regime_fixes.py`：活跃/支撑保留（2026-06-08 compare/analyze 复核已确认）
- `compare_market_proxy_variants.py`：活跃/支撑保留
- `compare_reduce_core_fine_scan.py`：活跃/支撑保留
- `compare_resource_guards.py`：活跃/支撑保留
- `compare_slope_combo_replacements.py`：活跃/支撑保留
- `compare_slope_hot_riskcap_candidates.py`：活跃/支撑保留
- `compare_strategy_refinements.py`：活跃/支撑保留
- `compare_tail_risk_bond_overlay.py`：活跃/支撑保留
- `compare_top2_riskcap_candidates.py`：活跃/支撑保留
- `compare_top2_split_candidates.py`：活跃/支撑保留
- `daily_monitor.py`：活跃/支撑保留
- `generate_candidate_decision_matrix.py`：活跃/支撑保留
- `generate_candidate_review.py`：活跃/支撑保留
- `generate_reduce_core_pair_review.py`：活跃/支撑保留
- `monitor_delivery.py`：活跃/支撑保留
- `monitor_pipeline.py`：活跃/支撑保留
- `monitor_render.py`：活跃/支撑保留
- `monitor_snapshot.py`：活跃/支撑保留
- `monitor_snapshot_builders.py`：活跃/支撑保留
- `official_candidate_runner.py`：活跃/支撑保留
- `official_strategy_core.py`：活跃/支撑保留
- `run_backtest.py`：活跃/支撑保留
- `runtime_env.py`：活跃/支撑保留
- `search_utils.py`：活跃/支撑保留
- `validate_strategy_outputs.py`：活跃/支撑保留

备注：当前根目录未发现“明显的一次性历史脚本”残留，剩余文件均属于正式入口、通知分层、活跃研究入口、分析脚本或共享支撑模块。

---

## 3. 已归档/删除的代表性无效代码

### 3.1 已从根目录归档到 `archive/experiments/`

- `compare_opportunistic_defensive_cap.py`
- `compare_zijin_candidate.py`
- `export_leader_margin_report.py`
- `annotate_drawdown_integral_definition.py`（后续进一步删除）

### 3.2 已直接删除的冗余历史工具

- `archive/experiments/plot_drawdown.py`
  - 已被 `run_backtest.py` 自动输出的分析图替代。
- `archive/experiments/annotate_drawdown_integral_definition.py`
  - 一次性口径迁移工具，当前字段已落盘完成。

---

## 4. archive 当前状态

当前 `archive/experiments/` 剩余 41 个 Python 脚本，用于保留历史实验复盘能力。

- 这批文件不再作为正式入口
- 默认输出应落到 `output/research/archive_flat/`
- 本轮未继续批量删除其余历史实验脚本，因为仍可能保留复盘价值

---

## 5. 静态清理证据

本轮已对 `momentum_backtest/**/*.py` 做 AST 扫描，结果：

- **remaining unused import count = 0**

说明当前目录下已无明显死 import / 死残留。

---

## 6. 回归验证证据

已运行：

```bash
python3 momentum_backtest/validate_strategy_outputs.py
```

结果：

- `validation_failed=0`

说明：

- 正式回测主链正常
- 通知/快照链路正常
- 分析输出与兼容接口正常

---

## 7. 当前结论

截至 2026-06-08：

1. 明显无效的一次性根目录脚本已归档
2. 已被当前主链完全替代的历史工具脚本已删除
3. 根目录与 archive 中的死 import 已清零
4. 当前剩余 archive 脚本默认视为“历史实验资产”，不是“无效代码”

后续若继续清理，应进入“历史实验价值审计”阶段，而不是继续做机械性死代码删除。
