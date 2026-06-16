# Historical Experiments

这个目录存放已经退出日常使用路径的历史实验脚本。

原则：

- 原则上保留可复盘的历史实验；但对已被当前主链完全替代的一次性迁移/绘图工具脚本，可直接删除。
- 不作为正式入口，不建议直接拿这些结果覆盖 `output/core/`。
- 这批脚本已经补了根目录引导，可以直接运行。
- 根目录当前只保留仍被正式入口、巡检脚本、活跃研究脚本直接依赖的文件；不再被依赖的一次性实验优先继续归档到这里。
- 这批旧实验在历史结构下写到 `output/research/` 根目录的散落结果，已统一搬到 `output/research/archive_flat/`。
- 新增整理后的旧实验，默认应写到 `output/research/archive_flat/<script_name>/` 这类子目录，避免再次在 `output/research/` 根目录生成散落文件。

已删除的冗余历史工具：

- `plot_drawdown.py`
  - 已被 `run_backtest.py` 自动输出的 `output/analysis/historical_drawdown.png` / `nav_and_drawdown.png` 替代。
- `annotate_drawdown_integral_definition.py`
  - 属于一次性口径迁移工具，当前字段已完成落盘，不再保留。

示例：

```bash
python3 momentum_backtest/archive/experiments/compare_lookback_range.py --years 10
python3 momentum_backtest/archive/experiments/tune_threshold_dual_momentum.py --years 10
```

当前已归档：

- `compare_opportunistic_defensive_cap.py`
- `compare_zijin_candidate.py`
- `export_leader_margin_report.py`
- `compare_china_internet_germany_variants.py`
- `compare_cash_overlay_candidates.py`
- `compare_amount_proxy_blends.py`
- `compare_amount_relief_by_etf_strength.py`
- `compare_dividend_defensive_additions.py`
- `compare_defensive_reentry_probe.py`
- `compare_defensive_bucket_blends.py`
- `compare_defensive_trigger_refinements.py`
- `compare_dual_momentum_variants.py`
- `compare_common_sense_enhancements.py`
- `compare_energy_income_candidates.py`
- `compare_effective_momentum_definitions.py`
- `compare_joint_treasury_trigger_search.py`
- `compare_lookback_range.py`
- `compare_more_candidates.py`
- `compare_multihorizon_combo.py`
- `compare_momentum_quality_variants.py`
- `compare_nonferrous_guards.py`
- `compare_overheat_exposure_cap.py`
- `compare_regime_mix_pool_variations.py`
- `compare_resource_baseline_with_china_internet.py`
- `compare_resource_energy_common_sense.py`
- `compare_simple_bond_overlay.py`
- `run_china_internet_cap70_backtest.py`
- `run_resource_abs08_backtest.py`
- `compare_structural_optimizations.py`
- `compare_stressbond_dual_objective_refine.py`
- `compare_stressbond_edge_refine.py`
- `compare_stressbond_nearmiss_repair.py`
- `compare_stressbond_persistence_current.py`
- `compare_stressbond_smooth_guard.py`
- `compare_stressbond_trigger_refine_current.py`
- `compare_stressbond_trigger_refine_rc182_local.py`
- `compare_stressbond_trigger_refine_rc184.py`
- `compare_stressbond_vm_refine.py`
- `compare_two_tier_defense.py`
- `compare_two_tier_defense_compact.py`
- `tune_threshold_dual_momentum.py`
