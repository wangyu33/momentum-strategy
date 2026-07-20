# Historical Experiments

这个目录存放已经退出日常使用路径的历史实验脚本。

原则：

- 原则上保留可复盘的历史实验；但对已被当前主链完全替代的一次性迁移/绘图工具脚本，可直接删除。
- 不作为正式入口，不建议直接拿这些结果覆盖 `output/core/`。
- 这批脚本已经补了根目录引导，可以直接运行。
- 根目录当前只保留仍被正式入口、巡检脚本、活跃研究脚本直接依赖的文件；不再被依赖的一次性实验优先继续归档到这里。
- 这批旧实验在历史结构下写到 `output/research/` 根目录的散落结果，已统一搬到 `output/research/archive_flat/`。
- 新增整理后的旧实验，默认应写到 `output/research/archive_flat/<script_name>/` 这类子目录，避免再次在 `output/research/` 根目录生成散落文件。
- 如果 archive 脚本只是写到 `archive_flat/<script_name>/` 这类标准目录，优先复用 `archive_data_loaders.build_archive_flat_output_dir()`，不要各脚本继续硬编码整段 `Path("momentum_backtest/output/research/archive_flat/...")`。
- 如果 archive 脚本还需要配套的 `best.json` / `notify_state.json` 路径，优先复用 `archive_data_loaders.build_archive_flat_strategy_paths()`，不要各脚本继续重复拼同一套状态文件路径。
- `archive_data_loaders.load_workspace_price_cache()` 现在也统一经 `ARCHIVE_FLAT_OUTPUT_DIR` 搜索 `archive_flat/*/prices.csv`；后续不要再在 helper 里单独硬编码同一段 archive_flat 根路径。
- `compare_simple_bond_overlay.py` 这条历史上游链在本轮没有 `valid_improvements` 时，也会把最佳可恢复 overlay 上下文写进 `best.json` 的 `strict_improvements`，避免下游 `compare_cash_overlay_candidates.py` / `compare_stressbond_nearmiss_repair.py` 因上游 payload 丢字段而断链。
- 少数“第二阶段”历史搜索脚本会读取上游实验目录下的 `notify_state.json` / `best.json` 作为继续搜参的基线；这类依赖现在应优先通过 `momentum_backtest/search_utils.py` 里的公共 helper 读取，而不是再去 import 上游脚本内部函数。
- 如果多份历史脚本只是共用同一套覆盖层策略实现，优先下沉到共享 helper（例如当前简单切债链使用 `overlay_strategy_helpers.py`），不要继续让 compare 脚本彼此 import 策略函数。
- 如果 archive 脚本只是复用 `run_backtest.py` 里已经稳定的常量、抓数函数或通用回放 helper，优先经 `archive_strategy_common.py` 转发，不要继续直接 import 正式入口模块。
- 阈值双动量历史链当前共用 `dual_momentum_helpers.py`；参数搜索脚本不要再直接 import `compare_dual_momentum_variants.py` 这类研究入口。
- 同一条阈值双动量历史链如果只是复用 `output/core/selected_etfs.csv` 和 `prices.csv`，也优先经 `dual_momentum_helpers.load_cached_data()` 读取，避免各脚本重复拼缓存路径。
- `dual_momentum_helpers.load_cached_data()` 现在底层统一走 `archive_data_loaders.load_core_cached_data()`；后续不要再在双动量链 helper 里直接手读 `output/core/*.csv`。
- 需要“给定 signal / target_exposure 后再统一回放净值、换手和交易明细”的历史脚本，优先复用 `dual_momentum_helpers.run_exposure_strategy()`，不要继续在各脚本里复制同一段持仓回放逻辑。
- 如果 archive 脚本本质上还是 exposure 类回放链，只是额外补几列占比/区间字段，优先复用 `dual_momentum_helpers.summarize_variant_result()`，不要再各自手写 `start_date` / `end_date` 包装。
- 只是在 archive 脚本里复用 `output/core` 已有价格面板并按候选池裁列时，优先走 `archive_data_loaders.load_recent_selected_prices()` 或 `load_core_selected_and_prices()`，不要再手写 `CORE_OUTPUT_DIR / "prices.csv"` 读取。
- 如果 archive 脚本只是从 `output/core` 读取给定起始日后的候选池价格，优先复用 `archive_data_loaders.load_selected_prices_from_start()`，不要再各自重复读 `prices.csv`。
- 如果 archive 脚本只是合并多份历史价格缓存（如旧 research 输出 + core 价格），优先复用 `archive_data_loaders.load_cached_price_panel()`，不要再在脚本里复制同一段 CSV 合并逻辑。
- 如果 archive 脚本只是想复用 workspace 里常见的 `core + archive_flat + 旧 research` 价格缓存，优先走 `archive_data_loaders.load_workspace_price_cache()`，不要每支脚本各自硬编码一串缓存路径。
- 如果 archive 脚本只是复用 `resource_abs08` / `china_internet_cap70` / `core` 这类 workspace 既有价格缓存，也优先直接走 `load_workspace_selected_prices()` 的默认搜索路径，不要再在脚本里重复手传同一组 `extra_paths`。
- 如果 archive 脚本需要“基础池 + 债券/现金候选”这类覆盖层价格准备，并且要统一处理缺历史 warning、可用候选过滤和必需基线资产校验，优先复用 `archive_data_loaders.load_candidate_prices_with_checks()`。
- 如果 archive 脚本要求“新增候选必须全部具备历史，否则直接失败”，优先复用 `archive_data_loaders.load_required_candidate_prices()`，不要继续在各脚本里重复拼 `missing_codes / fetch_error / rerun with network access` 这套报错模板。
- 如果 archive 脚本只是把同一个 `market_proxy + prices` 重新组装成命名代理，优先复用 `archive_data_loaders.build_named_market_proxy()` / `load_named_market_proxy()`，不要再各自展开 `build_proxy_catalog(...)`。
- 覆盖层 / 防守链历史脚本如果只是复用“market-proxy 最优上下文 + overlay 上游上下文 + treasury_row + params + 去掉 drop_codes 后的基础池”这套装配，优先走 `context_loaders.load_market_overlay_runtime()`，不要继续在各脚本里重复拼这段上游装配。
- 覆盖层 / 防守链历史脚本如果只是把 baseline 结果汇总成 summary，并写入 `rows/nav_compare/descriptions`，优先复用 `context_loaders.append_overlay_baseline()`；如果只是补 `treasury_code / risk_cap / ratio_cut / breadth_cut / mode / drawdown_cut` 这类 overlay 摘要字段，优先复用 `context_loaders.build_overlay_summary_fields()`，不要继续在各脚本里重复手填这段字段映射。
- 防守覆盖层链里共用的候选资产常量，统一放在 `overlay_candidate_catalog.py`，避免下游脚本再从其它 compare 脚本里借常量。
- 候选池加减替换类的通用池变更逻辑，历史脚本应优先复用 `pool_change_helpers.py`，不要再直接 import 活跃研究入口里的 `apply_pool_change`。
- `pool_change_helpers.py` 当前只是 archive 侧薄包装，底层统一复用根目录 `momentum_backtest/pool_change_common.py`；后续如果历史脚本还需要“默认正式策略评估外壳”，也优先继续下沉到这层公共模块，不要在 archive 和当前研究链各维护一份。
- 候选池实验共用的 ETF 常量、基础池代码和基础阈值双动量 helper，统一下沉到 `momentum_backtest/candidate_pool_common.py`，archive 脚本不要再直接依赖 `compare_candidate_pool_additions.py`。
- 像 `159930`、`510410`、`159985`、`508000`、`513300`、`513660` 这类已在多支候选池历史脚本复用的 ETF 常量，也优先继续补进 `candidate_pool_common.py`；只有同代码需要保留多套历史展示语义时，才单独补别名常量，不要在脚本里重复手写字典。
- 候选池对比类历史脚本如果只是复用同一套“跑候选组合 / 生成 summary 与 nav_compare / 画一张总览图”外壳，优先复用 `candidate_compare_helpers.py`，不要继续在各脚本里复制同样的循环、摘要包装和 diff 落盘逻辑；如果只是标准入口收尾，也优先走 `run_candidate_compare_entrypoint()` / `save_and_print_candidate_compare_artifacts()`。
- `candidate_compare_helpers.py` 现在默认优先复用本地 `output/core/prices.csv` / `output/research/archive_flat/*/prices.csv` 缓存，再按缺失标的补抓；如果当前环境无网且基线候选池本身也缺历史，脚本会明确报出缺失代码并提示用 `--refresh` 或先准备本地价格缓存，而不是只抛笼统的抓数失败。
- 多变体参数扫描类历史脚本如果只是复用“基础摘要 / baseline diff / summary 与 nav_compare 落盘 / 画总览图”外壳，优先复用 `variant_compare_helpers.py`；如果只是把单个候选的 `summary + strategy + extra_fields + nav_compare + description` 注册到结果集中，优先复用 `variant_compare_helpers.append_variant_result()`，不要继续在各脚本里重复拼 `rows.append(...)` / `nav_compare[...] = result["nav"]` / `descriptions[...] = ...` 这层外壳。
- 如果 archive 脚本只是往 `compare_df` / `nav_compare` 补一列沪深300基准净值，也优先复用 `variant_compare_helpers.add_hs300_benchmark()`，不要再各自手写 `build_benchmark_nav(...).reindex(...)`。
- 如果 archive 脚本只是想初始化一个按 `prices.index` 对齐的净值对比面板，并顺手补上沪深300基准列，优先复用 `variant_compare_helpers.build_compare_frame()`，不要继续重复写 `pd.DataFrame(index=prices.index)` + `add_hs300_benchmark(...)`。
- 如果 archive 脚本是在循环里等首个有效结果出来后才懒初始化 `compare_df`，优先复用 `variant_compare_helpers.ensure_compare_frame()`，不要继续散落 `if compare_df is None:` + `pd.DataFrame(...)` + `add_hs300_benchmark(...)`。
- 如果 archive 脚本只是想导出“策略 vs 沪深300”的年度收益表，也优先复用 `variant_compare_helpers.build_yearly_returns_df()`，不要再各自直接拼 `build_yearly_return_rows(...)`。
- 如果 archive 脚本只是先做 `finalize_baseline_diff_summary(...)`，再立刻按固定列 `sort_values(...)` 或 `reset_index(drop=True)`，优先直接用这个 helper 的 `sort_by=` / `ascending=` / `reset_index=` 参数，不要再把同一段排序尾巴散落在脚本末尾。
- 如果 archive 脚本是先给 `summary_df` 补一列 `is_soft_improvement` / `is_integral_valid_change` 这类临时标记，再按固定列排序，优先复用 `variant_compare_helpers.build_summary_frame(..., sort_by=..., ascending=...)`，不要继续直接写 `summary_df.sort_values(...)`。
- 如果 archive 脚本只是想从 `summary_df` 里取 baseline 那一行，或单策略摘要只想取第一行打印，优先复用 `variant_compare_helpers.get_summary_row()` / `get_summary_mapping()`，不要继续手写 `.loc[...] .iloc[0]` / `.iloc[0].to_dict()`。
- stress-bond、覆盖层以及其它参数搜索类 archive 脚本，只要核心输出仍是 `summary.csv` + `nav_compare.csv`，优先直接走 `variant_compare_helpers.save_variant_compare_outputs()`，不要再手写两次 CSV 落盘。
- 如果 archive 脚本只是顺手导出某个变体的 `*_nav.csv`、`*_trades.csv` 或 `*_nav.csv + *_trades.csv`，优先复用 `variant_compare_helpers.save_named_nav_output()` / `save_named_trades_output()` / `save_named_nav_and_trades()`；如果是循环里批量导出多组变体，也优先走 `save_named_nav_outputs()` / `save_named_nav_and_trades_outputs()`，不要在脚本里重复写同样的单文件导出样板。
- 如果 archive 脚本只是把 `print(summary.to_csv(...))`、`print(summary.to_string(...))` 和 `outputs saved to ...` 这层重复输出包一遍，也优先走 `variant_compare_helpers.print_summary_csv()` / `print_saved_summary()`；若脚本本来就保留了精简预览表或 `baseline + top candidates` 结构，则优先复用 `print_summary_preview(..., output_dir=...)` / `print_baseline_and_preview(..., output_dir=...)`，避免继续散落同样的末尾打印模板。
- 如果 archive 对比脚本只是复用“先画一张净值对比图，再落 `summary.csv`/`nav_compare.csv` 并打印摘要”这层标准收尾，优先走 `variant_compare_helpers.save_plot_and_print_variant_compare_outputs()`，不要继续把 `plot_variant_compare(...)` 和 `save_and_print_variant_compare_outputs(...)` 分开写在各脚本末尾。
- 如果 archive 对比脚本还要顺手导出 `yearly_returns.csv` 或一批 `*_nav.csv` / `*_trades.csv`，优先复用 `variant_compare_helpers.save_variant_compare_artifacts()`，不要继续在脚本尾部手串 `save_aux_csv_output()`、`save_named_nav_and_trades_outputs()` 和总览图导出。
- 如果 archive 对比脚本只是先写一组 `plot_lines` 再按 `compare_df` / `nav_compare` 过滤可用列，优先复用 `variant_compare_helpers.filter_available_plot_lines()`，不要继续在各脚本里散落同一行列表推导。
- 如果 archive baseline 预览表只是“少量策略参数前缀字段 + 统一绩效尾部（`annualized_return` / `sharpe_rf0` / `max_drawdown_integral` 及对应 diff）”，优先复用 `variant_compare_helpers.build_baseline_preview_columns()`，不要继续在各脚本里重复抄整段列名列表。
- 当前这套 `build_baseline_preview_columns()` 已覆盖大部分 overlay / stress-bond refine 链；新增脚本若只是补 `ratio_cut` / `breadth_cut` / `risk_cap` / `vg_cap` / `short_ratio_cut` 这类前缀参数字段，也应优先直接复用，不要再新开一份手写 columns 模板。
- 如果 archive 预览表还需要拼接特殊指标或筛选标记（例如 `max_drawdown`、`trigger_days`、`is_integral_valid_change`、`is_strict_drawdown_safe`），优先复用 `variant_compare_helpers.build_preview_columns()`，不要在脚本里继续手写整段混合列顺序。
- 如果 archive 搜参脚本只是复用“best.json 落盘 + 增量通知 + 通知结果打印”这层收尾，优先复用 `variant_compare_helpers.save_best_payload_and_notify_webhook()` / `save_best_payload_and_notify_ranked()`，不要继续在各脚本里重复拼 `save_best_payload(...)`、`load_incremental_notify_candidates(...)`、`notify_*` 和结果打印。
- 如果 archive 排名型增量通知只是想沿用 webhook 风格的结果打印，也优先复用 `variant_compare_helpers.save_best_payload_and_notify_ranked_webhook()`，不要继续在脚本里单独传 `print_result_fn=print_webhook_notify_result`。
- 如果 archive 搜参脚本只是从 `summary_df` 里筛 `is_valid_change` 候选，优先复用 `variant_compare_helpers.select_valid_change_rows()`；如果还需要给下游链保留“除 baseline 外的兜底上下文”，优先复用 `select_nonbaseline_rows()`，不要继续在各脚本里手写 `summary_df[...] .copy()` + rerank 模板。
- 如果 archive 脚本只是想在已有 `best.json` 上补一组非通知型兜底记录（例如给下游历史链保留 `strict_improvements` 上下文），优先复用 `variant_compare_helpers.append_best_payload_records()`，不要再各自手写 `json.loads(...)` / `write_json_atomic(...)`。
- 单策略导出/报表类 archive 脚本如果只是复用“selected / prices / backtest_nav / trades / historical_nav / 对比图表”这套落盘外壳，优先复用 `single_variant_report_helpers.py`，不要继续各自维护一整套文件导出与画图逻辑。
- 如果单策略 archive 脚本还需要复用沪深300净值序列做额外打印、衍生报表或二次图表，优先直接接 `save_outputs(...)` 的返回值 `benchmark_nav`，不要再把刚写出的 `strategy_vs_hs300.csv` 读回内存一次。
- 如果单策略 archive 脚本只是顺手补一份 `summary.csv`、`contribution_summary.csv` 这类附加表，也优先复用 `save_single_variant_tables(...)`；如果还需要顺带打印摘要，则优先走 `save_and_print_single_variant_tables(...)`，不要各脚本再各自维护重复的 CSV 落盘和末尾打印模板。
- 如果单策略 archive 脚本还需要统一生成一份 `summary.csv`、`contribution_summary.csv` 并打印摘要，优先复用 `single_variant_report_helpers.save_and_print_single_variant_summary()`；只在确实只想打印键值摘要时，再用 `print_summary()` 或 `variant_compare_helpers.print_mapping_summary()`。
- 如果单策略 archive 脚本只是在通用摘要上额外补一两个字段（例如 `leader_margin`、窗口参数、实验标签），优先通过 `summary_extra_fields=` 交给 `save_and_print_single_variant_summary()` / `save_and_print_single_variant_report()`，不要再单独手拼 `summary_df`。
- `compare_goal_optimizations.py` / `compare_hs300_regime_fixes.py` 里稳定复用的公共能力，现统一经 `goal_optimization_common.py` / `hs300_regime_common.py` 暴露，archive 脚本不要再直接 import 这两个 compare 入口。
- `compare_market_proxy_variants.py`、`compare_tail_risk_bond_overlay.py`、`compare_strategy_refinements.py`、`compare_resource_guards.py`、`compare_defensive_persistence.py` 里被历史脚本复用的稳定能力，现统一经对应 `*_common.py` wrapper 暴露，archive 脚本不要再直接 import 这些 compare 入口。
- 当前 `stress-bond` 细化链里复用的基础策略实现，也已经下沉到 `stressbond_strategy_helpers.py`；`compare_stressbond_nearmiss_repair.py` 不再充当其它脚本的实现出口。
- 当前正式 `stress-bond` 细化链如果只是复用“current context + params + 默认池 + recent prices + named proxy”这套运行时装配，优先走 `current_stressbond_context.load_current_stressbond_runtime()`，不要继续在各脚本里重复拼这段上下文准备。
- 如果当前正式 `stress-bond` 细化链只是把 baseline 结果锚回官方净值、汇总 summary 并写入 `rows/nav_compare/descriptions`，优先复用 `current_stressbond_context.append_current_stressbond_baseline()`，不要继续在各脚本里重复拼这段 baseline 初始化外壳。
- 如果某个 archive 脚本的 baseline 明确表示“当前正式基线”或 `base_pool`，现在应优先通过 `momentum_backtest/official_baseline.py` 把 baseline 净值锚回 `output/reference/official_baseline_nav.csv`，不要再直接把裸 `run_default_strategy_with_params(...)` 结果当正式基线口径。
- 只有在 baseline 本身就是其它历史最优实验、通知链上游上下文或覆盖层策略时，才不应该硬套官方锚点；否则会把研究链自己的基线改掉。
- 当前明确保留历史 baseline、故意不套官方 28.2691 锚点的脚本有：`compare_regime_mix_pool_variations.py`、`compare_resource_baseline_with_china_internet.py`、`run_china_internet_cap70_backtest.py`。
- 覆盖层 / stress-bond / 早期 threshold-dual 参数搜索这几条 archive 链，也都属于“延续上游历史赢家语义”的研究链；现在已通过脚本顶部说明或 `INTENTIONALLY_UNANCHORED_BASELINE = True` 显式标注，后续不要把它们误当正式基线脚本统一锚定。

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

当前已归档，建议按下面几组理解：

基础参数 / 结构研究：

- `compare_lookback_range.py`
- `tune_threshold_dual_momentum.py`
- `compare_dual_momentum_variants.py`
- `compare_multihorizon_combo.py`
- `compare_structural_optimizations.py`
- `compare_dynamic_thresholds.py`
- `compare_strategy_direction_bakeoff.py`
- `compare_common_sense_enhancements.py`
- `compare_effective_momentum_definitions.py`
- `compare_momentum_quality_variants.py`
- `compare_amount_proxy_blends.py`
- `compare_amount_relief_by_etf_strength.py`
- `compare_overheat_exposure_cap.py`
- `compare_opportunistic_defensive_cap.py`
- `compare_defensive_reentry_probe.py`

候选池 / 标的扩展研究：

- `compare_more_candidates.py`
- `compare_energy_income_candidates.py`
- `compare_nonferrous_guards.py`
- `compare_zijin_candidate.py`
- `compare_dividend_defensive_additions.py`
- `compare_china_internet_germany_variants.py`
- `compare_resource_baseline_with_china_internet.py`
- `compare_resource_energy_common_sense.py`
- `compare_regime_mix_pool_variations.py`
- `run_china_internet_cap70_backtest.py`
- `run_resource_abs08_backtest.py`

报表 / 导出类工具：

- `export_leader_margin_report.py`

防守覆盖层主链：

- 第一阶段基线：`compare_simple_bond_overlay.py`
- 第二阶段扩展：`compare_cash_overlay_candidates.py`、`compare_defensive_trigger_refinements.py`、`compare_two_tier_defense.py`、`compare_two_tier_defense_compact.py`、`compare_joint_treasury_trigger_search.py`
- 第三阶段衍生：`compare_defensive_bucket_blends.py`

stress-bond 细化链：

- 第一阶段：`compare_stressbond_nearmiss_repair.py`
- 第二阶段：`compare_stressbond_edge_refine.py`、`compare_stressbond_smooth_guard.py`
- 同链并行局部搜索：`compare_stressbond_dual_objective_refine.py`、`compare_stressbond_persistence_current.py`、`compare_stressbond_trigger_refine_current.py`、`compare_stressbond_trigger_refine_rc182_local.py`、`compare_stressbond_trigger_refine_rc184.py`、`compare_stressbond_vm_refine.py`

当前已知的历史依赖约定：

- `compare_cash_overlay_candidates.py` 及其下游链条，会读取 `compare_simple_bond_overlay/` 下的 `notify_state.json` / `best.json`。
- `compare_defensive_bucket_blends.py` 会读取 `compare_cash_overlay_candidates/` 下的 `notify_state.json` / `best.json`。
- `compare_stressbond_edge_refine.py` 会读取 `compare_stressbond_nearmiss_repair/` 下的 `notify_state.json` / `best.json`。
- 这类“阶段二/阶段三”脚本默认不是完全独立的单文件实验；如果单独运行，先确认上游实验目录已有状态文件。
