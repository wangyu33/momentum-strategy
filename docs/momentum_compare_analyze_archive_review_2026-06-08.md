# compare/analyze 脚本归档复核（2026-06-08）

## 目的

复核 `momentum_backtest/` 根目录下所有以 `analyze_*.py` / `compare_*.py` 命名的脚本，判断是否还有可以继续归档到 `archive/experiments/` 的对象。

本次复核只针对**当前工作区真实状态**，不依据历史印象做判断。

## 复核标准

满足以下任一条件的脚本，保留在根目录：

1. 被正式入口、通知链路或验证脚本直接 import。
2. 被其他活跃研究脚本当作共享能力复用。
3. 虽然未被别的脚本 import，但仍是当前活跃研究入口，并且：
   - 已有独立研究输出目录；
   - 在当前 README / 重构文档中仍被视为活跃研究对象；
   - 近期不是“已被主链完全替代的历史一次性实验”。

只有同时满足以下条件，才适合归档：

- 不被正式入口或活跃脚本依赖；
- 不再作为当前研究入口；
- 没有当前文档维护价值；
- 迁走后不会让根目录研究入口更混乱。

## 根目录 compare/analyze 全量复核结论

### 明确保留（存在正式依赖或共享依赖）

- `analyze_contribution.py`
- `analyze_drawdowns.py`
- `compare_candidate_pool_additions.py`
- `compare_china_internet_guards.py`
- `compare_current_best_fine_tune.py`
- `compare_current_best_pool_additions.py`
- `compare_defensive_persistence.py`
- `compare_goal_optimizations.py`
- `compare_hs300_regime_fixes.py`
- `compare_market_proxy_variants.py`
- `compare_reduce_core_fine_scan.py`
- `compare_resource_guards.py`
- `compare_strategy_refinements.py`
- `compare_tail_risk_bond_overlay.py`

保留原因：

- 这些文件要么被 `run_backtest.py` / `daily_monitor.py` / `monitor_pipeline.py` / `official_*` / `validate_strategy_outputs.py` 直接依赖；
- 要么是当前 README 明确保留的活跃研究入口；
- `compare_hs300_regime_fixes.py` 虽然名字像历史专项实验，但当前实际上承担了多个活跃脚本复用的 `summarize` / `run_target_weights_strategy` 等共享能力，不能归档。

### 人工复核后仍保留（未被别的活跃脚本直接 import，但仍属于当前活跃研究入口）

- `compare_domestic_idea_adoptions.py`
- `compare_slope_combo_replacements.py`
- `compare_slope_hot_riskcap_candidates.py`
- `compare_top2_riskcap_candidates.py`
- `compare_top2_split_candidates.py`

保留原因：

1. 这 5 个脚本都已有各自独立研究输出目录：
   - `output/research/domestic_idea_adoptions/`
   - `output/research/slope_combo_replacements/`
   - `output/research/slope_hot_riskcap_candidates/`
   - `output/research/top2_riskcap_candidates/`
   - `output/research/top2_split_candidates/`
2. 它们在当前重构文档中被明确列为：
   - 已迁移到统一主链 / 共享 runner 的研究脚本；或
   - 当前仍建议继续收口、但尚未退出使用路径的研究脚本。
3. 这类脚本虽然不像正式入口那样被 import，但它们本身就是研究入口；
   “没有被其它脚本引用”不能单独作为归档依据。
4. 结合当前输出目录与文档状态，这 5 个脚本更接近“活跃专题研究入口”，而不是“历史一次性实验”。

## 本次归档结果

**结论：本轮没有新增可安全归档的 compare/analyze 根目录脚本。**

因此：

- 不执行新的脚本迁移；
- 不对 `archive/experiments/` 新增 compare/analyze 文件；
- 保持当前根目录结构不变。

## 额外修正

相较于之前的清理审计，本次复核补充确认：

- `compare_hs300_regime_fixes.py` 不应再标记为“待人工复核”；
- 它在当前仓库里属于**共享研究核心模块**，应按“保留”处理。

## 可复用的判定经验

后续如果再审 compare/analyze 脚本，建议先问 4 个问题：

1. 它是不是正式/通知/验证链路的直接依赖？
2. 它是不是其他研究脚本复用的共享模块？
3. 它是不是当前仍在产出 `output/research/<name>/` 的研究入口？
4. 文档里是否仍把它当作当前结构的一部分在维护？

只要前 3 个问题里任意一个回答为“是”，通常就**不应直接归档**。
