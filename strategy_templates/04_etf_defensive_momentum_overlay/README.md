# 04 ETF 防守型动量轮动模板

## 1. 模板定位

这个模板不是“纯进攻型轮动”，而是参考当前仓库 `momentum_backtest/` 已有正式策略和研究脚本，总结出一套更适合 **宽基 ETF + 防守资产 + 弱市切债** 的防守型动量框架。

它要解决的核心问题不是单纯“今天买谁”，而是同时回答 3 个问题：

1. **进攻时买谁**：在风险资产池中选出最强 ETF。
2. **弱势时退到哪里**：在黄金 / 红利低波 / 商品 / 国债之间做防守切换。
3. **仓位该给多少**：当市场量能、广度、热度恶化时，不是只切换资产，还要主动压低风险暴露。

所以，这是一套 **信号选择 + 防守层 + 风险上限** 的复合模板。

---

## 2. 调研依据

本模板主要参考以下文件与结果：

### 2.1 正式入口与当前默认参数

- `momentum_backtest/run_backtest.py`
- `momentum_backtest/README.md`
- `momentum_backtest/output/core/backtest_nav.csv`

### 2.2 防守层专项研究脚本

- `momentum_backtest/compare_tail_risk_bond_overlay.py`
- `momentum_backtest/compare_defensive_persistence.py`
- `momentum_backtest/archive/experiments/compare_opportunistic_defensive_cap.py`
- `momentum_backtest/compare_top2_riskcap_candidates.py`
- `momentum_backtest/compare_goal_optimizations.py`

### 2.3 已落盘研究结果

- `momentum_backtest/output/research/simple_bond_overlay/summary.csv`
- `momentum_backtest/output/research/defensive_trigger_refinements/summary.csv`
- `momentum_backtest/output/research/defensive_persistence/summary.csv`
- `momentum_backtest/output/research/defensive_bucket_blends/summary.csv`
- `momentum_backtest/output/research/two_tier_defense/summary.csv`
- `momentum_backtest/output/research/joint_treasury_trigger_search/summary.csv`
- `momentum_backtest/output/research/stressbond_edge_refine/summary.csv`
- `momentum_backtest/output/research/stressbond_trigger_refine_current/summary.csv`
- `momentum_backtest/output/research/stressbond_persistence_current/summary.csv`
- `momentum_backtest/output/research/stressbond_vm_refine/summary.csv`
- `momentum_backtest/output/research/stressbond_dual_objective_refine/summary.csv`

---

## 3. 当前仓库里已经验证过的防守思路

结合代码和实验结果，仓库里的 ETF 防守策略大致分成 6 层。

### 3.1 第一层：风险资产池内先做“更稳的强者选择”

当前正式策略不是直接按原始 25 日涨幅选第一，而是：

- 使用 `lookback = 25`
- 原始动量仍看 `price / price.shift(25) - 1`
- 但实际排名时用 `slope_085` 做质量惩罚

也就是：

- 不是只看“涨了多少”
- 还看“这段上涨是不是最后几天陡拉出来的”

这一步本身就属于轻量防守，因为它能降低追高型误判。

### 3.2 第二层：弱趋势时先退到防守池，而不是直接空仓

当前体系的防守池不是单一资产，而是一个 **防守 ETF 桶**：

- 黄金 ETF：`518880`
- 红利低波 ETF：`512890`
- 豆粕 ETF：`159985`

当风险资产第一名动量落到弱趋势甚至转负时，策略先在这些防守资产中找相对更强者，而不是立刻完全空仓。

这个设计的意义是：

- 保留一定收益弹性
- 避免“股弱了但商品/黄金还强”时完全踏空
- 让防守不是单一方向押注

### 3.3 第三层：市场环境弱化时，不只切资产，还要直接砍风险仓位

当前正式参数里已经有一层 **市场量能 + 市场广度** 的风险开关：

- `20日/60日成交额 < 0.91`
- `5日/20日成交额 < 0.90`
- `市场广度代理 < -0.04`
- `effective_momentum <= 0.16`

命中后，风险仓位上限会被压到 `0`。

这层的关键价值在于：

- 防守不再仅依赖“某只防守 ETF 是否更强”
- 而是直接承认：市场整体风险预算已经不该给那么高

### 3.4 第四层：极弱市单独切到十年国债 ETF

仓库里对“更极端的弱市”又单独做了一层 stress bond overlay：

- 资产：`511260` 十年国债 ETF
- 当前正式默认触发条件：
  - `20日/60日成交额 < 0.90`
  - `市场广度代理 < -0.031`
- 当前正式默认参数：
  - `risk_cap = 0.0`
  - `enter_days = 1`
  - `exit_days = 1`

也就是说，正式版把这层当成 **极弱市防线**，触发后允许把风险仓直接压到 0，再由债券承接。

### 3.5 第五层：高位过热时不追满仓

当前正式策略的过热保护已经不是简单台阶式，而是连续降仓：

- `pre_overheat_start_cut = 0.15`
- `pre_overheat_end_cut = 0.20`
- `pre_overheat_end_exposure = 0.90`
- `overheat_momentum_cut = 0.25`
- `overheat_max_exposure = 0.30`
- `overheat_high_momentum_cut = 0.30`
- `overheat_high_max_exposure = 0.10`

它解决的是：

- 趋势还在，但热度已经偏高
- 这时不是立即反手做空，而是先降风险暴露

这也是一种非常关键的“防守”，本质上属于 **顺势但不重仓追热**。

### 3.6 第六层：候选胜者太接近时，主动压仓避免假突破

`run_backtest.py` 当前默认还有一层：

- `top2_close_gap = 0.005`
- `top2_close_risk_cap = 0.70`

如果风险池第一名和第二名太接近，说明领先优势不够明确，这时直接把风险仓上限压低到 70%。

这层尤其适合轮动策略，因为很多回撤恰恰发生在“冠军切换非常拥挤但并不稳定”的阶段。

---

## 4. 回测结果里哪些防守思路值得保留

下面只保留对模板设计真正有指导意义的结论。

### 4.1 纯“回撤触发切债”太保守，不建议做成主模板

`simple_bond_overlay/summary.csv` 里，纯 drawdown 驱动的切债思路虽然能把最大回撤压到更低，但收益损失很明显。

一个代表性结果：

- 策略：`drawbond_511260_dd03_cap50`
- 年化收益：约 `15.35%`
- 年化波动：约 `16.19%`
- Sharpe：约 `0.95`
- 最大回撤：约 `-16.04%`

结论：

- 如果目标是“极致防守、宁可少赚很多”，它有意义
- 但如果目标是做一套可长期实盘的 ETF 轮动模板，它过于保守
- 因此不建议把“仅看回撤就切债”作为主模板

### 4.2 “市场弱化 + stress bond” 是最值得保留的主防线

`defensive_trigger_refinements` 与 `joint_treasury_trigger_search` 里，最稳定的一类思路是：

- 当市场量能 / 广度进入弱市区间时
- 不是完全依赖防守池内部轮动
- 而是显式切一部分或全部仓位给十年国债 ETF

代表结果：

- 策略：`stressbond_511260_vr900_vb-10_cap15_vsna`
- 年化收益：约 `26.58%`
- 年化波动：约 `20.97%`
- Sharpe：约 `1.267`
- 最大回撤：约 `-23.08%`

它说明：

- **国债 overlay 本身是有效的**
- 但最优形式并不是“越激进越好”或“越频繁越好”
- 更关键的是触发阈值与风险上限的组合

### 4.3 persistence（进入/退出滞后）更像调参工具，不是第一优先级

`defensive_persistence/summary.csv` 与 `stressbond_persistence_current/summary.csv` 说明：

- 给弱市切债增加 `enter_days` / `exit_days`
- 确实可以减少噪音触发
- 但并不总能明显优于当前正式口径

比如：

- `prs_rc184_en01_ex02`
- 年化收益约 `23.62%`
- Sharpe 约 `1.178`
- 最大回撤约 `-21.99%`

这个结果说明：

- **退出稍慢** 有时能改善 Sharpe
- 但防守收益比并没有出现质变
- 因此 persistence 更适合作为“细化稳定性”的二级开关，而不是模板的核心卖点

### 4.4 防守桶里再混现金/短债，改善有限

`defensive_bucket_blends/summary.csv` 对比了：

- 只用 `511260`
- `511260 + 511880 + 现金`

结果显示：

- 数值上会有轻微变化
- 但并没有形成明显的结构性优势

这说明：

- 防守桶复杂化不一定带来明显提升
- 除非用户有明确需求（比如波动更低、久期更短、现金占比更高）
- 否则模板层面没必要默认引入太多防守细分资产

### 4.5 双层防守（weak + extreme）有研究价值，但不适合先做默认模板

`two_tier_defense/summary.csv` 做了两级触发：

- 一层弱市 cap
- 一层极弱市更低 cap

这类思路方向上是合理的，但当前结果并没有稳定压过更简单的 stress bond 方案。

结论：

- 可以保留为研究升级方向
- 但不建议一开始就作为 `strategy_templates` 里的默认模板
- 模板应优先选择“解释性强、参数更少、迁移成本低”的方案

### 4.6 当前正式策略已经是一套“三段式防守”框架

从当前 `run_backtest.py` 和 `output/core/backtest_nav.csv` 看，正式入口已经不是单一 overlay，而是组合后的防守结构：

1. **防守池轮动**：黄金 / 红利低波 / 豆粕
2. **市场弱化压风险仓**：volume + breadth + effective momentum guard
3. **极弱市切债**：`511260`
4. **高位过热降仓**：连续降 exposure
5. **胜者差距不明显时压仓**：top2 close cap

直接看当前正式输出（`output/core/backtest_nav.csv`）重新计算后的结果：

- 年化收益：约 `25.56%`
- 年化波动：约 `19.41%`
- Sharpe：约 `1.314`
- 最大回撤：约 `-18.98%`
- 平均风险暴露：约 `80.3%`

这说明当前正式版已经不是“只有收益没有防守”的进攻模板，而是一套防守层比较完整的 ETF 轮动框架。

---

## 5. 适合沉淀为模板的 3 套 ETF 防守方案

基于上面的研究，我建议 `strategy_templates` 里的 ETF 防守策略按“复杂度递增”沉淀成 3 档，而不是只给一个唯一答案。

## 模板 A：轻防守轮动

### 适用场景

- 用户刚开始做 ETF 轮动
- 希望逻辑简单、换手可控
- 不想一开始就堆很多风控参数

### 结构

1. 风险池里按 25 日动量选第一
2. 风险动量弱时切到防守池第一名
3. 不单独引入国债 overlay
4. 允许弱趋势区间保留部分防守仓

### 推荐资产池

- 风险资产：`510300 / 159949 / 159954 / 159941 / 513880`
- 防守资产：`518880 / 512890 / 159985`

### 评价

- 优点：实现最简单，解释性强
- 缺点：面对系统性弱市时，防守强度不够

### 是否推荐保留

**推荐保留**，适合做“入门版 ETF 防守模板”。

---

## 模板 B：标准防守轮动（推荐默认模板）

### 适用场景

- 希望兼顾收益、解释性、落地复杂度
- 接近当前仓库正式版本的设计
- 适合做长期维护的默认模板

### 结构

1. 风险池使用 `25 日动量 + slope_085` 选主信号
2. 弱趋势时退到防守池
3. 市场量能 / 广度弱化时，将风险仓上限压低
4. 极弱市单独切到 `511260`
5. 高位过热时连续降仓
6. top2 领先差距过小的时候压仓

### 推荐参数

- `lookback = 25`
- `signal_quality_method = slope`
- `signal_slope_penalty = 0.85`
- `absolute_threshold = 0.05`
- `weak_trend_defensive_weight = 0.80`
- 市场弱化：
  - `20/60 amount < 0.91`
  - `5/20 amount < 0.90`
  - `breadth < -0.04`
  - `effective momentum <= 0.16`
- stress bond：
  - `treasury = 511260`
  - `20/60 amount < 0.90`
  - `breadth < -0.031`
- 过热保护：
  - `0.15 -> 0.20` 开始预减仓
  - `0.25` 动量后压到 `30%`
  - `0.30` 动量后压到 `10%`
- close leader protection：
  - `top2 gap < 0.5%`
  - `risk cap = 70%`

### 评价

- 优点：
  - 已有完整代码与产出
  - 防守逻辑完整
  - 对实盘可解释性最强
- 缺点：
  - 参数比入门版多
  - 对市场代理数据质量有依赖

### 是否推荐保留

**强烈推荐保留，建议作为默认 ETF 防守模板。**

---

## 模板 C：增强防守轮动（研究保留版）

### 适用场景

- 希望继续优化风险收益比
- 接受更多参数和更复杂的维护成本
- 主要用于研究，不一定直接上生产

### 结构

在模板 B 上继续引入更细的边缘约束，例如：

- stress bond 风险上限精修（`risk_cap` 不一定直接到 0）
- volume guard 与 overheat cap 联合细化
- persistence / exit lag 只做局部修正

### 研究结论

从 `stressbond_edge_refine` 结果看，增强版能把收益进一步抬高，例如代表结果：

- 年化收益：约 `26.59%`
- 年化波动：约 `20.97%`
- Sharpe：约 `1.268`
- 最大回撤：约 `-23.10%`

但问题也很明显：

- 最大回撤未必优于当前正式输出
- 参数解释成本更高
- 更适合作为“研究储备版本”，而不是模板默认值

### 是否推荐保留

**建议保留为 research-only 模板，不建议直接作为默认模板。**

---

## 6. 最终建议

如果只保留一个 ETF 防守主模板，我建议选：

## **模板 B：标准防守轮动**

原因有 4 个：

1. **它和当前仓库正式实现最接近**，迁移成本最低。
2. **防守层次完整**，不是单一切债，也不是单一防守池轮动。
3. **研究结果已经证明方向有效**，并且当前正式输出的风险收益比已经比较均衡。
4. **可继续向模板 C 升级**，不会把后续研究路线堵死。

换句话说：

- 真正值得沉淀成模板的，不是“某一个最优参数组合”
- 而是 **风险池轮动 + 防守池承接 + 弱市切债 + 过热降仓 + 胜者不明确时压仓** 这一整套结构

---

## 7. 后续如果要继续落地，建议怎么做

如果后续要把这个模板继续工程化，我建议按下面顺序推进：

1. 先把模板 B 的参数写成一份独立配置文件
2. 再把风险池 / 防守池 / 国债承接池拆成可替换的 universe 配置
3. 再单独增加：
   - persistence 开关
   - two-tier defense 开关
   - defensive bucket blend 开关
4. 最后才做自动搜索和参数微调

这样可以避免一开始就把模板写成“研究脚本堆叠版”。

---

## 8. 一句话结论

这份调研的核心结论是：

> ETF 防守策略最有效的做法，不是单纯把风险资产切成债券，而是把“防守池轮动、弱市切债、市场弱化压仓、高位过热降仓”组合成一套多层防守框架；当前仓库里最值得沉淀为模板的方案，就是以当前正式版为基底的标准防守轮动模板。
