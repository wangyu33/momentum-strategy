# ETF Momentum Backtest

这个目录当前只服务三类事情：

- 正式基线回测
- 日常巡检与消息通知
- 研究、排查与对照实验

如果只想跑当前正式版本，只看两个入口：

- `run_backtest.py`
- `daily_monitor.py`

## 代码结构

建议按 4 层理解：

- 正式入口
  - `run_backtest.py`
  - `daily_monitor.py`

- 活跃研究入口
  - `compare_current_best_fine_tune.py`
  - `compare_market_proxy_variants.py`
  - `compare_current_best_pool_additions.py`
  - `compare_goal_optimizations.py`
  这些脚本仍会直接产出新一轮研究结果，保留在根目录。

- 支撑模块与专项对照
  - `compare_hs300_regime_fixes.py`
  - `compare_tail_risk_bond_overlay.py`
  - `compare_defensive_persistence.py`
  - `compare_candidate_pool_additions.py`
  - `compare_strategy_refinements.py`
  - `compare_china_internet_guards.py`
  - `compare_resource_guards.py`
  - `search_utils.py`
  - `runtime_env.py`
  - `validate_strategy_outputs.py`
  这批文件里有些会被正式入口、巡检脚本或活跃研究脚本直接 import。
  它们更多承担复用逻辑、专项回测或回归校验职责，不建议继续往 `archive/experiments/` 挪，除非先解除依赖。

- 历史归档
  - `archive/experiments/`
  已退出日常使用路径，但保留复盘价值。当前原则是：只有不再被正式入口、巡检脚本、活跃研究脚本引用的一次性实验，才继续归档进去。
  - `output/research/archive_flat/`
  历史归档实验在旧结构下写到 `output/research/` 根目录的散落结果，已统一搬到这里，避免和当前活跃结果混在一起。

补充说明：

- `run_backtest.py`
  负责拉取历史数据，运行默认正式基线，并同步刷新 `output/core/` 与依赖 core 推导出的 `output/analysis/`。

- `daily_monitor.py`
  先用历史收盘价刷新正式结果，再叠加盘中实时价格生成快照并发双通知。盘中快照不会反写 `output/core/`。
  当前会区分：
  - 最新确认持仓
  - 今日目标持仓
  - 根据今日盘中/盘后数据是否需要交易

## 指标口径

当前和回撤相关的两个指标，必须区分：

- `max_drawdown_integral`
  - 含义：全历史回撤积分
  - 口径：整个回测区间内，所有负回撤的累计面积
  - 适合衡量：长期处于水下的总痛苦程度

- `max_drawdown_episode_integral`
  - 含义：最大回撤区间积分
  - 口径：只对“最深的那一段回撤 episode”做积分
  - 适合衡量：单次最差回撤阶段的持续时间和深度

补充：

- 为兼容历史研究脚本，很多脚本仍使用 `max_drawdown_integral` 这个字段名
- 从当前版本开始，它指的是“全历史回撤积分”
- 如果要看旧语义中的“最差单段回撤面积”，请改看 `max_drawdown_episode_integral`
- 若研究结果里带有 `drawdown_integral_recomputed`
  - `true` 表示该文件已经在新口径下重新回刷
  - `false` 表示该文件只补了说明字段，数值本身仍可能来自旧口径结果

- `compare_*.py`
  是研究脚本或研究支撑模块。真正作为入口使用的脚本通常会把结果写到 `output/research/<experiment_name>/`，不应覆盖正式基线。

- `analyze_*.py` / `plot_*.py`
  对已有回测结果做归因、回撤拆解和图表分析。

## 当前默认基线

当前默认基线不是最早的 plain threshold dual，而是：

`baseline_cf60top2`

可以把它理解成 4 层规则叠加：

1. 底层信号选择
   先算风险资产的原始 `25` 日动量，再对“近 5 日涨得过陡”的标的做一层 `slope_085` 惩罚；同时要求候选标的也进入风险池 `60` 日动量前 `2` 名，最后再按修正后的质量分数选第一名。
   - 原始动量：`price / price.shift(25) - 1`
   - 质量分数：`raw_momentum - 0.85 * clip(ret5 - raw_momentum / 5, lower=0)`
   - 中期确认：风险池 `60` 日动量前 `2` 名才允许参与当天风险信号竞争
   - 对外展示和阈值判断仍使用原始 `25` 日动量

2. dual momentum 仓位框架
   - 风险池第一名近 `25` 日动量 `> 5%`：风险资产 `100%`
   - 风险池第一名近 `25` 日动量在 `0% ~ 5%`：切到防守池第一名，仓位 `80%`
   - 风险池第一名近 `25` 日动量 `<= 0%`：持有防守池第一名，仓位 `100%` 或空仓

3. regime mix + market proxy
   - 去掉了 `511580`、`513650`
   - 进攻核心仓 `28%`
   - 保守核心仓 `27%`
   - regime 切换阈值 `6%`
   - 市场量能过滤：
     - `20日/60日成交额 < 91%`
     - `5日/20日成交额 < 90%`
     - `市场广度代理 < -4%`
     - 且综合动量 `<= 16%`
   - 命中后，风险资产总仓位上限压到 `50%`
   - 如果风险池前二名信号质量差 `<= 0.5%`，且风险仓高于 `70%`，再额外压到 `70%`

4. 弱市切债 + 过热降仓
   - 若市场进入更强的弱市区间：
     - `20日/60日成交额 < 90%`
     - `市场广度代理 < -3.1%`
   - 当前正式版把风险仓上限压到 `0%`
   - 其余仓位切到 `511260 十年国债ETF`
   - 同时保留高位过热保护：
     - 回撤 `>= -2%` 且动量 `>= 25%` 时，风险仓再压到 `6.6%`
     - 极热动量 `>= 32%` 时，风险仓进一步压到 `3.0%`

当前默认基金池分三层：

- 风险资产池
  - `510300` 沪深300
  - `159949` 创业板50
  - `159954` H股ETF
  - `159941` 纳指ETF
  - `513880` 日经ETF

- 防守资产池
  - `518880` 黄金ETF
  - `512890` 红利低波ETF
  - `159985` 豆粕ETF

- 弱市切债资产
  - `511260` 十年国债ETF

## 正式基线策略详解

这套正式基线可以理解成一句话：

- 先在风险资产里用 `25` 日动量 + `slope_085` 质量筛选选出主信号
- 再用 risk / defensive 双核心仓位做平滑切换
- 然后用市场量能、市场广度、策略热度控制风险暴露
- 最后在极弱市切十年国债，在高位过热时进一步降仓

它不是“单一 ETF 满仓切来切去”的纯轮动，而是“信号选择 + 仓位管理 + 尾部防守”的组合结构。

### 1. 候选池怎么分

- 风险资产池
  - `510300` 沪深300ETF华泰柏瑞
  - `159949` 创业板50ETF华安
  - `159954` H股ETF
  - `159941` 纳指ETF广发
  - `513880` 日经225ETF华安

- 防守资产池
  - `518880` 黄金ETF华安
  - `512890` 红利低波ETF华泰柏瑞
  - `159985` 豆粕ETF

- 极弱市切债资产
  - `511260` 十年国债ETF

补充：

- 固定池原始名单一共 `10` 只，但正式默认基线会移除 `511580 政金ETF` 和 `513650 标普500ETF`
- 原因是这两只加入后，对长样本收益/夏普/回撤的综合性价比不如当前版本
- 所以正式默认基线实际使用的是“`5` 只风险 + `3` 只防守 + `1` 只弱市切债”

### 2. 整个策略怎么跑

按日频回测时，核心流程是：

1. 拉取候选池历史价格，并对 ETF 现金分红做本地前复权处理。
2. 计算风险池里每只 ETF 的原始 `25` 日动量。
3. 用 `slope_085` 质量分数在风险池里选出当前主信号。
4. 根据主信号原始动量强弱，先决定“偏进攻”还是“偏防守”的核心仓位框架。
5. 再读取大盘量能代理和市场广度代理，判断是否要压低风险总仓位。
6. 如果处于更弱的市场区间，则把超出的风险仓切到 `511260 十年国债ETF`。
7. 如果组合已经处在高位附近且热度过高，再把风险仓压得更低，防止追高后快速回撤。

所以正式基线不是简单回答“今天买哪只”，而是同时回答两个问题：

- 当前该买哪只主信号 ETF
- 当前最多应该给这只主信号多少风险仓位

### 3. 关键阈值是什么

#### 动量主窗口

- `lookback = 25`
- `signal_quality_method = slope`
- `signal_slope_penalty = 0.85`

原因：

- `25` 个交易日大约对应一个自然月，足够反映一段可交易趋势
- 比 `10` 日更稳，不容易被短期噪声带偏
- 比 `60` 日更快，不会明显钝化切换速度
- `slope_085` 不是替换动量，而是在同一窗口下抑制“最后几天突然急拉”的假强势
- 它解决的是相同 `25` 日涨幅里，哪种走势更平滑、更可持续

#### dual momentum 基础阈值

- `absolute momentum threshold = 5%`
- `weak trend defensive weight = 80%`

可以粗略理解为：

- 风险主信号 `> 5%`，趋势够强，允许更激进地配风险仓
- 风险主信号在弱趋势区间时，不直接满仓风险，保留更多防守成分
- 风险主信号转弱甚至转负时，优先切防守，不硬扛

#### regime mix 核心仓位

- 进攻核心仓 `aggressive_core_weight = 28%`
- 防守核心仓 `conservative_core_weight = 27%`
- regime 切换阈值 `regime_momentum_cut = 6%`

原因：

- 不直接在 `0%` 和 `100%` 两端跳变，而是保留中间态，让仓位变化更平滑
- 这样可以减少“刚切进去就反向”的抖动
- `6%` 相当于把“趋势确立”门槛略抬高，避免把一般性反弹误当成主升段

#### 市场量能保护

- `20日/60日成交额 < 91%`
- `5日/20日成交额 < 90%`
- `市场广度代理 < -4%`
- `策略热度 <= 16%`
- 命中后风险仓上限压到 `0%`

原因：

- `20/60` 看中期量能是否在持续走弱
- `5/20` 看短期量能是否还在进一步恶化
- 市场广度补充“只是缩量”还是“赚钱效应同步走弱”
- 只有当内部热度不高时才启用这层保护，避免强趋势行情被过早打断

#### 弱市切债

- `20日/60日成交额 < 90%`
- `市场广度代理 < -3.1%`
- `risk_cap = 0%`
- `enter_days = 1`
- `exit_days = 1`

命中后：

- 风险资产最多只保留 `0%`
- 超出的部分切到 `511260 十年国债ETF`

原因：

- 这层是尾部防守，不再只是“少买一点”，而是把多余仓位明确切到低波动债券
- `0%` 代表一旦确认进入更强的弱市区间，就把风险资产彻底切干净，避免在缩量弱广度阶段继续承受回撤面积

#### 高位过热保护

- `drawdown >= -2%` 且 `effective_momentum >= 25%`，风险仓压到 `6.6%`
- `effective_momentum >= 32%`，风险仓进一步压到 `3.0%`

原因：

- 真正容易追高受伤的阶段，往往不是“动量很高”本身，而是“净值接近历史高位 + 热度已经很极端”
- 所以这里同时看回撤位置和内部热度
- 先压到 `6.6%`，极热再压到 `3.0%`，目的都是缩小高位回撤冲击

### 4. 用到了哪些变量，分别是干什么的

- `current_momentum`
  给人看的“当前信号动量”，口径是当前信号 ETF 自身近 `25` 个交易日涨幅。
  之所以保留它，是因为它最符合人工阅读和盘中判断直觉。

- `signal_asset_momentum`
  和 `current_momentum` 同口径，主要是为了兼容外部读取和历史脚本。

- `effective_momentum`
  策略内部热度，用于量能保护和过热保护。
  之所以单独保留，是因为正式基线不是单一 ETF 纯轮动，内部风控更适合看组合热度而不是只看单一信号 ETF。

- `signal`
  当日选出的主信号 ETF 代码。
  它回答“今天最强的是谁”。
  当前正式基线里，`signal` 是按 `slope_085` 质量分数从风险池里选出来的。

- `holding`
  实际持有的 ETF 代码。
  有时 `holding` 和 `signal` 会不一样，因为执行层有 `T+1` 持仓滞后，还可能叠加防守资产或债券仓位。
  可以直接理解成：
  - `signal` 回答“今天系统想切去哪”
  - `holding` 回答“今天实际拿着什么”

- `target_exposure`
  当日目标风险仓位。
  它回答“即使信号没变，今天最多该给多少风险仓”。

- `market_amount_ratio_20_60`
  大盘成交额 `20` 日均值 / `60` 日均值。
  用它判断中期量能趋势是否转弱，优点是比单日量能稳定得多。

- `market_amount_ratio_5_20`
  大盘成交额 `5` 日均值 / `20` 日均值。
  用它补足 `20/60` 的滞后性，判断最近一段时间是否还在继续缩量。

- `market_breadth_proxy`
  市场广度代理。
  用它判断赚钱效应是否同步走弱，避免只看成交额带来误判。

- `drawdown`
  当前净值相对历史最高点的回撤。
  用它区分“高位过热”还是“已经在低位杀跌区”，因为同样的高动量，在不同净值位置风险完全不同。

- `risk_cap`
  弱市时允许保留的风险仓上限。
  这是“防守力度”的直接开关。

- `aggressive_core_weight` / `conservative_core_weight`
  风险核心仓和防守核心仓的基础比例。
  这两个变量让策略不是二元开关，而是能平滑切换。

### 5. 为什么最终选这几个变量

核心原因是要同时覆盖四类不同风险：

- 趋势风险
  用 `25` 日动量解决，回答“谁最强”。

- 市场环境风险
  用 `20/60`、`5/20` 量能比和 `market_breadth_proxy` 解决，回答“现在的大盘环境是否支持继续给风险仓”。

- 追高风险
  用 `effective_momentum + drawdown` 解决，回答“当前位置是不是太热了”。

- 尾部回撤风险
  用 `risk_cap + stress bond` 解决，回答“如果市场整体进入弱区间，仓位应该怎么防守”。

只用单一动量，会出现两个问题：

- 强趋势里赚得到，但高位回撤会偏大
- 弱市里经常在错误的趋势上来回切换

所以正式基线才会加入量能、广度、回撤和切债四层辅助变量。

### 6. 盘中和收盘后口径有什么区别

- `run_backtest.py`
  只用历史收盘价，输出正式回测结果，刷新 `output/core/`。

- `daily_monitor.py`
  会先用历史收盘价刷新一次正式结果，然后叠加当日实时 ETF 价格做盘中快照。

当前统一约定：

- 市场量能固定使用“上一交易日收盘值”
- 不再做盘中量能预估
- 通知里会标注当前是 `开盘前 / 盘中 / 午间休市 / 收盘后`
- 通知里把“当前持仓”和“当前信号”拆开显示，不再混用

这样做的原因是，ETF 实时价格适合盘中看信号变化，但大盘成交量盘中噪声太大，直接拿来做风控容易误判。

补充：`daily_momentum_board.py` 现在也遵循同一套价格口径。

- `盘中 / 午间休市`：榜单使用实时 ETF 价格计算 `25` 日动量；
- `收盘后`：榜单回到收盘价口径；
- 榜单消息会明确标注 `交易时段`、`价格口径`、`价格日期`，避免把盘中榜单和收盘榜单混在一起看。

## 运行环境

当前项目需要区分两套运行环境：

### 1. 开发 / 调试环境

开发调试时，默认使用仓库根目录下的 `.venv`，优先不要手工用系统 `python3` 直接跑编辑版代码。

首次进入项目建议先执行：

```bash
cd <项目编辑目录>
source .venv/bin/activate
```

如果不想激活虚拟环境，也可以直接使用：

```bash
<项目编辑目录>/.venv/bin/python
```

### 2. 定时运行环境

正式定时任务不直接读取编辑目录，而是读取稳定运行版目录：

- 编辑版目录：`/Users/bytedance/Documents/trae_projects/wy_test`
- 稳定运行版目录：`~/Library/Caches/wy_test_runtime`

当前 LaunchAgent 会从稳定运行版目录启动：

- `com.codex.etf-momentum-monitor` -> `~/Library/Caches/wy_test_runtime/momentum_backtest/daily_monitor.py`
- `com.codex.etf-momentum-board` -> `~/Library/Caches/wy_test_runtime/momentum_backtest/daily_momentum_board.py`
- `com.codex.momentum-backfill` -> `~/Library/Caches/wy_test_runtime/momentum_backtest/run_backtest.py`

当前调度时间是：

- monitor：工作日 `09:40`、`12:10`、`14:50`
- momentum board：工作日 `15:10`
- backfill：工作日 `15:20`

注意：LaunchAgent 当前实际调用的是系统 `/usr/bin/python3`，但它运行在稳定版目录，并以同步后的项目文件为准。

### 3. 改动如何上线到定时任务

- **平时改代码只改编辑版目录**。
- **任何影响正式通知 / 正式回测 / LaunchAgent 的改动，都要同步到稳定运行版目录**。

执行：

```bash
./scripts/sync_runtime_and_reload.sh
```

这个脚本会：

- 把编辑版同步到 `~/Library/Caches/wy_test_runtime`
- 运行 `validate_strategy_outputs.py`
- 重写并重载 LaunchAgent
- 补齐运行时日志路径

如果发现 `numpy` / `pandas` / `akshare` 导入报错，先区分自己是在：

- 开发环境误用了系统 Python；还是
- 定时运行环境没有完成同步 / 重载。

## 最常用命令

### 1. 跑正式基线回测

```bash
.venv/bin/python momentum_backtest/run_backtest.py
```

这会：

- 拉取默认池历史价格
- 运行默认基线
- 覆盖 `output/core/`

如果只想指定区间：

```bash
.venv/bin/python momentum_backtest/run_backtest.py --years 10
```

### 2. 运行每日巡检

```bash
.venv/bin/python momentum_backtest/daily_monitor.py
```

这会：

- 读取默认池
- 先用历史收盘价刷新 `output/core/`
- 再叠加当日实时 ETF 价格
- 市场量能固定使用上一交易日收盘值
- 重算当前策略信号、当前持仓和目标风险仓位
- 发送两条消息：
  - `ETF市场热度`：信号动量、策略热度、市场量能、建仓风险、历史同类区间摘要
  - `ETF交易与回撤`：当前确认持仓、今日目标组合、今日交易、本次操作原因、净值、当前回撤、历史最大回撤、历史高点

如果只是想看输出，不发直连飞书：

```bash
.venv/bin/python momentum_backtest/daily_monitor.py --disable-direct-feishu
```

如果想强制重发一次：

```bash
.venv/bin/python momentum_backtest/daily_monitor.py --force
```

如果是人工补发，但不想影响后续 `12:10` / `14:50` 的正式定时通知：

```bash
.venv/bin/python momentum_backtest/daily_monitor.py --preview-send
```

### 3. 做一致性校验

```bash
.venv/bin/python momentum_backtest/validate_strategy_outputs.py
```

这会校验三类事情：

- `backtest_nav.csv` 的收益、换手、净值、回撤是否能从价格和权重反推一致
- `trades.csv` 是否和权重变化逐日对得上
- A 股 `开盘前 / 盘中 / 午间休市 / 收盘后` 的边界，以及何时允许启用实时快照

## 参数说明

### `run_backtest.py`

最常用：

```bash
.venv/bin/python momentum_backtest/run_backtest.py
.venv/bin/python momentum_backtest/run_backtest.py --years 15
.venv/bin/python momentum_backtest/run_backtest.py --years 10 --fee-rate 0.0003 --slippage-rate 0.0002
```

参数说明：

- `--years`
  回测最近多少年。默认 `15`。
  但项目统一对齐到 `2012-01-01` 之后，不会再往前取更早历史。

- `--lookback`
  动量窗口，单位是交易日。默认 `25`。
  这个参数目前主要影响 plain threshold dual 分支；默认正式基线内部也以 `25` 日为核心窗口。

- `--fee-rate`
  单边手续费。默认 `0.0003`。

- `--slippage-rate`
  单边滑点。默认 `0.0002`。

- `--cash-threshold`
  当最强动量低于某阈值时空仓。默认不用。

- `--disable-overheat-cap`
  不跑默认正式基线，临时切回更简化的 plain dual momentum 版本。这个参数主要用于做对照，不建议当正式结果使用。

下面这些参数只在 `--disable-overheat-cap` 的简化分支下有意义：

- `--overheat-drawdown-cut`
- `--overheat-momentum-cut`
- `--overheat-max-exposure`
- `--overheat-high-momentum-cut`
- `--overheat-high-max-exposure`

它们控制“高位过热时的减仓阈值”，默认正式基线内部有自己独立的一套参数。

### `daily_monitor.py`

最常用：

```bash
.venv/bin/python momentum_backtest/daily_monitor.py
.venv/bin/python momentum_backtest/daily_monitor.py --force
.venv/bin/python momentum_backtest/daily_monitor.py --preview-send
.venv/bin/python momentum_backtest/daily_monitor.py --debug
```

参数说明：

- `--strategy`
  选择巡检哪个策略。
  可选值：
  - `default`
  - `china_internet_cap70`
  - `resource_abs08`

- `--force`
  忽略上次状态缓存，强制发送一次，并写入新的去重状态。

- `--preview-send`
  发送当前快照，但不读写 `daily_monitor_state.json`。
  人工补发默认使用这个参数，避免把后续正式定时通知去重掉。

- `--debug`
  打开调试日志。

- `--disable-direct-feishu`
  不走直连飞书私信发送，只保留 webhook。

- `--webhook-url`
  临时指定 webhook。

## 正式输出怎么理解

默认正式基线的输出都在 `output/core/`。

### 关键文件

- `output/core/selected_etfs.csv`
  当前正式回测使用的 ETF 列表。

- `output/core/prices.csv`
  价格面板。使用新浪 ETF 日线，并对现金分红做了本地前复权处理。

- `output/core/backtest_nav.csv`
  最重要的结果表。常用列：
  - `nav`：策略净值
  - `strategy_return`：当日策略收益率
  - `current_momentum`：当前信号标的的 25 日动量，面向图表、通知和人工阅读
  - `effective_momentum`：默认正式基线内部真正用于量能过滤/过热保护的组合动量
  - `signal_asset_momentum`：与 `current_momentum` 相同，单独保留是为了兼容外部读取
  - `signal`：当日目标标的
  - `holding`：实际持仓标的
  - `exposure`：实际仓位
  - `target_exposure`：目标仓位
  - `drawdown`：相对历史最高点的回撤

- `output/core/trades.csv`
  交易记录。

- `output/core/historical_nav.csv`
  只保留净值序列，方便外部读取。

- `output/core/yearly_returns.csv`
  自然年收益和相对沪深300超额收益。

- `output/core/strategy_vs_hs300.csv`
  策略和沪深300ETF的净值对比。

### 图表

- `output/core/nav_with_trades.png`
  净值曲线，并标出买卖点。

- `output/core/momentum_series.png`
  历史信号动量序列。

- `output/core/max_drawdown.png`
  历史回撤曲线，并标记最大回撤点。

- `output/core/strategy_vs_hs300.png`
  策略和沪深300对比图。

- `output/core/contribution_rate.png`
  各 ETF 对总收益的贡献度图。

- `output/core/contribution_rate_cn.png`
  同一张图的备用文件名，主要用于绕过本地图片缓存。

### 动量字段怎么理解

- `current_momentum`
  这是你在通知里看到的“当前信号动量”。口径是“当前信号标的自身的近 25 个交易日涨幅”。

- `effective_momentum`
  这是默认正式基线内部用于量能过滤、过热保护等规则判断的“策略热度”。它可能混合了 core/satellite 结构，所以不一定等于单一 ETF 的动量。

- 为什么要拆成两个字段
  之前默认基线把内部组合动量直接写到了 `current_momentum`，会让人工看盘时误以为“信号 ETF 的动量”比实际更低。现在已经拆开，策略逻辑不变，但输出口径更清晰。

- 当前口径校验
  以 `2026-05-26` 为例，`current_momentum` 和 `signal_asset_momentum` 都是 `14.31%`，而 `effective_momentum` 是 `11.71%`。这说明“用户看到的信号动量”和“策略内部热度”已经正确拆开。

## 本次基线切换说明

当前正式默认基线已经切到 `slope_085` 版本。

这次切换只改了一件核心事情：

- 风险池第一名不再只按原始 `25` 日动量排序
- 而是优先选“近 25 日够强，同时近 5 日没有陡峭过热”的标的

可以把它理解成：

- 旧版更容易把“最后几天突然急拉”的标的排到第一
- `slope_085` 会对这种短期斜率过高的信号扣分
- 因此更偏向“趋势仍强，但路径更平滑”的领涨资产

这层改动不会改变：

- dual momentum 的 `5%` 绝对阈值
- regime mix 的仓位框架
- 量能保护和市场广度保护
- 弱市切债和高位过热降仓

它只改变“风险池里谁是第一名”的排序方式。

## 交易记录怎么读

交易记录文件：

- `output/core/trades.csv`

常见字段：

- `date`
  发生交易的日期。

- `action`
  交易动作：
  - `BUY`
  - `SELL`
  - `ADD`
  - `REDUCE`

- `code`
  ETF 代码。

- `theme`
  主题名。

- `name`
  ETF 名称。

- `from_exposure`
  调整前仓位。

- `to_exposure`
  调整后仓位。

- `nav`
  当日交易发生时的策略净值。

阅读方式：

- 同一天出现 `SELL` + `BUY`
  通常表示换仓。

- 只出现 `ADD` / `REDUCE`
  表示标的不变，只是仓位变了。

- `from_exposure` 到 `to_exposure`
  就是那次仓位变化的具体数值。

## `output/` 目录分工

- `output/core/`
  正式基线的主输出。`run_backtest.py` 会直接覆盖这里；默认 `daily_monitor.py` 只会用历史收盘数据刷新这里，不会把盘中实时快照写进来。

- `output/analysis/`
  归因、回撤分析等派生结果。

- `output/research/`
  各种实验、对比、搜参结果。不会影响正式基线。
  当前活跃研究结果优先放在各自子目录或少量仍在使用的根目录平面文件中。
  历史归档实验留下的散落根目录文件已经移到 `output/research/archive_flat/`。

- `output/monitor/`
  每日巡检缓存，例如去重状态、交易日历缓存，以及盘中快照运行时依赖的数据。

- `archive/experiments/`
  已退出日常使用路径的历史实验脚本。用于回看过去结论，不建议作为新基线研究入口。

- `output/research/archive_flat/`
  存放旧实验在历史目录结构下生成的散落结果文件，不作为当前策略分析入口。

## 研究脚本怎么用

共同特征：

- 默认不覆盖 `output/core/`
- 活跃研究脚本通常写到 `output/research/<name>/` 这类独立子目录，少量仍在使用的旧入口会继续写在 `output/research/` 根目录平面文件
- 已归档到 `archive/experiments/` 的历史脚本，整理后默认写到 `output/research/archive_flat/<script_name>/`
- 常见输出是 `summary.csv`、`best.json`、`nav_compare.csv`

常用入口：

- `compare_current_best_fine_tune.py`
  围绕当前正式默认策略做小范围局部搜索。优先用它验证小改动是否真的有效。

- `compare_market_proxy_variants.py`
  对比不同的大盘量能/广度代理构造方式。已经切到当前正式默认策略，不再围绕旧通知基线。

- `compare_current_best_pool_additions.py`
  对当前正式默认策略测试候选池增删替换，不会直接改动正式基线。

- `compare_goal_optimizations.py`
  更大的增量搜索入口。候选很多，运行时间更长，适合做阶段性全局搜索，不适合日常频繁跑。

下面这些脚本更多是支撑模块或专项对照：

- `compare_defensive_persistence.py`
- 以及其他未在这里单独点名的 `compare_*.py`

当前已经归档的历史脚本在 `archive/experiments/`，例如：

- `archive/experiments/compare_lookback_range.py`
- `archive/experiments/compare_multihorizon_combo.py`
- `archive/experiments/compare_momentum_quality_variants.py`
- `archive/experiments/compare_cash_overlay_candidates.py`
- `archive/experiments/compare_defensive_bucket_blends.py`
- `archive/experiments/compare_defensive_trigger_refinements.py`
- `archive/experiments/compare_energy_income_candidates.py`
- `archive/experiments/compare_more_candidates.py`
- `archive/experiments/compare_simple_bond_overlay.py`
- `archive/experiments/compare_resource_energy_common_sense.py`
- `archive/experiments/compare_stressbond_nearmiss_repair.py`
- `archive/experiments/compare_two_tier_defense.py`
- `archive/experiments/tune_threshold_dual_momentum.py`

常见参数：

- `--years`
  研究区间长度。

- `--refresh`
  不用本地缓存，重新抓价格。

- `--fee-rate`
  手续费。

- `--slippage-rate`
  滑点。

- `--notify`
  如果找到比当前脚本自身比较基线更优的策略，就通过 webhook 发送机器人消息。

- `--webhook-url`
  临时指定 webhook。

示例：

```bash
.venv/bin/python momentum_backtest/compare_current_best_fine_tune.py --years 15
.venv/bin/python momentum_backtest/archive/experiments/compare_lookback_range.py --years 10
```

## 分析脚本怎么用

- 收益贡献：

```bash
.venv/bin/python momentum_backtest/analyze_contribution.py
```

- `output/analysis/contribution_summary.csv`

- 回撤拆解：

```bash
.venv/bin/python momentum_backtest/analyze_drawdowns.py
```

- `output/analysis/drawdown_episodes.csv`

- 回撤图：

  当前默认由 `.venv/bin/python momentum_backtest/run_backtest.py` 自动刷新到 `output/analysis/historical_drawdown.png` 与 `output/analysis/nav_and_drawdown.png`。

- 历史分支完整落盘：

```bash
.venv/bin/python momentum_backtest/archive/experiments/run_china_internet_cap70_backtest.py
.venv/bin/python momentum_backtest/archive/experiments/run_resource_abs08_backtest.py
```

## 典型工作流

- 看正式结果：
  - 运行 `.venv/bin/python momentum_backtest/run_backtest.py`
  - 重点读 `output/core/backtest_nav.csv`、`output/core/trades.csv`

- 看当天信号：
  - 运行 `.venv/bin/python momentum_backtest/daily_monitor.py --force --disable-direct-feishu`
  - 直接看终端打印或飞书消息

- 做小范围策略试验：
  - 运行 `.venv/bin/python momentum_backtest/compare_current_best_fine_tune.py --years 15`
  - 重点读 `output/research/goal_optimizations/current_best_fine_tune/summary.csv`

## 当前目录里最值得直接看的文件

- `run_backtest.py`
  默认正式基线定义。

- `daily_monitor.py`
  实盘巡检和消息格式。

- `output/core/backtest_nav.csv`
  正式基线的完整时间序列。

- `output/core/trades.csv`
  正式基线交易记录。

- `output/research/current_best_pool_additions/summary.csv`
  当前正式基线加减候选池后的有效变化汇总。

- `output/research/goal_optimizations/current_best_fine_tune/summary.csv`
  围绕当前正式基线的小范围微调结果。

- `output/research/goal_optimizations/market_proxy_variants/summary.csv`
  大盘量能/广度代理口径的对照结果。
