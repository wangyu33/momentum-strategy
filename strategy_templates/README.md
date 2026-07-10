# strategy_templates

这个目录用于沉淀可复用的策略模板与调研结论。

当前已补充：

- `04_etf_defensive_momentum_overlay/README.md`
  - 基于 `momentum_backtest/` 当前正式实现与研究结果整理的 ETF 防守策略调研
  - 重点总结：防守池轮动、弱市切债、量能/广度压仓、过热降仓、top2 接近压仓
- `05_global_etf_defensive_basic_strategy/README.md`
  - 国际上宽基ETF/债券ETF常见防守策略综述，并给出一版基础策略草案
- `05_global_etf_defensive_basic_strategy/strategy.yaml`
  - 对应的基础策略配置模板
- `06_domestic_etf_defensive_basic_strategy/README.md`
  - 结合当前仓库与国内ETF市场特征整理的A股/国内ETF防守基础策略
- `06_domestic_etf_defensive_basic_strategy/strategy.yaml`
  - 对应的国内版基础策略配置模板
- `07_domestic_etf_defensive_conservative_strategy/README.md`
  - 更保守版国内ETF防守策略：加入511580、现金替代、周频调仓
- `07_domestic_etf_defensive_conservative_strategy/strategy.yaml`
  - 对应的保守版国内策略配置模板
