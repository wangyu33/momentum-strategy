# domestic_etf_defensive_backtest

这是一套独立于 `momentum_backtest/` 主目录的国内 ETF 防守策略回测目录。

包含：

- `run_backtest.py`
  - 独立回测入口，支持 `base` / `conservative` 两个预设
- `configs/base.yaml`
  - 国内 ETF 防守基础版配置
- `configs/conservative.yaml`
  - 更保守版配置（加入 511580 / 现金替代 / 周频调仓）
- `output/<preset>/core/`
  - 类似 momentum_backtest/output/core 的主结果目录
- `output/<preset>/analysis/`
  - 类似 momentum_backtest/output/analysis 的分析目录

示例：

```bash
python3 domestic_etf_defensive_backtest/run_backtest.py --preset base --years 15
python3 domestic_etf_defensive_backtest/run_backtest.py --preset conservative --years 15
python3 domestic_etf_defensive_backtest/run_backtest.py --preset all --years 15
```
