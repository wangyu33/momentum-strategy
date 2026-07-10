# 动量策略任务路径与同步说明

更新时间：2026-06-02

## 当前文档版 / 原工作版位置
- 原工作目录（当前）：`/Users/bytedance/Documents/trae_projects/wy_test`
- 策略主目录：`/Users/bytedance/Documents/trae_projects/wy_test/momentum_backtest`

## 迁移目的
当前 LaunchAgent 直接读取 `Documents/` 下脚本时，触发了 macOS 权限限制（`Operation not permitted`）。
因此需要迁移到非 `Documents/` 目录作为稳定运行版。

## 稳定运行版位置
- 目标稳定目录：`~/Library/Caches/wy_test_runtime`
- 迁移完成后，需要把下面两项一起更新：
  1. LaunchAgent plist 路径
  2. 本文档里的“稳定运行版位置”

## 后续修改同步规则
在迁移完成后，若存在“文档版/编辑版”和“稳定运行版”两个目录，修改时必须注意：

1. **先确认当前改动发生在哪个目录**
   - 文档版/编辑版
   - 稳定运行版

2. **凡是影响正式通知、正式回测、LaunchAgent 的改动**
   都必须同步到稳定运行版。

3. **同步后要做最少校验**
   - `python3 momentum_backtest/validate_strategy_outputs.py`
   - 如涉及正式输出，再跑正式回测
   - 如涉及通知，再做一次人工校验；人工补发默认使用 `python3 momentum_backtest/daily_monitor.py --preview-send`

4. **如果只改文档或研究脚本**
   不必强制同步到稳定运行版，但要在提交说明里写清楚。

## 迁移后待补充内容
- 稳定运行版真实路径：`~/Library/Caches/wy_test_runtime`
- LaunchAgent monitor 实际脚本路径：`~/Library/Caches/wy_test_runtime/momentum_backtest/daily_monitor.py`
- LaunchAgent board 实际脚本路径：`~/Library/Caches/wy_test_runtime/momentum_backtest/daily_momentum_board.py`
- LaunchAgent backfill 实际脚本路径：`~/Library/Caches/wy_test_runtime/momentum_backtest/run_backtest.py`
- 若保留双目录，谁是“主编辑目录”：当前仍以 Documents 下目录作为主编辑目录；每次影响正式运行的改动，需同步到 `~/Library/Caches/wy_test_runtime` 后再重载 LaunchAgent


## 当前定时触发时间
- monitor 触发时间：工作日 09:40、12:10、14:50
- momentum board 触发时间：工作日 15:10
- backfill 触发时间：工作日 15:20

- 重载定时任务若已错过当日定时点，`scripts/sync_runtime_and_reload.sh` 会按 09:40 / 12:10 / 14:50 / 15:10 / 15:20 五个正式时点做一次补跑检查，并通过运行版输出文件 mtime + 当日补跑 stamp 避免重复补发。
