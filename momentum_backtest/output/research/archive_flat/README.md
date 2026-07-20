# Archive Flat Outputs

这个目录存放历史归档实验在旧目录结构下直接写到 `output/research/` 根目录的散落结果。

这批文件保留，但默认不建议当成当前策略分析入口，原因是：

- 它们不属于当前正式基线输出。
- 它们大多来自 `archive/experiments/` 下已经退出日常使用路径的旧实验脚本。
- 如果继续留在 `output/research/` 根目录，会和当前活跃研究结果混在一起，阅读噪声很大。

如果需要复盘某个历史实验，优先运行对应的 `archive/experiments/*.py` 脚本，而不是直接把这里的旧文件当成最新结论。
