# SEO 页面分流实验分析：72025174

## 实验说明

- `flight_id=72025174`
- 实验名：`KEP_BOT_多因子测试v6`
- 模块：`SEO 页面分流`
- 页面模块：`discover_kw`
- 机房范围：`SG`、`VA`、`TX`、`EU_TTP`
- 分析重点：`DAU`、`DNU`、`Google impression`、`Google click`
- 机房映射：
  - `ROW = other`
  - `EU = eu_ttp`
  - `TTP = tx`
- 当前报表口径：`2026-05-24` 单日累计数据

## 版本定义

- `v0`：对照组，`76144135`
- `v1`：实验组 1，`76144136`
- `v2`：实验组 2，`76144137`
- `v3`：实验组 3，`76144138`

## 先说结论

### 1. 这次实验没有拿到支持上线的证据

如果按 SEO 页面分流的核心目标看，这次实验没有形成正向闭环：

- `PC Web DAU/Page` 在整体和三地机房都下降
- `APP DNU/Page` 在整体和三地机房都下降
- `v3` 在三地几乎都是最差版本
- 只有 `TTP` 的 `v1` 在 `Mobile Web DAU/Page` 上有明显正向，但没有带动 `DNU`

因此，这个实验不能支持“SEO 页面分流能稳定提升 DAU / DNU”。

### 2. Google 指标证据不完整

`Google impression/click` 在实验里是有挂载的，但数据返回存在明显地区差异：

- `ROW` 有值，且三个实验组全部下降
- `EU` 返回 `null`
- `TTP` 返回 `null`

所以目前只能确认：

- `ROW` 的 Google 曝光和点击都在下降
- `EU` / `TTP` 还不能做 Google 流量判断

这意味着目前无法证明 SEO 页面分流带来了更好的搜索上游流量。

### 3. 如果一定要选版本，`v2` 相对最平，但仍不建议上线

- `v1`：只在 `TTP Mobile Web DAU/Page` 上有明显收益，但 `PC DAU`、`DNU`、`ROW Google` 都是负向
- `v2`：整体最平，移动 DAU 不再明显增长，但 `DNU` 下降相对更轻
- `v3`：DAU / DNU / ROW Google 指标都明显更差

结论是：

- 不建议上线 `v1`
- 不建议上线 `v2`
- 明确不建议上线 `v3`

如果只是做相对排序，`v2 > v1 >> v3`。

## 整体结果

### DAU / DNU

以 `v0` 为基线：

| 指标 | v1 vs v0 | p-value | v2 vs v0 | p-value | v3 vs v0 | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| PC Web DAU/Page | `-4.47%` | `0` | `-4.44%` | `0` | `-10.55%` | `0` |
| Mobile Web DAU/Page | `+11.20%` | `0.0014` | `-0.80%` | `0.9938` | `-9.08%` | `0.0159` |
| APP DNU/Page | `-2.84%` | `0.0103` | `-2.42%` | `0.0404` | `-10.77%` | `0` |
| APP DC DNU/Page | `-3.22%` | `0.0440` | `-1.74%` | `0.4896` | `-8.61%` | `1.58e-11` |

解读：

- `v1` 的主要优点是整体 `Mobile Web DAU/Page` 有提升，但它没有传导到 `DNU`
- `v2` 是更接近中性的版本，但 `PC DAU` 和 `DNU` 仍然在跌
- `v3` 是明确负向版本

### Google Crawled Count/Page

这个指标只能作为补充信号，不能替代真正的搜索流量结果。

| 指标 | v1 vs v0 | p-value | v2 vs v0 | p-value | v3 vs v0 | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| Google Crawled Count/Page | `+0.22%` | `1.15e-9` | `+0.24%` | `1.28e-11` | `+0.82%` | `0` |
| Google Mobile Crawled Count/Page | `+0.21%` | `1.27e-11` | `+0.23%` | `5.77e-14` | `+0.84%` | `0` |
| Google PC Crawled Count/Page | `+0.16%` | `0.1356` | `+0.18%` | `0.0700` | `+0.55%` | `3.54e-13` |

解读：

- 抓取量整体确实有轻微提升，`v3` 提升最大
- 但抓取量提升并没有转成更好的 `DAU / DNU`
- 因为核心 `Google impression/click` 数据不完整，所以不能把抓取提升当成成功证据

## 分机房分析

## ROW

### DAU / DNU

| 指标 | v1 vs v0 | p-value | v2 vs v0 | p-value | v3 vs v0 | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| PC Web DAU/Page | `-4.27%` | `0` | `-4.17%` | `0` | `-9.52%` | `0` |
| Mobile Web DAU/Page | `-0.54%` | `0.8861` | `+0.18%` | `0.9947` | `-7.30%` | `0` |
| APP DNU/Page | `-2.20%` | `0.4316` | `-1.52%` | `0.7236` | `-9.39%` | `6.66e-10` |

### Google impression / click

| 指标 | v1 vs v0 | p-value | v2 vs v0 | p-value | v3 vs v0 | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| GG mobile impression/Page | `-7.13%` | `4.44e-16` | `-6.99%` | `2.66e-15` | `-12.64%` | `0` |
| GG PC impression/Page | `-9.84%` | `0` | `-9.24%` | `0` | `-17.68%` | `0` |
| GG mobile clicks/Page | `-2.36%` | `0.00036` | `-2.37%` | `0.00034` | `-8.79%` | `0` |
| GG PC clicks/Page | `-5.57%` | `0` | `-5.04%` | `0` | `-11.42%` | `0` |

### ROW 结论

- `ROW` 是这次实验最明确的负向地区
- `PC DAU` 下滑明显
- `Google impression/click` 四个核心指标全部下降，且显著
- `v3` 在 ROW 明显最差

## EU

### DAU / DNU

| 指标 | v1 vs v0 | p-value | v2 vs v0 | p-value | v3 vs v0 | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| PC Web DAU/Page | `-3.22%` | `1.86e-13` | `-3.30%` | `4.66e-14` | `-7.03%` | `0` |
| Mobile Web DAU/Page | `-2.65%` | `0.0000035` | `-3.23%` | `6.84e-9` | `-8.46%` | `0` |
| APP DNU/Page | `-6.91%` | `0.0852` | `-3.80%` | `0.5633` | `-9.50%` | `0.0064` |

### Google impression / click

- `EU` 下 `Google impression/click` 当前全部返回 `null`
- 因此无法判断 SEO 页面分流对 EU 搜索曝光和点击是否有改善

### 补充看抓取量

- `Google Crawled Count/Page`
  - `v1`: `+0.37%`, `p=0.2822`
  - `v2`: `+0.43%`, `p=0.1509`
  - `v3`: `+1.40%`, `p=5.95e-11`

### EU 结论

- `EU` 是全面走弱地区
- `PC DAU`、`Mobile DAU`、`DNU` 方向都偏负
- `v3` 在 EU 也最差
- 即使抓取量有提升，也没有转成活跃和新增结果

## TTP

### DAU / DNU

| 指标 | v1 vs v0 | p-value | v2 vs v0 | p-value | v3 vs v0 | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| PC Web DAU/Page | `-4.76%` | `0` | `-4.74%` | `0` | `-11.45%` | `0` |
| Mobile Web DAU/Page | `+18.67%` | `0.0030` | `-0.76%` | `0.9990` | `-9.56%` | `0.2867` |
| APP DNU/Page | `-2.58%` | `0.1810` | `-2.73%` | `0.1416` | `-11.24%` | `0` |

### Google impression / click

- `TTP` 下 `Google impression/click` 当前全部返回 `null`
- 因此无法判断 SEO 页面分流对 TTP 搜索曝光和点击是否有改善

### 补充看抓取量

- `Google Crawled Count/Page`
  - `v1`: `+0.03%`, `p=0.7439`
  - `v2`: `+0.05%`, `p=0.4462`
  - `v3`: `+0.21%`, `p=4.73e-9`

### TTP 结论

- `TTP` 唯一的亮点是 `v1` 的 `Mobile Web DAU/Page` 明显增长
- 但这个增长没有带来 `DNU` 提升，且 `PC DAU` 仍然明显下降
- 因此 `TTP` 也不能支持上线

## 版本判断

### v1

- 优点：整体和 `TTP` 的 `Mobile Web DAU/Page` 有正向
- 缺点：`PC DAU`、`DNU` 走弱，`ROW Google impression/click` 全部显著下降
- 结论：不建议上线

### v2

- 优点：整体最平，属于“损伤最小”的版本
- 缺点：没有清晰正向收益，`PC DAU` 和 `DNU` 仍是负向
- 结论：相对最优，但仍不建议上线

### v3

- 优点：抓取量提升最大
- 缺点：`DAU`、`DNU`、`ROW Google` 都最差
- 结论：明确不建议上线

## 最终结论

从 SEO 页面分流目标看，这次实验当前只能得出：

1. `DAU / DNU` 没有形成可支持上线的正向结果。
2. `ROW` 的 `Google impression/click` 明确下降，是强负向信号。
3. `EU` / `TTP` 的 Google 指标仍为 `null`，证据链不完整。
4. `Google crawl` 提升只能视为补充信号，不能覆盖核心业务指标的负向。

因此当前建议是：

- 不上线当前三个实验组
- 如果后续还要继续试，优先基于 `v2` 做更小步改动
- 下一轮必须补齐 `EU / TTP` 的 `Google impression/click` 数据，再做 SEO 成效判断
