# SEO 页面分流实验分析：72022691

## 说明

本分析基于当前上下文，默认实验为：

- `flight_id=72022691`
- 实验名：`[SEO] video页 add sim text recall`

分析重点按你的要求收敛到：

- `DAU`
- `DNU`
- `google impression`
- `google click`

并按机房拆分：

- `ROW` = `other`
- `EU` = `eu_ttp`
- `TTP` = `tx`

## 先说结论

### 1. 这次实验在 SEO 关注指标上，没有拿到支持上线的证据

从 Libra 当前挂载指标来看，`SEO DNU/User`、`SEO Backreflow DAU/User`、`SEO Monthly-active First DAU/User` 在整体和分机房口径下都没有形成统计显著的正向收益。

也就是说：

- 没有证据证明实验稳定提升了 SEO 相关 DNU
- 也没有证据证明实验稳定提升了 SEO 相关 DAU

### 2. `google impression / google click` 当前在该实验挂载指标里没有找到

我扫描了当前实验挂载的所有 metric group，未发现直接对应以下指标的 metric：

- `google impression`
- `google click`

因此，**当前无法仅凭这个 Libra 实验直接评估 Google 入口流量是否提升**。  
如果这两个指标是本次功能的核心验收目标，那么这次实验的证据链是不完整的。

### 3. 分机房看，没有哪个机房给出强正向信号

- `ROW`：整体稳定，略偏中性
- `EU`：部分 DNU 方向微正，但 DAU 方向偏弱，且都不显著
- `TTP`：DNU 方向偏负，DAU 也没有稳定正向

所以，按 SEO 页面分流的目标来看，这次实验更接近“**无显著收益**”，而不是“可确认有效”。

## 实验版本

- `v1`：对照组
- `v2`：使用 tier3 作为细分召回
- `v3`：不使用 tier3，开启 first_video_emb
- `v4`：在 v3 基础上增加相关性 boost

## 指标口径说明

当前实验挂出来的 SEO 相关重点指标主要在 `metric_group=7050696`：

- `DNU/User`
- `SEO DNU/User`
- `Backreflow DAU/User`
- `SEO Backreflow DAU/User`
- `Monthly-active First DAU/User`
- `SEO Monthly-active First DAU/User`

这里最贴近 SEO 页面分流目标的，我重点看：

- `SEO DNU/User`
- `SEO Backreflow DAU/User`
- `SEO Monthly-active First DAU/User`

同时保留：

- `DNU/User`
- `Backreflow DAU/User`
- `Monthly-active First DAU/User`

作为辅助参考。

## 整体结果

### SEO DNU/User

基线 `v1`：

- `0.0046903`

实验组：

- `v2`: `0.0047005`, `+0.22%`, `p=0.9937`
- `v3`: `0.0047075`, `+0.37%`, `p=0.9712`
- `v4`: `0.0047202`, `+0.64%`, `p=0.8682`

解读：

- 三个实验组方向上都略正，但全部不显著。
- 不能据此认定 SEO DNU 有提升。

### SEO Backreflow DAU/User

基线 `v1`：

- `0.0023579`

实验组：

- `v2`: `0.0023699`, `+0.51%`, `p=0.9712`
- `v3`: `0.0023407`, `-0.73%`, `p=0.9210`
- `v4`: `0.0023255`, `-1.37%`, `p=0.6322`

解读：

- `v2` 略正，`v3/v4` 略负，但都不显著。
- 说明 SEO 回流类 DAU 没有稳定提升。

### SEO Monthly-active First DAU/User

基线 `v1`：

- `0.0084151`

实验组：

- `v2`: `0.0083858`, `-0.35%`, `p=0.9447`
- `v3`: `0.0084277`, `+0.15%`, `p=0.9952`
- `v4`: `0.0083964`, `-0.22%`, `p=0.9846`

解读：

- 几乎没有变化。
- 无法支持“SEO 首次活跃 DAU 被改善”的结论。

## 分机房分析

## ROW 机房

### SEO DNU/User

- `v2`: `+0.89%`, `p=0.8146`
- `v3`: `+0.72%`, `p=0.8903`
- `v4`: `+0.60%`, `p=0.9347`

### SEO Backreflow DAU/User

- `v2`: `+0.74%`, `p=0.9576`
- `v3`: `-1.15%`, `p=0.8577`
- `v4`: `-2.13%`, `p=0.4592`

### SEO Monthly-active First DAU/User

- `v2`: `-0.49%`, `p=0.9249`
- `v3`: `+0.65%`, `p=0.8452`
- `v4`: `+0.46%`, `p=0.9373`

### ROW 结论

- `ROW` 机房整体最接近“中性”。
- DNU 有轻微正向，但 DAU 没有同步且稳定改善。
- 没有一个版本在 ROW 上同时把 DNU 和 DAU 都做成可确认的正向。

## EU 机房

### SEO DNU/User

- `v2`: `+1.88%`, `p=0.8211`
- `v3`: `-0.46%`, `p=0.9966`
- `v4`: `+3.03%`, `p=0.5019`

### SEO Backreflow DAU/User

- `v2`: `+2.39%`, `p=0.7606`
- `v3`: `-0.52%`, `p=0.9965`
- `v4`: `-0.83%`, `p=0.9862`

### SEO Monthly-active First DAU/User

- `v2`: `-0.56%`, `p=0.9774`
- `v3`: `+0.35%`, `p=0.9942`
- `v4`: `-2.19%`, `p=0.3876`

### EU 结论

- `EU` 机房没有形成稳定正向。
- `v4` 的 SEO DNU 方向最好，但 DAU 并没有同步改善。
- 从 SEO 业务角度看，EU 更像是“方向混杂、证据不足”，不能支持上线。

## TTP 机房

### SEO DNU/User

- `v2`: `-3.67%`, `p=0.2365`
- `v3`: `-0.12%`, `p=0.9999`
- `v4`: `-1.03%`, `p=0.9518`

### SEO Backreflow DAU/User

- `v2`: `-3.26%`, `p=0.6907`
- `v3`: `+0.77%`, `p=0.9938`
- `v4`: `+1.09%`, `p=0.9831`

### SEO Monthly-active First DAU/User

- `v2`: `+0.42%`, `p=0.9922`
- `v3`: `-1.72%`, `p=0.6495`
- `v4`: `-0.16%`, `p=0.9996`

### TTP 结论

- `TTP` 是 SEO DNU 最弱的机房。
- `v2` 在 TTP 上 SEO DNU 明显偏负。
- `v3/v4` 虽然把 DAU 拉回一点，但也没有显著性支撑。

## 版本判断

### v2

- 优点：ROW / EU 的 SEO DNU 略正
- 问题：TTP 的 SEO DNU 明显偏负；DAU 没有同步改善
- 结论：不建议作为上线版本

### v3

- 优点：整体最平，不算极端；SEO Monthly-active First DAU 在整体和 ROW/EU 上略正
- 问题：SEO DNU 提升幅度很小，且全不显著
- 结论：三组里相对最均衡，但仍然不能支持上线

### v4

- 优点：整体 SEO DNU 数值最高
- 问题：DAU 不跟随，EU/TTP 也没有形成可确认改善
- 结论：不建议上线

## 关于 Google 指标的结论

当前实验在 Libra 挂载的 metric group 中，没有发现直接对应：

- `google impression`
- `google click`

因此本次分析不能回答下面这个关键问题：

- SEO 页面分流是否真的给 Google 入口带来了更多曝光和点击？

这意味着：

1. 如果这两个 Google 指标是核心目标，那么当前实验配置不完整。
2. 即使 DNU / DAU 勉强持平，也不能说明 SEO 页面分流真的成功。
3. 后续决策前，建议补齐 Google 侧指标再做判断。

## 最终结论

从“SEO 页面分流”这个模块目标看，这次实验目前只能得出：

- `SEO DNU/User` 没有显著提升
- `SEO 相关 DAU` 没有显著提升
- `google impression / google click` 当前未挂载，无法验证 SEO 上游收益

所以，**这次实验不具备明确正向证据，不建议据此直接上线。**

## 建议动作

1. 后续实验必须补挂 `google impression` 和 `google click` 指标。
2. 若继续沿当前版本迭代，优先参考 `v3`，因为它相对更均衡。
3. 单独复查 `TTP` 机房的 SEO DNU，避免在该区域继续放大负向风险。
4. 在下一轮实验中，把 SEO 目标拆成两层：
   - 上游：Google impression / click
   - 下游：SEO DNU / SEO DAU

只有两层都成立，才能证明 SEO 页面分流真的有效。
