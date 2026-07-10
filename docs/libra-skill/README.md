# Libra 看数流程总结

## 目标

把这次从无法访问 Libra，到完成 `i18n-tt` 授权并成功读取实验和报表的流程沉淀成可复用步骤。

## 结论

- 当前环境里，优先使用 `bytedcli` 访问 Libra，不再依赖 `gdpa-cli`。
- 访问 TikTok ROW / `tiktok-row` 相关实验时，优先使用 `--site i18n-tt`。
- 先完成 `i18n-tt` 授权，再读取实验和报表。
- Libra 报表命令使用 `--metric-group`，不是 `--metric-group-id`。
- 刚启动的实验即使能读到报表结构，也可能因为数据包未导入而所有值为 `null`。

## 适用场景

- 用户给出 Libra 实验链接，希望确认是否能访问。
- 用户给出 `flight_id`，希望查看实验详情、traffic、report。
- 用户要看 TikTok Web / ROW 实验数据。
- 用户要确认 DNU、CTR、CVR 这类指标在哪个 metric group 里。

## 实操步骤

### 1. 判断站点和实验标识

从链接中提取：

- `flight_id`
- `group_id`
- 站点信息

如果链接或上下文出现：

- `tiktok-row`
- `libra-sg`
- `ROW`
- `TikTok`

优先按 `i18n-tt` 处理。

### 2. 检查 bytedcli 和授权状态

先确认 `bytedcli` 可用：

```bash
bytedcli --version
```

再检查授权状态：

```bash
bytedcli --json --site i18n-tt auth status
```

如果返回 `need_login`，继续授权。

### 3. 发起 i18n 授权

使用非阻塞登录：

```bash
bytedcli --json --site i18n-tt auth login --begin
```

关键返回字段：

- `verification_uri_complete`
- `complete_token`

打开授权页，用户在浏览器完成确认。

### 4. 完成授权

浏览器确认后执行：

```bash
bytedcli --json --site i18n-tt auth login --complete <complete_token>
```

成功后再次检查：

```bash
bytedcli --json --site i18n-tt auth status
```

预期结果是 `authenticated: true`。

### 5. 读取实验详情

```bash
bytedcli --json --site i18n-tt libra experiment get --flight-id <flight_id>
```

重点关注：

- 实验名
- owner
- app / app_name
- versions
- metrics
- truly_effected_regions
- start_time

### 6. 读取报表

先看命令帮助，避免猜参数：

```bash
bytedcli libra experiment report --help
```

正确命令：

```bash
bytedcli --json --site i18n-tt libra experiment report --flight-id <flight_id> --metric-group <group_id>
```

注意：

- 参数名是 `--metric-group`
- 不是 `--metric-group-id`

### 7. 判断为什么没有数

如果报表返回：

- `merge_data` 里值全是 `null`
- `miss_data_pack` 有地区缺数
- `marked_rows` 提示机房数据未导入
- `data_status` 表示数据未 ready

说明链路没问题，只是数据尚未产出。

这次实验就是这种情况：

- 实验在 `2026-05-25` 启动
- 报表可读，但当天数据未导入
- 因此当前所有值是 `null`

## 这次实验的关键指标定位

实验：

- `flight_id=72036188`
- 名称：`[SEO] video页 add sim text recall-new`

### DNU 相关

metric group:

- `7039914` `seo_page_server_call_to_dnu_cvr`

核心指标：

- `dnu`
- `dc_dnu`
- `server_call_to_dnu_cvr`
- `server_call_to_dc_dnu_cvr`
- `base_user_to_dnu_cvr`
- `onelink_install_to_dnu_cvr`

### CTR / CVR 相关

metric group:

- `7039720` `Share Reflow Page Tap Target Conversion`

核心指标：

- `CTR (UV)` = `click UV / page view UV`
- `CTR (PV)` = `click PV / page view PV`
- `CVR (UV)` = `Onelink Install UV / Onelink Click UV`
- `CVR (PV)` = `Onelink Install PV / Onelink Click PV`

可进一步按 `tap_target_group` 拆分：

- `ALL Targets`
- `Launch Popup`
- `Login Popup`

## 常见坑

1. 只做了 CN 授权，没有做 `i18n-tt` 授权，导致 TikTok ROW 的 Libra 仍然无法访问。
2. 误以为命令参数是 `--metric-group-id`，实际应为 `--metric-group`。
3. 看到报表全是 `null` 就以为权限有问题，实际上可能只是离线数据尚未导入。
4. 只看 `important` 指标组，忽略了真正业务关注的 `watching` 指标组。

## 推荐命令模板

查看授权状态：

```bash
bytedcli --json --site i18n-tt auth status
```

查看实验：

```bash
bytedcli --json --site i18n-tt libra experiment get --flight-id 72036188
```

查看 DNU 组：

```bash
bytedcli --json --site i18n-tt libra experiment report --flight-id 72036188 --metric-group 7039914
```

查看 CTR / CVR 组：

```bash
bytedcli --json --site i18n-tt libra experiment report --flight-id 72036188 --metric-group 7039720
```

## 判断标准

先判断“能不能读”：

- 授权是否成功
- experiment get 是否成功
- report 是否返回结构

再判断“有没有数”：

- 数据日期是否 ready
- 是否有地区数据缺失
- `merge_data` 是否仍为 `null`

这样可以把“权限问题”和“数据未产出”分开。
