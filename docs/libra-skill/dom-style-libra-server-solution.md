# DOM Style Libra 服务端自动化方案

## 目标

把 DOM 样式探索里的服务端部分落成一条可执行链路：

1. 接收算法生成的候选样式策略。
2. 自动创建 Libra 实验草稿。
3. 自动提交 review，并按配置自动开实验。
4. 回收实验详情、traffic、报表和维度拆解结果。
5. 通过 SQL 将 `flight_id` / `review_id` / 批次信息写入 Hive。
6. 将负向结果、止损结果回写给算法做下一轮剪枝。

## 推荐分层

- `experiment_orchestrator`
  负责批次编排、参数校验、调用 CLI 或 Libra OpenAPI。
- `config_dispatcher`
  负责把 `strategy_id -> SmartUI config` 的映射落到配置中心。
- `result_collector`
  负责读取 Libra 元数据、traffic、report 和维度拆解。
- `risk_controller`
  负责止损、pause / close、异常回写。

## 输入输出约定

### 输入

```json
{
  "scene": "video_page_dom_style",
  "site": "i18n-tt",
  "app_id": 22,
  "phase": "cold_start",
  "owner": "alice",
  "layer_name": "seo_dom_style_video",
  "metric_groups": [7039914, 7039720],
  "strategies": [
    {
      "strategy_id": "s001",
      "version_name": "v2_h3_strong",
      "traffic_ratio": 0.001,
      "unit_template_mapping": {
        "video_title": "h2",
        "video_desc": "strong"
      }
    }
  ]
}
```

### 输出

```json
{
  "batch_id": "dom_style_20260527_round1",
  "flight_id": 72036188,
  "review_id": 88001234,
  "launch_mode": "auto",
  "version_strategy_mapping": {
    "v2_h3_strong": "s001"
  },
  "smartui_config_version": "20260527_01"
}
```

## CLI 工作流

### 1. 鉴权

```bash
bytedcli --json --site i18n-tt auth status
```

如果未授权，先完成登录，再继续。

### 2. 参考已有实验结构

新建实验不要从零拼 payload。优先拿一个同 app、同 layer、同实验类型的实验做结构参考：

```bash
mkdir -p artifacts/libra

bytedcli --json --site i18n-tt libra experiment get \
  --flight-id 72036188 \
  > artifacts/libra/source-flight.json
```

重点观察：

- `name`
- `owner`
- `description`
- `duration`
- `layer_info`
- `versions`
- `filter_rule`
- `metrics`

### 3. 生成创建请求

优先用 `--request-file`，不要把大 JSON 直接塞进命令行：

```bash
bytedcli --json --site i18n-tt libra experiment create \
  --app-id 22 \
  --request-file docs/libra-skill/examples/dom_style_libra_create_request.json \
  > artifacts/libra/create-response.json
```

创建完成后，从响应里提取：

- `flight_id`
- `review_id`，如果返回里已有
- `app_id`
- `versions`

### 4. 自动开实验

如果希望 review 通过后自动开始，直接在提交 review 时指定：

```bash
bytedcli --json --site i18n-tt libra experiment submit-review \
  --flight-id <flight_id> \
  --reviewers alice,bob \
  --description 'dom style batch round1' \
  --auto-launch-mode auto \
  > artifacts/libra/review-response.json
```

说明：

- `auto`：review 通过且 block check 全部完成后自动 running。
- `manual`：review 通过后仍需手动 `release`。
- `timer`：到指定时间自动启动，需要配 `scheduled_start_time`。

如果是 `manual` 模式，再执行：

```bash
bytedcli --json --site i18n-tt libra experiment release --flight-id <flight_id>
```

### 5. 轮询 review 状态

```bash
bytedcli --json --site i18n-tt libra experiment review-status \
  --flight-id <flight_id> \
  --review-id <review_id> \
  > artifacts/libra/review-status.json
```

只要有 `is_block=true` 且状态未通过，就不能认为实验已经成功上线。

## 如何获取 Libra 实验结果

### 1. 拉实验详情

```bash
bytedcli --json --site i18n-tt libra experiment get --flight-id <flight_id> \
  > artifacts/libra/flight-<flight_id>.json
```

### 2. 拉 traffic

```bash
bytedcli --json --site i18n-tt libra experiment traffic --flight-id <flight_id> \
  > artifacts/libra/traffic-<flight_id>.json
```

### 3. 先拉 report meta，再拉具体指标组

```bash
bytedcli --json --site i18n-tt libra experiment report --flight-id <flight_id> \
  > artifacts/libra/report-meta-<flight_id>.json

bytedcli --json --site i18n-tt libra experiment report \
  --flight-id <flight_id> \
  --metric-group 7039914 \
  > artifacts/libra/report-<flight_id>-7039914.json
```

### 4. 需要维度拆解时先看维度

```bash
bytedcli --json --site i18n-tt libra experiment report \
  --flight-id <flight_id> \
  --metric-group 7039720 \
  --list-dimensions
```

然后按实际维度值继续查：

```bash
bytedcli --json --site i18n-tt libra experiment report \
  --flight-id <flight_id> \
  --metric-group 7039720 \
  --dimension 171948:29484162 \
  --trend
```

## Hive 落表建议

### 表 1. flight registry

一行一个实验：

- `scene`
- `batch_id`
- `flight_id`
- `review_id`
- `app_id`
- `site`
- `phase`
- `owner`
- `layer_name`
- `smartui_config_version`
- `strategy_version_mapping_json`

### 表 2. metric result

一行一个 `flight_id + metric_group + metric + version + dt + dimension`：

- `flight_id`
- `metric_group_id`
- `metric_name`
- `version_id`
- `version_name`
- `raw_value`
- `diff_value`
- `p_value`
- `is_significant`
- `dimension_json`

## 如何通过 SQL 把 Libra ID 写入 Hive

推荐两种方式：

### 方式 A. 服务端先解析变量，再拼 SQL

先从创建响应里解析：

- `flight_id`
- `review_id`
- `app_id`
- `site`
- `batch_id`

然后执行：

```sql
INSERT INTO seo.dom_style_libra_flight_registry_di PARTITION (dt='${biz_date}')
SELECT
  '${scene}'                         AS scene,
  '${batch_id}'                      AS batch_id,
  CAST('${flight_id}' AS BIGINT)     AS flight_id,
  CAST('${review_id}' AS BIGINT)     AS review_id,
  CAST('${app_id}' AS BIGINT)        AS app_id,
  '${site}'                          AS site,
  '${phase}'                         AS phase,
  '${owner}'                         AS owner,
  '${layer_name}'                    AS layer_name,
  '${smartui_config_version}'        AS smartui_config_version,
  '${strategy_version_mapping_json}' AS strategy_version_mapping_json,
  current_timestamp()                AS created_at,
  current_timestamp()                AS updated_at;
```

### 方式 B. 先落原始 JSON，再用 SQL 解析

如果服务端只把 `create-response.json` 直接写进 ODS，也可以二次解析：

```sql
INSERT INTO seo.dom_style_libra_flight_registry_di PARTITION (dt='${biz_date}')
SELECT
  get_json_object(resp_json, '$.scene')                           AS scene,
  get_json_object(resp_json, '$.batch_id')                        AS batch_id,
  CAST(COALESCE(get_json_object(resp_json, '$.data.flight_id'),
                get_json_object(resp_json, '$.flight_id')) AS BIGINT) AS flight_id,
  CAST(COALESCE(get_json_object(resp_json, '$.data.review_id'),
                get_json_object(resp_json, '$.review_id')) AS BIGINT) AS review_id,
  CAST(get_json_object(resp_json, '$.app_id') AS BIGINT)          AS app_id,
  get_json_object(resp_json, '$.site')                            AS site,
  get_json_object(resp_json, '$.phase')                           AS phase,
  current_timestamp()                                             AS created_at,
  current_timestamp()                                             AS updated_at
FROM seo.dom_style_libra_create_resp_ods
WHERE dt='${biz_date}';
```

## 推荐默认策略

- 默认用 `submit-review --auto-launch-mode auto`，减少人工点开实验。
- 默认保存原始 JSON，不只保存结论。
- 默认将 `flight_id` 写入 Hive registry，作为后续回收的唯一入口。
- 默认把显著负向、渲染异常、无收益三类策略写入黑名单或降权表。
