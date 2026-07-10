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
