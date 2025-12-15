CREATE OR REPLACE TABLE `telco_ml.customer_history` AS
WITH base AS (
  SELECT
    CONCAT('C', CAST(100000 + x AS STRING)) AS customer_id,
    CAST(1 + MOD(x, 60) AS INT64) AS tenure_months,
    ROUND(30 + RAND()*70, 2) AS monthly_charges,
    ROUND(RAND()*200, 2) AS total_data_gb,
    CAST(RAND()*5 AS INT64) AS support_tickets_30d,
    CAST(RAND()*6 AS INT64) AS late_payments_6m
  FROM UNNEST(GENERATE_ARRAY(1, 20000)) AS x
),
labeled AS (
  SELECT
    *,
    IF(
      (support_tickets_30d >= 3 AND late_payments_6m >= 2)
      OR (tenure_months <= 3 AND support_tickets_30d >= 2)
      OR (late_payments_6m >= 4),
      1, 0
    ) AS churn_next_month
  FROM base
)
SELECT * FROM labeled;

CREATE OR REPLACE TABLE `telco_ml.customers_current_month` AS
SELECT
  customer_id,
  tenure_months,
  monthly_charges,
  total_data_gb,
  support_tickets_30d,
  late_payments_6m
FROM `telco_ml.customer_history`;
