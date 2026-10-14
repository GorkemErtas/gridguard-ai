-- Run against: gridguard export-sqlite --input data/demo.csv
-- Readings represent daily energy use in kWh, not confirmed illicit behavior.
-- SQLite syntax.

WITH customer_stats AS (
    SELECT
        customer_id,
        COUNT(*) AS reading_days,
        ROUND(AVG(kwh), 2) AS avg_daily_kwh,
        ROUND(MAX(kwh), 2) AS max_daily_kwh,
        ROUND(
            1.0 * SUM(CASE WHEN kwh = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
            3
        ) AS observed_zero_ratio
    FROM consumption
    GROUP BY customer_id
)
SELECT *
FROM customer_stats
ORDER BY observed_zero_ratio DESC, avg_daily_kwh DESC
LIMIT 50;
