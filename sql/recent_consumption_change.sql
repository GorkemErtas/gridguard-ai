-- SQLite: compare last 30 calendar dates to the prior 30 calendar dates.
-- This is a descriptive filter, not a trained theft detection model.
WITH max_date AS (
    SELECT MAX(reading_date) AS last_date FROM consumption
), periods AS (
    SELECT
        c.customer_id,
        AVG(CASE
            WHEN c.reading_date > date(m.last_date, '-30 days') THEN c.kwh
        END) AS recent_kwh,
        AVG(CASE
            WHEN c.reading_date <= date(m.last_date, '-30 days')
             AND c.reading_date > date(m.last_date, '-60 days') THEN c.kwh
        END) AS previous_kwh
    FROM consumption AS c
    CROSS JOIN max_date AS m
    GROUP BY c.customer_id
)
SELECT customer_id,
       ROUND(previous_kwh, 2) AS previous_kwh,
       ROUND(recent_kwh, 2) AS recent_kwh,
       ROUND((previous_kwh - recent_kwh) / (previous_kwh + 1), 3) AS drop_ratio
FROM periods
WHERE previous_kwh IS NOT NULL AND recent_kwh IS NOT NULL
ORDER BY drop_ratio DESC
LIMIT 50;
