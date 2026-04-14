{{ config(materialized='table') }}

/*
  Analytical summary of XGBoost mood predictions.
  Does NOT retrain the model — exposes prediction results for dashboards.
  CROSS JOIN adds rolling model-level accuracy stats to every row.
*/

WITH predictions AS (
    SELECT * FROM {{ ref('stg_ml_predictions') }}
    WHERE period IS NULL          -- training data only (validated history)
),

summary AS (
    SELECT
        COUNT(*)                                AS total_weeks,
        COUNTIF(correct)                        AS correct_predictions,
        SAFE_DIVIDE(COUNTIF(correct), COUNT(*)) AS overall_accuracy,
        AVG(confidence)                         AS avg_confidence,
        MIN(week_start)                         AS first_week,
        MAX(week_start)                         AS last_week
    FROM predictions
),

forward AS (
    SELECT * FROM {{ ref('stg_ml_predictions') }}
    WHERE period IS NOT NULL      -- forward (unvalidated) predictions
)

SELECT
    p.week_start,
    p.region,
    p.actual_mood,
    p.predicted_mood,
    p.correct,
    ROUND(p.confidence, 4)           AS confidence,
    ROUND(p.anxiety_index, 6)        AS anxiety_index,
    ROUND(p.tension_index, 6)        AS tension_index,
    ROUND(p.positivity_index, 6)     AS positivity_index,
    s.total_weeks,
    s.correct_predictions,
    ROUND(s.overall_accuracy, 4)     AS overall_accuracy,
    ROUND(s.avg_confidence, 4)       AS avg_confidence,
    s.first_week,
    s.last_week,
    FALSE                            AS is_forward,
    CURRENT_TIMESTAMP()              AS dbt_loaded_at
FROM predictions p
CROSS JOIN summary s

UNION ALL

SELECT
    f.week_start,
    f.region,
    f.actual_mood,
    f.predicted_mood,
    f.correct,
    ROUND(f.confidence, 4)           AS confidence,
    ROUND(f.anxiety_index, 6)        AS anxiety_index,
    ROUND(f.tension_index, 6)        AS tension_index,
    ROUND(f.positivity_index, 6)     AS positivity_index,
    s.total_weeks,
    s.correct_predictions,
    ROUND(s.overall_accuracy, 4)     AS overall_accuracy,
    ROUND(s.avg_confidence, 4)       AS avg_confidence,
    s.first_week,
    s.last_week,
    TRUE                             AS is_forward,
    CURRENT_TIMESTAMP()              AS dbt_loaded_at
FROM forward f
CROSS JOIN summary s

ORDER BY week_start
