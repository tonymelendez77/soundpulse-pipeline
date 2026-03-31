WITH source AS (
    SELECT * FROM {{ source('raw', 'ml_predictions') }}
),

cleaned AS (
    SELECT
        CAST(week_start AS DATE) AS week_start,
        actual_mood,
        predicted_mood,
        correct,
        confidence,
        avg_fear,
        avg_anger,
        avg_joy,
        avg_sadness,
        anxiety_index,
        tension_index,
        positivity_index,
        ingested_at,
        CURRENT_TIMESTAMP()      AS dbt_loaded_at
    FROM source
    WHERE week_start IS NOT NULL
)

SELECT * FROM cleaned
