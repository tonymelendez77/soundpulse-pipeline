WITH source AS (
    SELECT * FROM {{ source('raw', 'news_sentiment_weekly') }}
),

cleaned AS (
    SELECT
        CAST(week_start AS DATE) AS week_start,
        topic,
        article_count,
        dominant_emotion,
        avg_fear,
        avg_anger,
        avg_joy,
        avg_sadness,
        avg_surprise,
        avg_disgust,
        avg_neutral,
        anxiety_index,
        tension_index,
        positivity_index,
        ingested_at,
        CURRENT_TIMESTAMP()      AS dbt_loaded_at
    FROM source
    WHERE week_start IS NOT NULL
      AND article_count > 0
)

SELECT * FROM cleaned
