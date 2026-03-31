WITH source AS (
    SELECT * FROM {{ source('raw', 'news_sentiment') }}
),

cleaned AS (
    SELECT
        CAST(date AS DATE)           AS article_date,
        topic,
        title,
        emotion,
        fear_score,
        anger_score,
        joy_score,
        sadness_score,
        surprise_score,
        disgust_score,
        neutral_score,
        fear_score + sadness_score   AS article_anxiety_index,
        anger_score + disgust_score  AS article_tension_index,
        joy_score + surprise_score   AS article_positivity_index,
        ingested_at,
        CURRENT_TIMESTAMP()          AS dbt_loaded_at
    FROM source
    WHERE title IS NOT NULL
)

SELECT * FROM cleaned
