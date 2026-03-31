WITH source AS (
    SELECT * FROM {{ source('raw', 'news_historical') }}
),

cleaned AS (
    SELECT
        date,
        topic,
        title,
        description,
        url,
        published_at,
        source,
        ingested_at,
        CURRENT_TIMESTAMP() AS dbt_loaded_at
    FROM source
)

SELECT * FROM cleaned