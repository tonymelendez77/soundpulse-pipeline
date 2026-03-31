WITH source AS (
    SELECT * FROM {{ source('raw', 'news_articles') }}
),

cleaned AS (
    SELECT
        url,
        title,
        description,
        source,
        author,
        content,
        PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', published_at) AS published_at,
        ingested_at,
        CURRENT_TIMESTAMP() AS dbt_loaded_at
    FROM source
)

SELECT * FROM cleaned