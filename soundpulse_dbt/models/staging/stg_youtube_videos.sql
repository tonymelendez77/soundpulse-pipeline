WITH source AS (
    SELECT * FROM {{ source('raw', 'youtube_videos') }}
),

deduplicated AS (
    SELECT * FROM source
    QUALIFY ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY ingested_at DESC) = 1
),

cleaned AS (
    SELECT
        video_id,
        title,
        channel_title AS channel,
        CAST(view_count AS INT64) AS view_count,
        CAST(like_count AS INT64) AS like_count,
        CAST(comment_count AS INT64) AS comment_count,
        duration,
        description,
        PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', published_at) AS published_at,
        ingested_at,
        CURRENT_TIMESTAMP() AS dbt_loaded_at
    FROM deduplicated
)

SELECT * FROM cleaned