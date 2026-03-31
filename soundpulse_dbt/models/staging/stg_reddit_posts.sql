WITH source AS (
    SELECT * FROM {{ source('raw', 'reddit_posts') }}
),

cleaned AS (
    SELECT
        id AS post_id,
        subreddit,
        title,
        body,
        score,
        CAST(num_comments AS INT64) AS num_comments,
        author,
        url,
        PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%S%Ez', created_utc) AS created_at,
        ingested_at,
        CURRENT_TIMESTAMP() AS dbt_loaded_at
    FROM source
)

SELECT * FROM cleaned