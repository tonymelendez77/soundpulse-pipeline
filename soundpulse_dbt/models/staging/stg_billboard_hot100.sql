WITH source AS (
    SELECT * FROM {{ source('raw', 'billboard_hot100') }}
),

cleaned AS (
    SELECT
        rank,
        title AS song_title,
        artist AS artist_name,
        last_week,
        peak_position,
        weeks_on_chart,
        PARSE_DATE('%Y-%m-%d', chart_date) AS chart_date,
        ingested_at,
        CURRENT_TIMESTAMP() AS dbt_loaded_at
    FROM source
)

SELECT * FROM cleaned