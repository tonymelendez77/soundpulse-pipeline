WITH source AS (
    SELECT * FROM {{ source('raw', 'audio_mood_weekly') }}
),

cleaned AS (
    SELECT
        CAST(week_start AS DATE) AS week_start,
        chart_name               AS chart_source,
        dominant_mood,
        track_count,
        euphoric_pct,
        melancholic_pct,
        aggressive_pct,
        peaceful_pct,
        groovy_pct,
        avg_valence,
        avg_energy,
        avg_danceability,
        avg_tempo,
        ingested_at,
        CURRENT_TIMESTAMP()      AS dbt_loaded_at
    FROM source
    WHERE week_start IS NOT NULL
      AND track_count > 0
)

SELECT * FROM cleaned
