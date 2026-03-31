WITH source AS (
    SELECT * FROM {{ source('raw', 'audio_mood_clusters') }}
),

cleaned AS (
    SELECT
        title,
        artist,
        CAST(chart_date AS DATE)     AS chart_date,
        source                       AS chart_source,
        chart_rank,
        genre,
        cluster_id,
        mood_archetype,
        valence,
        energy,
        danceability,
        acousticness,
        tempo,
        loudness,
        instrumentalness,
        speechiness,
        CASE
            WHEN valence >= 0.6 THEN 'happy'
            WHEN valence <= 0.4 THEN 'sad'
            ELSE 'neutral'
        END AS valence_mood,
        ingested_at,
        CURRENT_TIMESTAMP()          AS dbt_loaded_at
    FROM source
    WHERE title IS NOT NULL
      AND mood_archetype IS NOT NULL
)

SELECT * FROM cleaned
