WITH source AS (
    SELECT * FROM {{ source('raw', 'weekly_features') }}
),

cleaned AS (
    SELECT
        CAST(week_start AS DATE) AS week_start,
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
        dominant_emotion,
        euphoric_pct,
        melancholic_pct,
        aggressive_pct,
        peaceful_pct,
        groovy_pct,
        dominant_mood,
        avg_valence,
        avg_energy,
        avg_danceability,
        avg_tempo,
        ingested_at,
        CURRENT_TIMESTAMP()      AS dbt_loaded_at
    FROM source
    WHERE week_start IS NOT NULL
)

SELECT * FROM cleaned
