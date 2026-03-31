WITH source AS (
    SELECT * FROM {{ source('raw', 'generated_tracks') }}
),

cleaned AS (
    SELECT
        generation_id,
        CAST(week_start AS DATE)         AS week_start,
        mood_archetype,
        prompt_text,
        similar_tracks_json,
        audio_gcs_path,
        ROUND(duration_seconds, 2)       AS duration_seconds,
        CAST(generated_at AS TIMESTAMP)  AS generated_at,
        CURRENT_TIMESTAMP()              AS dbt_loaded_at
    FROM source
    WHERE generation_id IS NOT NULL
      AND mood_archetype IS NOT NULL
)

SELECT * FROM cleaned
