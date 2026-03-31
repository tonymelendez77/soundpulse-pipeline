WITH source AS (
    SELECT * FROM {{ source('raw', 'shap_importance') }}
),

cleaned AS (
    SELECT
        feature,
        mood_archetype,
        mean_shap_value,
        mean_abs_shap,
        rank             AS importance_rank,
        ingested_at,
        CURRENT_TIMESTAMP() AS dbt_loaded_at
    FROM source
    WHERE feature IS NOT NULL
      AND mood_archetype IS NOT NULL
)

SELECT * FROM cleaned
