{{ config(materialized='table') }}

/*
  Rebuilds emotion_music_correlation in pure BigQuery SQL using CORR().

  Python Layer 3 uses scipy.stats.pearsonr — BigQuery's CORR(x, y) is
  mathematically equivalent. The Python-produced source table remains
  available if p-values are needed (CORR() doesn't provide them).

  'notable' (|r| >= 0.3) replaces the Python 'significant' flag
  because a t-distribution CDF is not available in BigQuery SQL.
*/

WITH news_by_week AS (
    SELECT
        week_start,
        AVG(avg_fear)         AS avg_fear,
        AVG(avg_anger)        AS avg_anger,
        AVG(avg_joy)          AS avg_joy,
        AVG(avg_sadness)      AS avg_sadness,
        AVG(avg_surprise)     AS avg_surprise,
        AVG(avg_disgust)      AS avg_disgust,
        AVG(avg_neutral)      AS avg_neutral,
        AVG(anxiety_index)    AS anxiety_index,
        AVG(tension_index)    AS tension_index,
        AVG(positivity_index) AS positivity_index
    FROM {{ ref('stg_news_sentiment_weekly') }}
    GROUP BY week_start
),

audio_by_week AS (
    SELECT
        week_start,
        AVG(euphoric_pct)    AS euphoric_pct,
        AVG(melancholic_pct) AS melancholic_pct,
        AVG(aggressive_pct)  AS aggressive_pct,
        AVG(peaceful_pct)    AS peaceful_pct,
        AVG(groovy_pct)      AS groovy_pct
    FROM {{ ref('stg_audio_mood_weekly') }}
    GROUP BY week_start
),

joined AS (
    SELECT n.*, a.*
    FROM news_by_week n
    INNER JOIN audio_by_week a USING (week_start)
),

correlations AS (
    SELECT 'avg_fear'         AS emotion, 'euphoric'    AS mood_archetype, CORR(avg_fear,         euphoric_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_fear'         AS emotion, 'melancholic' AS mood_archetype, CORR(avg_fear,         melancholic_pct) AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_fear'         AS emotion, 'aggressive'  AS mood_archetype, CORR(avg_fear,         aggressive_pct)  AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_fear'         AS emotion, 'peaceful'    AS mood_archetype, CORR(avg_fear,         peaceful_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_fear'         AS emotion, 'groovy'      AS mood_archetype, CORR(avg_fear,         groovy_pct)      AS pearson_r, COUNT(*) AS n FROM joined UNION ALL

    SELECT 'avg_anger'        AS emotion, 'euphoric'    AS mood_archetype, CORR(avg_anger,        euphoric_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_anger'        AS emotion, 'melancholic' AS mood_archetype, CORR(avg_anger,        melancholic_pct) AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_anger'        AS emotion, 'aggressive'  AS mood_archetype, CORR(avg_anger,        aggressive_pct)  AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_anger'        AS emotion, 'peaceful'    AS mood_archetype, CORR(avg_anger,        peaceful_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_anger'        AS emotion, 'groovy'      AS mood_archetype, CORR(avg_anger,        groovy_pct)      AS pearson_r, COUNT(*) AS n FROM joined UNION ALL

    SELECT 'avg_joy'          AS emotion, 'euphoric'    AS mood_archetype, CORR(avg_joy,          euphoric_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_joy'          AS emotion, 'melancholic' AS mood_archetype, CORR(avg_joy,          melancholic_pct) AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_joy'          AS emotion, 'aggressive'  AS mood_archetype, CORR(avg_joy,          aggressive_pct)  AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_joy'          AS emotion, 'peaceful'    AS mood_archetype, CORR(avg_joy,          peaceful_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_joy'          AS emotion, 'groovy'      AS mood_archetype, CORR(avg_joy,          groovy_pct)      AS pearson_r, COUNT(*) AS n FROM joined UNION ALL

    SELECT 'avg_sadness'      AS emotion, 'euphoric'    AS mood_archetype, CORR(avg_sadness,      euphoric_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_sadness'      AS emotion, 'melancholic' AS mood_archetype, CORR(avg_sadness,      melancholic_pct) AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_sadness'      AS emotion, 'aggressive'  AS mood_archetype, CORR(avg_sadness,      aggressive_pct)  AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_sadness'      AS emotion, 'peaceful'    AS mood_archetype, CORR(avg_sadness,      peaceful_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_sadness'      AS emotion, 'groovy'      AS mood_archetype, CORR(avg_sadness,      groovy_pct)      AS pearson_r, COUNT(*) AS n FROM joined UNION ALL

    SELECT 'avg_surprise'     AS emotion, 'euphoric'    AS mood_archetype, CORR(avg_surprise,     euphoric_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_surprise'     AS emotion, 'melancholic' AS mood_archetype, CORR(avg_surprise,     melancholic_pct) AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_surprise'     AS emotion, 'aggressive'  AS mood_archetype, CORR(avg_surprise,     aggressive_pct)  AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_surprise'     AS emotion, 'peaceful'    AS mood_archetype, CORR(avg_surprise,     peaceful_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_surprise'     AS emotion, 'groovy'      AS mood_archetype, CORR(avg_surprise,     groovy_pct)      AS pearson_r, COUNT(*) AS n FROM joined UNION ALL

    SELECT 'avg_disgust'      AS emotion, 'euphoric'    AS mood_archetype, CORR(avg_disgust,      euphoric_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_disgust'      AS emotion, 'melancholic' AS mood_archetype, CORR(avg_disgust,      melancholic_pct) AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_disgust'      AS emotion, 'aggressive'  AS mood_archetype, CORR(avg_disgust,      aggressive_pct)  AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_disgust'      AS emotion, 'peaceful'    AS mood_archetype, CORR(avg_disgust,      peaceful_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_disgust'      AS emotion, 'groovy'      AS mood_archetype, CORR(avg_disgust,      groovy_pct)      AS pearson_r, COUNT(*) AS n FROM joined UNION ALL

    SELECT 'avg_neutral'      AS emotion, 'euphoric'    AS mood_archetype, CORR(avg_neutral,      euphoric_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_neutral'      AS emotion, 'melancholic' AS mood_archetype, CORR(avg_neutral,      melancholic_pct) AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_neutral'      AS emotion, 'aggressive'  AS mood_archetype, CORR(avg_neutral,      aggressive_pct)  AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_neutral'      AS emotion, 'peaceful'    AS mood_archetype, CORR(avg_neutral,      peaceful_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'avg_neutral'      AS emotion, 'groovy'      AS mood_archetype, CORR(avg_neutral,      groovy_pct)      AS pearson_r, COUNT(*) AS n FROM joined UNION ALL

    SELECT 'anxiety_index'    AS emotion, 'euphoric'    AS mood_archetype, CORR(anxiety_index,    euphoric_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'anxiety_index'    AS emotion, 'melancholic' AS mood_archetype, CORR(anxiety_index,    melancholic_pct) AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'anxiety_index'    AS emotion, 'aggressive'  AS mood_archetype, CORR(anxiety_index,    aggressive_pct)  AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'anxiety_index'    AS emotion, 'peaceful'    AS mood_archetype, CORR(anxiety_index,    peaceful_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'anxiety_index'    AS emotion, 'groovy'      AS mood_archetype, CORR(anxiety_index,    groovy_pct)      AS pearson_r, COUNT(*) AS n FROM joined UNION ALL

    SELECT 'tension_index'    AS emotion, 'euphoric'    AS mood_archetype, CORR(tension_index,    euphoric_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'tension_index'    AS emotion, 'melancholic' AS mood_archetype, CORR(tension_index,    melancholic_pct) AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'tension_index'    AS emotion, 'aggressive'  AS mood_archetype, CORR(tension_index,    aggressive_pct)  AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'tension_index'    AS emotion, 'peaceful'    AS mood_archetype, CORR(tension_index,    peaceful_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'tension_index'    AS emotion, 'groovy'      AS mood_archetype, CORR(tension_index,    groovy_pct)      AS pearson_r, COUNT(*) AS n FROM joined UNION ALL

    SELECT 'positivity_index' AS emotion, 'euphoric'    AS mood_archetype, CORR(positivity_index, euphoric_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'positivity_index' AS emotion, 'melancholic' AS mood_archetype, CORR(positivity_index, melancholic_pct) AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'positivity_index' AS emotion, 'aggressive'  AS mood_archetype, CORR(positivity_index, aggressive_pct)  AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'positivity_index' AS emotion, 'peaceful'    AS mood_archetype, CORR(positivity_index, peaceful_pct)    AS pearson_r, COUNT(*) AS n FROM joined UNION ALL
    SELECT 'positivity_index' AS emotion, 'groovy'      AS mood_archetype, CORR(positivity_index, groovy_pct)      AS pearson_r, COUNT(*) AS n FROM joined
)

SELECT
    emotion,
    mood_archetype,
    ROUND(pearson_r, 6)                              AS pearson_r,
    n                                                AS weeks_compared,
    CASE
        WHEN pearson_r > 0 THEN 'positive'
        WHEN pearson_r < 0 THEN 'negative'
        ELSE 'zero'
    END                                              AS direction,
    ABS(pearson_r) >= 0.3                            AS notable,
    CURRENT_TIMESTAMP()                              AS dbt_loaded_at
FROM correlations
WHERE pearson_r IS NOT NULL
ORDER BY ABS(pearson_r) DESC
