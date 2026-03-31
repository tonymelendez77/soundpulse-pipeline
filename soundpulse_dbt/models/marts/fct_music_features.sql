SELECT
    t.track_id,
    t.track_name,
    t.artist_name,
    t.album_name,
    t.release_date,
    t.genre,
    t.preview_url,
    t.itunes_track_id,
    t.chart_rank,
    t.chart_date,
    t.match_layer,
    t.duration_ms,
    t.danceability,
    t.energy,
    t.valence,
    t.acousticness,
    t.instrumentalness,
    t.liveness,
    t.speechiness,
    t.tempo,
    t.loudness,
    t.key,
    t.mode,
    t.time_signature,
    t.mfcc_1,
    t.mfcc_2,
    t.mfcc_5,
    t.mfcc_13,
    t.chroma_C,
    t.chroma_C_sharp,
    t.chroma_D,
    t.chroma_D_sharp,
    t.chroma_E,
    t.chroma_F,
    t.chroma_F_sharp,
    t.chroma_G,
    t.chroma_G_sharp,
    t.chroma_A,
    t.chroma_A_sharp,
    t.chroma_B,
    t.spectral_centroid,
    t.harmonic_percussive_ratio,
    CASE
        WHEN t.valence >= 0.6 THEN 'happy'
        WHEN t.valence <= 0.4 THEN 'sad'
        ELSE 'neutral'
    END AS mood,
    CASE
        WHEN t.energy >= 0.6 AND t.tempo >= 120 THEN 'high_energy'
        WHEN t.energy <= 0.4 AND t.tempo <= 100 THEN 'low_energy'
        ELSE 'medium_energy'
    END AS energy_level,
    CASE
        WHEN t.mfcc_1 > 0 AND t.mfcc_2 > 0 THEN 'bright'
        WHEN t.mfcc_1 < 0 AND t.mfcc_2 < 0 THEN 'dark'
        ELSE 'balanced'
    END AS timbre_profile,
    CASE
        WHEN t.harmonic_percussive_ratio >= 1.5 THEN 'melodic'
        WHEN t.harmonic_percussive_ratio <= 0.5 THEN 'percussive'
        ELSE 'balanced'
    END AS harmonic_character,
    c.cluster_id     AS kmeans_cluster_id,
    c.mood_archetype AS kmeans_mood_archetype,
    t.dbt_loaded_at
FROM {{ ref('stg_spotify_tracks') }} t
LEFT JOIN (
    -- Deduplicate: one row per title+artist, take the most recent week
    SELECT * EXCEPT (row_num)
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY LOWER(TRIM(title)), LOWER(TRIM(artist))
                ORDER BY week_start DESC
            ) AS row_num
        FROM {{ ref('stg_audio_mood_clusters') }}
    )
    WHERE row_num = 1
) c
    ON  LOWER(TRIM(t.track_name))  = LOWER(TRIM(c.title))
    AND LOWER(TRIM(t.artist_name)) = LOWER(TRIM(c.artist))
