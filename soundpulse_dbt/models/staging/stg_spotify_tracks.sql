{{ config(materialized='table') }}

WITH source AS (
    SELECT * FROM {{ source('raw', 'trending_tracks') }}
),

deduplicated AS (
    SELECT * FROM source
    QUALIFY ROW_NUMBER() OVER (PARTITION BY title, artist ORDER BY ingested_at DESC) = 1
),

cleaned AS (
    SELECT
        TO_HEX(MD5(CONCAT(LOWER(TRIM(title)), '|', LOWER(TRIM(artist))))) AS track_id,
        title AS track_name,
        artist AS artist_name,
        itunes_album AS album_name,
        itunes_release_date AS release_date,
        preview_url,
        itunes_track_id,
        itunes_genre AS genre,
        chart_rank,
        chart_date,
        match_layer,
        source,
        country,
        source_count,
        listeners,
        playcount,
        COALESCE(itunes_duration_ms, 0) AS duration_ms,
        explicit,
        spotify_track_id,
        spotify_url,
        ingested_at,
        COALESCE(tempo, 120.0) AS tempo,
        COALESCE(energy, 0.5) AS energy,
        COALESCE(danceability, 0.5) AS danceability,
        COALESCE(valence, 0.5) AS valence,
        COALESCE(acousticness, 0.5) AS acousticness,
        COALESCE(instrumentalness, 0.0) AS instrumentalness,
        COALESCE(liveness, 0.5) AS liveness,
        COALESCE(loudness, -10.0) AS loudness,
        COALESCE(speechiness, 0.1) AS speechiness,
        CAST(COALESCE(key, 0) AS INT64) AS key,
        CAST(COALESCE(mode, 1) AS INT64) AS mode,
        CAST(COALESCE(time_signature, 4) AS INT64) AS time_signature,
        COALESCE(mfcc_1, 0.0) AS mfcc_1,
        COALESCE(mfcc_2, 0.0) AS mfcc_2,
        COALESCE(mfcc_5, 0.0) AS mfcc_5,
        COALESCE(mfcc_13, 0.0) AS mfcc_13,
        COALESCE(chroma_C, 0.5) AS chroma_C,
        COALESCE(chroma_C_sharp, 0.5) AS chroma_C_sharp,
        COALESCE(chroma_D, 0.5) AS chroma_D,
        COALESCE(chroma_D_sharp, 0.5) AS chroma_D_sharp,
        COALESCE(chroma_E, 0.5) AS chroma_E,
        COALESCE(chroma_F, 0.5) AS chroma_F,
        COALESCE(chroma_F_sharp, 0.5) AS chroma_F_sharp,
        COALESCE(chroma_G, 0.5) AS chroma_G,
        COALESCE(chroma_G_sharp, 0.5) AS chroma_G_sharp,
        COALESCE(chroma_A, 0.5) AS chroma_A,
        COALESCE(chroma_A_sharp, 0.5) AS chroma_A_sharp,
        COALESCE(chroma_B, 0.5) AS chroma_B,
        COALESCE(spectral_centroid, 2000.0) AS spectral_centroid,
        COALESCE(harmonic_percussive_ratio, 1.0) AS harmonic_percussive_ratio,
        CURRENT_TIMESTAMP() AS dbt_loaded_at
    FROM deduplicated
)

SELECT * FROM cleaned