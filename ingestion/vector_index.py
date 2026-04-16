"""Pinecone vector index builder for track similarity search."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from loguru import logger
from pinecone import Pinecone
from sklearn.preprocessing import StandardScaler

import os

# Config
PROJECT        = "soundpulse-production"
DATASET        = "music_analytics"
PINECONE_INDEX = "soundpulse-tracks"
SCALER_PATH    = Path(__file__).parent / "scaler_params.json"

FEATURE_COLS = [
    "tempo", "energy", "danceability", "valence", "acousticness",
    "instrumentalness", "liveness", "loudness", "speechiness",
    "key", "mode", "time_signature",
    "mfcc_1", "mfcc_2", "mfcc_5", "mfcc_13",
    "chroma_C", "chroma_C_sharp", "chroma_D", "chroma_D_sharp",
    "chroma_E", "chroma_F", "chroma_F_sharp", "chroma_G",
    "chroma_G_sharp", "chroma_A", "chroma_A_sharp", "chroma_B",
    "spectral_centroid", "harmonic_percussive_ratio",
]


# BQ queries

def fetch_historical_tracks(client: bigquery.Client) -> pd.DataFrame:
    """Fetch deduped historical tracks with mood archetypes."""
    feature_cols_sql = ",\n    ".join(f"h.{c}" for c in FEATURE_COLS)
    query = f"""
        SELECT
            h.title,
            h.artist,
            c.mood_archetype,
            h.chart_name,
            CAST(h.week_start AS STRING) AS week_start,
            {feature_cols_sql}
        FROM `{PROJECT}.{DATASET}.trending_historical` h
        JOIN `{PROJECT}.{DATASET}.audio_mood_clusters` c
            ON  LOWER(TRIM(h.title))  = LOWER(TRIM(c.title))
            AND LOWER(TRIM(h.artist)) = LOWER(TRIM(c.artist))
            AND CAST(h.week_start AS DATE) = c.week_start
        WHERE h.tempo IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY LOWER(TRIM(h.title)), LOWER(TRIM(h.artist))
            ORDER BY h.week_start DESC
        ) = 1
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Fetched {len(df):,} unique historical tracks (title+artist deduped)")
    return df


def fetch_trending_tracks(client: bigquery.Client) -> pd.DataFrame:
    """Fetch current trending tracks from BigQuery."""
    feature_cols_sql = ", ".join(FEATURE_COLS)
    query = f"""
        SELECT title, artist, source AS chart_name, {feature_cols_sql}
        FROM `{PROJECT}.{DATASET}.trending_tracks`
        WHERE tempo IS NOT NULL
    """
    df = client.query(query).to_dataframe()
    df["mood_archetype"] = None
    df["week_start"] = pd.Timestamp.now().date().isoformat()
    logger.info(f"Fetched {len(df):,} current trending tracks")
    return df


# Scaler

def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, dict]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    params = {
        "mean_":       scaler.mean_.tolist(),
        "scale_":      scaler.scale_.tolist(),
        "feature_cols": FEATURE_COLS,
    }
    return X_scaled, params


def save_scaler_params(params: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Scaler params saved to {path}")


# Centroid assignment

def assign_mood_by_centroid(
    X_scaled: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> list[str]:
    """Assign mood archetype by nearest centroid."""
    archetype_names = list(centroids.keys())
    centroid_matrix = np.stack([centroids[a] for a in archetype_names])  # (n_archetypes, 30)

    # Normalise for cosine similarity
    X_norm = X_scaled / (np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-9)
    C_norm = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-9)

    sims = X_norm @ C_norm.T   # (n_rows, n_archetypes)
    best_idx = np.argmax(sims, axis=1)
    return [archetype_names[i] for i in best_idx]


# Pinecone

def track_id(title: str, artist: str) -> str:
    """Stable track ID from title+artist."""
    key = f"{title.lower().strip()}|{artist.lower().strip()}"
    return hashlib.md5(key.encode()).hexdigest()


def build_pinecone_vectors(df: pd.DataFrame, X_scaled: np.ndarray) -> list[dict]:
    vectors = []
    for i, (_, row) in enumerate(df.iterrows()):
        vec_id = track_id(str(row["title"]), str(row["artist"]))
        vectors.append({
            "id":     vec_id,
            "values": X_scaled[i].tolist(),
            "metadata": {
                "title":         str(row["title"]),
                "artist":        str(row["artist"]),
                "mood_archetype": str(row["mood_archetype"]) if row["mood_archetype"] else "unknown",
                "chart_name":    str(row.get("chart_name", "")),
                "week_start":    str(row.get("week_start", "")),
            },
        })
    return vectors


def upsert_to_pinecone(index, vectors: list[dict], batch_size: int = 100) -> None:
    total = len(vectors)
    for i in range(0, total, batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        logger.debug(f"  Upserted rows {i}–{i + len(batch) - 1}")
    logger.info(f"Upserted {total:,} vectors to Pinecone")


# Main

def main():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not found in .env")

    client = bigquery.Client(project=PROJECT)

    # Fetch data
    df_hist  = fetch_historical_tracks(client)
    df_trend = fetch_trending_tracks(client)

    # Combine — historical first so trending overwrites on same track ID
    df_all = pd.concat([df_hist, df_trend], ignore_index=True)
    before = len(df_all)
    df_all = df_all.dropna(subset=FEATURE_COLS)
    if len(df_all) < before:
        logger.warning(f"Dropped {before - len(df_all)} rows with null features")

    # Fit scaler on combined set
    X = df_all[FEATURE_COLS].values.astype(float)
    X_scaled, scaler_params = fit_scaler(X)
    save_scaler_params(scaler_params, SCALER_PATH)

    # Compute centroids from historical tracks (known archetypes)
    hist_mask = df_all.index < len(df_hist)
    archetypes = df_all.loc[hist_mask, "mood_archetype"].dropna().unique()
    centroids: dict[str, np.ndarray] = {}
    for arch in archetypes:
        mask = (df_all["mood_archetype"] == arch).values & hist_mask
        centroids[arch] = X_scaled[mask].mean(axis=0)
    logger.info(f"Computed centroids for archetypes: {list(centroids.keys())}")

    # Assign mood to trending_tracks rows (no archetype)
    trend_mask = ~hist_mask
    if trend_mask.any() and centroids:
        assigned = assign_mood_by_centroid(X_scaled[trend_mask], centroids)
        df_all.loc[trend_mask, "mood_archetype"] = assigned
        logger.info(f"Assigned archetypes to {trend_mask.sum()} trending tracks")

    # Build and upsert vectors
    vectors = build_pinecone_vectors(df_all, X_scaled)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX)
    upsert_to_pinecone(index, vectors)

    # Summary
    arch_dist = df_all["mood_archetype"].value_counts().to_dict()
    logger.info("VECTOR INDEX COMPLETE ")
    logger.info(f"  Total vectors upserted : {len(vectors):,}")
    logger.info(f"  Archetype distribution : {arch_dist}")
    logger.info(f"  Scaler params saved to : {SCALER_PATH}")
    logger.info(f"  Pinecone index         : {PINECONE_INDEX}")


if __name__ == "__main__":
    main()
