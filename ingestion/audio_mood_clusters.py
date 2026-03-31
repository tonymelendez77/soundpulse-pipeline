"""
SoundPulse — Module 12, Layer 2
Audio Mood Clustering using KMeans on 30 Librosa features

Input:  music_analytics.trending_historical  (7,110 rows, 12 weeks)
Output: music_analytics.audio_mood_clusters  (per-track cluster assignment)
        music_analytics.audio_mood_weekly    (dominant mood archetype per week × source)

Mood archetypes (auto-named from centroids):
    euphoric   — high valence, high energy, high tempo
    melancholic— low valence, low energy, high acousticness
    aggressive — low valence, high energy, high loudness
    peaceful   — high acousticness, low energy, high instrumentalness
    groovy     — high danceability, medium energy, medium valence

Install first (if not already):
    pip install scikit-learn
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from google.cloud import bigquery
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────────────────────────
PROJECT   = "soundpulse-production"
DATASET   = "music_analytics"
SRC_TABLE = f"{PROJECT}.{DATASET}.trending_historical"
DST_TABLE = f"{PROJECT}.{DATASET}.audio_mood_clusters"
AGG_TABLE = f"{PROJECT}.{DATASET}.audio_mood_weekly"

K_RANGE   = range(3, 8)     # try k=3..7, pick best silhouette
RANDOM_STATE = 42

# 30 Librosa features (match BigQuery schema exactly)
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

# Columns to carry through to output
# Actual trending_historical columns: week_start, chart_name, rank, itunes_genre
META_COLS = ["title", "artist", "week_start", "chart_name", "rank", "itunes_genre"]

DST_SCHEMA = [
    bigquery.SchemaField("title",              "STRING"),
    bigquery.SchemaField("artist",             "STRING"),
    bigquery.SchemaField("week_start",         "DATE"),
    bigquery.SchemaField("chart_name",         "STRING"),
    bigquery.SchemaField("rank",               "INTEGER"),
    bigquery.SchemaField("genre",              "STRING"),
    bigquery.SchemaField("cluster_id",         "INTEGER"),
    bigquery.SchemaField("mood_archetype",     "STRING"),
    bigquery.SchemaField("valence",            "FLOAT64"),
    bigquery.SchemaField("energy",             "FLOAT64"),
    bigquery.SchemaField("danceability",       "FLOAT64"),
    bigquery.SchemaField("acousticness",       "FLOAT64"),
    bigquery.SchemaField("tempo",              "FLOAT64"),
    bigquery.SchemaField("loudness",           "FLOAT64"),
    bigquery.SchemaField("instrumentalness",   "FLOAT64"),
    bigquery.SchemaField("speechiness",        "FLOAT64"),
    bigquery.SchemaField("ingested_at",        "TIMESTAMP"),
]

AGG_SCHEMA = [
    bigquery.SchemaField("week_start",         "DATE"),
    bigquery.SchemaField("chart_name",         "STRING"),
    bigquery.SchemaField("dominant_mood",      "STRING"),
    bigquery.SchemaField("track_count",        "INTEGER"),
    bigquery.SchemaField("euphoric_pct",       "FLOAT64"),
    bigquery.SchemaField("melancholic_pct",    "FLOAT64"),
    bigquery.SchemaField("aggressive_pct",     "FLOAT64"),
    bigquery.SchemaField("peaceful_pct",       "FLOAT64"),
    bigquery.SchemaField("groovy_pct",         "FLOAT64"),
    bigquery.SchemaField("avg_valence",        "FLOAT64"),
    bigquery.SchemaField("avg_energy",         "FLOAT64"),
    bigquery.SchemaField("avg_danceability",   "FLOAT64"),
    bigquery.SchemaField("avg_tempo",          "FLOAT64"),
    bigquery.SchemaField("ingested_at",        "TIMESTAMP"),
]


# ── Cluster naming ──────────────────────────────────────────────────────────────

ARCHETYPE_NAMES = ["euphoric", "melancholic", "aggressive", "peaceful", "groovy"]


def name_clusters(centroids_df: pd.DataFrame, k: int) -> dict[int, str]:
    """
    Assign archetype names to cluster IDs using centroid feature scores.
    Each archetype wins the cluster it best matches (greedy assignment).
    """
    # Normalise centroids to 0-1 range for scoring
    norm = (centroids_df - centroids_df.min()) / (centroids_df.max() - centroids_df.min() + 1e-9)

    # Scoring rules per archetype (higher = more likely to be that archetype)
    def score(row, archetype):
        v  = row.get("valence", 0.5)
        e  = row.get("energy", 0.5)
        t  = row.get("tempo", 0.5)          # already normalised
        ac = row.get("acousticness", 0.5)
        d  = row.get("danceability", 0.5)
        lo = row.get("loudness", 0.5)
        ins = row.get("instrumentalness", 0.5)
        if archetype == "euphoric":
            return v * 0.4 + e * 0.3 + t * 0.15 + d * 0.15
        if archetype == "melancholic":
            return (1 - v) * 0.4 + (1 - e) * 0.3 + ac * 0.2 + (1 - t) * 0.1
        if archetype == "aggressive":
            return (1 - v) * 0.3 + e * 0.35 + lo * 0.25 + (1 - ac) * 0.1
        if archetype == "peaceful":
            return ac * 0.35 + (1 - e) * 0.25 + ins * 0.25 + (1 - lo) * 0.15
        if archetype == "groovy":
            return d * 0.45 + e * 0.25 + v * 0.2 + (1 - ins) * 0.1
        return 0.0

    archetypes_to_assign = ARCHETYPE_NAMES[:k]  # only as many as clusters

    # Build score matrix
    score_matrix = pd.DataFrame(
        {arch: [score(norm.iloc[i], arch) for i in range(k)]
         for arch in archetypes_to_assign}
    )

    # Greedy assignment: highest score first
    assignment: dict[int, str] = {}
    used_archetypes: set[str] = set()
    used_clusters: set[int] = set()

    flat = score_matrix.stack().sort_values(ascending=False)
    for (cluster_idx, archetype), _ in flat.items():
        if cluster_idx in used_clusters or archetype in used_archetypes:
            continue
        assignment[int(cluster_idx)] = archetype
        used_clusters.add(cluster_idx)
        used_archetypes.add(archetype)
        if len(assignment) == k:
            break

    # Fallback: any unassigned cluster gets a generic name
    for i in range(k):
        if i not in assignment:
            assignment[i] = f"cluster_{i}"

    return assignment


# ── BQ helpers ──────────────────────────────────────────────────────────────────

def ensure_table(client, table_id, schema):
    client.delete_table(table_id, not_found_ok=True)
    table = bigquery.Table(table_id, schema=schema)
    client.create_table(table)
    logger.info(f"Table ready: {table_id}")


def streaming_insert(client, table_id, rows, chunk=500):
    for i in range(0, len(rows), chunk):
        errors = client.insert_rows_json(table_id, rows[i : i + chunk])
        if errors:
            logger.error(f"BQ insert errors at row {i}: {errors[:2]}")
    logger.info(f"Inserted {len(rows):,} rows → {table_id}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    client = bigquery.Client(project=PROJECT)
    now_ts = datetime.now(timezone.utc).isoformat()

    # 1. Load trending_historical
    logger.info("Reading trending_historical …")
    select_cols = ", ".join(META_COLS + FEATURE_COLS)
    df = client.query(
        f"SELECT {select_cols} FROM `{SRC_TABLE}` WHERE tempo IS NOT NULL"
    ).to_dataframe()
    logger.info(f"Loaded {len(df):,} tracks")

    # 2. Drop rows with any null feature (should be minimal)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with null features")

    # 3. Scale features
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Pick best k via silhouette score
    logger.info(f"Fitting KMeans for k in {list(K_RANGE)} …")
    best_k, best_score, best_model = None, -1.0, None
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(3000, len(X_scaled)))
        logger.info(f"  k={k}  silhouette={sil:.4f}")
        if sil > best_score:
            best_k, best_score, best_model = k, sil, km

    logger.info(f"Best k={best_k} (silhouette={best_score:.4f})")

    # 5. Assign clusters + archetypes
    df["cluster_id"] = best_model.labels_

    # Centroids in original feature space
    centroids_orig = pd.DataFrame(
        scaler.inverse_transform(best_model.cluster_centers_),
        columns=FEATURE_COLS,
    )
    cluster_name_map = name_clusters(centroids_orig, best_k)
    df["mood_archetype"] = df["cluster_id"].map(cluster_name_map)

    logger.info("Cluster → archetype mapping:")
    for cid, name in sorted(cluster_name_map.items()):
        n = (df["cluster_id"] == cid).sum()
        logger.info(f"  {cid} → {name:12s}  ({n:,} tracks)")

    # 6. Write audio_mood_clusters
    ensure_table(client, DST_TABLE, DST_SCHEMA)
    track_rows = [
        {
            "title":            str(r["title"]),
            "artist":           str(r["artist"]),
            "week_start":       str(pd.to_datetime(r["week_start"]).date()),
            "chart_name":       str(r["chart_name"]) if pd.notna(r["chart_name"]) else None,
            "rank":             int(r["rank"]) if pd.notna(r["rank"]) else None,
            "genre":            str(r["itunes_genre"]) if pd.notna(r["itunes_genre"]) else None,
            "cluster_id":       int(r["cluster_id"]),
            "mood_archetype":   r["mood_archetype"],
            "valence":          round(float(r["valence"]), 6),
            "energy":           round(float(r["energy"]), 6),
            "danceability":     round(float(r["danceability"]), 6),
            "acousticness":     round(float(r["acousticness"]), 6),
            "tempo":            round(float(r["tempo"]), 6),
            "loudness":         round(float(r["loudness"]), 6),
            "instrumentalness": round(float(r["instrumentalness"]), 6),
            "speechiness":      round(float(r["speechiness"]), 6),
            "ingested_at":      now_ts,
        }
        for _, r in df.iterrows()
    ]
    streaming_insert(client, DST_TABLE, track_rows)

    # 7. Weekly aggregation
    logger.info("Building weekly mood aggregates …")
    # week_start already exists in the source table — just normalise to string
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.date.astype(str)

    agg_rows = []
    for (week_start, chart_name), grp in df.groupby(["week_start", "chart_name"]):
        mood_counts  = grp["mood_archetype"].value_counts()
        total        = len(grp)
        dominant     = mood_counts.index[0] if len(mood_counts) else "unknown"
        pcts = {
            f"{arch}_pct": round(mood_counts.get(arch, 0) / total, 6)
            for arch in ARCHETYPE_NAMES
        }
        agg_rows.append({
            "week_start":       str(week_start),
            "chart_name":       str(chart_name),
            "dominant_mood":    dominant,
            "track_count":      total,
            **pcts,
            "avg_valence":      round(float(grp["valence"].mean()),      6),
            "avg_energy":       round(float(grp["energy"].mean()),       6),
            "avg_danceability": round(float(grp["danceability"].mean()), 6),
            "avg_tempo":        round(float(grp["tempo"].mean()),        6),
            "ingested_at":      now_ts,
        })

    ensure_table(client, AGG_TABLE, AGG_SCHEMA)
    streaming_insert(client, AGG_TABLE, agg_rows)

    # 8. Summary
    logger.info("─── LAYER 2 COMPLETE ───")
    logger.info(f"  Tracks clustered     : {len(df):,}")
    logger.info(f"  Optimal k            : {best_k}")
    logger.info(f"  Silhouette score     : {best_score:.4f}")
    logger.info(f"  Weekly agg rows      : {len(agg_rows):,}")
    arch_dist = df["mood_archetype"].value_counts().to_dict()
    logger.info(f"  Archetype distribution: {arch_dist}")


if __name__ == "__main__":
    main()
