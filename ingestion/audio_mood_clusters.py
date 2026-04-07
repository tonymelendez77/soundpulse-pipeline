"""
M10: KMeans clustering on 30 audio features from trending_historical.
Outputs per-track clusters + weekly/regional mood aggregates.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from google.cloud import bigquery
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT = "soundpulse-production"
DATASET = "music_analytics"
SRC_TABLE = f"{PROJECT}.{DATASET}.trending_historical"
DST_TABLE = f"{PROJECT}.{DATASET}.audio_mood_clusters"
AGG_TABLE = f"{PROJECT}.{DATASET}.audio_mood_weekly"
REGIONAL_TABLE = f"{PROJECT}.{DATASET}.audio_mood_regional"

K_RANGE = range(3, 8)
RANDOM_STATE = 42

CHART_REGION_MAP = {
    "Hot 100": "north_america",
    "Pop Songs": "north_america",
    "Dance Club Play": "north_america",
    "Rhythmic 40": "north_america",
    "Hot Latin Songs": "latin_america",
    "Latin": "latin_america",
    "Latin Pop Airplay": "latin_america",
    "Regional Mexican Airplay": "latin_america",
    "Tropical Airplay": "latin_america",
    "Global 200": "global",
    "Billboard Global 200": "global",
}

MARKET_REGION_MAP = {
    "usa": "north_america",
    "latin_america": "latin_america",
    "central_america": "latin_america",
    "europe": "europe",
    "global": "global",
}

REGIONS = ["north_america", "latin_america", "europe", "global"]

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

META_COLS = ["title", "artist", "week_start", "chart_name", "rank", "itunes_genre"]

DST_SCHEMA = [
    bigquery.SchemaField("title", "STRING"),
    bigquery.SchemaField("artist", "STRING"),
    bigquery.SchemaField("week_start", "DATE"),
    bigquery.SchemaField("chart_name", "STRING"),
    bigquery.SchemaField("rank", "INTEGER"),
    bigquery.SchemaField("genre", "STRING"),
    bigquery.SchemaField("cluster_id", "INTEGER"),
    bigquery.SchemaField("mood_archetype", "STRING"),
    bigquery.SchemaField("valence", "FLOAT64"),
    bigquery.SchemaField("energy", "FLOAT64"),
    bigquery.SchemaField("danceability", "FLOAT64"),
    bigquery.SchemaField("acousticness", "FLOAT64"),
    bigquery.SchemaField("tempo", "FLOAT64"),
    bigquery.SchemaField("loudness", "FLOAT64"),
    bigquery.SchemaField("instrumentalness", "FLOAT64"),
    bigquery.SchemaField("speechiness", "FLOAT64"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP"),
]

AGG_SCHEMA = [
    bigquery.SchemaField("week_start", "DATE"),
    bigquery.SchemaField("chart_name", "STRING"),
    bigquery.SchemaField("dominant_mood", "STRING"),
    bigquery.SchemaField("track_count", "INTEGER"),
    bigquery.SchemaField("euphoric_pct", "FLOAT64"),
    bigquery.SchemaField("melancholic_pct", "FLOAT64"),
    bigquery.SchemaField("aggressive_pct", "FLOAT64"),
    bigquery.SchemaField("peaceful_pct", "FLOAT64"),
    bigquery.SchemaField("groovy_pct", "FLOAT64"),
    bigquery.SchemaField("avg_valence", "FLOAT64"),
    bigquery.SchemaField("avg_energy", "FLOAT64"),
    bigquery.SchemaField("avg_danceability", "FLOAT64"),
    bigquery.SchemaField("avg_tempo", "FLOAT64"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP"),
]

REGIONAL_SCHEMA = [
    bigquery.SchemaField("week_start", "DATE"),
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("dominant_mood", "STRING"),
    bigquery.SchemaField("track_count", "INTEGER"),
    bigquery.SchemaField("euphoric_pct", "FLOAT64"),
    bigquery.SchemaField("melancholic_pct", "FLOAT64"),
    bigquery.SchemaField("aggressive_pct", "FLOAT64"),
    bigquery.SchemaField("peaceful_pct", "FLOAT64"),
    bigquery.SchemaField("groovy_pct", "FLOAT64"),
    bigquery.SchemaField("avg_valence", "FLOAT64"),
    bigquery.SchemaField("avg_energy", "FLOAT64"),
    bigquery.SchemaField("avg_danceability", "FLOAT64"),
    bigquery.SchemaField("avg_tempo", "FLOAT64"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP"),
]

ARCHETYPE_NAMES = ["euphoric", "melancholic", "aggressive", "peaceful", "groovy"]


def name_clusters(centroids_df: pd.DataFrame, k: int) -> dict[int, str]:
    """Map cluster IDs to mood archetypes using Russell's Circumplex quadrants.
    Uses raw centroid values (not normalised) to avoid all clusters looking the same."""

    def _quadrant_score(v, e):
        high_v = max(0.0, (v - 0.5) * 2.0)
        low_v = max(0.0, (0.5 - v) * 2.0)
        high_e = max(0.0, (e - 0.5) * 2.0)
        low_e = max(0.0, (0.5 - e) * 2.0)
        return high_v, low_v, high_e, low_e

    def score(row, archetype):
        v = float(row.get("valence", 0.5))
        e = float(row.get("energy", 0.5))
        d = float(row.get("danceability", 0.5))
        ac = float(row.get("acousticness", 0.1))
        lo = float(row.get("loudness", -10.0))
        ins = float(row.get("instrumentalness", 0.0))

        lo_n = max(0.0, min(1.0, (lo + 20.0) / 20.0))
        high_v, low_v, high_e, low_e = _quadrant_score(v, e)

        if archetype == "euphoric":
            base = high_v * 0.45 + high_e * 0.35 + d * 0.20
            gate = max(0.1, high_v)
            return base * gate
        if archetype == "aggressive":
            base = low_v * 0.50 + high_e * 0.35 + lo_n * 0.15
            gate = max(0.05, low_v)
            return base * gate
        if archetype == "melancholic":
            base = low_v * 0.45 + low_e * 0.30 + ac * 0.15 + (1.0 - d) * 0.10
            gate = max(0.05, low_v * low_e)
            return base * (1.0 + gate)
        if archetype == "peaceful":
            base = high_v * 0.40 + low_e * 0.30 + ac * 0.20 + ins * 0.10
            gate = max(0.05, high_v)
            return base * gate
        if archetype == "groovy":
            e_groove = max(0.0, 1.0 - abs(e - 0.65) * 3.0)
            return d * 0.55 + e_groove * 0.25 + max(0.0, v - 0.35) * 0.20
        return 0.0

    archetypes_to_assign = ARCHETYPE_NAMES[:k]

    score_matrix = pd.DataFrame(
        {arch: [score(centroids_df.iloc[i], arch) for i in range(k)]
         for arch in archetypes_to_assign}
    )

    # greedy: pick highest-scoring (cluster, archetype) pairs
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

    for i in range(k):
        if i not in assignment:
            assignment[i] = f"cluster_{i}"

    return assignment


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
    logger.info(f"Inserted {len(rows):,} rows into {table_id}")


def main():
    client = bigquery.Client(project=PROJECT)
    now_ts = datetime.now(timezone.utc).isoformat()

    select_cols = ", ".join(META_COLS + FEATURE_COLS)
    df = client.query(
        f"SELECT {select_cols} FROM `{SRC_TABLE}` WHERE tempo IS NOT NULL"
    ).to_dataframe()
    logger.info(f"Loaded {len(df):,} tracks")

    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with null features")

    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, best_score, best_model = None, -1.0, None
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(3000, len(X_scaled)))
        logger.info(f"  k={k}  silhouette={sil:.4f}")
        if sil > best_score:
            best_k, best_score, best_model = k, sil, km

    logger.info(f"Best k={best_k} (silhouette={best_score:.4f})")

    df["cluster_id"] = best_model.labels_
    centroids_orig = pd.DataFrame(
        scaler.inverse_transform(best_model.cluster_centers_),
        columns=FEATURE_COLS,
    )
    cluster_name_map = name_clusters(centroids_orig, best_k)
    df["mood_archetype"] = df["cluster_id"].map(cluster_name_map)

    for cid, name in sorted(cluster_name_map.items()):
        n = (df["cluster_id"] == cid).sum()
        logger.info(f"  {cid} -> {name:12s}  ({n:,} tracks)")

    # write per-track clusters
    ensure_table(client, DST_TABLE, DST_SCHEMA)
    track_rows = [
        {
            "title": str(r["title"]),
            "artist": str(r["artist"]),
            "week_start": str(pd.to_datetime(r["week_start"]).date()),
            "chart_name": str(r["chart_name"]) if pd.notna(r["chart_name"]) else None,
            "rank": int(r["rank"]) if pd.notna(r["rank"]) else None,
            "genre": str(r["itunes_genre"]) if pd.notna(r["itunes_genre"]) else None,
            "cluster_id": int(r["cluster_id"]),
            "mood_archetype": r["mood_archetype"],
            "valence": round(float(r["valence"]), 6),
            "energy": round(float(r["energy"]), 6),
            "danceability": round(float(r["danceability"]), 6),
            "acousticness": round(float(r["acousticness"]), 6),
            "tempo": round(float(r["tempo"]), 6),
            "loudness": round(float(r["loudness"]), 6),
            "instrumentalness": round(float(r["instrumentalness"]), 6),
            "speechiness": round(float(r["speechiness"]), 6),
            "ingested_at": now_ts,
        }
        for _, r in df.iterrows()
    ]
    streaming_insert(client, DST_TABLE, track_rows)

    # weekly aggregation by chart
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.date.astype(str)

    agg_rows = []
    for (week_start, chart_name), grp in df.groupby(["week_start", "chart_name"]):
        mood_counts = grp["mood_archetype"].value_counts()
        total = len(grp)
        dominant = mood_counts.index[0] if len(mood_counts) else "unknown"
        pcts = {
            f"{arch}_pct": round(mood_counts.get(arch, 0) / total, 6)
            for arch in ARCHETYPE_NAMES
        }
        agg_rows.append({
            "week_start": str(week_start),
            "chart_name": str(chart_name),
            "dominant_mood": dominant,
            "track_count": total,
            **pcts,
            "avg_valence": round(float(grp["valence"].mean()), 6),
            "avg_energy": round(float(grp["energy"].mean()), 6),
            "avg_danceability": round(float(grp["danceability"].mean()), 6),
            "avg_tempo": round(float(grp["tempo"].mean()), 6),
            "ingested_at": now_ts,
        })

    ensure_table(client, AGG_TABLE, AGG_SCHEMA)
    streaming_insert(client, AGG_TABLE, agg_rows)

    # regional aggregation with z-score labeling per region
    df["region"] = df["chart_name"].map(CHART_REGION_MAP).fillna("global")

    regional_rows = []
    for region, rdf in df.groupby("region"):
        week_pcts = []
        for week_start, grp in rdf.groupby("week_start"):
            mood_counts = grp["mood_archetype"].value_counts()
            total = len(grp)
            pcts = {f"{arch}_pct": mood_counts.get(arch, 0) / total for arch in ARCHETYPE_NAMES}
            pcts["week_start"] = str(week_start)
            pcts["track_count"] = total
            pcts["avg_valence"] = float(grp["valence"].mean())
            pcts["avg_energy"] = float(grp["energy"].mean())
            pcts["avg_danceability"] = float(grp["danceability"].mean())
            pcts["avg_tempo"] = float(grp["tempo"].mean())
            week_pcts.append(pcts)

        if not week_pcts:
            continue
        wpdf = pd.DataFrame(week_pcts)

        active_archetypes = ["euphoric", "melancholic", "aggressive"]
        z_cols = []
        for arch in active_archetypes:
            col = f"{arch}_pct"
            mean_v = wpdf[col].mean()
            std_v = wpdf[col].std()
            z_col = f"{col}_z"
            wpdf[z_col] = (wpdf[col] - mean_v) / std_v if std_v > 1e-4 else 0.0
            z_cols.append(z_col)

        z_name_map = {f"{a}_pct_z": a for a in active_archetypes}
        wpdf["dominant_mood"] = wpdf[z_cols].idxmax(axis=1).map(z_name_map).fillna("aggressive")

        mood_dist = wpdf["dominant_mood"].value_counts().to_dict()
        logger.info(f"  [{region}] {len(wpdf)} weeks, moods: {mood_dist}")

        for _, row in wpdf.iterrows():
            regional_rows.append({
                "week_start": str(row["week_start"]),
                "region": region,
                "dominant_mood": row["dominant_mood"],
                "track_count": int(row["track_count"]),
                "euphoric_pct": round(float(row["euphoric_pct"]), 6),
                "melancholic_pct": round(float(row["melancholic_pct"]), 6),
                "aggressive_pct": round(float(row["aggressive_pct"]), 6),
                "peaceful_pct": round(float(row.get("peaceful_pct", 0)), 6),
                "groovy_pct": round(float(row.get("groovy_pct", 0)), 6),
                "avg_valence": round(float(row["avg_valence"]), 6),
                "avg_energy": round(float(row["avg_energy"]), 6),
                "avg_danceability": round(float(row["avg_danceability"]), 6),
                "avg_tempo": round(float(row["avg_tempo"]), 6),
                "ingested_at": now_ts,
            })

    ensure_table(client, REGIONAL_TABLE, REGIONAL_SCHEMA)
    streaming_insert(client, REGIONAL_TABLE, regional_rows)

    logger.info(f"Done: {len(df):,} tracks, k={best_k}, {len(agg_rows)} weekly, {len(regional_rows)} regional")
    logger.info(f"Archetypes: {df['mood_archetype'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
