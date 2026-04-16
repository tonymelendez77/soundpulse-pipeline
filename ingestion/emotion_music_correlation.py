"""Weekly emotion-mood correlation and feature table builder."""

import io
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from google.cloud import bigquery
from loguru import logger
from scipy.stats import pearsonr

PROJECT = "soundpulse-production"
DATASET = "music_analytics"
NEWS_AGG = f"{PROJECT}.{DATASET}.news_sentiment_weekly"
AUDIO_REGIONAL = f"{PROJECT}.{DATASET}.audio_mood_regional"
CORR_TABLE = f"{PROJECT}.{DATASET}.emotion_music_correlation"
JOINED_TABLE = f"{PROJECT}.{DATASET}.weekly_features"

REGIONS = ["north_america", "latin_america", "europe", "global"]

EMOTION_COLS = ["avg_fear", "avg_anger", "avg_joy", "avg_sadness",
                "avg_surprise", "avg_disgust", "avg_neutral",
                "anxiety_index", "tension_index", "positivity_index"]

MOOD_PCT_COLS = ["euphoric_pct", "melancholic_pct", "aggressive_pct",
                 "peaceful_pct", "groovy_pct"]

AUDIO_FEATURE_COLS = ["avg_valence", "avg_energy", "avg_danceability", "avg_tempo"]

CORR_SCHEMA = [
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("emotion", "STRING"),
    bigquery.SchemaField("mood_archetype", "STRING"),
    bigquery.SchemaField("pearson_r", "FLOAT64"),
    bigquery.SchemaField("p_value", "FLOAT64"),
    bigquery.SchemaField("significant", "BOOLEAN"),
    bigquery.SchemaField("direction", "STRING"),
    bigquery.SchemaField("weeks_compared", "INTEGER"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP"),
]

JOINED_SCHEMA = [
    bigquery.SchemaField("week_start", "DATE"),
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("avg_fear", "FLOAT64"),
    bigquery.SchemaField("avg_anger", "FLOAT64"),
    bigquery.SchemaField("avg_joy", "FLOAT64"),
    bigquery.SchemaField("avg_sadness", "FLOAT64"),
    bigquery.SchemaField("avg_surprise", "FLOAT64"),
    bigquery.SchemaField("avg_disgust", "FLOAT64"),
    bigquery.SchemaField("avg_neutral", "FLOAT64"),
    bigquery.SchemaField("anxiety_index", "FLOAT64"),
    bigquery.SchemaField("tension_index", "FLOAT64"),
    bigquery.SchemaField("positivity_index", "FLOAT64"),
    bigquery.SchemaField("dominant_emotion", "STRING"),
    bigquery.SchemaField("euphoric_pct", "FLOAT64"),
    bigquery.SchemaField("melancholic_pct", "FLOAT64"),
    bigquery.SchemaField("aggressive_pct", "FLOAT64"),
    bigquery.SchemaField("peaceful_pct", "FLOAT64"),
    bigquery.SchemaField("groovy_pct", "FLOAT64"),
    bigquery.SchemaField("dominant_mood", "STRING"),
    bigquery.SchemaField("avg_valence", "FLOAT64"),
    bigquery.SchemaField("avg_energy", "FLOAT64"),
    bigquery.SchemaField("avg_danceability", "FLOAT64"),
    bigquery.SchemaField("avg_tempo", "FLOAT64"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP"),
]


def ensure_table(client, table_id, schema):
    client.delete_table(table_id, not_found_ok=True)
    table = bigquery.Table(table_id, schema=schema)
    client.create_table(table)
    time.sleep(10)
    logger.info(f"Table ready: {table_id}")


def load_rows(client, table_id, rows):
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    ndjson = "\n".join(json.dumps(r) for r in rows)
    job = client.load_table_from_file(
        io.BytesIO(ndjson.encode()),
        table_id,
        job_config=job_config,
    )
    job.result()
    logger.info(f"Loaded {len(rows):,} rows -> {table_id}")


def main():
    client = bigquery.Client(project=PROJECT)
    now_ts = datetime.now(timezone.utc).isoformat()

    # load news sentiment and average across topics per week
    logger.info("Reading news_sentiment_weekly ...")
    news_df = client.query(f"SELECT * FROM `{NEWS_AGG}`").to_dataframe()
    logger.info(f"  {len(news_df):,} rows (week x topic)")

    news_weekly = (
        news_df.groupby("week_start")[
            EMOTION_COLS + ["dominant_emotion"]
        ]
        .agg(
            {col: "mean" for col in EMOTION_COLS} | {"dominant_emotion": lambda x: x.mode()[0]}
        )
        .reset_index()
    )
    news_weekly["week_start"] = pd.to_datetime(news_weekly["week_start"])

    logger.info("Reading audio_mood_regional ...")
    audio_df = client.query(f"SELECT * FROM `{AUDIO_REGIONAL}`").to_dataframe()
    logger.info(f"  {len(audio_df):,} rows (week x region)")
    audio_df["week_start"] = pd.to_datetime(audio_df["week_start"])

    if audio_df.empty:
        logger.error("audio_mood_regional is empty -- run audio_mood_clusters.py first.")
        return

    all_joined = []
    corr_rows = []
    z_map = {"euphoric_pct_z": "euphoric", "melancholic_pct_z": "melancholic",
              "aggressive_pct_z": "aggressive"}

    for region in REGIONS:
        region_df = audio_df[audio_df["region"] == region].copy()
        if region_df.empty:
            logger.warning(f"  [{region}] No audio data -- skipping")
            continue

        logger.info(f"  [{region}] {len(region_df)} weeks of audio data")

        # z-score dominant_mood within this region's history
        for col in ["euphoric_pct", "melancholic_pct", "aggressive_pct"]:
            mean_v = region_df[col].mean()
            std_v = region_df[col].std()
            z_col = f"{col}_z"
            if pd.notna(std_v) and std_v > 1e-4:
                region_df[z_col] = (region_df[col] - mean_v) / std_v
            else:
                region_df[z_col] = 0.0
        region_df["dominant_mood"] = (
            region_df[list(z_map.keys())].idxmax(axis=1).map(z_map)
        )
        mood_dist = region_df["dominant_mood"].value_counts().to_dict()
        logger.info(f"  [{region}] dominant_mood (z-score): {mood_dist}")

        # join with global news (no regional split on news yet)
        joined = pd.merge(region_df, news_weekly, on="week_start", how="left")
        for col in EMOTION_COLS:
            col_mean = joined[col].mean()
            joined[col] = joined[col].fillna(col_mean if pd.notna(col_mean) else 0.0)
        joined["dominant_emotion"] = joined["dominant_emotion"].fillna("neutral")
        n_imputed = joined["dominant_emotion"].eq("neutral").sum()
        logger.info(
            f"  [{region}] {len(joined)} joined weeks "
            f"({n_imputed} emotion-imputed from column means)"
        )
        joined["region"] = region
        all_joined.append(joined)

        # pearson correlations per region
        if len(joined) < 3:
            logger.warning(f"  [{region}] Too few weeks for correlation -- skipping")
            continue
        for emotion in EMOTION_COLS:
            for mood_pct in MOOD_PCT_COLS:
                x = joined[emotion].values
                y = joined[mood_pct].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 3:
                    continue
                r, p = pearsonr(x[mask], y[mask])
                if np.isnan(r) or np.isnan(p):
                    continue
                corr_rows.append({
                    "region": region,
                    "emotion": emotion,
                    "mood_archetype": mood_pct.replace("_pct", ""),
                    "pearson_r": round(float(r), 6),
                    "p_value": round(float(p), 6),
                    "significant": bool(p < 0.05),
                    "direction": "positive" if r > 0 else "negative",
                    "weeks_compared": int(mask.sum()),
                    "ingested_at": now_ts,
                })

    if not all_joined:
        logger.error("No regional data found -- aborting.")
        return

    full_joined = pd.concat(all_joined, ignore_index=True)

    # sort correlations by absolute strength
    corr_rows.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
    logger.info(f"Top 10 correlations (all regions):")
    for row in corr_rows[:10]:
        sig = "Y" if row["significant"] else " "
        logger.info(
            f"  [{sig}] {row['region']:14s}  {row['emotion']:20s} <> {row['mood_archetype']:12s}  "
            f"r={row['pearson_r']:+.3f}  p={row['p_value']:.3f}"
        )

    # write correlation table
    ensure_table(client, CORR_TABLE, CORR_SCHEMA)
    load_rows(client, CORR_TABLE, corr_rows)

    # write weekly_features (4 rows per week, one per region)
    joined_rows = []
    for _, r in full_joined.iterrows():
        row_dict = {
            "week_start": str(r["week_start"].date()),
            "region": str(r["region"]),
            "ingested_at": now_ts,
        }
        for col in EMOTION_COLS:
            row_dict[col] = round(float(r[col]), 6) if pd.notna(r[col]) else None
        row_dict["dominant_emotion"] = str(r.get("dominant_emotion", ""))
        for col in MOOD_PCT_COLS + AUDIO_FEATURE_COLS:
            row_dict[col] = round(float(r[col]), 6) if pd.notna(r[col]) else None
        row_dict["dominant_mood"] = str(r.get("dominant_mood", ""))
        joined_rows.append(row_dict)

    ensure_table(client, JOINED_TABLE, JOINED_SCHEMA)
    load_rows(client, JOINED_TABLE, joined_rows)

    sig_count = sum(1 for r in corr_rows if r["significant"])
    logger.info(f"Done -- {len(all_joined)} regions, {len(joined_rows)} feature rows, "
                f"{len(corr_rows)} correlations ({sig_count} significant)")


if __name__ == "__main__":
    main()
