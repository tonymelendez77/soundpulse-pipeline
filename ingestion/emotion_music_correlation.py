"""
SoundPulse — Module 12, Layer 3
Emotion–Music Correlation Analysis

Input:  music_analytics.news_sentiment_weekly  (Layer 1 output)
        music_analytics.audio_mood_weekly       (Layer 2 output)
Output: music_analytics.emotion_music_correlation

Answers: which world emotions drive which audio mood archetypes?
  — Pearson correlation: anxiety_index → euphoric_pct, etc.
  — Weekly joined table for Layer 4 (XGBoost features)
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from google.cloud import bigquery
from loguru import logger
from scipy.stats import pearsonr

# ── Config ──────────────────────────────────────────────────────────────────────
PROJECT       = "soundpulse-production"
DATASET       = "music_analytics"
NEWS_AGG      = f"{PROJECT}.{DATASET}.news_sentiment_weekly"
AUDIO_AGG     = f"{PROJECT}.{DATASET}.audio_mood_weekly"
CORR_TABLE    = f"{PROJECT}.{DATASET}.emotion_music_correlation"
JOINED_TABLE  = f"{PROJECT}.{DATASET}.weekly_features"   # feed for Layer 4

EMOTION_COLS  = ["avg_fear", "avg_anger", "avg_joy", "avg_sadness",
                 "avg_surprise", "avg_disgust", "avg_neutral",
                 "anxiety_index", "tension_index", "positivity_index"]

MOOD_PCT_COLS = ["euphoric_pct", "melancholic_pct", "aggressive_pct",
                 "peaceful_pct", "groovy_pct"]

AUDIO_FEATURE_COLS = ["avg_valence", "avg_energy", "avg_danceability", "avg_tempo"]

CORR_SCHEMA = [
    bigquery.SchemaField("emotion",         "STRING"),
    bigquery.SchemaField("mood_archetype",  "STRING"),
    bigquery.SchemaField("pearson_r",       "FLOAT64"),
    bigquery.SchemaField("p_value",         "FLOAT64"),
    bigquery.SchemaField("significant",     "BOOLEAN"),   # p < 0.05
    bigquery.SchemaField("direction",       "STRING"),    # positive / negative
    bigquery.SchemaField("weeks_compared",  "INTEGER"),
    bigquery.SchemaField("ingested_at",     "TIMESTAMP"),
]

JOINED_SCHEMA = [
    bigquery.SchemaField("week_start",           "DATE"),
    # news emotion scores (averaged across all topics)
    bigquery.SchemaField("avg_fear",             "FLOAT64"),
    bigquery.SchemaField("avg_anger",            "FLOAT64"),
    bigquery.SchemaField("avg_joy",              "FLOAT64"),
    bigquery.SchemaField("avg_sadness",          "FLOAT64"),
    bigquery.SchemaField("avg_surprise",         "FLOAT64"),
    bigquery.SchemaField("avg_disgust",          "FLOAT64"),
    bigquery.SchemaField("avg_neutral",          "FLOAT64"),
    bigquery.SchemaField("anxiety_index",        "FLOAT64"),
    bigquery.SchemaField("tension_index",        "FLOAT64"),
    bigquery.SchemaField("positivity_index",     "FLOAT64"),
    bigquery.SchemaField("dominant_emotion",     "STRING"),
    # audio mood distribution (averaged across all sources)
    bigquery.SchemaField("euphoric_pct",         "FLOAT64"),
    bigquery.SchemaField("melancholic_pct",      "FLOAT64"),
    bigquery.SchemaField("aggressive_pct",       "FLOAT64"),
    bigquery.SchemaField("peaceful_pct",         "FLOAT64"),
    bigquery.SchemaField("groovy_pct",           "FLOAT64"),
    bigquery.SchemaField("dominant_mood",        "STRING"),
    bigquery.SchemaField("avg_valence",          "FLOAT64"),
    bigquery.SchemaField("avg_energy",           "FLOAT64"),
    bigquery.SchemaField("avg_danceability",     "FLOAT64"),
    bigquery.SchemaField("avg_tempo",            "FLOAT64"),
    bigquery.SchemaField("ingested_at",          "TIMESTAMP"),
]


# ── BQ helpers ──────────────────────────────────────────────────────────────────

def ensure_table(client, table_id, schema):
    client.delete_table(table_id, not_found_ok=True)
    table = bigquery.Table(table_id, schema=schema)
    client.create_table(table)
    time.sleep(3)   # BQ streaming insert needs a moment after table (re)creation
    logger.info(f"Table ready: {table_id}")


def streaming_insert(client, table_id, rows, chunk=500):
    for i in range(0, len(rows), chunk):
        errors = client.insert_rows_json(table_id, rows[i : i + chunk])
        if errors:
            logger.error(f"Insert error at {i}: {errors[:2]}")
    logger.info(f"Inserted {len(rows):,} rows → {table_id}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    client = bigquery.Client(project=PROJECT)
    now_ts = datetime.now(timezone.utc).isoformat()

    # 1. Load news_sentiment_weekly — average across topics per week
    logger.info("Reading news_sentiment_weekly …")
    news_df = client.query(f"SELECT * FROM `{NEWS_AGG}`").to_dataframe()
    logger.info(f"  {len(news_df):,} rows (week × topic)")

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

    # 2. Load audio_mood_weekly — average across sources per week
    logger.info("Reading audio_mood_weekly …")
    audio_df = client.query(f"SELECT * FROM `{AUDIO_AGG}`").to_dataframe()
    logger.info(f"  {len(audio_df):,} rows (week × source)")

    audio_weekly = (
        audio_df.groupby("week_start")
        .agg(
            {col: "mean" for col in MOOD_PCT_COLS + AUDIO_FEATURE_COLS}
        )
        .reset_index()
    )
    audio_weekly["week_start"] = pd.to_datetime(audio_weekly["week_start"])

    # Dominant mood via relative z-score ranking.
    # Because one mood archetype (e.g. "aggressive") can dominate every week in
    # absolute terms, mode-based winner-take-all produces a single class label for
    # ALL weeks, making training useless.  Z-score ranking labels each week by
    # WHICH MOOD IS MOST ABOVE ITS OWN HISTORICAL AVERAGE — giving the classifier
    # genuine class diversity even when absolute distributions are similar.
    z_map = {"euphoric_pct_z": "euphoric", "melancholic_pct_z": "melancholic",
             "aggressive_pct_z": "aggressive"}
    for col in ["euphoric_pct", "melancholic_pct", "aggressive_pct"]:
        mean_val = audio_weekly[col].mean()
        std_val  = audio_weekly[col].std()
        z_col    = f"{col}_z"
        if std_val > 1e-4:
            audio_weekly[z_col] = (audio_weekly[col] - mean_val) / std_val
        else:
            audio_weekly[z_col] = 0.0
    audio_weekly["dominant_mood"] = (
        audio_weekly[list(z_map.keys())].idxmax(axis=1).map(z_map)
    )
    # Log distribution so we can see diversity
    mood_dist = audio_weekly["dominant_mood"].value_counts().to_dict()
    logger.info(f"  dominant_mood distribution (z-score): {mood_dist}")

    # 3. Left join: keep ALL audio weeks, fill missing news emotions with column means
    # This ensures the model sees all 52 weeks of diverse audio data even when
    # news backfill is incomplete (e.g. Guardian API rate-limited to 13 weeks).
    joined = pd.merge(audio_weekly, news_weekly, on="week_start", how="left")
    # Mean-impute missing emotion features
    for col in EMOTION_COLS:
        col_mean = joined[col].mean()
        joined[col] = joined[col].fillna(col_mean if pd.notna(col_mean) else 0.0)
    joined["dominant_emotion"] = joined["dominant_emotion"].fillna("neutral")
    n_imputed = joined["dominant_emotion"].eq("neutral").sum()
    logger.info(
        f"Joined {len(joined):,} weeks (audio-anchored left join, "
        f"{n_imputed} weeks had news emotion imputed from column means)"
    )

    if len(joined) < 3:
        logger.error("Too few overlapping weeks for correlation — check that both Layer 1 and Layer 2 ran successfully.")
        return

    # 4. Pearson correlations: every emotion × every mood pct
    logger.info("Computing Pearson correlations …")
    corr_rows = []
    for emotion in EMOTION_COLS:
        for mood_pct in MOOD_PCT_COLS:
            x = joined[emotion].values
            y = joined[mood_pct].values
            # Drop pairs where either is NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 3:
                continue
            r, p = pearsonr(x[mask], y[mask])
            # Skip constant-input pairs — NaN is not valid JSON for BigQuery
            if np.isnan(r) or np.isnan(p):
                logger.debug(f"  Skipping {emotion} ↔ {mood_pct} (constant input, r=NaN)")
                continue
            corr_rows.append({
                "emotion":        emotion,
                "mood_archetype": mood_pct.replace("_pct", ""),
                "pearson_r":      round(float(r), 6),
                "p_value":        round(float(p), 6),
                "significant":    bool(p < 0.05),
                "direction":      "positive" if r > 0 else "negative",
                "weeks_compared": int(mask.sum()),
                "ingested_at":    now_ts,
            })

    # Sort by absolute correlation strength
    corr_rows.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    logger.info("Top 10 correlations:")
    for row in corr_rows[:10]:
        sig = "✓" if row["significant"] else " "
        logger.info(
            f"  [{sig}] {row['emotion']:20s} ↔ {row['mood_archetype']:12s}  "
            f"r={row['pearson_r']:+.3f}  p={row['p_value']:.3f}"
        )

    # 5. Write emotion_music_correlation
    ensure_table(client, CORR_TABLE, CORR_SCHEMA)
    streaming_insert(client, CORR_TABLE, corr_rows)

    # 6. Write weekly_features (joined table — input for Layer 4 XGBoost)
    joined_rows = []
    for _, r in joined.iterrows():
        row_dict = {"week_start": str(r["week_start"].date()), "ingested_at": now_ts}
        for col in EMOTION_COLS:
            row_dict[col] = round(float(r[col]), 6) if pd.notna(r[col]) else None
        row_dict["dominant_emotion"] = str(r.get("dominant_emotion", ""))
        for col in MOOD_PCT_COLS + AUDIO_FEATURE_COLS:
            row_dict[col] = round(float(r[col]), 6) if pd.notna(r[col]) else None
        row_dict["dominant_mood"] = str(r.get("dominant_mood", ""))
        joined_rows.append(row_dict)

    ensure_table(client, JOINED_TABLE, JOINED_SCHEMA)
    streaming_insert(client, JOINED_TABLE, joined_rows)

    # 7. Summary
    sig_count = sum(1 for r in corr_rows if r["significant"])
    logger.info("─── LAYER 3 COMPLETE ───")
    logger.info(f"  Correlation pairs computed : {len(corr_rows)}")
    logger.info(f"  Significant (p<0.05)       : {sig_count}")
    logger.info(f"  Weeks in analysis          : {len(joined)}")
    logger.info(f"  weekly_features table ready for Layer 4 XGBoost")


if __name__ == "__main__":
    main()
