"""
SoundPulse — Module 12, Layer 1
News Sentiment Classification using DistilRoBERTa

Model: j-hartmann/emotion-english-distilroberta-base
Labels: anger, disgust, fear, joy, neutral, sadness, surprise
Input:  music_analytics.news_historical  (6,348 rows, 83 days, 8 topics)
Output: music_analytics.news_sentiment   (per-article emotions)
        music_analytics.news_sentiment_weekly (aggregated by week + topic)

Install first:
    pip install transformers torch
"""

import math
import time
from datetime import datetime, timezone

import pandas as pd
from google.cloud import bigquery
from loguru import logger
from transformers import pipeline

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT   = "soundpulse-production"
DATASET   = "music_analytics"
SRC_TABLE = f"{PROJECT}.{DATASET}.news_historical"
DST_TABLE = f"{PROJECT}.{DATASET}.news_sentiment"
AGG_TABLE = f"{PROJECT}.{DATASET}.news_sentiment_weekly"

BATCH_SIZE  = 64          # articles per inference batch (adjust down if OOM)
MAX_TOKENS  = 128         # truncate input to this many tokens
WRITE_CHUNK = 500         # rows per BigQuery streaming insert

DST_SCHEMA = [
    bigquery.SchemaField("date",           "DATE"),
    bigquery.SchemaField("topic",          "STRING"),
    bigquery.SchemaField("title",          "STRING"),
    bigquery.SchemaField("emotion",        "STRING"),   # primary label
    bigquery.SchemaField("fear_score",     "FLOAT64"),
    bigquery.SchemaField("anger_score",    "FLOAT64"),
    bigquery.SchemaField("joy_score",      "FLOAT64"),
    bigquery.SchemaField("sadness_score",  "FLOAT64"),
    bigquery.SchemaField("surprise_score", "FLOAT64"),
    bigquery.SchemaField("disgust_score",  "FLOAT64"),
    bigquery.SchemaField("neutral_score",  "FLOAT64"),
    bigquery.SchemaField("ingested_at",    "TIMESTAMP"),
]

AGG_SCHEMA = [
    bigquery.SchemaField("week_start",           "DATE"),
    bigquery.SchemaField("topic",                "STRING"),
    bigquery.SchemaField("article_count",        "INTEGER"),
    bigquery.SchemaField("dominant_emotion",     "STRING"),
    bigquery.SchemaField("avg_fear",             "FLOAT64"),
    bigquery.SchemaField("avg_anger",            "FLOAT64"),
    bigquery.SchemaField("avg_joy",              "FLOAT64"),
    bigquery.SchemaField("avg_sadness",          "FLOAT64"),
    bigquery.SchemaField("avg_surprise",         "FLOAT64"),
    bigquery.SchemaField("avg_disgust",          "FLOAT64"),
    bigquery.SchemaField("avg_neutral",          "FLOAT64"),
    bigquery.SchemaField("anxiety_index",        "FLOAT64"),  # fear + sadness
    bigquery.SchemaField("tension_index",        "FLOAT64"),  # anger + disgust
    bigquery.SchemaField("positivity_index",     "FLOAT64"),  # joy + surprise
    bigquery.SchemaField("ingested_at",          "TIMESTAMP"),
]

EMOTION_LABELS = ["fear", "anger", "joy", "sadness", "surprise", "disgust", "neutral"]


# ── BQ helpers ─────────────────────────────────────────────────────────────────

def ensure_table(client: bigquery.Client, table_id: str, schema: list, drop_first: bool = True):
    if drop_first:
        client.delete_table(table_id, not_found_ok=True)
        logger.info(f"Dropped {table_id}")
    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table, exists_ok=True)
    logger.info(f"Table ready: {table_id}")
    return table


def streaming_insert(client: bigquery.Client, table_id: str, rows: list[dict]):
    """Insert rows in chunks using streaming API."""
    for i in range(0, len(rows), WRITE_CHUNK):
        chunk = rows[i : i + WRITE_CHUNK]
        errors = client.insert_rows_json(table_id, chunk)
        if errors:
            logger.error(f"BQ insert errors (chunk {i}): {errors[:3]}")
        else:
            logger.debug(f"Inserted rows {i}–{i + len(chunk) - 1}")


# ── Inference ──────────────────────────────────────────────────────────────────

def load_model():
    logger.info("Loading DistilRoBERTa emotion model …")
    clf = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,          # return all label scores
        truncation=True,
        max_length=MAX_TOKENS,
        device=-1,           # CPU; change to 0 for GPU
    )
    logger.info("Model loaded.")
    return clf


def build_text(row: pd.Series) -> str:
    """Combine title + description into a single string for the model."""
    title = str(row.get("title", "") or "").strip()
    desc  = str(row.get("description", "") or "").strip()
    if desc and desc.lower() != "none":
        return f"{title}. {desc}"
    return title


def scores_from_output(result: list[dict]) -> dict:
    """Convert [{label, score}, …] → {fear_score, anger_score, …}."""
    lookup = {item["label"].lower(): item["score"] for item in result}
    return {f"{lbl}_score": round(lookup.get(lbl, 0.0), 6) for lbl in EMOTION_LABELS}


def run_inference(clf, texts: list[str]) -> list[dict]:
    """Run model on a list of texts; returns list of score dicts."""
    results = clf(texts, batch_size=BATCH_SIZE)
    return [scores_from_output(r) for r in results]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    client = bigquery.Client(project=PROJECT)
    now_ts = datetime.now(timezone.utc).isoformat()

    # 1. Load news_historical
    logger.info("Reading news_historical …")
    query = f"""
        SELECT date, topic, title, description
        FROM `{SRC_TABLE}`
        WHERE title IS NOT NULL
        ORDER BY date, topic
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df):,} articles")

    # 2. Prepare input texts
    df["_text"] = df.apply(build_text, axis=1)
    texts = df["_text"].tolist()

    # 3. Load model + run inference in batches
    clf = load_model()
    all_scores: list[dict] = []
    n_batches = math.ceil(len(texts) / BATCH_SIZE)
    logger.info(f"Running inference: {len(texts):,} articles, {n_batches} batches …")

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        scores = run_inference(clf, batch)
        all_scores.extend(scores)
        if (i // BATCH_SIZE + 1) % 10 == 0:
            logger.info(f"  batch {i // BATCH_SIZE + 1}/{n_batches} done")

    logger.info("Inference complete.")

    # 4. Build output rows
    score_df = pd.DataFrame(all_scores)
    df = df.reset_index(drop=True)
    df = pd.concat([df, score_df], axis=1)

    # Primary emotion = label with highest score
    score_cols = [f"{lbl}_score" for lbl in EMOTION_LABELS]
    df["emotion"] = df[score_cols].idxmax(axis=1).str.replace("_score", "", regex=False)

    # Ensure date is a string (DATE-compatible)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    rows = [
        {
            "date":           row["date"],
            "topic":          row["topic"],
            "title":          str(row["title"])[:500],
            "emotion":        row["emotion"],
            "fear_score":     float(row["fear_score"]),
            "anger_score":    float(row["anger_score"]),
            "joy_score":      float(row["joy_score"]),
            "sadness_score":  float(row["sadness_score"]),
            "surprise_score": float(row["surprise_score"]),
            "disgust_score":  float(row["disgust_score"]),
            "neutral_score":  float(row["neutral_score"]),
            "ingested_at":    now_ts,
        }
        for _, row in df.iterrows()
    ]

    # 5. Write news_sentiment
    ensure_table(client, DST_TABLE, DST_SCHEMA)
    logger.info(f"Writing {len(rows):,} rows to {DST_TABLE} …")
    streaming_insert(client, DST_TABLE, rows)
    logger.info("news_sentiment written.")

    # 6. Weekly aggregation (in-memory, then write)
    logger.info("Building weekly aggregates …")
    df["date_dt"] = pd.to_datetime(df["date"])
    df["week_start"] = (df["date_dt"] - pd.to_timedelta(df["date_dt"].dt.dayofweek, unit="d")).dt.date.astype(str)

    agg = (
        df.groupby(["week_start", "topic"])
        .agg(
            article_count =("title", "count"),
            avg_fear      =("fear_score",     "mean"),
            avg_anger     =("anger_score",    "mean"),
            avg_joy       =("joy_score",      "mean"),
            avg_sadness   =("sadness_score",  "mean"),
            avg_surprise  =("surprise_score", "mean"),
            avg_disgust   =("disgust_score",  "mean"),
            avg_neutral   =("neutral_score",  "mean"),
        )
        .reset_index()
    )

    agg["anxiety_index"]    = (agg["avg_fear"]    + agg["avg_sadness"]).round(6)
    agg["tension_index"]    = (agg["avg_anger"]   + agg["avg_disgust"]).round(6)
    agg["positivity_index"] = (agg["avg_joy"]     + agg["avg_surprise"]).round(6)

    # Dominant emotion per week+topic = highest avg score column
    avg_cols = {col: col.replace("avg_", "") for col in
                ["avg_fear","avg_anger","avg_joy","avg_sadness","avg_surprise","avg_disgust","avg_neutral"]}
    agg["dominant_emotion"] = agg[list(avg_cols.keys())].idxmax(axis=1).map(avg_cols)

    agg_rows = [
        {
            "week_start":       row["week_start"],
            "topic":            row["topic"],
            "article_count":    int(row["article_count"]),
            "dominant_emotion": row["dominant_emotion"],
            "avg_fear":         round(float(row["avg_fear"]),    6),
            "avg_anger":        round(float(row["avg_anger"]),   6),
            "avg_joy":          round(float(row["avg_joy"]),     6),
            "avg_sadness":      round(float(row["avg_sadness"]), 6),
            "avg_surprise":     round(float(row["avg_surprise"]),6),
            "avg_disgust":      round(float(row["avg_disgust"]), 6),
            "avg_neutral":      round(float(row["avg_neutral"]), 6),
            "anxiety_index":    round(float(row["anxiety_index"]),    6),
            "tension_index":    round(float(row["tension_index"]),    6),
            "positivity_index": round(float(row["positivity_index"]), 6),
            "ingested_at":      now_ts,
        }
        for _, row in agg.iterrows()
    ]

    ensure_table(client, AGG_TABLE, AGG_SCHEMA)
    logger.info(f"Writing {len(agg_rows):,} weekly rows to {AGG_TABLE} …")
    streaming_insert(client, AGG_TABLE, agg_rows)
    logger.info("news_sentiment_weekly written.")

    # 7. Summary
    logger.info("─── LAYER 1 COMPLETE ───")
    logger.info(f"  Articles classified : {len(rows):,}")
    logger.info(f"  Weekly aggregates   : {len(agg_rows):,} (week × topic combos)")
    emotion_counts = df["emotion"].value_counts().to_dict()
    logger.info(f"  Emotion distribution: {emotion_counts}")


if __name__ == "__main__":
    main()
