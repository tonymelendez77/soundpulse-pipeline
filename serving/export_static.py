"""
SoundPulse — Module 14
Static export: queries BigQuery and writes JSON snapshots to docs/data/
Also downloads the latest generated WAV from GCS to docs/audio/track.wav

Run (from repo root, venv active):
    python serving/export_static.py

Output:
    docs/data/correlation.json
    docs/data/timeline.json
    docs/data/shap.json
    docs/data/predictions.json
    docs/data/mood_weekly.json
    docs/data/news_sentiment.json
    docs/data/generated_tracks.json
    docs/audio/track.wav          (latest generated clip)
    docs/data/meta.json           (export timestamp + summary stats)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.cloud import bigquery, storage
from loguru import logger

PROJECT = "soundpulse-production"
DATASET = f"{PROJECT}.dbt_transformed"
GCS_BUCKET = "soundpulse-prod-raw-lake"

DOCS = Path(__file__).parent.parent / "docs"
DATA_DIR = DOCS / "data"
AUDIO_DIR = DOCS / "audio"

DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def bq_to_json(client: bigquery.Client, query: str) -> list[dict]:
    df = client.query(query).to_dataframe()
    for col in df.columns:
        if hasattr(df[col], "dt"):
            df[col] = df[col].astype(str)
        elif df[col].dtype == object:
            df[col] = df[col].astype(str)
    df = df.where(pd.notna(df), other=None)
    return json.loads(df.to_json(orient="records"))


def write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info(f"  → {path.relative_to(DOCS.parent)} ({len(data)} rows)")


def export_correlation(client):
    data = bq_to_json(client, f"""
        SELECT emotion, mood_archetype,
               CAST(pearson_r AS FLOAT64) AS pearson_r,
               direction,
               CAST(notable AS BOOL) AS notable
        FROM `{DATASET}.fct_emotion_music_correlation`
        ORDER BY mood_archetype, emotion
    """)
    write_json(DATA_DIR / "correlation.json", data)


def export_timeline(client):
    data = bq_to_json(client, f"""
        SELECT
            CAST(week_start AS STRING) AS week_start,
            CAST(avg_fear AS FLOAT64) AS avg_fear,
            CAST(avg_anger AS FLOAT64) AS avg_anger,
            CAST(avg_joy AS FLOAT64) AS avg_joy,
            CAST(avg_sadness AS FLOAT64) AS avg_sadness,
            CAST(avg_surprise AS FLOAT64) AS avg_surprise,
            CAST(avg_disgust AS FLOAT64) AS avg_disgust,
            CAST(avg_neutral AS FLOAT64) AS avg_neutral,
            CAST(anxiety_index AS FLOAT64) AS anxiety_index,
            CAST(tension_index AS FLOAT64) AS tension_index,
            CAST(positivity_index AS FLOAT64) AS positivity_index,
            dominant_emotion, dominant_mood,
            CAST(avg_valence AS FLOAT64) AS avg_valence,
            CAST(avg_energy AS FLOAT64) AS avg_energy,
            CAST(avg_danceability AS FLOAT64) AS avg_danceability,
            CAST(avg_tempo AS FLOAT64) AS avg_tempo
        FROM `{DATASET}.stg_weekly_features`
        ORDER BY week_start ASC
    """)
    write_json(DATA_DIR / "timeline.json", data)


def export_shap(client):
    data = bq_to_json(client, f"""
        SELECT feature, mood_archetype,
               CAST(mean_shap_value AS FLOAT64) AS mean_shap_value,
               CAST(mean_abs_shap AS FLOAT64) AS mean_abs_shap,
               importance_rank
        FROM `{DATASET}.stg_shap_importance`
        ORDER BY mood_archetype, importance_rank ASC
    """)
    write_json(DATA_DIR / "shap.json", data)


def export_predictions(client):
    data = bq_to_json(client, f"""
        SELECT
            CAST(week_start AS STRING) AS week_start,
            actual_mood, predicted_mood,
            CAST(correct AS BOOL) AS correct,
            CAST(confidence AS FLOAT64) AS confidence,
            CAST(overall_accuracy AS FLOAT64) AS overall_accuracy,
            CAST(avg_confidence AS FLOAT64) AS avg_confidence,
            total_weeks, correct_predictions,
            CAST(first_week AS STRING) AS first_week,
            CAST(last_week AS STRING) AS last_week
        FROM `{DATASET}.fct_mood_prediction_summary`
        ORDER BY week_start ASC
    """)
    write_json(DATA_DIR / "predictions.json", data)


def export_mood_weekly(client):
    data = bq_to_json(client, f"""
        SELECT
            CAST(week_start AS STRING) AS week_start,
            chart_source, dominant_mood, track_count,
            CAST(euphoric_pct AS FLOAT64) AS euphoric_pct,
            CAST(melancholic_pct AS FLOAT64) AS melancholic_pct,
            CAST(aggressive_pct AS FLOAT64) AS aggressive_pct,
            CAST(peaceful_pct AS FLOAT64) AS peaceful_pct,
            CAST(groovy_pct AS FLOAT64) AS groovy_pct,
            CAST(avg_valence AS FLOAT64) AS avg_valence,
            CAST(avg_energy AS FLOAT64) AS avg_energy,
            CAST(avg_danceability AS FLOAT64) AS avg_danceability,
            CAST(avg_tempo AS FLOAT64) AS avg_tempo
        FROM `{DATASET}.stg_audio_mood_weekly`
        ORDER BY week_start ASC, chart_source ASC
    """)
    write_json(DATA_DIR / "mood_weekly.json", data)


def export_news_sentiment(client):
    data = bq_to_json(client, f"""
        SELECT
            CAST(week_start AS STRING) AS week_start,
            topic, article_count, dominant_emotion,
            CAST(avg_fear AS FLOAT64) AS avg_fear,
            CAST(avg_anger AS FLOAT64) AS avg_anger,
            CAST(avg_joy AS FLOAT64) AS avg_joy,
            CAST(avg_sadness AS FLOAT64) AS avg_sadness,
            CAST(avg_surprise AS FLOAT64) AS avg_surprise,
            CAST(avg_disgust AS FLOAT64) AS avg_disgust,
            CAST(avg_neutral AS FLOAT64) AS avg_neutral,
            CAST(anxiety_index AS FLOAT64) AS anxiety_index,
            CAST(tension_index AS FLOAT64) AS tension_index,
            CAST(positivity_index AS FLOAT64) AS positivity_index
        FROM `{DATASET}.stg_news_sentiment_weekly`
        ORDER BY week_start ASC, topic ASC
    """)
    write_json(DATA_DIR / "news_sentiment.json", data)


def export_generated_tracks(client) -> dict | None:
    data = bq_to_json(client, f"""
        SELECT
            generation_id,
            CAST(week_start AS STRING) AS week_start,
            mood_archetype, prompt_text,
            similar_tracks_json, audio_gcs_path,
            CAST(duration_seconds AS FLOAT64) AS duration_seconds,
            CAST(generated_at AS STRING) AS generated_at
        FROM `{DATASET}.stg_generated_tracks`
        ORDER BY generated_at DESC
        LIMIT 10
    """)
    write_json(DATA_DIR / "generated_tracks.json", data)
    return data[0] if data else None


def download_wav(latest: dict | None) -> None:
    if not latest:
        logger.warning("No generated tracks found — skipping WAV download")
        return

    gcs_path = latest.get("audio_gcs_path", "")
    if not gcs_path.startswith("gs://"):
        logger.warning(f"Unexpected GCS path: {gcs_path}")
        return

    # gs://soundpulse-prod-raw-lake/generated/2026-03-16_aggressive.wav
    parts = gcs_path[5:].split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]

    dest = AUDIO_DIR / "track.wav"
    gcs = storage.Client(project=PROJECT)
    bucket = gcs.bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(str(dest))
    size_kb = dest.stat().st_size // 1024
    logger.info(f"  → docs/audio/track.wav ({size_kb} KB) from {gcs_path}")


def write_meta(exports: dict) -> None:
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "counts": exports,
    }
    with open(DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  → docs/data/meta.json")


def main():
    logger.info("SoundPulse — exporting static data for GitHub Pages")
    client = bigquery.Client(project=PROJECT)

    counts = {}

    logger.info("Exporting correlation...")
    export_correlation(client)
    counts["correlation"] = 50

    logger.info("Exporting timeline...")
    export_timeline(client)
    with open(DATA_DIR / "timeline.json") as f:
        counts["timeline"] = len(json.load(f))

    logger.info("Exporting SHAP importance...")
    export_shap(client)
    with open(DATA_DIR / "shap.json") as f:
        counts["shap"] = len(json.load(f))

    logger.info("Exporting predictions...")
    export_predictions(client)
    with open(DATA_DIR / "predictions.json") as f:
        counts["predictions"] = len(json.load(f))

    logger.info("Exporting mood weekly...")
    export_mood_weekly(client)
    with open(DATA_DIR / "mood_weekly.json") as f:
        counts["mood_weekly"] = len(json.load(f))

    logger.info("Exporting news sentiment...")
    export_news_sentiment(client)
    with open(DATA_DIR / "news_sentiment.json") as f:
        counts["news_sentiment"] = len(json.load(f))

    logger.info("Exporting generated tracks...")
    latest = export_generated_tracks(client)
    counts["generated_tracks"] = 1 if latest else 0

    logger.info("Downloading WAV from GCS...")
    download_wav(latest)

    write_meta(counts)

    logger.info("─── EXPORT COMPLETE ───")
    for k, v in counts.items():
        logger.info(f"  {k}: {v} rows")
    logger.info(f"  Site ready at: docs/index.html")
    logger.info("  Enable GitHub Pages: repo Settings → Pages → /docs")


if __name__ == "__main__":
    main()
