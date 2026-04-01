"""
SoundPulse — Module 14
FastAPI backend — queries BigQuery dbt_transformed views/tables and returns JSON.

Run:
    uvicorn serving.api:app --reload --port 8000
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery
from google.oauth2 import service_account
import math

PROJECT = "soundpulse-production"
DATASET = f"{PROJECT}.dbt_transformed"

app = FastAPI(title="SoundPulse API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_client() -> bigquery.Client:
    """
    Supports three credential modes (in priority order):
    1. GCP_SERVICE_ACCOUNT_JSON env var — JSON string (Render secret env var)
    2. GOOGLE_APPLICATION_CREDENTIALS env var — path to key file (local / GH Actions)
    3. Application Default Credentials (local dev with gcloud auth)
    """
    raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if raw:
        info = json.loads(raw)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(project=PROJECT, credentials=creds)
    # Falls through to ADC / GOOGLE_APPLICATION_CREDENTIALS automatically
    return bigquery.Client(project=PROJECT)


client = _make_client()


def _run(query: str) -> list[dict]:
    """Execute a BQ query and return rows as a list of dicts."""
    df = client.query(query).to_dataframe()
    # Convert non-serialisable types (date, Timestamp, Decimal) to plain Python
    for col in df.select_dtypes(include=["dbdate", "datetime64[ns]", "datetime64[ns, UTC]",
                                          "object"]).columns:
        df[col] = df[col].astype(str)


    return [
    {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
    for row in df.to_dict(orient="records")
]


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Correlation ──────────────────────────────────────────────────────────────

@app.get("/correlation")
def correlation():
    """50 Pearson r values: 10 emotions × 5 mood archetypes."""
    try:
        rows = _run(f"""
            SELECT emotion, mood_archetype, pearson_r, direction, notable
            FROM `{DATASET}.fct_emotion_music_correlation`
            ORDER BY mood_archetype, emotion
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Timeline ─────────────────────────────────────────────────────────────────

@app.get("/timeline")
def timeline():
    """Week-level emotion indices + dominant mood — XGBoost input table."""
    try:
        rows = _run(f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
                avg_fear, avg_anger, avg_joy, avg_sadness,
                avg_surprise, avg_disgust, avg_neutral,
                anxiety_index, tension_index, positivity_index,
                dominant_emotion, dominant_mood,
                avg_valence, avg_energy, avg_danceability, avg_tempo
            FROM `{DATASET}.stg_weekly_features`
            ORDER BY week_start ASC
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── SHAP importance ───────────────────────────────────────────────────────────

@app.get("/shap")
def shap(mood: Optional[str] = Query(default=None, description="Filter by mood_archetype")):
    """SHAP feature importance per mood archetype. Optional ?mood= filter."""
    try:
        where = f"WHERE mood_archetype = '{mood}'" if mood else ""
        rows = _run(f"""
            SELECT feature, mood_archetype, mean_shap_value, mean_abs_shap, importance_rank
            FROM `{DATASET}.stg_shap_importance`
            {where}
            ORDER BY mood_archetype, importance_rank ASC
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Mood weekly ───────────────────────────────────────────────────────────────

@app.get("/mood-weekly")
def mood_weekly():
    """Week × chart_source dominant mood with archetype percentages."""
    try:
        rows = _run(f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
                chart_source, dominant_mood, track_count,
                euphoric_pct, melancholic_pct, aggressive_pct,
                peaceful_pct, groovy_pct,
                avg_valence, avg_energy, avg_danceability, avg_tempo
            FROM `{DATASET}.stg_audio_mood_weekly`
            ORDER BY week_start ASC, chart_source ASC
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Predictions ───────────────────────────────────────────────────────────────

@app.get("/predictions")
def predictions():
    """XGBoost mood predictions with per-week accuracy + model-level stats."""
    try:
        rows = _run(f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
                actual_mood, predicted_mood, correct,
                confidence, anxiety_index, tension_index, positivity_index,
                total_weeks, correct_predictions, overall_accuracy,
                avg_confidence,
                CAST(first_week AS STRING) AS first_week,
                CAST(last_week AS STRING) AS last_week
            FROM `{DATASET}.fct_mood_prediction_summary`
            ORDER BY week_start ASC
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── News sentiment ─────────────────────────────────────────────────────────────

@app.get("/news-sentiment")
def news_sentiment(topic: Optional[str] = Query(default=None, description="Filter by topic")):
    """Week × topic aggregated emotion scores. Optional ?topic= filter."""
    try:
        where = f"WHERE topic = '{topic}'" if topic else ""
        rows = _run(f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
                topic, article_count, dominant_emotion,
                avg_fear, avg_anger, avg_joy, avg_sadness,
                avg_surprise, avg_disgust, avg_neutral,
                anxiety_index, tension_index, positivity_index
            FROM `{DATASET}.stg_news_sentiment_weekly`
            {where}
            ORDER BY week_start ASC, topic ASC
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Generated tracks ──────────────────────────────────────────────────────────

@app.get("/generated-tracks")
def generated_tracks():
    """Most recent MusicGen generation runs (up to 10)."""
    try:
        rows = _run(f"""
            SELECT
                generation_id,
                CAST(week_start AS STRING) AS week_start,
                mood_archetype, prompt_text,
                similar_tracks_json, audio_gcs_path,
                duration_seconds,
                CAST(generated_at AS STRING) AS generated_at
            FROM `{DATASET}.stg_generated_tracks`
            ORDER BY generated_at DESC
            LIMIT 10
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
