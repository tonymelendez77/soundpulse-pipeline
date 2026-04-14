"""
SoundPulse FastAPI backend. Queries BigQuery and returns JSON for dashboards.
"""

import json
import math
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT = "soundpulse-production"
DATASET = f"{PROJECT}.dbt_transformed"
RAW = f"{PROJECT}.music_analytics"

app = FastAPI(title="SoundPulse API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_client() -> bigquery.Client:
    raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if raw:
        info = json.loads(raw)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(project=PROJECT, credentials=creds)
    return bigquery.Client(project=PROJECT)


client = _make_client()


def _run(query: str) -> list[dict]:
    df = client.query(query).to_dataframe()
    for col in df.select_dtypes(include=["dbdate", "datetime64[ns]", "datetime64[ns, UTC]",
                                          "object"]).columns:
        df[col] = df[col].astype(str)
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _region_clause(region: Optional[str], prefix: str = "") -> str:
    if not region or region == "all":
        return ""
    col = f"{prefix}region" if prefix else "region"
    return f"WHERE {col} = '{region}'"


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/correlation")
def correlation(region: Optional[str] = Query(default=None)):
    try:
        where = _region_clause(region)
        rows = _run(f"""
            SELECT region, emotion, mood_archetype, pearson_r, direction, significant AS notable
            FROM `{RAW}.emotion_music_correlation`
            {where}
            ORDER BY region, mood_archetype, emotion
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/timeline")
def timeline(region: Optional[str] = Query(default=None)):
    try:
        where = _region_clause(region)
        rows = _run(f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
                region,
                avg_fear, avg_anger, avg_joy, avg_sadness,
                avg_surprise, avg_disgust, avg_neutral,
                anxiety_index, tension_index, positivity_index,
                dominant_emotion, dominant_mood,
                avg_valence, avg_energy, avg_danceability, avg_tempo
            FROM `{RAW}.weekly_features`
            {where}
            ORDER BY week_start ASC
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/shap")
def shap(
    mood: Optional[str] = Query(default=None),
    region: Optional[str] = Query(default=None),
):
    try:
        conditions = []
        if mood:
            conditions.append(f"mood_archetype = '{mood}'")
        if region and region != "all":
            conditions.append(f"region = '{region}'")
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        rows = _run(f"""
            SELECT region, feature, mood_archetype, mean_shap_value, mean_abs_shap, rank AS importance_rank
            FROM `{RAW}.shap_importance`
            {where}
            ORDER BY region, mood_archetype, rank ASC
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mood-weekly")
def mood_weekly():
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


@app.get("/mood-regional")
def mood_regional(region: Optional[str] = Query(default=None)):
    try:
        where = _region_clause(region)
        rows = _run(f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
                region, dominant_mood, track_count,
                euphoric_pct, melancholic_pct, aggressive_pct,
                peaceful_pct, groovy_pct,
                avg_valence, avg_energy, avg_danceability, avg_tempo
            FROM `{RAW}.audio_mood_regional`
            {where}
            ORDER BY week_start ASC, region ASC
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
def predictions(region: Optional[str] = Query(default=None)):
    try:
        where = f"WHERE region = '{region}'" if region else ""
        rows = _run(f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
                region, actual_mood, predicted_mood, correct,
                confidence, anxiety_index, tension_index, positivity_index,
                total_weeks, correct_predictions, overall_accuracy,
                avg_confidence,
                CAST(first_week AS STRING) AS first_week,
                CAST(last_week AS STRING) AS last_week,
                is_forward
            FROM `{DATASET}.fct_mood_prediction_summary`
            {where}
            ORDER BY week_start ASC
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/news-sentiment")
def news_sentiment(topic: Optional[str] = Query(default=None)):
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


@app.get("/generated-tracks")
def generated_tracks(region: Optional[str] = Query(default=None)):
    try:
        where = _region_clause(region)
        rows = _run(f"""
            SELECT
                generation_id,
                CAST(week_start AS STRING) AS week_start,
                region, period,
                mood_archetype, prompt_text, mood_blend_json,
                similar_tracks_json, audio_gcs_path,
                duration_seconds,
                CAST(generated_at AS STRING) AS generated_at
            FROM `{RAW}.generated_tracks`
            {where}
            ORDER BY generated_at DESC
            LIMIT 20
        """)
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
