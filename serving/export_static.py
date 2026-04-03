"""
SoundPulse — Module 14
Static export: queries BigQuery and writes JSON snapshots to docs/data/
Downloads latest generated WAVs from GCS to docs/audio/ (today/weekly/monthly)
Maintains a timestamped history in docs/audio/history/ (free on GitHub Pages)

Run (from repo root, venv active):
    python serving/export_static.py

Output:
    docs/data/correlation.json
    docs/data/timeline.json
    docs/data/shap.json
    docs/data/predictions.json
    docs/data/mood_weekly.json
    docs/data/news_sentiment.json
    docs/data/generated_tracks.json    (latest per period)
    docs/data/song_history.json        (full archive index)
    docs/data/meta.json
    docs/audio/today.wav               (today's prediction)
    docs/audio/weekly.wav              (this week's prediction)
    docs/audio/monthly.wav             (this month's prediction)
    docs/audio/track.wav               (backwards-compat copy of today.wav)
    docs/audio/history/YYYY-MM-DD_period.wav  (one per period per run, if new)
"""

import json
import os
import shutil
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from loguru import logger

PROJECT = "soundpulse-production"


def _make_clients():
    raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if raw:
        info = json.loads(raw)
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        return (bigquery.Client(project=PROJECT, credentials=creds),
                storage.Client(project=PROJECT, credentials=creds))
    return bigquery.Client(project=PROJECT), storage.Client(project=PROJECT)


DATASET   = f"{PROJECT}.dbt_transformed"
GCS_BUCKET = "soundpulse-prod-raw-lake"

DOCS      = Path(__file__).parent.parent / "docs"
DATA_DIR  = DOCS / "data"
AUDIO_DIR = DOCS / "audio"
HIST_DIR  = AUDIO_DIR / "history"

DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

WAV_MAP = {
    "today":   "today.wav",
    "weekly":  "weekly.wav",
    "monthly": "monthly.wav",
}


# ── Generic helpers ───────────────────────────────────────────────────────────────

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


# ── Per-table exports (unchanged from original) ──────────────────────────────────

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


# ── Generated tracks export (updated for multi-period) ───────────────────────────

def export_generated_tracks(client: bigquery.Client) -> dict:
    """Return latest row per period as a dict keyed by period name."""
    data = bq_to_json(client, f"""
        SELECT
            generation_id,
            CAST(week_start AS STRING) AS week_start,
            period,
            mood_archetype, prompt_text,
            mood_blend_json,
            similar_tracks_json, audio_gcs_path,
            CAST(duration_seconds AS FLOAT64) AS duration_seconds,
            CAST(generated_at AS STRING) AS generated_at
        FROM `{DATASET}.stg_generated_tracks`
        WHERE period IN ('today', 'weekly', 'monthly')
        QUALIFY ROW_NUMBER() OVER (PARTITION BY period ORDER BY generated_at DESC) = 1
        ORDER BY generated_at DESC
    """)
    write_json(DATA_DIR / "generated_tracks.json", data)

    by_period = {}
    for row in data:
        p = row.get("period")
        if p and p not in by_period:
            by_period[p] = row

    # Backwards-compat fallback: if no period rows at all, grab latest row (old schema)
    if not by_period:
        fallback = bq_to_json(client, f"""
            SELECT
                generation_id,
                CAST(week_start AS STRING) AS week_start,
                mood_archetype, prompt_text,
                similar_tracks_json, audio_gcs_path,
                CAST(duration_seconds AS FLOAT64) AS duration_seconds,
                CAST(generated_at AS STRING) AS generated_at
            FROM `{DATASET}.stg_generated_tracks`
            ORDER BY generated_at DESC
            LIMIT 1
        """)
        if fallback:
            by_period["today"] = fallback[0]

    logger.info(f"  generated_tracks: periods found = {list(by_period.keys())}")
    return by_period


# ── WAV download ─────────────────────────────────────────────────────────────────

def download_wavs(by_period: dict, gcs_client: storage.Client) -> None:
    """Download live WAVs for each period + backwards-compat track.wav."""
    bucket = gcs_client.bucket(GCS_BUCKET)

    for period, row in by_period.items():
        gcs_path = row.get("audio_gcs_path", "")
        if not gcs_path.startswith("gs://"):
            logger.warning(f"  [{period}] unexpected GCS path: {gcs_path}")
            continue

        blob_name = gcs_path[5:].split("/", 1)[1]
        dest_name = WAV_MAP.get(period, f"{period}.wav")
        dest = AUDIO_DIR / dest_name

        try:
            bucket.blob(blob_name).download_to_filename(str(dest))
            size_kb = dest.stat().st_size // 1024
            logger.info(f"  → docs/audio/{dest_name} ({size_kb} KB) from {gcs_path}")
        except Exception as e:
            logger.error(f"  [{period}] WAV download failed: {e}")

    # Backwards-compat: track.wav = today.wav
    today_wav = AUDIO_DIR / "today.wav"
    track_wav = AUDIO_DIR / "track.wav"
    if today_wav.exists():
        shutil.copy(today_wav, track_wav)
        logger.info(f"  → docs/audio/track.wav (copy of today.wav, backwards-compat)")


# ── History management ────────────────────────────────────────────────────────────

def update_history(by_period: dict) -> None:
    """Copy each period's live WAV into docs/audio/history/ with a datestamp.
    Only creates the file if it doesn't already exist for that date+period.
    This means the pipeline can run daily without duplicating files.
    """
    today = _date.today()
    monday = today - timedelta(days=today.weekday())
    month_str = today.strftime("%Y-%m")

    stamp_map = {
        "today":   f"{today.isoformat()}_today.wav",
        "weekly":  f"{monday.isoformat()}_weekly.wav",
        "monthly": f"{month_str}-01_monthly.wav",
    }

    for period, fname in stamp_map.items():
        src = AUDIO_DIR / WAV_MAP.get(period, f"{period}.wav")
        dest = HIST_DIR / fname
        if src.exists() and not dest.exists():
            shutil.copy(src, dest)
            logger.info(f"  → docs/audio/history/{fname} (archived)")
        elif dest.exists():
            logger.info(f"  history/{fname} already exists — skipping")
        else:
            logger.warning(f"  src {src} not found — cannot archive {fname}")


def export_song_history(client: bigquery.Client) -> None:
    """Build song_history.json index from docs/audio/history/ + BQ metadata."""
    # Fetch all generated_tracks metadata with period info
    try:
        all_tracks = bq_to_json(client, f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
                period,
                mood_archetype, prompt_text,
                CAST(duration_seconds AS FLOAT64) AS duration_seconds,
                CAST(generated_at AS STRING) AS generated_at
            FROM `{DATASET}.stg_generated_tracks`
            WHERE period IS NOT NULL
            ORDER BY generated_at DESC
        """)
    except Exception as e:
        logger.warning(f"Could not load stg_generated_tracks for history: {e}")
        all_tracks = []

    # Index history WAV files
    history = []
    for wav in sorted(HIST_DIR.glob("*.wav"), reverse=True):
        stem_parts = wav.stem.split("_", 1)
        if len(stem_parts) != 2:
            continue
        date_str, period = stem_parts

        # Match metadata: find row for this period (best-effort proximity)
        meta = next((t for t in all_tracks if t.get("period") == period), {})

        label_map = {
            "today":   date_str,
            "weekly":  f"Week of {date_str}",
            "monthly": date_str[:7],  # YYYY-MM
        }
        history.append({
            "period":         period,
            "date":           date_str,
            "label":          label_map.get(period, date_str),
            "mood_archetype": meta.get("mood_archetype") or "",
            "confidence":     0,   # not stored in stg — kept as 0 for now
            "prompt_text":    meta.get("prompt_text") or "",
            "duration_seconds": meta.get("duration_seconds") or 0,
            "audio_path":     f"audio/history/{wav.name}",
        })

    write_json(DATA_DIR / "song_history.json", history)


# ── Prediction accuracy (feedback loop) ──────────────────────────────────────────

def export_prediction_accuracy(client: bigquery.Client) -> None:
    """Export rolling prediction accuracy from prediction_accuracy table.
    Falls back to computing it live from ml_predictions if the table is empty."""
    try:
        data = bq_to_json(client, f"""
            SELECT
                CAST(week_start AS STRING)  AS week_start,
                period,
                predicted_mood,
                actual_mood,
                correct,
                confidence,
                rolling_8w_acc,
                rolling_8w_n,
                CAST(validated_at AS STRING) AS validated_at
            FROM `{PROJECT}.music_analytics.prediction_accuracy`
            ORDER BY week_start DESC
            LIMIT 100
        """)
    except Exception:
        data = []

    if not data:
        # Fallback: compute from ml_predictions directly
        data = bq_to_json(client, f"""
            SELECT
                CAST(week_start AS STRING)  AS week_start,
                period,
                predicted_mood,
                actual_mood,
                correct,
                confidence,
                NULL AS rolling_8w_acc,
                NULL AS rolling_8w_n,
                CAST(ingested_at AS STRING)  AS validated_at
            FROM `{PROJECT}.music_analytics.ml_predictions`
            WHERE period IS NOT NULL
              AND correct IS NOT NULL
            ORDER BY week_start DESC
            LIMIT 100
        """)

    write_json(DATA_DIR / "prediction_accuracy.json", data)


# ── Meta ─────────────────────────────────────────────────────────────────────────

def write_meta(exports: dict) -> None:
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "counts": exports,
    }
    with open(DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("  → docs/data/meta.json")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    logger.info("SoundPulse — exporting static data for GitHub Pages")
    bq_client, gcs_client = _make_clients()

    counts = {}

    logger.info("Exporting correlation...")
    export_correlation(bq_client)
    counts["correlation"] = 50

    logger.info("Exporting timeline...")
    export_timeline(bq_client)
    with open(DATA_DIR / "timeline.json") as f:
        counts["timeline"] = len(json.load(f))

    logger.info("Exporting SHAP importance...")
    export_shap(bq_client)
    with open(DATA_DIR / "shap.json") as f:
        counts["shap"] = len(json.load(f))

    logger.info("Exporting predictions...")
    export_predictions(bq_client)
    with open(DATA_DIR / "predictions.json") as f:
        counts["predictions"] = len(json.load(f))

    logger.info("Exporting mood weekly...")
    export_mood_weekly(bq_client)
    with open(DATA_DIR / "mood_weekly.json") as f:
        counts["mood_weekly"] = len(json.load(f))

    logger.info("Exporting news sentiment...")
    export_news_sentiment(bq_client)
    with open(DATA_DIR / "news_sentiment.json") as f:
        counts["news_sentiment"] = len(json.load(f))

    logger.info("Exporting prediction accuracy (rolling feedback loop)...")
    export_prediction_accuracy(bq_client)
    with open(DATA_DIR / "prediction_accuracy.json") as f:
        counts["prediction_accuracy"] = len(json.load(f))

    logger.info("Exporting generated tracks...")
    by_period = export_generated_tracks(bq_client)
    counts["generated_tracks"] = len(by_period)

    logger.info("Downloading WAVs from GCS...")
    download_wavs(by_period, gcs_client)

    logger.info("Updating song history archive...")
    update_history(by_period)

    logger.info("Exporting song history index...")
    export_song_history(bq_client)
    with open(DATA_DIR / "song_history.json") as f:
        counts["song_history"] = len(json.load(f))

    write_meta(counts)

    logger.info("─── EXPORT COMPLETE ───")
    for k, v in counts.items():
        logger.info(f"  {k}: {v} rows")
    logger.info("  Site ready at: docs/index.html")


if __name__ == "__main__":
    main()
