"""
SoundPulse static export (Module 14).
Queries BigQuery, writes JSON snapshots to docs/data/, and syncs WAVs from GCS.
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


DATASET = f"{PROJECT}.dbt_transformed"
RAW = f"{PROJECT}.music_analytics"
GCS_BUCKET = "soundpulse-prod-raw-lake"

DOCS = Path(__file__).parent.parent / "docs"
DATA_DIR = DOCS / "data"
AUDIO_DIR = DOCS / "audio"
HIST_DIR = AUDIO_DIR / "history"

DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["north_america", "latin_america", "europe", "global"]

WAV_MAP = {
    "today": "today.wav",
    "weekly": "weekly.wav",
    "monthly": "monthly.wav",
}


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
        SELECT region, emotion, mood_archetype,
               CAST(pearson_r AS FLOAT64) AS pearson_r,
               direction,
               CAST(significant AS BOOL) AS notable
        FROM `{RAW}.emotion_music_correlation`
        ORDER BY region, mood_archetype, emotion
    """)
    write_json(DATA_DIR / "correlation.json", data)


def export_timeline(client):
    data = bq_to_json(client, f"""
        SELECT
            CAST(week_start AS STRING) AS week_start,
            region,
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
        FROM `{RAW}.weekly_features`
        ORDER BY week_start ASC, region ASC
    """)
    write_json(DATA_DIR / "timeline.json", data)


def export_shap(client):
    data = bq_to_json(client, f"""
        SELECT region, feature, mood_archetype,
               CAST(mean_shap_value AS FLOAT64) AS mean_shap_value,
               CAST(mean_abs_shap AS FLOAT64) AS mean_abs_shap,
               rank AS importance_rank
        FROM `{RAW}.shap_importance`
        ORDER BY region, mood_archetype, rank ASC
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


def export_mood_regional(client: bigquery.Client) -> None:
    """Export audio_mood_regional -> mood_regional.json for the region-tab charts."""
    data = bq_to_json(client, f"""
        SELECT
            CAST(week_start AS STRING) AS week_start,
            region, dominant_mood, track_count,
            CAST(euphoric_pct AS FLOAT64) AS euphoric_pct,
            CAST(melancholic_pct AS FLOAT64) AS melancholic_pct,
            CAST(aggressive_pct AS FLOAT64) AS aggressive_pct,
            CAST(peaceful_pct AS FLOAT64) AS peaceful_pct,
            CAST(groovy_pct AS FLOAT64) AS groovy_pct,
            CAST(avg_valence AS FLOAT64) AS avg_valence,
            CAST(avg_energy AS FLOAT64) AS avg_energy,
            CAST(avg_danceability AS FLOAT64) AS avg_danceability,
            CAST(avg_tempo AS FLOAT64) AS avg_tempo
        FROM `{RAW}.audio_mood_regional`
        ORDER BY week_start ASC, region ASC
    """)
    write_json(DATA_DIR / "mood_regional.json", data)


def export_generated_tracks(client: bigquery.Client) -> dict:
    """Return latest row per (region, period). Writes generated_tracks.json.
    Returns dict: {region: {period: row}} for WAV download step."""
    data = bq_to_json(client, f"""
        SELECT
            generation_id,
            CAST(week_start AS STRING) AS week_start,
            region, period,
            mood_archetype, prompt_text,
            mood_blend_json,
            similar_tracks_json, audio_gcs_path,
            CAST(duration_seconds AS FLOAT64) AS duration_seconds,
            CAST(generated_at AS STRING) AS generated_at
        FROM `{RAW}.generated_tracks`
        WHERE period IN ('today', 'weekly', 'monthly')
        QUALIFY ROW_NUMBER() OVER (PARTITION BY region, period ORDER BY generated_at DESC) = 1
        ORDER BY region ASC, generated_at DESC
    """)
    write_json(DATA_DIR / "generated_tracks.json", data)

    by_region_period: dict[str, dict] = {}
    for row in data:
        rgn = row.get("region") or "global"
        prd = row.get("period")
        if prd:
            by_region_period.setdefault(rgn, {})[prd] = row

    logger.info(f"  generated_tracks: regions = {list(by_region_period.keys())}")
    return by_region_period


def download_wavs(by_region_period: dict, gcs_client: storage.Client) -> None:
    """Download WAVs into docs/audio/{region}/{period}.wav.
    Keeps docs/audio/{period}.wav pointing at the global region for the site.
    """
    bucket = gcs_client.bucket(GCS_BUCKET)

    for region in REGIONS:
        period_rows = by_region_period.get(region, {})
        if not period_rows:
            logger.warning(f"  [{region}] No generated tracks — skipping WAV download")
            continue

        region_dir = AUDIO_DIR / region
        region_dir.mkdir(parents=True, exist_ok=True)

        for period, row in period_rows.items():
            gcs_path = row.get("audio_gcs_path", "")
            if not gcs_path.startswith("gs://"):
                logger.warning(f"  [{region}/{period}] unexpected GCS path: {gcs_path}")
                continue

            blob_name = gcs_path[5:].split("/", 1)[1]
            dest_name = WAV_MAP.get(period, f"{period}.wav")
            dest = region_dir / dest_name

            try:
                bucket.blob(blob_name).download_to_filename(str(dest))
                size_kb = dest.stat().st_size // 1024
                logger.info(f"  → docs/audio/{region}/{dest_name} ({size_kb} KB)")
            except Exception as e:
                logger.error(f"  [{region}/{period}] WAV download failed: {e}")

    # docs/audio/{period}.wav = global region's copy
    global_periods = by_region_period.get("global", {})
    for period, dest_name in WAV_MAP.items():
        src = AUDIO_DIR / "global" / dest_name
        dest = AUDIO_DIR / dest_name
        if src.exists():
            shutil.copy(src, dest)
            logger.info(f"  → docs/audio/{dest_name} (copy of global/{dest_name})")
        elif not global_periods:
            # no global region — use first available region
            for rgn in REGIONS:
                alt = AUDIO_DIR / rgn / dest_name
                if alt.exists():
                    shutil.copy(alt, dest)
                    logger.info(f"  → docs/audio/{dest_name} (copy of {rgn}/{dest_name})")
                    break

    # track.wav legacy copy
    today_wav = AUDIO_DIR / "today.wav"
    track_wav = AUDIO_DIR / "track.wav"
    if today_wav.exists():
        shutil.copy(today_wav, track_wav)
        logger.info(f"  → docs/audio/track.wav (copy of today.wav)")


def update_history(by_region_period: dict) -> None:
    """Copy each region+period WAV into docs/audio/history/ with a datestamp."""
    today = _date.today()
    monday = today - timedelta(days=today.weekday())
    month_str = today.strftime("%Y-%m")

    stamp_map = {
        "today": today.isoformat(),
        "weekly": monday.isoformat(),
        "monthly": f"{month_str}-01",
    }

    for region in REGIONS:
        region_dir = AUDIO_DIR / region
        for period, date_str in stamp_map.items():
            src = region_dir / WAV_MAP.get(period, f"{period}.wav")
            fname = f"{date_str}_{region}_{period}.wav"
            dest = HIST_DIR / fname
            if src.exists() and not dest.exists():
                shutil.copy(src, dest)
                logger.info(f"  → docs/audio/history/{fname} (archived)")
            elif dest.exists():
                logger.info(f"  history/{fname} already exists — skipping")


def export_song_history(client: bigquery.Client) -> None:
    """Build the song_history.json index from history WAVs and BQ metadata."""
    try:
        all_tracks = bq_to_json(client, f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
                region, period,
                mood_archetype, prompt_text,
                CAST(duration_seconds AS FLOAT64) AS duration_seconds,
                CAST(generated_at AS STRING) AS generated_at
            FROM `{RAW}.generated_tracks`
            WHERE period IS NOT NULL
            ORDER BY generated_at DESC
        """)
    except Exception as e:
        logger.warning(f"Could not load generated_tracks for history: {e}")
        all_tracks = []

    history = []
    for wav in sorted(HIST_DIR.glob("*.wav"), reverse=True):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        period = parts[-1]
        date_str = parts[0]
        region = "_".join(parts[1:-1])

        meta = next(
            (t for t in all_tracks
             if t.get("period") == period and t.get("region") == region),
            {}
        )
        label_map = {
            "today": date_str,
            "weekly": f"Week of {date_str}",
            "monthly": date_str[:7],
        }
        region_label = {
            "north_america": "N. America",
            "latin_america": "Latin America",
            "europe": "Europe",
            "global": "Global",
        }.get(region, region.replace("_", " ").title())
        history.append({
            "period": period,
            "region": region,
            "region_label": region_label,
            "date": date_str,
            "label": f"{label_map.get(period, date_str)} — {region_label}",
            "mood_archetype": meta.get("mood_archetype") or "",
            "prompt_text": meta.get("prompt_text") or "",
            "duration_seconds": meta.get("duration_seconds") or 0,
            "audio_path": f"audio/history/{wav.name}",
        })

    write_json(DATA_DIR / "song_history.json", history)


def export_prediction_accuracy(client: bigquery.Client) -> None:
    """Export rolling prediction accuracy, falling back to ml_predictions if needed."""
    try:
        data = bq_to_json(client, f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
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
        data = bq_to_json(client, f"""
            SELECT
                CAST(week_start AS STRING) AS week_start,
                period,
                predicted_mood,
                actual_mood,
                correct,
                confidence,
                NULL AS rolling_8w_acc,
                NULL AS rolling_8w_n,
                CAST(ingested_at AS STRING) AS validated_at
            FROM `{PROJECT}.music_analytics.ml_predictions`
            WHERE period IS NOT NULL
              AND correct IS NOT NULL
            ORDER BY week_start DESC
            LIMIT 100
        """)

    write_json(DATA_DIR / "prediction_accuracy.json", data)


def write_meta(exports: dict) -> None:
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "counts": exports,
    }
    with open(DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("  → docs/data/meta.json")


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

    logger.info("Exporting mood regional (4 regions)...")
    export_mood_regional(bq_client)
    with open(DATA_DIR / "mood_regional.json") as f:
        counts["mood_regional"] = len(json.load(f))

    logger.info("Exporting news sentiment...")
    export_news_sentiment(bq_client)
    with open(DATA_DIR / "news_sentiment.json") as f:
        counts["news_sentiment"] = len(json.load(f))

    logger.info("Exporting prediction accuracy...")
    export_prediction_accuracy(bq_client)
    with open(DATA_DIR / "prediction_accuracy.json") as f:
        counts["prediction_accuracy"] = len(json.load(f))

    logger.info("Exporting generated tracks...")
    by_region_period = export_generated_tracks(bq_client)
    counts["generated_tracks"] = sum(len(v) for v in by_region_period.values())

    logger.info("Downloading WAVs from GCS...")
    download_wavs(by_region_period, gcs_client)

    logger.info("Updating song history archive...")
    update_history(by_region_period)

    logger.info("Exporting song history index...")
    export_song_history(bq_client)
    with open(DATA_DIR / "song_history.json") as f:
        counts["song_history"] = len(json.load(f))

    write_meta(counts)

    logger.info("EXPORT COMPLETE")
    for k, v in counts.items():
        logger.info(f"  {k}: {v} rows")
    logger.info("  Site ready at: docs/index.html")


if __name__ == "__main__":
    main()
