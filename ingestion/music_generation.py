"""MusicGen audio generation from predicted mood profiles."""

import json
import math
import os
import tempfile
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from google.cloud import bigquery, storage
from loguru import logger
from pinecone import Pinecone
from transformers import AutoProcessor, MusicgenForConditionalGeneration

PROJECT = "soundpulse-production"
DATASET = "music_analytics"
BUCKET_NAME = "soundpulse-prod-raw-lake"
GCS_PREFIX = "generated"
PRED_TABLE = f"{PROJECT}.{DATASET}.ml_predictions"
GEN_TABLE = f"{PROJECT}.{DATASET}.generated_tracks"
PINECONE_INDEX = "soundpulse-tracks"
SCALER_PATH = Path(__file__).parent / "scaler_params.json"
MUSICGEN_MODEL = "facebook/musicgen-small"
TOP_K = 10
MAX_NEW_TOKENS = 500    # ~10 seconds of audio

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

REGIONS = ["north_america", "latin_america", "europe", "global"]

GEN_SCHEMA = [
    bigquery.SchemaField("generation_id", "STRING"),
    bigquery.SchemaField("week_start", "DATE"),
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("period", "STRING"),
    bigquery.SchemaField("mood_archetype", "STRING"),
    bigquery.SchemaField("mood_blend_json", "STRING"),
    bigquery.SchemaField("prompt_text", "STRING"),
    bigquery.SchemaField("similar_tracks_json", "STRING"),
    bigquery.SchemaField("audio_gcs_path", "STRING"),
    bigquery.SchemaField("duration_seconds", "FLOAT64"),
    bigquery.SchemaField("generated_at", "TIMESTAMP"),
]

MOOD_PREFIXES = {
    "euphoric": "euphoric pop anthem, celebratory and soaring,",
    "melancholic": "melancholic indie ballad, introspective and aching,",
    "aggressive": "aggressive rock, intense and visceral,",
    "peaceful": "peaceful ambient, meditative and spacious,",
    "groovy": "groovy funk, smooth and hypnotic,",
}

BLEND_HINTS = {
    "euphoric": "with uplifting hooks",
    "melancholic": "tinged with longing",
    "aggressive": "with raw intensity",
    "peaceful": "with spacious restraint",
    "groovy": "with a rhythmic swagger",
}

SEASON_TEXTURE = {
    "winter": "cold, stark, driving",
    "spring": "fresh, building energy, hopeful",
    "summer": "sun-drenched, open, vibrant",
    "autumn": "wistful, cinematic, rich",
}


def ensure_gen_table(client: bigquery.Client) -> None:
    """Create generated_tracks with current schema. If the table exists but is
    missing the 'region' or 'period' column (old schema), drop and recreate it."""
    try:
        existing = client.get_table(GEN_TABLE)
        existing_cols = {f.name for f in existing.schema}
        if "period" not in existing_cols or "region" not in existing_cols:
            logger.warning("generated_tracks missing schema column(s) — dropping and recreating")
            client.delete_table(GEN_TABLE)
            import time as _time; _time.sleep(5)
    except Exception:
        pass  # table doesn't exist yet — create below
    table = bigquery.Table(GEN_TABLE, schema=GEN_SCHEMA)
    client.create_table(table, exists_ok=True)
    logger.info(f"Table ready: {GEN_TABLE}")


def song_exists(client: bigquery.Client, period: str, week_start: str, region: str) -> bool:
    """Return True if a song for the given period/week/region already exists."""
    query = f"""
        SELECT COUNT(*) AS n
        FROM `{GEN_TABLE}`
        WHERE period = '{period}'
          AND region = '{region}'
          AND CAST(week_start AS STRING) = '{week_start}'
    """
    try:
        rows = list(client.query(query).result())
        return rows[0]["n"] > 0 if rows else False
    except Exception as e:
        logger.warning(f"song_exists check failed ({e}) — assuming no")
        return False


def fetch_predictions_by_region_period(client: bigquery.Client) -> dict:
    """Return the most recent prediction row per (region, period).
    Returns dict keyed as {region: {period: {...}}}."""
    query = f"""
        SELECT region, period, predicted_mood, confidence, target_date, mood_blend_json
        FROM `{PRED_TABLE}`
        WHERE period IN ('today', 'weekly', 'monthly')
        QUALIFY ROW_NUMBER() OVER (PARTITION BY region, period ORDER BY ingested_at DESC) = 1
    """
    rows = list(client.query(query).result())
    if not rows:
        raise RuntimeError("No period predictions found. Run ml_predictions.py first.")
    result = {}
    for r in rows:
        rgn = str(r["region"])
        prd = str(r["period"])
        result.setdefault(rgn, {})[prd] = {
            "predicted_mood": str(r["predicted_mood"]),
            "confidence": float(r["confidence"]),
            "target_date": str(r["target_date"]),
            "mood_blend_json": str(r["mood_blend_json"] or "{}"),
        }
    logger.info(f"Loaded predictions for regions: {list(result.keys())}")
    return result


def compute_avg_features(
    client: bigquery.Client,
    similar_tracks: list[dict],
    centroid_features: dict,
) -> dict:
    """Average audio features of the top-K similar Pinecone tracks, falling back to centroid."""
    titles_artists = [(t["title"], t["artist"]) for t in similar_tracks]
    if not titles_artists:
        return centroid_features

    def _safe_sql_str(s):
        return s.lower().replace("\\", "\\\\").replace("'", "\\'")

    try:
        conditions = " OR ".join(
            f"(LOWER(TRIM(title)) = '{_safe_sql_str(t)}' "
            f"AND LOWER(TRIM(artist)) = '{_safe_sql_str(a)}')"
            for t, a in titles_artists
        )
        feature_cols_sql = ", ".join(f"AVG({c}) AS {c}" for c in FEATURE_COLS)
        query = f"""
            SELECT {feature_cols_sql}
            FROM `{PROJECT}.{DATASET}.trending_historical`
            WHERE {conditions}
        """
        rows = list(client.query(query).result())
        if not rows or all(rows[0][c] is None for c in FEATURE_COLS):
            query2 = f"""
                SELECT {feature_cols_sql}
                FROM `{PROJECT}.{DATASET}.trending_tracks`
                WHERE {conditions}
            """
            rows = list(client.query(query2).result())

        if rows and rows[0]["tempo"] is not None:
            found = {c: float(rows[0][c] or centroid_features.get(c, 0.5)) for c in FEATURE_COLS}
            logger.info("avg_features: resolved from similar tracks in BQ")
            return found
    except Exception as e:
        logger.warning(f"avg_features query failed ({e}), using centroid")

    logger.info("avg_features: using mood centroid features")
    return centroid_features


def log_to_bigquery(
    client: bigquery.Client,
    generation_id: str,
    week_start: str,
    region: str,
    period: str,
    mood: str,
    mood_blend_json: str,
    prompt: str,
    similar_tracks: list[dict],
    gcs_path: str,
    duration_s: float,
) -> None:
    now_ts = datetime.now(timezone.utc).isoformat()
    similar_json = json.dumps([
        {"title": t["title"], "artist": t["artist"], "score": round(t["score"], 4)}
        for t in similar_tracks
    ])
    row = {
        "generation_id": generation_id,
        "week_start": week_start,
        "region": region,
        "period": period,
        "mood_archetype": mood,
        "mood_blend_json": mood_blend_json,
        "prompt_text": prompt,
        "similar_tracks_json": similar_json,
        "audio_gcs_path": gcs_path,
        "duration_seconds": round(duration_s, 2),
        "generated_at": now_ts,
    }
    errors = client.insert_rows_json(GEN_TABLE, [row])
    if errors:
        logger.error(f"BQ insert errors: {errors}")
    else:
        logger.info(f"Logged [{region}/{period}] generation to BigQuery")


def load_scaler_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler params not found at {path}. Run vector_index.py first."
        )
    with open(path) as f:
        params = json.load(f)
    assert len(params["feature_cols"]) == len(FEATURE_COLS), \
        "Dimension mismatch between scaler_params.json and FEATURE_COLS"
    return params


def scale_vector(raw: list[float], params: dict) -> list[float]:
    mean_ = np.array(params["mean_"])
    scale_ = np.array(params["scale_"])
    return ((np.array(raw) - mean_) / scale_).tolist()


def get_mood_centroid(
    client: bigquery.Client,
    mood: str,
    scaler_params: dict,
) -> tuple[list[float], dict]:
    """Return (scaled_vector, raw_feature_dict) for the given mood archetype."""
    feature_avgs_sql = ", ".join(f"AVG(h.{c}) AS {c}" for c in FEATURE_COLS)
    query = f"""
        SELECT {feature_avgs_sql}
        FROM `{PROJECT}.{DATASET}.trending_historical` h
        JOIN `{PROJECT}.{DATASET}.audio_mood_clusters` c
            ON  LOWER(TRIM(h.title))  = LOWER(TRIM(c.title))
            AND LOWER(TRIM(h.artist)) = LOWER(TRIM(c.artist))
            AND CAST(h.week_start AS DATE) = c.week_start
        WHERE c.mood_archetype = '{mood}'
          AND h.tempo IS NOT NULL
    """
    rows = list(client.query(query).result())

    if rows and rows[0]["tempo"] is not None:
        raw_centroid = [float(rows[0][c] or 0.0) for c in FEATURE_COLS]
        logger.info(f"Centroid for mood='{mood}' from BQ")
    else:
        logger.warning(f"Mood '{mood}' not found — falling back to any available mood")
        fallback_query = f"""
            SELECT c.mood_archetype,
                   {", ".join(f"AVG(h.{c}) AS {c}" for c in FEATURE_COLS)}
            FROM `{PROJECT}.{DATASET}.trending_historical` h
            JOIN `{PROJECT}.{DATASET}.audio_mood_clusters` c
                ON  LOWER(TRIM(h.title))  = LOWER(TRIM(c.title))
                AND LOWER(TRIM(h.artist)) = LOWER(TRIM(c.artist))
                AND CAST(h.week_start AS DATE) = c.week_start
            WHERE h.tempo IS NOT NULL
            GROUP BY c.mood_archetype
            LIMIT 1
        """
        fb_rows = list(client.query(fallback_query).result())
        if not fb_rows:
            logger.error("No mood centroids available — using zero vector")
            return [0.0] * len(FEATURE_COLS), {c: 0.5 for c in FEATURE_COLS}
        raw_centroid = [float(fb_rows[0][c] or 0.0) for c in FEATURE_COLS]
        logger.info(f"Fallback mood='{fb_rows[0]['mood_archetype']}'")

    raw_feature_dict = dict(zip(FEATURE_COLS, raw_centroid))
    return scale_vector(raw_centroid, scaler_params), raw_feature_dict


def query_pinecone_top_k(
    index,
    query_vector: list[float],
    mood: str,
    top_k: int = TOP_K,
) -> list[dict]:
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"mood_archetype": {"$eq": mood}},
    )
    matches = result.get("matches", [])

    if len(matches) < 3:
        logger.warning(f"Only {len(matches)} filtered matches — falling back to unfiltered")
        result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        matches = result.get("matches", [])

    tracks = []
    for m in matches:
        meta = m.get("metadata", {})
        tracks.append({
            "id": m["id"],
            "score": float(m["score"]),
            "title": meta.get("title", ""),
            "artist": meta.get("artist", ""),
            "mood_archetype": meta.get("mood_archetype", ""),
        })
    return tracks


def _season_for_date(d: date) -> str:
    m = d.month
    if m in (12, 1, 2):  return "winter"
    if m in (3, 4, 5):   return "spring"
    if m in (6, 7, 8):   return "summer"
    return "autumn"


# Canonical valence/energy per archetype so prompts stay internally consistent.
MOOD_VALENCE_ENERGY = {
    "euphoric": (0.80, 0.78),
    "melancholic": (0.22, 0.32),
    "aggressive": (0.22, 0.82),
    "peaceful": (0.75, 0.22),
    "groovy": (0.65, 0.68),
}


KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def build_prompt(
    primary_mood: str,
    mood_blend: dict,
    avg_features: dict,
    target_date: date,
) -> str:
    """Build a season-aware MusicGen prompt from mood prediction and audio features.

    Uses actual audio features from similar tracks so that prompts vary day-to-day
    rather than being identical for the same mood archetype.
    """
    season = _season_for_date(target_date)
    season_desc = SEASON_TEXTURE.get(season, "")
    prefix = MOOD_PREFIXES.get(primary_mood, f"{primary_mood} music,")

    # pick secondary flavour from blend
    secondary_flavor = ""
    for mood, prob in sorted(mood_blend.items(), key=lambda x: -x[1]):
        if mood == primary_mood or prob < 0.15:
            continue
        secondary_flavor = BLEND_HINTS.get(mood, "")
        break

    # Use actual audio features; fall back to canonical only when missing
    canon_valence, canon_energy = MOOD_VALENCE_ENERGY.get(
        primary_mood, (0.5, 0.5)
    )
    valence = avg_features.get("valence", canon_valence)
    energy = avg_features.get("energy", canon_energy)

    tempo = avg_features.get("tempo", 120)
    if tempo < 90:
        tempo_desc = f"slow {tempo:.0f} BPM"
    elif tempo < 115:
        tempo_desc = f"moderate {tempo:.0f} BPM"
    elif tempo < 140:
        tempo_desc = f"driving {tempo:.0f} BPM"
    else:
        tempo_desc = f"frenetic {tempo:.0f} BPM"

    if energy < 0.3:
        energy_desc = "hushed and delicate"
    elif energy < 0.5:
        energy_desc = "restrained energy"
    elif energy < 0.7:
        energy_desc = "charged energy"
    else:
        energy_desc = "explosive high energy"

    if valence < 0.25:
        valence_desc = "deeply minor key"
    elif valence < 0.45:
        valence_desc = "melancholic minor key"
    elif valence < 0.6:
        valence_desc = "bittersweet tonality"
    else:
        valence_desc = "uplifting major key"

    dance = avg_features.get("danceability", 0.5)
    if dance < 0.3:
        dance_desc = "steady rhythmic pulse"
    elif dance < 0.55:
        dance_desc = "rhythmic pulse"
    elif dance < 0.75:
        dance_desc = "locked-in groove"
    else:
        dance_desc = "infectious danceable groove"

    acoustic = avg_features.get("acousticness", 0.3)
    if acoustic > 0.6:
        acoustic_desc = "warm acoustic instrumentation"
    elif acoustic < 0.2:
        acoustic_desc = "dense electronic production"
    else:
        acoustic_desc = "hybrid acoustic-electronic"

    # Key signature from similar tracks
    key_desc = ""
    key_val = avg_features.get("key")
    mode_val = avg_features.get("mode")
    if key_val is not None and mode_val is not None:
        key_idx = int(round(key_val)) % 12
        mode_str = "major" if round(mode_val) == 1 else "minor"
        key_desc = f"{KEY_NAMES[key_idx]} {mode_str}"

    # Spectral brightness
    brightness_desc = ""
    centroid = avg_features.get("spectral_centroid")
    if centroid is not None:
        if centroid < 2000:
            brightness_desc = "warm dark timbre"
        elif centroid < 3500:
            brightness_desc = "balanced mid-range"
        else:
            brightness_desc = "bright shimmering highs"

    # Loudness character
    loud_desc = ""
    loudness = avg_features.get("loudness")
    if loudness is not None:
        if loudness > -5:
            loud_desc = "loud and compressed"
        elif loudness > -10:
            loud_desc = "punchy mix"
        else:
            loud_desc = "dynamic and spacious mix"

    parts = [
        prefix,
        f"{season}, {season_desc}",
        tempo_desc,
        energy_desc,
        valence_desc,
        key_desc,
        dance_desc,
        acoustic_desc,
        brightness_desc,
        loud_desc,
    ]
    if secondary_flavor:
        parts.append(secondary_flavor)

    return ", ".join(p for p in parts if p)


def generate_audio(prompt: str) -> tuple[np.ndarray, int]:
    logger.info(f"Loading {MUSICGEN_MODEL} ...")
    processor = AutoProcessor.from_pretrained(MUSICGEN_MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL)
    model.eval()

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")

    logger.info(f"Generating {MAX_NEW_TOKENS} tokens (~10s) ... prompt: {prompt}")
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    audio_np = audio_values[0, 0].numpy()
    sample_rate = model.config.audio_encoder.sampling_rate
    duration_s = len(audio_np) / sample_rate
    logger.info(f"Generated {duration_s:.1f}s at {sample_rate}Hz")
    return audio_np, sample_rate


def upload_wav_to_gcs(
    audio_np: np.ndarray,
    sample_rate: int,
    mood: str,
    week_start: str,
    region: str,
    period: str,
) -> str:
    gcs_filename = f"{GCS_PREFIX}/{week_start}_{region}_{period}_{mood}.wav"
    gcs_uri = f"gs://{BUCKET_NAME}/{gcs_filename}"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio_np, sample_rate)
        gcs_client = storage.Client(project=PROJECT)
        bucket = gcs_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_filename)
        blob.upload_from_filename(tmp_path, content_type="audio/wav")
        logger.info(f"Uploaded {gcs_uri}")
    finally:
        os.unlink(tmp_path)

    return gcs_uri


def main():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not found in .env")

    bq_client = bigquery.Client(project=PROJECT)
    ensure_gen_table(bq_client)

    region_predictions = fetch_predictions_by_region_period(bq_client)
    scaler_params = load_scaler_params(SCALER_PATH)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX)

    today = date.today()
    this_monday = today - timedelta(days=today.weekday())
    week_start_str = str(this_monday)

    base_periods_due = {"today"}
    if today.day == 1:
        base_periods_due.add("monthly")
    logger.info(f"Base periods due today ({today}): {base_periods_due}")

    songs_generated = 0
    month_start_str = str(today.replace(day=1))

    for region in REGIONS:
        preds = region_predictions.get(region)
        if not preds:
            logger.warning(f"  [{region}] No predictions found — skipping")
            continue

        periods_due = set(base_periods_due)
        if today.weekday() == 0 or not song_exists(bq_client, "weekly", week_start_str, region):
            periods_due.add("weekly")
            if today.weekday() != 0:
                logger.info(f"  [{region}] weekly: no song this week — generating catch-up")
        if today.day != 1 and not song_exists(bq_client, "monthly", month_start_str, region):
            periods_due.add("monthly")
            logger.info(f"  [{region}] monthly: no song this month — generating catch-up")

        for period_name in ("today", "weekly", "monthly"):
            if period_name not in periods_due:
                logger.info(f"  [{region}/{period_name}] Skipping — not due today")
                continue
            pred = preds.get(period_name)
            if not pred:
                logger.warning(f"  [{region}/{period_name}] No prediction row — skipping")
                continue

            mood = pred["predicted_mood"]
            mood_blend_str = pred["mood_blend_json"]
            target_dt = date.fromisoformat(pred["target_date"])

            try:
                mood_blend = json.loads(mood_blend_str)
            except (json.JSONDecodeError, TypeError):
                mood_blend = {mood: 1.0}

            logger.info(f"[{region}] {period_name} | mood: {mood} | target: {target_dt}")

            centroid, centroid_features = get_mood_centroid(bq_client, mood, scaler_params)
            similar_tracks = query_pinecone_top_k(index, centroid, mood)
            logger.info(f"  Top {len(similar_tracks)} similar tracks:")
            for t in similar_tracks:
                logger.info(f"    {t['score']:.3f}  {t['title']} - {t['artist']}")

            avg_features = compute_avg_features(bq_client, similar_tracks, centroid_features)
            prompt = build_prompt(mood, mood_blend, avg_features, target_dt)
            logger.info(f"  Prompt: {prompt}")

            audio_np, sr = generate_audio(prompt)
            duration_s = len(audio_np) / sr

            gcs_path = upload_wav_to_gcs(audio_np, sr, mood, week_start_str, region, period_name)

            generation_id = str(uuid.uuid4())
            log_to_bigquery(
                bq_client,
                generation_id=generation_id,
                week_start=week_start_str,
                region=region,
                period=period_name,
                mood=mood,
                mood_blend_json=mood_blend_str,
                prompt=prompt,
                similar_tracks=similar_tracks,
                gcs_path=gcs_path,
                duration_s=duration_s,
            )
            logger.info(f"  Done [{region}/{period_name}]: {mood} | {duration_s:.1f}s | {gcs_path}")
            songs_generated += 1

    logger.info(f"Module 13 complete, {songs_generated} songs generated")


if __name__ == "__main__":
    main()
