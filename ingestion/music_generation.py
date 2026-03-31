"""
SoundPulse — Module 13, Layer 2
MusicGen Audio Generation Pipeline

Reads the latest XGBoost mood prediction → queries Pinecone for the 10 most
sonically similar real tracks → builds a text prompt → generates 10 seconds
of audio with MusicGen-small → uploads WAV to GCS → logs to BigQuery.

Run AFTER vector_index.py (requires ingestion/scaler_params.json).

Install first:
    pip install "pinecone-client>=3.0,<4.0" soundfile
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from google.cloud import bigquery, storage
from loguru import logger
from pinecone import Pinecone
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# ── Config ──────────────────────────────────────────────────────────────────────
PROJECT        = "soundpulse-production"
DATASET        = "music_analytics"
BUCKET_NAME    = "soundpulse-prod-raw-lake"
GCS_PREFIX     = "generated"
PRED_TABLE     = f"{PROJECT}.{DATASET}.ml_predictions"
GEN_TABLE      = f"{PROJECT}.{DATASET}.generated_tracks"
PINECONE_INDEX = "soundpulse-tracks"
SCALER_PATH    = Path(__file__).parent / "scaler_params.json"
MUSICGEN_MODEL = "facebook/musicgen-small"
TOP_K          = 10
MAX_NEW_TOKENS = 500    # 500 tokens / 50 fps = 10 seconds of audio

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

GEN_SCHEMA = [
    bigquery.SchemaField("generation_id",       "STRING"),
    bigquery.SchemaField("week_start",          "DATE"),
    bigquery.SchemaField("mood_archetype",      "STRING"),
    bigquery.SchemaField("prompt_text",         "STRING"),
    bigquery.SchemaField("similar_tracks_json", "STRING"),
    bigquery.SchemaField("audio_gcs_path",      "STRING"),
    bigquery.SchemaField("duration_seconds",    "FLOAT64"),
    bigquery.SchemaField("generated_at",        "TIMESTAMP"),
]

MOOD_PREFIXES = {
    "euphoric":    "euphoric pop music, celebratory anthem,",
    "melancholic": "melancholic indie ballad, emotional and introspective,",
    "aggressive":  "aggressive rock music, intense and driving,",
    "peaceful":    "peaceful ambient music, calm and meditative,",
    "groovy":      "groovy funk music, smooth and rhythmic,",
}


# ── BQ helpers ──────────────────────────────────────────────────────────────────

def ensure_gen_table(client: bigquery.Client) -> None:
    """Create generated_tracks if it doesn't exist. NEVER drops — preserves history."""
    table = bigquery.Table(GEN_TABLE, schema=GEN_SCHEMA)
    client.create_table(table, exists_ok=True)
    logger.info(f"Table ready (exists_ok): {GEN_TABLE}")


def fetch_latest_prediction(client: bigquery.Client) -> dict:
    query = f"""
        SELECT week_start, predicted_mood, confidence
        FROM `{PRED_TABLE}`
        ORDER BY week_start DESC
        LIMIT 1
    """
    rows = list(client.query(query).result())
    if not rows:
        raise RuntimeError("No rows in ml_predictions — run ml_predictions.py first")
    row = rows[0]
    return {
        "week_start":     str(row["week_start"]),
        "predicted_mood": str(row["predicted_mood"]),
        "confidence":     float(row["confidence"]),
    }


def compute_avg_features(client: bigquery.Client, similar_tracks: list[dict]) -> dict:
    """Look up raw audio features for the top-K tracks and compute per-feature mean."""
    titles_artists = [(t["title"], t["artist"]) for t in similar_tracks]
    if not titles_artists:
        return {col: 0.5 for col in FEATURE_COLS}

    conditions = " OR ".join(
        f"(LOWER(TRIM(title)) = '{t.lower().replace(chr(39), chr(39)*2)}' "
        f"AND LOWER(TRIM(artist)) = '{a.lower().replace(chr(39), chr(39)*2)}')"
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
        # Fallback to trending_tracks
        query2 = f"""
            SELECT {feature_cols_sql}
            FROM `{PROJECT}.{DATASET}.trending_tracks`
            WHERE {conditions}
        """
        rows = list(client.query(query2).result())

    if rows and rows[0]["tempo"] is not None:
        return {c: float(rows[0][c] or 0.5) for c in FEATURE_COLS}
    return {col: 0.5 for col in FEATURE_COLS}


def log_to_bigquery(
    client: bigquery.Client,
    generation_id: str,
    week_start: str,
    mood: str,
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
        "generation_id":       generation_id,
        "week_start":          week_start,
        "mood_archetype":      mood,
        "prompt_text":         prompt,
        "similar_tracks_json": similar_json,
        "audio_gcs_path":      gcs_path,
        "duration_seconds":    round(duration_s, 2),
        "generated_at":        now_ts,
    }
    errors = client.insert_rows_json(GEN_TABLE, [row])
    if errors:
        logger.error(f"BQ insert errors: {errors}")
    else:
        logger.info("Logged generation run to BigQuery")


# ── Scaler ──────────────────────────────────────────────────────────────────────

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


# ── Pinecone ────────────────────────────────────────────────────────────────────

def get_mood_centroid(
    client: bigquery.Client,
    mood: str,
    scaler_params: dict,
) -> list[float]:
    """
    Compute the centroid of all tracks with the given mood_archetype by averaging
    their raw 30-feature vectors from trending_historical JOIN audio_mood_clusters,
    then scale using the saved scaler params.

    Falls back to the nearest available archetype if the target mood has no data.
    """
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
        logger.info(f"Computed centroid for mood='{mood}' from BQ")
    else:
        logger.warning(f"Mood '{mood}' not found in BQ — falling back to any available mood")
        fallback_query = f"""
            SELECT c.mood_archetype, {", ".join(f"AVG(h.{c}) AS {c}" for c in FEATURE_COLS)}
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
            logger.error("No mood centroids available at all — using zero vector")
            return [0.0] * len(FEATURE_COLS)
        raw_centroid = [float(fb_rows[0][c] or 0.0) for c in FEATURE_COLS]
        logger.info(f"Using fallback mood='{fb_rows[0]['mood_archetype']}'")

    return scale_vector(raw_centroid, scaler_params)


def query_pinecone_top_k(
    index,
    query_vector: list[float],
    mood: str,
    top_k: int = TOP_K,
) -> list[dict]:
    # Try filtered query first
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"mood_archetype": {"$eq": mood}},
    )
    matches = result.get("matches", [])

    if len(matches) < 3:
        logger.warning(f"Only {len(matches)} filtered matches for mood='{mood}' — falling back to unfiltered")
        result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        matches = result.get("matches", [])

    tracks = []
    for m in matches:
        meta = m.get("metadata", {})
        tracks.append({
            "id":            m["id"],
            "score":         float(m["score"]),
            "title":         meta.get("title", ""),
            "artist":        meta.get("artist", ""),
            "mood_archetype": meta.get("mood_archetype", ""),
        })
    return tracks


# ── Prompt builder ───────────────────────────────────────────────────────────────

def build_prompt(mood: str, avg_features: dict) -> str:
    prefix = MOOD_PREFIXES.get(mood, f"{mood} music,")

    tempo = avg_features.get("tempo", 120)
    if tempo < 90:
        tempo_desc = "slow tempo"
    elif tempo < 120:
        tempo_desc = "moderate tempo"
    elif tempo < 140:
        tempo_desc = f"upbeat, {tempo:.0f} BPM"
    else:
        tempo_desc = f"fast-paced, {tempo:.0f} BPM"

    energy = avg_features.get("energy", 0.5)
    if energy < 0.4:
        energy_desc = "soft and gentle"
    elif energy < 0.7:
        energy_desc = "moderate energy"
    else:
        energy_desc = "high energy"

    valence = avg_features.get("valence", 0.5)
    if valence < 0.35:
        valence_desc = "melancholic, minor key"
    elif valence < 0.6:
        valence_desc = "bittersweet"
    else:
        valence_desc = "uplifting, major key"

    dance = avg_features.get("danceability", 0.5)
    if dance > 0.7:
        dance_desc = "danceable groove"
    elif dance > 0.4:
        dance_desc = "rhythmic"
    else:
        dance_desc = "ambient"

    acoustic = avg_features.get("acousticness", 0.3)
    if acoustic > 0.6:
        acoustic_desc = "acoustic instruments"
    elif acoustic < 0.2:
        acoustic_desc = "electronic production"
    else:
        acoustic_desc = None

    parts = [prefix, tempo_desc, energy_desc, valence_desc, dance_desc]
    if acoustic_desc:
        parts.append(acoustic_desc)

    return " ".join(parts)


# ── MusicGen ─────────────────────────────────────────────────────────────────────

def generate_audio(prompt: str) -> tuple[np.ndarray, int]:
    logger.info(f"Loading {MUSICGEN_MODEL} …")
    processor = AutoProcessor.from_pretrained(MUSICGEN_MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL)
    model.eval()

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")

    logger.info(f"Generating {MAX_NEW_TOKENS} tokens (~10 seconds) on CPU …")
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    # audio_values shape: (1, 1, num_samples)
    audio_np = audio_values[0, 0].numpy()
    sample_rate = model.config.audio_encoder.sampling_rate
    duration_s = len(audio_np) / sample_rate
    logger.info(f"Generated {duration_s:.1f}s of audio at {sample_rate}Hz")
    return audio_np, sample_rate


# ── GCS upload ───────────────────────────────────────────────────────────────────

def upload_wav_to_gcs(
    audio_np: np.ndarray,
    sample_rate: int,
    mood: str,
    week_start: str,
) -> str:
    gcs_filename = f"{GCS_PREFIX}/{week_start}_{mood}.wav"
    gcs_uri = f"gs://{BUCKET_NAME}/{gcs_filename}"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio_np, sample_rate)
        gcs_client = storage.Client(project=PROJECT)
        bucket = gcs_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_filename)
        blob.upload_from_filename(tmp_path, content_type="audio/wav")
        logger.info(f"Uploaded WAV to {gcs_uri}")
    finally:
        os.unlink(tmp_path)

    return gcs_uri


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not found in .env")

    bq_client = bigquery.Client(project=PROJECT)

    # 1. Ensure output table exists
    ensure_gen_table(bq_client)

    # 2. Get latest prediction
    prediction = fetch_latest_prediction(bq_client)
    mood       = prediction["predicted_mood"]
    week_start = prediction["week_start"]
    confidence = prediction["confidence"]
    logger.info(f"Latest prediction: mood='{mood}', confidence={confidence:.1%}, week={week_start}")

    # 3. Load scaler
    scaler_params = load_scaler_params(SCALER_PATH)

    # 4. Get mood centroid → query Pinecone
    centroid = get_mood_centroid(bq_client, mood, scaler_params)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX)
    similar_tracks = query_pinecone_top_k(index, centroid, mood)

    logger.info(f"Top {len(similar_tracks)} similar tracks:")
    for t in similar_tracks:
        logger.info(f"  {t['score']:.3f}  {t['title']} — {t['artist']} [{t['mood_archetype']}]")

    # 5. Build prompt from avg features of neighbors
    avg_features = compute_avg_features(bq_client, similar_tracks)
    prompt = build_prompt(mood, avg_features)
    logger.info(f"MusicGen prompt: {prompt}")

    # 6. Generate audio
    audio_np, sample_rate = generate_audio(prompt)
    duration_s = len(audio_np) / sample_rate

    # 7. Upload to GCS
    gcs_path = upload_wav_to_gcs(audio_np, sample_rate, mood, week_start)

    # 8. Log to BigQuery
    generation_id = str(uuid.uuid4())
    log_to_bigquery(
        bq_client,
        generation_id=generation_id,
        week_start=week_start,
        mood=mood,
        prompt=prompt,
        similar_tracks=similar_tracks,
        gcs_path=gcs_path,
        duration_s=duration_s,
    )

    # 9. Summary
    logger.info("─── MODULE 13 COMPLETE ───")
    logger.info(f"  Predicted mood     : {mood} ({confidence:.1%} confidence)")
    logger.info(f"  Prompt             : {prompt}")
    logger.info(f"  Audio duration     : {duration_s:.1f}s")
    logger.info(f"  GCS path           : {gcs_path}")
    logger.info(f"  Generation ID      : {generation_id}")


if __name__ == "__main__":
    main()
