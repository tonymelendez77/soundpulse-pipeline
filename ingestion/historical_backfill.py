"""
SoundPulse - Historical Backfill
- Guardian news: daily, 3 months x 8 topics -> news_historical
- Billboard charts: weekly, 13 weeks x 10 charts -> trending_historical
- iTunes + Librosa: audio features for all unique Billboard songs
One-time run. Parallelized Guardian fetching.
"""

import os
import re
import json
import time
import math
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from difflib import SequenceMatcher
from dotenv import load_dotenv
from loguru import logger
from google.cloud import bigquery, storage
from bs4 import BeautifulSoup

load_dotenv()

PROJECT          = "soundpulse-production"
DATASET          = "music_analytics"
BUCKET           = "soundpulse-prod-raw-lake"
BASE_DIR         = Path(__file__).parent.parent
DAYS_BACK_START  = 365   # max news history to fetch (smart resume skips already-collected)
DAYS_BACK_END    = 8
WEEKS_BACK_START = 52    # max music history to fetch (smart resume skips already-collected)
WEEKS_BACK_END   = 2

SKIP_NEWS  = False        # Guardian daily quota exhausted — re-run tomorrow with False
SKIP_MUSIC = True       # set True to skip Billboard/iTunes/Librosa fetch

GUARDIAN_KEY = os.getenv("GUARDIAN_API_KEY")
GUARDIAN_URL = "https://content.guardianapis.com/search"

BILLBOARD_CHARTS = [
    {"slug": "hot-100",                  "name": "Hot 100"},
    {"slug": "billboard-global-200",     "name": "Global 200"},
    {"slug": "latin",                    "name": "Latin"},
    {"slug": "latin-pop-airplay",        "name": "Latin Pop Airplay"},
    {"slug": "regional-mexican-airplay", "name": "Regional Mexican Airplay"},
    {"slug": "tropical-airplay",         "name": "Tropical Airplay"},
    {"slug": "pop-songs",                "name": "Pop Songs"},
    {"slug": "rhythmic-40",              "name": "Rhythmic 40"},
    {"slug": "dance-club-play",          "name": "Dance Club Play"},
    {"slug": "hot-latin-songs",          "name": "Hot Latin Songs"},
]

BILLBOARD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

GUARDIAN_TOPICS = [
    {"query": "war conflict violence attack",       "topic": "conflict"},
    {"query": "mental health anxiety depression",   "topic": "mental_health"},
    {"query": "election politics government",       "topic": "politics"},
    {"query": "economic crisis recession inflation","topic": "economy"},
    {"query": "natural disaster earthquake flood",  "topic": "disaster"},
    {"query": "celebration festival victory award", "topic": "celebration"},
    {"query": "artificial intelligence technology", "topic": "technology"},
    {"query": "crime murder arrest corruption",     "topic": "crime"},
]

NEWS_SCHEMA = [
    bigquery.SchemaField("date",         "STRING"),
    bigquery.SchemaField("topic",        "STRING"),
    bigquery.SchemaField("title",        "STRING"),
    bigquery.SchemaField("description",  "STRING"),
    bigquery.SchemaField("url",          "STRING"),
    bigquery.SchemaField("published_at", "STRING"),
    bigquery.SchemaField("source",       "STRING"),
    bigquery.SchemaField("ingested_at",  "STRING"),
]

TRENDING_SCHEMA = [
    bigquery.SchemaField("week_start",              "STRING"),
    bigquery.SchemaField("week_number",             "INTEGER"),
    bigquery.SchemaField("chart_slug",              "STRING"),
    bigquery.SchemaField("chart_name",              "STRING"),
    bigquery.SchemaField("rank",                    "INTEGER"),
    bigquery.SchemaField("title",                   "STRING"),
    bigquery.SchemaField("artist",                  "STRING"),
    bigquery.SchemaField("itunes_track_id",         "INTEGER"),
    bigquery.SchemaField("itunes_title",            "STRING"),
    bigquery.SchemaField("itunes_artist",           "STRING"),
    bigquery.SchemaField("preview_url",             "STRING"),
    bigquery.SchemaField("itunes_duration_ms",      "FLOAT"),
    bigquery.SchemaField("itunes_genre",            "STRING"),
    bigquery.SchemaField("itunes_album",            "STRING"),
    bigquery.SchemaField("itunes_release_date",     "STRING"),
    bigquery.SchemaField("match_layer",             "INTEGER"),
    bigquery.SchemaField("tempo",                   "FLOAT"),
    bigquery.SchemaField("energy",                  "FLOAT"),
    bigquery.SchemaField("danceability",            "FLOAT"),
    bigquery.SchemaField("valence",                 "FLOAT"),
    bigquery.SchemaField("acousticness",            "FLOAT"),
    bigquery.SchemaField("instrumentalness",        "FLOAT"),
    bigquery.SchemaField("liveness",                "FLOAT"),
    bigquery.SchemaField("loudness",                "FLOAT"),
    bigquery.SchemaField("speechiness",             "FLOAT"),
    bigquery.SchemaField("key",                     "FLOAT"),
    bigquery.SchemaField("mode",                    "FLOAT"),
    bigquery.SchemaField("time_signature",          "FLOAT"),
    bigquery.SchemaField("mfcc_1",                  "FLOAT"),
    bigquery.SchemaField("mfcc_2",                  "FLOAT"),
    bigquery.SchemaField("mfcc_5",                  "FLOAT"),
    bigquery.SchemaField("mfcc_13",                 "FLOAT"),
    bigquery.SchemaField("chroma_C",                "FLOAT"),
    bigquery.SchemaField("chroma_C_sharp",          "FLOAT"),
    bigquery.SchemaField("chroma_D",                "FLOAT"),
    bigquery.SchemaField("chroma_D_sharp",          "FLOAT"),
    bigquery.SchemaField("chroma_E",                "FLOAT"),
    bigquery.SchemaField("chroma_F",                "FLOAT"),
    bigquery.SchemaField("chroma_F_sharp",          "FLOAT"),
    bigquery.SchemaField("chroma_G",                "FLOAT"),
    bigquery.SchemaField("chroma_G_sharp",          "FLOAT"),
    bigquery.SchemaField("chroma_A",                "FLOAT"),
    bigquery.SchemaField("chroma_A_sharp",          "FLOAT"),
    bigquery.SchemaField("chroma_B",                "FLOAT"),
    bigquery.SchemaField("spectral_centroid",       "FLOAT"),
    bigquery.SchemaField("harmonic_percussive_ratio","FLOAT"),
    bigquery.SchemaField("ingested_at",             "STRING"),
]


# ============================================================
# HELPERS
# ============================================================

def clean_record(record: dict) -> dict:
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


def upload_to_gcs(records: list, filename: str, prefix: str):
    if not records:
        return
    jsonl_path = BASE_DIR / f"{filename}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(clean_record(record), ensure_ascii=True, default=str) + "\n")
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob   = bucket.blob(f"{prefix}/{filename}.jsonl")
    blob.upload_from_filename(jsonl_path)
    logger.info(f"[OK] Uploaded gs://{BUCKET}/{prefix}/{filename}.jsonl ({len(records)} records)")


def load_to_bigquery(client, records: list, table_id: str, schema: list):
    if not records:
        logger.warning(f"No records to load for {table_id}")
        return
    full_table = f"{PROJECT}.{DATASET}.{table_id}"
    jsonl_path = BASE_DIR / f"{table_id}_temp.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(clean_record(record), ensure_ascii=True, default=str) + "\n")
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        ignore_unknown_values=True,
    )
    with open(jsonl_path, "rb") as f:
        job = client.load_table_from_file(f, full_table, job_config=job_config)
        job.result()
    logger.info(f"[OK] Loaded {len(records)} rows into {full_table}")


# ============================================================
# PHASE 1: GUARDIAN DAILY NEWS (PARALLEL)
# ============================================================

def fetch_guardian_topic(date_str: str, item: dict, ingested_at: str) -> list[dict]:
    try:
        r = requests.get(GUARDIAN_URL, params={
            "q":           item["query"],
            "api-key":     GUARDIAN_KEY,
            "from-date":   date_str,
            "to-date":     date_str,
            "page-size":   10,
            "show-fields": "headline,bodyText",
            "order-by":    "relevance",
            "format":      "json",
        }, timeout=15)
        r.raise_for_status()
        results = r.json().get("response", {}).get("results", [])
        articles = []
        for article in results:
            articles.append({
                "date":         date_str,
                "topic":        item["topic"],
                "title":        article.get("webTitle"),
                "description":  article.get("fields", {}).get("bodyText", "")[:300],
                "url":          article.get("webUrl"),
                "published_at": article.get("webPublicationDate"),
                "source":       "guardian",
                "ingested_at":  ingested_at,
            })
        return articles
    except Exception as e:
        logger.error(f"Guardian failed for {item['topic']} on {date_str}: {e}")
        return []


def fetch_guardian_day(date_str: str) -> list[dict]:
    if not GUARDIAN_KEY:
        return []
    ingested_at  = datetime.now(tz=timezone.utc).isoformat()
    all_articles = []
    # max_workers=2 (not 4) to stay well under Guardian's burst rate limit
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_guardian_topic, date_str, item, ingested_at): item
                   for item in GUARDIAN_TOPICS}
        for future in as_completed(futures):
            all_articles.extend(future.result())
    return all_articles


def run_news_backfill(existing_dates: set) -> list[dict]:
    total_days = DAYS_BACK_START - DAYS_BACK_END
    logger.info(f"Starting Guardian news backfill: scanning {total_days} days, skipping already-collected")
    all_articles = []
    batch        = []

    for day_num in range(DAYS_BACK_START, DAYS_BACK_END, -1):
        date_dt  = datetime.now(tz=timezone.utc) - timedelta(days=day_num)
        date_str = date_dt.strftime("%Y-%m-%d")
        if date_str in existing_dates:
            continue   # already in BQ — skip without sleeping
        articles = fetch_guardian_day(date_str)
        all_articles.extend(articles)
        batch.extend(articles)
        time.sleep(0.5)   # 0.5s between NEW fetches keeps under Guardian's rate limit

        days_done = DAYS_BACK_START - day_num
        if days_done % 7 == 0:
            logger.info(f"News progress: day {days_done}/{total_days} | total: {len(all_articles)}")

        if len(batch) >= 500:
            week_label = days_done // 7
            upload_to_gcs(batch, f"news_historical_batch_{week_label:02d}", "raw/historical/news")
            batch = []

    if batch:
        upload_to_gcs(batch, "news_historical_batch_final", "raw/historical/news")

    logger.info(f"[OK] News backfill complete: {len(all_articles)} articles")
    return all_articles


# ============================================================
# PHASE 2: BILLBOARD WEEKLY CHARTS
# ============================================================

def fetch_billboard_week(date_str: str, week_number: int) -> list[dict]:
    songs       = []
    ingested_at = datetime.now(tz=timezone.utc).isoformat()
    for chart in BILLBOARD_CHARTS:
        try:
            url      = f"https://www.billboard.com/charts/{chart['slug']}/{date_str}/"
            response = requests.get(url, headers=BILLBOARD_HEADERS, timeout=15)
            response.raise_for_status()
            soup    = BeautifulSoup(response.text, "html.parser")
            entries = soup.select("ul.o-chart-results-list-row")
            for rank, entry in enumerate(entries, start=1):
                title_el  = entry.select_one("h3#title-of-a-story")
                artist_el = entry.select_one("span.c-label.a-no-trucate")
                if not title_el:
                    continue
                songs.append({
                    "week_start":  date_str,
                    "week_number": week_number,
                    "chart_slug":  chart["slug"],
                    "chart_name":  chart["name"],
                    "rank":        rank,
                    "title":       title_el.get_text(strip=True),
                    "artist":      artist_el.get_text(strip=True) if artist_el else None,
                    "ingested_at": ingested_at,
                })
            logger.info(f"Week {week_number} {chart['name']}: {len(entries)} songs")
            time.sleep(3)
        except Exception as e:
            logger.error(f"Billboard failed for {chart['name']} week {week_number}: {e}")
    return songs


def run_billboard_backfill(existing_weeks: set) -> list[dict]:
    total_weeks = WEEKS_BACK_START - WEEKS_BACK_END + 1
    logger.info(f"Starting Billboard backfill: scanning {total_weeks} weeks, skipping already-collected")
    all_songs = []
    for week_num in range(WEEKS_BACK_START, WEEKS_BACK_END - 1, -1):
        week_start = datetime.now(tz=timezone.utc) - timedelta(weeks=week_num)
        date_str   = week_start.strftime("%Y-%m-%d")
        week_index = WEEKS_BACK_START - week_num + 1
        if date_str in existing_weeks:
            logger.info(f"Week {week_index}/{total_weeks}: {date_str} — already in BQ, skipping")
            continue
        logger.info(f"Fetching week {week_index}/{total_weeks}: {date_str}")
        songs = fetch_billboard_week(date_str, week_index)
        all_songs.extend(songs)
    logger.info(f"[OK] Billboard backfill complete: {len(all_songs)} chart entries")
    return all_songs


# ============================================================
# PHASE 3: ITUNES MATCHING + LIBROSA
# ============================================================

def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\bfeat\.?\b.*$', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def search_itunes_tracks(artist_name: str, limit: int = 15) -> list[dict]:
    url = "https://itunes.apple.com/search"
    params = {"term": artist_name, "media": "music", "entity": "song", "limit": limit}
    for attempt in range(3):
        try:
            time.sleep(0.5)
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            tracks = []
            for item in r.json().get("results", []):
                if item.get("previewUrl"):
                    tracks.append({
                        "itunes_track_id": item.get("trackId"),
                        "title":           item.get("trackName"),
                        "artist":          item.get("artistName"),
                        "album":           item.get("collectionName"),
                        "preview_url":     item.get("previewUrl"),
                        "duration_ms":     item.get("trackTimeMillis"),
                        "release_date":    item.get("releaseDate"),
                        "genre":           item.get("primaryGenreName"),
                    })
            return tracks
        except Exception as e:
            logger.error(f"iTunes search failed for {artist_name}: {e}")
            return []
    return []


def match_song_to_itunes(title: str, artist: str, itunes_catalog: list) -> dict:
    bb_title_norm  = normalize_text(title)
    bb_artist_norm = normalize_text(artist)

    # Layer 1: Exact
    for track in itunes_catalog:
        if normalize_text(track['title']) == bb_title_norm and normalize_text(track['artist']) == bb_artist_norm:
            return {**track, 'match_layer': 1}

    # Layer 2: Fuzzy 0.60
    best_score, best_match = 0, None
    for track in itunes_catalog:
        title_sim  = SequenceMatcher(None, bb_title_norm, normalize_text(track['title'])).ratio()
        artist_sim = SequenceMatcher(None, bb_artist_norm, normalize_text(track['artist'])).ratio()
        combined   = (0.7 * title_sim) + (0.3 * artist_sim)
        if combined >= 0.60 and combined > best_score:
            best_score, best_match = combined, track
    if best_match:
        return {**best_match, 'match_layer': 2}

    # Layer 3: Fuzzy 0.45
    best_score, best_match = 0, None
    for track in itunes_catalog:
        title_sim  = SequenceMatcher(None, bb_title_norm, normalize_text(track['title'])).ratio()
        artist_sim = SequenceMatcher(None, bb_artist_norm, normalize_text(track['artist'])).ratio()
        combined   = (0.7 * title_sim) + (0.3 * artist_sim)
        if combined >= 0.45 and combined > best_score:
            best_score, best_match = combined, track
    if best_match:
        return {**best_match, 'match_layer': 3}

    # Layer 4: Artist fallback
    best_score, best_match = 0, None
    for track in itunes_catalog:
        if bb_artist_norm in normalize_text(track['artist']):
            title_sim = SequenceMatcher(None, bb_title_norm, normalize_text(track['title'])).ratio()
            if title_sim >= 0.40 and title_sim > best_score:
                best_score, best_match = title_sim, track
    if best_match:
        return {**best_match, 'match_layer': 4}

    # Layer 5: Direct API
    try:
        time.sleep(0.5)
        r = requests.get(
            f"https://itunes.apple.com/search?term={requests.utils.quote(artist + ' ' + title)}&limit=1&media=music",
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            if data.get('resultCount', 0) > 0:
                result = data['results'][0]
                return {
                    'itunes_track_id': result.get('trackId'),
                    'title':           result.get('trackName'),
                    'artist':          result.get('artistName'),
                    'preview_url':     result.get('previewUrl'),
                    'duration_ms':     result.get('trackTimeMillis'),
                    'genre':           result.get('primaryGenreName'),
                    'album':           result.get('collectionName'),
                    'release_date':    result.get('releaseDate', '')[:10] if result.get('releaseDate') else None,
                    'match_layer':     5
                }
    except Exception:
        pass
    return {}


def enrich_with_audio_features(songs: list) -> list:
    from audio_features_librosa import enrich_with_librosa_features
    import pandas as pd

    unique = {}
    for song in songs:
        key = f"{song['title']}|{song['artist']}"
        if key not in unique and song.get('preview_url'):
            unique[key] = song

    if not unique:
        return songs

    logger.info(f"Extracting audio features for {len(unique)} unique songs...")
    df          = pd.DataFrame(list(unique.values()))
    df_enriched = enrich_with_librosa_features(df)

    audio_cols = ['tempo','energy','danceability','valence','acousticness','instrumentalness',
                  'liveness','loudness','speechiness','key','mode','time_signature',
                  'mfcc_1','mfcc_2','mfcc_5','mfcc_13','chroma_C','chroma_C_sharp',
                  'chroma_D','chroma_D_sharp','chroma_E','chroma_F','chroma_F_sharp',
                  'chroma_G','chroma_G_sharp','chroma_A','chroma_A_sharp','chroma_B',
                  'spectral_centroid','harmonic_percussive_ratio']

    features_map = {}
    for _, row in df_enriched.iterrows():
        key = f"{row['title']}|{row['artist']}"
        features_map[key] = {col: row.get(col) for col in audio_cols if col in row}

    enriched = []
    for song in songs:
        key      = f"{song['title']}|{song['artist']}"
        features = features_map.get(key, {})
        enriched.append({**song, **features})
    return enriched


def run_audio_backfill(billboard_songs: list) -> list:
    logger.info("Starting iTunes matching + Librosa audio extraction...")
    unique_artists = list({s['artist'] for s in billboard_songs if s.get('artist')})
    logger.info(f"Searching iTunes for {len(unique_artists)} unique artists...")

    itunes_catalog = []
    for idx, artist in enumerate(unique_artists):
        tracks = search_itunes_tracks(artist, limit=15)
        itunes_catalog.extend(tracks)
        if (idx + 1) % 20 == 0:
            logger.info(f"iTunes search: {idx+1}/{len(unique_artists)}")

    seen, deduped_catalog = set(), []
    for t in itunes_catalog:
        tid = t.get('itunes_track_id')
        if tid and tid not in seen:
            seen.add(tid)
            deduped_catalog.append(t)

    logger.info(f"iTunes catalog: {len(deduped_catalog)} unique tracks")

    unique_songs = {}
    for song in billboard_songs:
        key = f"{song['title']}|{song['artist']}"
        if key not in unique_songs:
            unique_songs[key] = song

    logger.info(f"Matching {len(unique_songs)} unique songs...")
    matched_map = {}
    for idx, (key, song) in enumerate(unique_songs.items()):
        match = match_song_to_itunes(song['title'], song['artist'], deduped_catalog)
        if match:
            matched_map[key] = match
        if (idx + 1) % 100 == 0:
            logger.info(f"Matching: {idx+1}/{len(unique_songs)}")

    matched = sum(1 for k in unique_songs if k in matched_map)
    logger.info(f"Match rate: {matched}/{len(unique_songs)} ({matched/len(unique_songs)*100:.1f}%)")

    matched_songs = []
    for key, song in unique_songs.items():
        if key in matched_map:
            m = matched_map[key]
            matched_songs.append({
                'title':               song['title'],
                'artist':              song['artist'],
                'preview_url':         m.get('preview_url'),
                'itunes_track_id':     m.get('itunes_track_id'),
                'itunes_title':        m.get('title'),
                'itunes_artist':       m.get('artist'),
                'itunes_duration_ms':  m.get('duration_ms'),
                'itunes_genre':        m.get('genre'),
                'itunes_album':        m.get('album'),
                'itunes_release_date': m.get('release_date'),
                'match_layer':         m.get('match_layer'),
            })

    enriched_songs   = enrich_with_audio_features(matched_songs)
    features_by_key  = {f"{s['title']}|{s['artist']}": s for s in enriched_songs}

    result = []
    for song in billboard_songs:
        key      = f"{song['title']}|{song['artist']}"
        features = features_by_key.get(key, {})
        result.append({**song, **features})
    return result


# ============================================================
# GAP DETECTION — fetch the set of already-collected dates/weeks
# ============================================================

def get_existing_news_dates(client: bigquery.Client) -> set:
    """Return the set of date strings already in news_historical (e.g. '2025-10-06').
    Used to skip dates we already have, wherever they fall in the timeline."""
    try:
        rows = list(client.query(
            f"SELECT DISTINCT date FROM `{PROJECT}.{DATASET}.news_historical`"
        ).result())
        dates = {r["date"] for r in rows}
        logger.info(f"news_historical: {len(dates)} unique dates already in BQ")
        return dates
    except Exception as e:
        logger.warning(f"Could not load existing news dates ({e}) — will fetch everything")
        return set()


def get_existing_music_weeks(client: bigquery.Client) -> set:
    """Return the set of week_start strings already in trending_historical (e.g. '2025-12-29').
    Used to skip weeks we already have, wherever they fall in the timeline."""
    try:
        rows = list(client.query(
            f"SELECT DISTINCT week_start FROM `{PROJECT}.{DATASET}.trending_historical`"
        ).result())
        weeks = {r["week_start"] for r in rows}
        logger.info(f"trending_historical: {len(weeks)} unique weeks already in BQ")
        return weeks
    except Exception as e:
        logger.warning(f"Could not load existing music weeks ({e}) — will fetch everything")
        return set()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("SOUNDPULSE - HISTORICAL BACKFILL (smart resume)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)

    bq_client = bigquery.Client(project=PROJECT)

    print("\n[1/6] Ensuring BigQuery tables exist...")
    for table_id, schema in [("news_historical", NEWS_SCHEMA), ("trending_historical", TRENDING_SCHEMA)]:
        full_table = f"{PROJECT}.{DATASET}.{table_id}"
        try:
            bq_client.get_table(full_table)
            logger.info(f"[OK] Table exists, will append: {full_table}")
        except Exception:
            bq_client.create_table(bigquery.Table(full_table, schema=schema))
            logger.info(f"[OK] Created {full_table}")

    # -- Smart resume: load existing date/week sets, skip what's already in BQ ─
    print("\n[1b/6] Scanning existing data in BQ...")
    existing_news_dates  = get_existing_news_dates(bq_client)
    existing_music_weeks = get_existing_music_weeks(bq_client)
    news_to_fetch  = (DAYS_BACK_START - DAYS_BACK_END) - len(
        {d for d in existing_news_dates
         if (datetime.now(tz=timezone.utc).date() - datetime.strptime(d, "%Y-%m-%d").date()).days
            in range(DAYS_BACK_END, DAYS_BACK_START)}
    )
    music_to_fetch = (WEEKS_BACK_START - WEEKS_BACK_END + 1) - len(
        {w for w in existing_music_weeks
         if (datetime.now(tz=timezone.utc).date() - datetime.strptime(w, "%Y-%m-%d").date()).days // 7
            in range(WEEKS_BACK_END, WEEKS_BACK_START + 1)}
    )
    print(f"  News  : ~{max(news_to_fetch, 0)} days missing out of {DAYS_BACK_START - DAYS_BACK_END}")
    print(f"  Music : ~{max(music_to_fetch, 0)} weeks missing out of {WEEKS_BACK_START - WEEKS_BACK_END + 1}")

    if SKIP_NEWS:
        print("\n[2/6] SKIP_NEWS=True — skipping Guardian fetch.")
        news_articles = []
    else:
        print("\n[2/6] Fetching Guardian historical news (missing dates only)...")
        news_articles = run_news_backfill(existing_news_dates)
        print(f"[OK] New news articles fetched: {len(news_articles)}")

    if SKIP_MUSIC:
        print("\n[3/6] SKIP_MUSIC=True — skipping Billboard fetch.")
        billboard_songs = []
    else:
        print("\n[3/6] Fetching Billboard historical charts (missing weeks only)...")
        billboard_songs = run_billboard_backfill(existing_music_weeks)
        print(f"[OK] New Billboard entries fetched: {len(billboard_songs)}")

    if billboard_songs:
        print("\n[4/6] iTunes matching + Librosa audio features...")
        enriched_songs = run_audio_backfill(billboard_songs)
        print(f"[OK] Enriched songs: {len(enriched_songs)}")
    else:
        print("\n[4/6] No new Billboard songs — skipping Librosa.")
        enriched_songs = []

    print("\n[5/6] Uploading to GCS...")
    if news_articles:
        upload_to_gcs(news_articles,  "news_historical_final",     "raw/historical/news")
    if enriched_songs:
        upload_to_gcs(enriched_songs, "trending_historical_final", "raw/historical/trending")

    print("\n[6/6] Loading to BigQuery...")
    if news_articles:
        load_to_bigquery(bq_client, news_articles,  "news_historical",    NEWS_SCHEMA)
    if enriched_songs:
        load_to_bigquery(bq_client, enriched_songs, "trending_historical", TRENDING_SCHEMA)

    print("\n[OK] Verification:")
    for table_id in ["news_historical", "trending_historical"]:
        rows = list(bq_client.query(
            f"SELECT COUNT(*) as cnt FROM {PROJECT}.{DATASET}.{table_id}"
        ).result())
        print(f"  {table_id}: {rows[0].cnt} rows")

    print("\n" + "=" * 60)
    print("HISTORICAL BACKFILL COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()