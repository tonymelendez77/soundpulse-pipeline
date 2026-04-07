"""
SoundPulse Unified Pipeline - Module 10 (Redesigned)
Sources: Billboard + iTunes Charts + Last.fm
Filters: Release date > 30 days + Artist newness + Cross-platform
8-layer matching system with duration constraints
Sentiment: Reddit + News + YouTube saved separately
"""

import math
import pandas as pd
import requests
import time
import re
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from difflib import SequenceMatcher
from google.cloud import bigquery, storage

# Import modules
from itunes_ingestion import fetch_all_itunes_charts, search_itunes_tracks
from lastfm_ingestion import run_lastfm_ingestion
from billboard_ingestion import run_billboard_ingestion
from spotify_ingestion import enrich_with_spotify_metadata
from audio_features_librosa import enrich_with_librosa_features
from youtube_ingestion import run_youtube_ingestion
from reddit_ingestion import run_reddit_ingestion
from news_ingestion import run_news_ingestion

# BigQuery config
PROJECT = "soundpulse-production"
DATASET = "music_analytics"

# TEST MODE
TEST_MODE = True

# Filter thresholds
RELEASE_AGE_MIN_DAYS = 30
ARTIST_NEWNESS_DAYS  = 14
CROSS_PLATFORM_MIN   = 2

# Paths
BASE_DIR        = Path(__file__).parent.parent
DIAGNOSTIC_FILE = BASE_DIR / "diagnostic_matching_results.json"
BUCKET_NAME     = "soundpulse-prod-raw-lake"


# ============================================================
# SHARED CLEANER
# ============================================================

def clean_record(record: dict) -> dict:
    """Replace NaN, Inf, and None-like floats with None for JSON safety."""
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            cleaned[k] = None
        elif v is None:
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


# ============================================================
# FILTER FUNCTIONS
# ============================================================

def apply_release_date_filter(df: pd.DataFrame, date_col: str = "release_date") -> pd.DataFrame:
    """Filter 1: Remove songs released less than 30 days ago."""
    if date_col not in df.columns:
        return df

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=RELEASE_AGE_MIN_DAYS)

    def is_old_enough(date_str):
        if not date_str or pd.isna(date_str):
            return True
        try:
            d = pd.to_datetime(date_str, utc=True)
            return d <= cutoff
        except Exception:
            return True

    before = len(df)
    df = df[df[date_col].apply(is_old_enough)].copy()
    removed = before - len(df)
    print(f"[Filter 1] Release date: removed {removed} new releases, kept {len(df)} tracks")
    return df


def apply_artist_newness_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter 2: Remove all songs by artists who released anything in last 14 days."""
    if "release_date" not in df.columns:
        return df

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=ARTIST_NEWNESS_DAYS)

    def has_recent_release(date_str):
        if not date_str or pd.isna(date_str):
            return False
        try:
            d = pd.to_datetime(date_str, utc=True)
            return d >= cutoff
        except Exception:
            return False

    recent_mask    = df["release_date"].apply(has_recent_release)
    recent_artists = set(df[recent_mask]["artist"].str.lower().unique())

    before  = len(df)
    df      = df[~df["artist"].str.lower().isin(recent_artists)].copy()
    removed = before - len(df)
    print(f"[Filter 2] Artist newness: removed {removed} tracks from {len(recent_artists)} active artists, kept {len(df)}")
    return df


def apply_cross_platform_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter 3: Keep only songs appearing on 2+ platforms."""
    def normalize(text):
        if pd.isna(text):
            return ""
        return re.sub(r'[^\w\s]', '', str(text).lower().strip())

    df         = df.copy()
    df["_key"] = df["title"].apply(normalize) + "|" + df["artist"].apply(normalize)
    counts     = df.groupby("_key")["source"].nunique().reset_index()
    counts.columns = ["_key", "source_count"]
    df         = df.merge(counts, on="_key", how="left")
    df         = df.drop(columns=["_key"])

    before  = len(df)
    df      = df[df["source_count"] >= CROSS_PLATFORM_MIN].copy()
    removed = before - len(df)
    print(f"[Filter 3] Cross-platform: removed {removed} single-source tracks, kept {len(df)} tracks on 2+ platforms")
    return df


# ============================================================
# 8-LAYER MATCHING
# ============================================================

def match_to_itunes(source_df, itunes_df):
    matched     = []
    unmatched   = []
    match_stats = {f'layer_{i}': 0 for i in range(1, 9)}

    def normalize_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\bfeat\.?\b.*$', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())

    def duration_match(d1, d2, tolerance_ms):
        if pd.isna(d1) or pd.isna(d2):
            return False
        return abs(d1 - d2) <= tolerance_ms

    # Build exact match hash table
    exact_match_table = {}
    for _, row in itunes_df.iterrows():
        key = f"{normalize_text(row['title'])}|{normalize_text(row['artist'])}"
        if key not in exact_match_table:
            exact_match_table[key] = []
        exact_match_table[key].append(row)

    # Pre-filter by first letter
    itunes_by_letter = {}
    for _, row in itunes_df.iterrows():
        norm         = normalize_text(row['title'])
        first_letter = norm[0] if norm else 'other'
        if first_letter not in itunes_by_letter:
            itunes_by_letter[first_letter] = []
        itunes_by_letter[first_letter].append(row)

    print(f"\n[OK] Starting 8-layer matching for {len(source_df)} tracks...")

    for idx, source_row in source_df.iterrows():
        bb_title_norm  = normalize_text(source_row['title'])
        bb_artist_norm = normalize_text(source_row['artist'])
        bb_duration    = source_row.get('duration_ms', pd.NA)
        bb_genre       = source_row.get('genre', '') or ''

        matched_track = None
        match_layer   = None

        # LAYER 1: Exact match + duration
        key = f"{bb_title_norm}|{bb_artist_norm}"
        if key in exact_match_table:
            for candidate in exact_match_table[key]:
                if duration_match(bb_duration, candidate['duration_ms'], 10000):
                    matched_track = candidate.to_dict()
                    match_layer   = 1
                    break
            if matched_track is None:
                matched_track = exact_match_table[key][0].to_dict()
                match_layer   = 1

        # LAYER 2: Fuzzy + duration (threshold 0.60)
        if matched_track is None:
            first_letter = bb_title_norm[0] if bb_title_norm else 'other'
            filtered     = itunes_by_letter.get(first_letter, [])
            best_score   = 0
            best_match   = None
            for itunes_row in filtered:
                if not duration_match(bb_duration, itunes_row['duration_ms'], 10000):
                    continue
                title_sim  = SequenceMatcher(None, bb_title_norm, normalize_text(itunes_row['title'])).ratio()
                artist_sim = SequenceMatcher(None, bb_artist_norm, normalize_text(itunes_row['artist'])).ratio()
                combined   = (0.7 * title_sim) + (0.3 * artist_sim)
                if combined >= 0.60 or (title_sim >= 0.70 and artist_sim >= 0.50):
                    if combined > best_score:
                        best_score = combined
                        best_match = itunes_row
            if best_match is not None:
                matched_track = best_match.to_dict()
                match_layer   = 2

        # LAYER 3: Aggressive fuzzy + duration (threshold 0.45)
        if matched_track is None:
            first_letter = bb_title_norm[0] if bb_title_norm else 'other'
            filtered     = itunes_by_letter.get(first_letter, [])
            best_score   = 0
            best_match   = None
            for itunes_row in filtered:
                if not duration_match(bb_duration, itunes_row['duration_ms'], 10000):
                    continue
                title_sim  = SequenceMatcher(None, bb_title_norm, normalize_text(itunes_row['title'])).ratio()
                artist_sim = SequenceMatcher(None, bb_artist_norm, normalize_text(itunes_row['artist'])).ratio()
                combined   = (0.7 * title_sim) + (0.3 * artist_sim)
                if combined >= 0.45:
                    if combined > best_score:
                        best_score = combined
                        best_match = itunes_row
            if best_match is not None:
                matched_track = best_match.to_dict()
                match_layer   = 3

        # LAYER 4: Genre + title + duration
        if matched_track is None and bb_genre:
            genre_filtered = itunes_df[
                (itunes_df['genre'].str.lower() == bb_genre.lower()) &
                (itunes_df['duration_ms'].notna())
            ].copy()
            best_score = 0
            best_match = None
            for _, itunes_row in genre_filtered.iterrows():
                if not duration_match(bb_duration, itunes_row['duration_ms'], 15000):
                    continue
                title_sim = SequenceMatcher(None, bb_title_norm, normalize_text(itunes_row['title'])).ratio()
                if title_sim >= 0.70 and title_sim > best_score:
                    best_score = title_sim
                    best_match = itunes_row
            if best_match is not None:
                matched_track = best_match.to_dict()
                match_layer   = 4

        # LAYER 5: Artist fallback with minimum title similarity (0.40)
        if matched_track is None:
            artist_tracks = itunes_df[
                itunes_df['artist'].str.lower().str.contains(bb_artist_norm, case=False, na=False, regex=False)
            ].copy()
            if not artist_tracks.empty:
                best_score = 0
                best_match = None
                for _, itunes_row in artist_tracks.iterrows():
                    title_sim = SequenceMatcher(None, bb_title_norm, normalize_text(itunes_row['title'])).ratio()
                    if title_sim >= 0.40 and title_sim > best_score:
                        best_score = title_sim
                        best_match = itunes_row
                if best_match is not None:
                    matched_track = best_match.to_dict()
                    match_layer   = 5

	# LAYER 6: Collaborator fallback + duration preference (no title constraint)
        if matched_track is None:
            collabs = [c.strip() for c in re.split(r'&|,|x|featuring|feat|ft|and', bb_artist_norm)
                       if c.strip() and len(c.strip()) >= 3]
            for collab in collabs:
                collab_tracks = itunes_df[
                    itunes_df['artist'].str.lower().str.contains(collab, case=False, na=False, regex=False)
                ].copy()
                if not collab_tracks.empty:
                    duration_matches = collab_tracks[
                        collab_tracks['duration_ms'].apply(lambda x: duration_match(bb_duration, x, 30000))
                    ]
                    matched_track = duration_matches.iloc[0].to_dict() if not duration_matches.empty else collab_tracks.iloc[0].to_dict()
                    match_layer   = 6
                    break

        # LAYER 7: Partial title + duration
        if matched_track is None:
            bb_words = set(w for w in bb_title_norm.split() if len(w) >= 4)
            if bb_words:
                best_overlap = 0
                best_match   = None
                for _, itunes_row in itunes_df.iterrows():
                    if not duration_match(bb_duration, itunes_row['duration_ms'], 15000):
                        continue
                    itunes_title_norm = normalize_text(itunes_row['title'])
                    itunes_words      = set(w for w in itunes_title_norm.split() if len(w) >= 4)
                    if not itunes_words:
                        continue
                    overlap = len(bb_words & itunes_words)
                    if overlap / len(bb_words) >= 0.5 and bb_artist_norm == normalize_text(itunes_row['artist']):
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match   = itunes_row
                if best_match is not None:
                    matched_track = best_match.to_dict()
                    match_layer   = 7

        # LAYER 8: Direct iTunes API search (no duration constraint)
        if matched_track is None:
            search_term = f"{source_row['artist']} {source_row['title']}"
            search_url  = f"https://itunes.apple.com/search?term={requests.utils.quote(search_term)}&limit=1&media=music"
            try:
                time.sleep(0.5)
                response = requests.get(search_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('resultCount', 0) > 0:
                        result        = data['results'][0]
                        matched_track = {
                            'itunes_track_id': result.get('trackId'),
                            'title':           result.get('trackName'),
                            'artist':          result.get('artistName'),
                            'preview_url':     result.get('previewUrl'),
                            'duration_ms':     result.get('trackTimeMillis'),
                            'genre':           result.get('primaryGenreName'),
                            'album':           result.get('collectionName'),
                            'release_date':    result.get('releaseDate', '')[:10] if result.get('releaseDate') else None
                        }
                        match_layer = 8
            except Exception:
                pass

        if matched_track:
            result_row = source_row.to_dict()
            result_row.update({
                'itunes_track_id':     matched_track.get('itunes_track_id'),
                'itunes_title':        matched_track.get('title'),
                'itunes_artist':       matched_track.get('artist'),
                'preview_url':         matched_track.get('preview_url'),
                'itunes_duration_ms':  matched_track.get('duration_ms'),
                'itunes_genre':        matched_track.get('genre'),
                'itunes_album':        matched_track.get('album'),
                'itunes_release_date': matched_track.get('release_date'),
                'match_layer':         match_layer
            })
            matched.append(result_row)
            match_stats[f'layer_{match_layer}'] += 1
        else:
            unmatched.append(source_row.to_dict())

    print("\n[OK] Matching complete:")
    for layer, count in match_stats.items():
        if count > 0:
            print(f"  {layer}: {count} matches ({count/len(source_df)*100:.1f}%)")
    print(f"  Unmatched: {len(unmatched)} ({len(unmatched)/len(source_df)*100:.1f}%)")

    matched_df = pd.DataFrame(matched)

    # Deduplication: keep best match (lowest layer) per itunes_track_id
    if not matched_df.empty and 'itunes_track_id' in matched_df.columns:
        before     = len(matched_df)
        matched_df = matched_df.sort_values('match_layer').drop_duplicates(subset=['itunes_track_id'], keep='first')
        removed    = before - len(matched_df)
        if removed > 0:
            print(f"  Deduped: removed {removed} duplicate iTunes matches")

    return matched_df, pd.DataFrame(unmatched)


# ============================================================
# HELPERS
# ============================================================

def save_diagnostic_json(matched_df, unmatched_df):
    matched_records   = [clean_record(r) for r in matched_df.to_dict('records')]
    unmatched_records = [clean_record(r) for r in unmatched_df.to_dict('records')]

    diagnostic_data = {
        "timestamp":        datetime.now().isoformat(),
        "total_tracks":     len(matched_df) + len(unmatched_df),
        "matched":          len(matched_df),
        "unmatched":        len(unmatched_df),
        "match_rate":       f"{len(matched_df)/(len(matched_df)+len(unmatched_df))*100:.1f}%" if (len(matched_df)+len(unmatched_df)) > 0 else "0%",
        "layer_breakdown":  {},
        "matched_tracks":   matched_records,
        "unmatched_tracks": unmatched_records
    }

    if 'match_layer' in matched_df.columns:
        for layer in range(1, 9):
            count = len(matched_df[matched_df['match_layer'] == layer])
            if count > 0:
                diagnostic_data["layer_breakdown"][f"layer_{layer}"] = count

    with open(DIAGNOSTIC_FILE, 'w', encoding='utf-8') as f:
        json.dump(diagnostic_data, f, indent=2, ensure_ascii=True, default=str)
    print(f"[OK] Diagnostic JSON saved: {DIAGNOSTIC_FILE}")


def upload_to_gcs(data, filename, prefix="raw"):
    if isinstance(data, pd.DataFrame):
        records = data.to_dict('records')
    else:
        records = data

    jsonl_path = BASE_DIR / f"{filename}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(clean_record(record), ensure_ascii=True, default=str) + '\n')

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob   = bucket.blob(f"{prefix}/{filename}.jsonl")
    blob.upload_from_filename(jsonl_path)
    print(f"[OK] Uploaded to GCS: gs://{BUCKET_NAME}/{prefix}/{filename}.jsonl")


# ============================================================
# SMART RESUME — gap detection + backfill for missed pipeline runs
# ============================================================

def _gcs_files_for_date(gcs_client: storage.Client, prefix: str, date_compact: str) -> list:
    """Return GCS blobs whose name contains the compact date string (YYYYMMDD)."""
    try:
        bucket = gcs_client.bucket(BUCKET_NAME)
        return [b for b in bucket.list_blobs(prefix=prefix) if date_compact in b.name]
    except Exception:
        return []


def get_missing_dates(gcs_client: storage.Client, gcs_prefix: str,
                      days_back: int = 7) -> list[str]:
    """Return YYYY-MM-DD strings for days in the last N days with no GCS file.
    Used to detect missed pipeline runs."""
    today   = datetime.now(tz=timezone.utc).date()
    missing = []
    for i in range(1, days_back + 1):
        d       = today - timedelta(days=i)
        compact = d.strftime("%Y%m%d")
        iso     = d.isoformat()
        if not _gcs_files_for_date(gcs_client, gcs_prefix, compact):
            missing.append(iso)
    if missing:
        print(f"[smart-resume] {gcs_prefix}: missing {len(missing)} days → {missing}")
    return missing


def backfill_gaps(gcs_client: storage.Client, missing_dates: list[str]) -> None:
    """For each missing date, re-fetch Billboard, News, and Reddit data and upload to GCS.
    YouTube/iTunes/Last.fm are current-only and cannot be backfilled."""
    if not missing_dates:
        return

    for date_str in missing_dates:
        compact = date_str.replace("-", "")
        print(f"\n[smart-resume] Backfilling {date_str}")

        # Billboard (supports historical date URLs)
        try:
            df = run_billboard_ingestion(date_str=date_str)
            if not df.empty:
                df["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()
                upload_to_gcs(df, f"billboard_backfill_{compact}", prefix="raw")
                print(f"[smart-resume] Billboard {date_str}: {len(df)} songs uploaded")
        except Exception as e:
            print(f"[smart-resume] Billboard backfill failed for {date_str}: {e}")

        # News — Guardian supports from-date/to-date
        try:
            news_df = run_news_ingestion(date_str=date_str)
            if not news_df.empty:
                upload_to_gcs(news_df, f"news_backfill_{compact}", prefix="raw/sentiment")
                print(f"[smart-resume] News {date_str}: {len(news_df)} articles uploaded")
            time.sleep(0.5)   # Guardian rate limit
        except Exception as e:
            print(f"[smart-resume] News backfill failed for {date_str}: {e}")

        # Reddit — new.json with epoch window gives posts from specific day
        try:
            reddit_posts = run_reddit_ingestion(date_str=date_str)
            if reddit_posts:
                upload_to_gcs(reddit_posts, f"reddit_backfill_{compact}", prefix="raw/sentiment")
                print(f"[smart-resume] Reddit {date_str}: {len(reddit_posts)} posts uploaded")
        except Exception as e:
            print(f"[smart-resume] Reddit backfill failed for {date_str}: {e}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("SOUNDPULSE UNIFIED PIPELINE - MODULE 10")
    print("=" * 60)
    print(f"Test Mode: {TEST_MODE}")
    print(f"Timestamp: {datetime.now()}")
    print(f"Filters: release>{RELEASE_AGE_MIN_DAYS}d, artist_newness>{ARTIST_NEWNESS_DAYS}d, cross_platform>={CROSS_PLATFORM_MIN}")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # Smart resume: detect + fill gaps from missed runs
    gcs_client = storage.Client()

    # Check if charts already ran today (prevents duplicate ingestion on re-trigger)
    charts_already_done = bool(
        _gcs_files_for_date(gcs_client, "raw/trending_tracks_", today_str.replace("-", ""))
    )
    if charts_already_done:
        print("[smart-resume] trending_tracks already in GCS for today — skipping chart fetch")

    # Find and backfill any days missed in the last 7 days (Billboard + News + Reddit)
    missing_dates = get_missing_dates(gcs_client, "raw/trending_tracks_", days_back=7)
    backfill_gaps(gcs_client, missing_dates)

# STEP 1: Fetch all trending sources
    if charts_already_done:
        print("\n[1-4/9] Charts already fetched today — loading from GCS skipped, using empty frames.")
        itunes_charts_df    = pd.DataFrame(columns=["title","artist","release_date","genre","source","country_code","chart_rank"])
        lastfm_df           = pd.DataFrame(columns=["title","artist","source","country","chart_rank","listeners","playcount"])
        billboard_df        = pd.DataFrame(columns=["title","artist","source","chart_date","rank"])
        youtube_trending_df = pd.DataFrame(columns=["title","channel","source","country_code"])
        itunes_charts_df["source"] = "itunes_chart"
        lastfm_df["source"]        = "lastfm_chart"
        billboard_df["source"]     = "billboard_chart"
        youtube_trending_df["source"] = "youtube_chart"
    else:
        print("\n[1/9] Fetching iTunes charts (10 countries)...")
        itunes_charts_df           = fetch_all_itunes_charts()
        itunes_charts_df["source"] = "itunes_chart"
        print(f"[OK] iTunes: {len(itunes_charts_df)} songs")

        print("\n[2/9] Fetching Last.fm charts (10 countries + global)...")
        lastfm_df           = run_lastfm_ingestion()
        lastfm_df["source"] = "lastfm_chart"
        print(f"[OK] Last.fm: {len(lastfm_df)} songs")

        print("\n[3/9] Fetching Billboard charts...")
        billboard_df           = run_billboard_ingestion()
        billboard_df["source"] = "billboard_chart"
        print(f"[OK] Billboard: {len(billboard_df)} songs")

        print("\n[4/9] Fetching YouTube trending music (10 countries)...")
        youtube_trending_df           = run_youtube_ingestion()
        youtube_trending_df["source"] = "youtube_chart"
        print(f"[OK] YouTube: {len(youtube_trending_df)} videos")

    # STEP 2: Combine into master list
    print("\n[6/9] Building master trending list...")

    itunes_cols = itunes_charts_df[["title", "artist", "release_date", "genre", "source", "country_code", "chart_rank"]].copy()
    itunes_cols.rename(columns={"country_code": "country"}, inplace=True)
    itunes_cols["listeners"]  = None
    itunes_cols["playcount"]  = None
    itunes_cols["chart_date"] = None

    lastfm_cols = lastfm_df[["title", "artist", "source", "country", "chart_rank", "listeners", "playcount"]].copy()
    lastfm_cols["release_date"] = None
    lastfm_cols["genre"]        = None
    lastfm_cols["chart_date"]   = None

    billboard_cols = billboard_df[["title", "artist", "source", "chart_date", "rank"]].copy()
    billboard_cols.rename(columns={"rank": "chart_rank"}, inplace=True)
    billboard_cols["release_date"] = None
    billboard_cols["genre"]        = None
    billboard_cols["country"]      = "US"
    billboard_cols["listeners"]    = None
    billboard_cols["playcount"]    = None

    youtube_cols = youtube_trending_df[["title", "source", "country_code"]].copy()
    youtube_cols.rename(columns={"country_code": "country"}, inplace=True)
    youtube_cols["artist"]       = youtube_trending_df["channel"]
    youtube_cols["release_date"] = None
    youtube_cols["genre"]        = None
    youtube_cols["chart_date"]   = None
    youtube_cols["chart_rank"]   = None
    youtube_cols["listeners"]    = None
    youtube_cols["playcount"]    = None

    master_df = pd.concat([itunes_cols, lastfm_cols, billboard_cols, youtube_cols], ignore_index=True)
    print(f"[OK] Master list before filters: {len(master_df)} rows")

    # STEP 3: Apply filters
    print("\n[7/9] Applying filters...")
    master_df     = apply_release_date_filter(master_df, date_col="release_date")
    master_df     = apply_artist_newness_filter(master_df)
    master_df     = apply_cross_platform_filter(master_df)
    unique_tracks = master_df.drop_duplicates(subset=["title", "artist"]).copy()
    print(f"[OK] Unique tracks after filters: {len(unique_tracks)}")

    if TEST_MODE:
        unique_tracks = unique_tracks.head(30).copy()
        print(f"[TEST MODE] Processing first 30 tracks only")

    # STEP 4: Get iTunes preview URLs
    print("\n[7/9] Fetching iTunes preview URLs...")
    unique_artists        = unique_tracks["artist"].unique()[:100]
    itunes_search_results = []
    for artist in unique_artists:
        tracks = search_itunes_tracks(artist, limit=15)
        itunes_search_results.extend(tracks)
    itunes_search_df = pd.DataFrame(itunes_search_results).drop_duplicates(subset=["itunes_track_id"]) if itunes_search_results else pd.DataFrame()
    print(f"[OK] iTunes search catalog: {len(itunes_search_df)} tracks")

    # STEP 5: 8-layer matching
    print("\n[9/9] Matching trending songs to iTunes previews...")
    if itunes_search_df.empty:
        print("[WARN] No iTunes search results — skipping matching")
        matched_df   = unique_tracks.copy()
        unmatched_df = pd.DataFrame()
    else:
        matched_df, unmatched_df = match_to_itunes(unique_tracks, itunes_search_df)

    save_diagnostic_json(matched_df, unmatched_df if not unmatched_df.empty else pd.DataFrame())

    # STEP 6: Spotify metadata enrichment
    print("\nEnriching with Spotify metadata...")
    matched_df = enrich_with_spotify_metadata(matched_df)
    matched_df["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()

    # STEP 7: Librosa audio features
    print("\nExtracting Librosa audio features...")
    enriched_df = enrich_with_librosa_features(matched_df)
    print(f"[OK] Enriched tracks: {len(enriched_df)}")

# STEP 8: Upload trending tracks to GCS
    print("\nUploading trending tracks to GCS...")
    upload_to_gcs(enriched_df, f"trending_tracks_{timestamp}", prefix="raw")

    # STEP 9: Upload YouTube trending to GCS
    print("\nUploading YouTube trending to GCS...")
    upload_to_gcs(youtube_trending_df, f"youtube_{timestamp}", prefix="raw/trending")

    # STEP 10: Fetch and upload sentiment sources
    print("\nFetching sentiment sources...")

    print("  Fetching Reddit posts...")
    try:
        reddit_posts = run_reddit_ingestion()
        upload_to_gcs(reddit_posts, f"reddit_{timestamp}", prefix="raw/sentiment")
        print(f"  [OK] Reddit: {len(reddit_posts)} posts")
    except Exception as e:
        print(f"  [WARN] Reddit failed: {e}")

    print("  Fetching news articles...")
    try:
        news_df = run_news_ingestion()
        upload_to_gcs(news_df, f"news_{timestamp}", prefix="raw/sentiment")
        print(f"  [OK] News: {len(news_df)} articles")
    except Exception as e:
        print(f"  [WARN] News failed: {e}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Trending tracks processed: {len(enriched_df)}")
    total = len(matched_df) + len(unmatched_df)
    print(f"Match rate: {len(matched_df)/total*100:.1f}%" if total > 0 else "Match rate: N/A")
    print(f"Diagnostic file: {DIAGNOSTIC_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()