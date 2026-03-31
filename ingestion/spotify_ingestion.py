import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from loguru import logger
import time

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")


def get_access_token() -> str:
    url = "https://accounts.spotify.com/api/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type":    "client_credentials",
        "client_id":     SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET,
    }
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    token = response.json()["access_token"]
    logger.info("Spotify access token obtained")
    return token


def search_track_metadata(access_token: str, title: str, artist: str) -> dict:
    """Search Spotify for a specific song and return metadata."""
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    query = f"track:{title} artist:{artist}"
    params = {"q": query, "type": "track", "limit": 1}

    for attempt in range(3):
        try:
            time.sleep(0.2 * (attempt + 1))
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            items = response.json().get("tracks", {}).get("items", [])
            if not items:
                return {}
            track = items[0]
            return {
                "spotify_track_id": track.get("id"),
                "explicit":         track.get("explicit"),
                "duration_ms":      track.get("duration_ms"),
                "spotify_url":      track.get("external_urls", {}).get("spotify"),
            }
        except Exception as e:
            if attempt < 2:
                logger.warning(f"Spotify retry {attempt+1}/3 for {title} - {artist}: {e}")
                time.sleep(2)
            else:
                logger.error(f"Spotify search failed for {title} - {artist}: {e}")
                return {}
    return {}


def enrich_with_spotify_metadata(df: pd.DataFrame, max_workers: int = 5) -> pd.DataFrame:
    """
    Given a DataFrame with 'title' and 'artist' columns,
    enrich each row with Spotify metadata using parallel threads.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if df.empty:
        logger.warning("Empty DataFrame passed to Spotify enrichment")
        return df

    logger.info(f"Enriching {len(df)} tracks with Spotify metadata ({max_workers} parallel workers)")
    access_token = get_access_token()

    rows = list(df.iterrows())
    results = {}

    def fetch(item):
        idx, row = item
        return idx, search_track_metadata(access_token, row["title"], row["artist"])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, item): item for item in rows}
        completed = 0
        for future in as_completed(futures):
            idx, metadata = future.result()
            results[idx] = metadata
            completed += 1
            if completed % 50 == 0:
                logger.info(f"Spotify enrichment progress: {completed}/{len(df)}")

    spotify_data = [results[idx] for idx, _ in rows]
    spotify_df = pd.DataFrame(spotify_data, index=df.index)

    for col in ["spotify_track_id", "explicit", "duration_ms", "spotify_url"]:
        df[col] = spotify_df[col] if col in spotify_df.columns else None

    enriched = df["spotify_track_id"].notna().sum()
    logger.info(f"Spotify enrichment complete: {enriched}/{len(df)} tracks matched")
    return df

def run_spotify_ingestion():
    """Kept for backward compatibility with unified_pipeline.py"""
    logger.info("Spotify is now used for metadata enrichment only")
    return pd.DataFrame()


def save_to_local(df: pd.DataFrame) -> str:
    output_dir = "data/raw/spotify"
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    filename = f"{output_dir}/spotify_metadata_{today}.json"
    df.to_json(filename, orient="records", indent=2)
    logger.info(f"Saved {len(df)} tracks to {filename}")
    return filename


if __name__ == "__main__":
    # Test enrichment with 3 sample songs
    test_df = pd.DataFrame([
        {"title": "Choosin' Texas", "artist": "Ella Langley"},
        {"title": "Espresso",       "artist": "Sabrina Carpenter"},
        {"title": "FATHER",         "artist": "Kanye West"},
    ])
    token = get_access_token()
    for _, row in test_df.iterrows():
        result = search_track_metadata(token, row["title"], row["artist"])
        print(row["title"], "-", row["artist"], "->", result)