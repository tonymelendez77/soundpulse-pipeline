import requests
import pandas as pd
from loguru import logger
from datetime import datetime, timezone
import os
import time
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_URL = "https://ws.audioscrobbler.com/2.0/"

COUNTRIES = {
    "united states":  "usa",
    "united kingdom": "europe",
    "germany":        "europe",
    "france":         "europe",
    "spain":          "europe",
    "mexico":         "latin_america",
    "argentina":      "latin_america",
    "brazil":         "latin_america",
    "colombia":       "latin_america",
}


def fetch_top_tracks_by_country(country: str, market: str, limit: int = 100) -> list[dict]:
    if not LASTFM_API_KEY:
        logger.warning("LASTFM_API_KEY not set - skipping Last.fm")
        return []

    params = {
        "method":  "geo.gettoptracks",
        "country": country,
        "api_key": LASTFM_API_KEY,
        "format":  "json",
        "limit":   limit,
    }

    try:
        response = requests.get(LASTFM_URL, params=params, timeout=10)
        response.raise_for_status()
        tracks_data = response.json().get("tracks", {}).get("track", [])

        tracks = []
        for idx, track in enumerate(tracks_data):
            tracks.append({
                "title":        track.get("name"),
                "artist":       track.get("artist", {}).get("name"),
                "listeners":    int(track.get("listeners", 0)),
                "playcount":    int(track.get("playcount", 0) or 0),
                "lastfm_url":   track.get("url"),
                "chart_rank":   idx + 1,
                "country":      country,
                "market":       market,
                "source":       "lastfm_chart",
                "ingested_at":  datetime.now(tz=timezone.utc).isoformat(),
            })

        logger.info(f"Last.fm {country}: {len(tracks)} tracks")
        return tracks

    except Exception as e:
        logger.error(f"Last.fm failed for {country}: {e}")
        return []


def fetch_global_top_tracks(limit: int = 100) -> list[dict]:
    if not LASTFM_API_KEY:
        logger.warning("LASTFM_API_KEY not set - skipping Last.fm global")
        return []

    params = {
        "method":  "chart.gettoptracks",
        "api_key": LASTFM_API_KEY,
        "format":  "json",
        "limit":   limit,
    }

    try:
        response = requests.get(LASTFM_URL, params=params, timeout=10)
        response.raise_for_status()
        tracks_data = response.json().get("tracks", {}).get("track", [])

        tracks = []
        for idx, track in enumerate(tracks_data):
            tracks.append({
                "title":       track.get("name"),
                "artist":      track.get("artist", {}).get("name"),
                "listeners":   int(track.get("listeners", 0)),
                "playcount":   int(track.get("playcount", 0) or 0),
                "lastfm_url":  track.get("url"),
                "chart_rank":  idx + 1,
                "country":     "global",
                "market":      "global",
                "source":      "lastfm_global",
                "ingested_at": datetime.now(tz=timezone.utc).isoformat(),
            })

        logger.info(f"Last.fm global: {len(tracks)} tracks")
        return tracks

    except Exception as e:
        logger.error(f"Last.fm global chart failed: {e}")
        return []


def run_lastfm_ingestion() -> pd.DataFrame:
    logger.info("Starting Last.fm ingestion")
    all_tracks = []

    # Global chart
    global_tracks = fetch_global_top_tracks(limit=100)
    all_tracks.extend(global_tracks)

    # Per country
    for country, market in COUNTRIES.items():
        tracks = fetch_top_tracks_by_country(country, market, limit=100)
        all_tracks.extend(tracks)
        time.sleep(0.5)

    df = pd.DataFrame(all_tracks)
    if not df.empty:
        logger.info(f"Last.fm ingestion complete. Total rows: {len(df)}")
    else:
        logger.warning("No tracks fetched from Last.fm")
    return df


def save_to_local(df: pd.DataFrame) -> str:
    output_dir = "data/raw/lastfm"
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    filename = f"{output_dir}/lastfm_{today}.json"
    df.to_json(filename, orient="records", indent=2)
    logger.info(f"Saved {len(df)} tracks to {filename}")
    return filename


if __name__ == "__main__":
    df = run_lastfm_ingestion()
    save_to_local(df)
    print(f"\nFetched {len(df)} tracks")
    if not df.empty:
        print(df[["country", "chart_rank", "artist", "title"]].head(10))