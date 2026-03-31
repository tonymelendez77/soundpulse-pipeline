import requests
import pandas as pd
from loguru import logger
from datetime import datetime, timezone
import os
import time


ITUNES_COUNTRIES = {
    "us": "usa",
    "mx": "latin_america",
    "co": "latin_america",
    "ar": "latin_america",
    "gt": "central_america",
    "sv": "central_america",
    "gb": "europe",
    "de": "europe",
    "fr": "europe",
    "es": "europe",
}

RSS_URL = "https://rss.applemarketingtools.com/api/v2/{country}/music/most-played/100/songs.json"


def extract_primary_artist(artist_string):
    if not artist_string or pd.isna(artist_string):
        return None
    separators = ['&', 'Featuring', 'featuring', ',', 'X', 'x', ' and ', ' And ']
    result = str(artist_string).strip()
    for sep in separators:
        if sep in result:
            result = result.split(sep)[0].strip()
    return result


def fetch_itunes_chart(country_code: str, market: str) -> list[dict]:
    url = RSS_URL.format(country=country_code.lower())
    try:
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=15)
                break
            except requests.exceptions.Timeout:
                if attempt < 2:
                    logger.warning(f"iTunes chart timeout for {country_code}, retry {attempt+1}/3")
                    time.sleep(3)
                else:
                    raise
        response.raise_for_status()
        results = response.json().get("feed", {}).get("results", [])
        tracks = []
        for idx, item in enumerate(results):
            tracks.append({
                "itunes_track_id":  item.get("id"),
                "title":            item.get("name"),
                "artist":           item.get("artistName"),
                "album":            item.get("collectionName", ""),
                "genre":            item.get("genres", [{}])[0].get("name", "") if item.get("genres") else "",
                "release_date":     item.get("releaseDate", ""),
                "itunes_url":       item.get("url", ""),
                "chart_rank":       idx + 1,
                "country_code":     country_code.upper(),
                "market":           market,
                "source":           "itunes_chart",
                "preview_url":      None,
                "duration_ms":      None,
                "ingested_at":      datetime.now(tz=timezone.utc).isoformat(),
            })
        logger.info(f"iTunes chart {country_code.upper()}: {len(tracks)} songs")
        return tracks
    except Exception as e:
        logger.error(f"iTunes chart failed for {country_code}: {e}")
        return []


def fetch_all_itunes_charts() -> pd.DataFrame:
    logger.info("Starting iTunes chart ingestion across all markets")
    all_tracks = []

    for country_code, market in ITUNES_COUNTRIES.items():
        tracks = fetch_itunes_chart(country_code, market)
        all_tracks.extend(tracks)
        time.sleep(0.3)

    df = pd.DataFrame(all_tracks)
    if not df.empty:
        df = df.drop_duplicates(subset=["itunes_track_id", "country_code"], keep="first")
        logger.info(f"iTunes ingestion complete. Total rows: {len(df)} across {df['country_code'].nunique()} countries")
    return df


def search_itunes_tracks(artist_name, limit=10):
    url = "https://itunes.apple.com/search"
    params = {
        "term":   artist_name,
        "media":  "music",
        "entity": "song",
        "limit":  limit
    }
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            time.sleep(0.5)
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 429:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Rate limited for {artist_name}, waiting {wait_time}s")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            tracks = []
            for item in response.json().get("results", []):
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
            if tracks:
                logger.info(f"iTunes: Found {len(tracks)} tracks with previews for {artist_name}")
            return tracks
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                continue
            logger.error(f"iTunes search failed for {artist_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"iTunes search failed for {artist_name}: {e}")
            return []

    logger.warning(f"Max retries reached for {artist_name}")
    return []


def run_itunes_ingestion(track_list=None):
    if track_list is not None and not track_list.empty:
        logger.info("Running iTunes search mode for preview URL enrichment")
        track_list["primary_artist"] = track_list["artist"].apply(extract_primary_artist)
        unique_artists = track_list["primary_artist"].value_counts().head(100).index.tolist()
        all_tracks = []
        for idx, artist in enumerate(unique_artists):
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx+1}/{len(unique_artists)} artists searched")
            tracks = search_itunes_tracks(artist, limit=15)
            all_tracks.extend(tracks)
        df = pd.DataFrame(all_tracks)
        if not df.empty:
            df = df.drop_duplicates(subset=["itunes_track_id"], keep="first")
        return df
    else:
        return fetch_all_itunes_charts()


def save_to_local(df: pd.DataFrame) -> str:
    output_dir = "data/raw/itunes"
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    filename = f"{output_dir}/itunes_{today}.json"
    df.to_json(filename, orient="records", indent=2)
    logger.info(f"Saved {len(df)} iTunes tracks to {filename}")
    return filename


if __name__ == "__main__":
    df = fetch_all_itunes_charts()
    save_to_local(df)
    print(f"\nFetched {len(df)} tracks across {df['country_code'].nunique()} countries")
    print(df[["country_code", "chart_rank", "artist", "title"]].head(10))