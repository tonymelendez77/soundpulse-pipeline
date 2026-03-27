import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

YOUTUBE_API_KEY   = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_URL       = "https://www.googleapis.com/youtube/v3/videos"

COUNTRY_CODES = {
    "US": "usa",
    "MX": "latin_america",
    "CO": "latin_america",
    "AR": "latin_america",
    "GT": "central_america",
    "SV": "central_america",
    "GB": "europe",
    "DE": "europe",
    "FR": "europe",
    "ES": "europe",
}

MUSIC_CATEGORY_ID = "10"


def fetch_trending_music(country_code: str, market: str) -> list[dict]:
    if not YOUTUBE_API_KEY:
        logger.warning("YOUTUBE_API_KEY not set — skipping YouTube")
        return []

    params = {
        "part":            "snippet,statistics",
        "chart":           "mostPopular",
        "regionCode":      country_code,
        "videoCategoryId": MUSIC_CATEGORY_ID,
        "maxResults":      50,
        "key":             YOUTUBE_API_KEY,
    }

    response = requests.get(YOUTUBE_URL, params=params, timeout=10)
    response.raise_for_status()
    items    = response.json().get("items", [])
    videos   = []

    for item in items:
        snippet    = item.get("snippet", {})
        statistics = item.get("statistics", {})

        videos.append({
            "video_id":      item.get("id"),
            "country_code":  country_code,
            "market":        market,
            "title":         snippet.get("title"),
            "channel":       snippet.get("channelTitle"),
            "published_at":  snippet.get("publishedAt"),
            "view_count":    int(statistics.get("viewCount", 0)),
            "like_count":    int(statistics.get("likeCount", 0)),
            "comment_count": int(statistics.get("commentCount", 0)),
            "tags":          snippet.get("tags", []),
        })

    logger.info(f"Fetched {len(videos)} trending videos for {country_code}")
    return videos


def run_youtube_ingestion() -> pd.DataFrame:
    logger.info("Starting YouTube ingestion run")
    all_videos = []

    for country_code, market in COUNTRY_CODES.items():
        try:
            videos = fetch_trending_music(country_code, market)
            all_videos.extend(videos)
        except Exception as e:
            logger.error(f"YouTube failed for {country_code}: {e}")

    df = pd.DataFrame(all_videos)

    if df.empty:
        logger.warning("No videos fetched")
        return df

    df["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()
    logger.info(f"YouTube ingestion complete. Total videos: {len(df)}")
    return df


def save_to_local(df: pd.DataFrame) -> str:
    output_dir = "data/raw/youtube"
    os.makedirs(output_dir, exist_ok=True)
    date_str   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    filename   = f"{output_dir}/youtube_{date_str}.json"
    df.to_json(filename, orient="records", indent=2)
    logger.info(f"Saved {len(df)} videos to {filename}")
    return filename


if __name__ == "__main__":
    df = run_youtube_ingestion()
    save_to_local(df)
    from upload_helper import upload_to_gcs
    upload_to_gcs(df.to_dict('records'), 'youtube')