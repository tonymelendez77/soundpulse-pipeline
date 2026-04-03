import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from loguru import logger

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

CHARTS = [
    {"slug": "hot-100",                    "name": "Hot 100"},
    {"slug": "billboard-global-200",       "name": "Global 200"},
    {"slug": "latin",                      "name": "Latin"},
    {"slug": "latin-pop-airplay",          "name": "Latin Pop Airplay"},
    {"slug": "regional-mexican-airplay",   "name": "Regional Mexican Airplay"},
    {"slug": "tropical-airplay",           "name": "Tropical Airplay"},
    {"slug": "pop-songs",                  "name": "Pop Songs"},
    {"slug": "rhythmic-40",               "name": "Rhythmic 40"},
    {"slug": "dance-club-play",            "name": "Dance Club Play"},
    {"slug": "hot-latin-songs",            "name": "Hot Latin Songs"},
]

REQUEST_DELAY = 3


def fetch_chart(slug: str, chart_name: str, date_str: str | None = None) -> list[dict]:
    """Fetch a Billboard chart. Pass date_str (YYYY-MM-DD) to retrieve a historical week."""
    if date_str:
        url = f"https://www.billboard.com/charts/{slug}/{date_str}/"
    else:
        url = f"https://www.billboard.com/charts/{slug}/"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup    = BeautifulSoup(response.text, "html.parser")
    entries = soup.select("ul.o-chart-results-list-row")
    songs   = []
    chart_date = date_str or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    for rank, entry in enumerate(entries, start=1):
        title_el  = entry.select_one("h3#title-of-a-story")
        artist_el = entry.select_one("span.c-label.a-no-trucate")

        if not title_el:
            continue

        songs.append({
            "chart_slug":  slug,
            "chart_name":  chart_name,
            "rank":        rank,
            "title":       title_el.get_text(strip=True),
            "artist":      artist_el.get_text(strip=True) if artist_el else None,
            "chart_date":  chart_date,
        })

    logger.info(f"Fetched {len(songs)} songs from {chart_name}" + (f" [{date_str}]" if date_str else ""))
    return songs


def run_billboard_ingestion(date_str: str | None = None) -> pd.DataFrame:
    """Fetch all Billboard charts. Pass date_str (YYYY-MM-DD) to backfill a specific week."""
    label = f" [{date_str}]" if date_str else " [current]"
    logger.info(f"Starting Billboard ingestion{label}")
    all_songs = []

    for chart in CHARTS:
        try:
            songs = fetch_chart(chart["slug"], chart["name"], date_str=date_str)
            all_songs.extend(songs)
        except Exception as e:
            logger.error(f"Billboard failed for '{chart['name']}'{label}: {e}")
        finally:
            time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_songs)

    if df.empty:
        logger.warning("No songs fetched")
        return df

    df["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()
    logger.info(f"Billboard ingestion complete. Total entries: {len(df)}")
    return df


def save_to_local(df: pd.DataFrame) -> str:
    output_dir = "data/raw/billboard"
    os.makedirs(output_dir, exist_ok=True)
    date_str   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    filename   = f"{output_dir}/billboard_{date_str}.json"
    df.to_json(filename, orient="records", indent=2)
    logger.info(f"Saved {len(df)} chart entries to {filename}")
    return filename


if __name__ == "__main__":
    df = run_billboard_ingestion()
    save_to_local(df)
    from upload_helper import upload_to_gcs
    upload_to_gcs(df.to_dict('records'), 'billboard')