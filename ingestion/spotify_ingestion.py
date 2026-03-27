import os
import base64
import requests
import pandas as pd
import kagglehub
from datetime import datetime, timezone
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_TOKEN_URL     = "https://accounts.spotify.com/api/token"
SPOTIFY_API_URL       = "https://api.spotify.com/v1"

KAGGLE_DATASET = "maharshipandya/-spotify-tracks-dataset"

SPOTIFY_TOP_PLAYLISTS = {
    "37i9dQZEVXbLRQDuF5jeBp": "usa",
    "37i9dQZEVXbO3qyFxbkOE1": "latin_america",
    "37i9dQZEVXbMMy2roB9myp": "latin_america",
    "37i9dQZEVXbMDoHDwVN2tF": "latin_america",
    "37i9dQZEVXbIaRFqZjlkYy": "central_america",
    "37i9dQZEVXbNFJfN1Vw8d9": "europe",
    "37i9dQZEVXbJiZcmkrIHGU": "europe",
    "37i9dQZEVXbIPWwFssbupI": "europe",
}

CHART_SEARCHES = [
    {"query": "Bad Bunny",  "market": "usa"},
    {"query": "Drake",      "market": "usa"},
    {"query": "Bad Bunny",  "market": "latin_america"},
    {"query": "J Balvin",   "market": "latin_america"},
    {"query": "Shakira",    "market": "central_america"},
    {"query": "Ed Sheeran", "market": "europe"},
    {"query": "Dua Lipa",   "market": "europe"},
]

AUDIO_FEATURE_COLS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "liveness",
    "loudness", "speechiness", "key", "mode", "time_signature",
]


def get_access_token() -> str:
    credentials = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    encoded     = base64.b64encode(credentials.encode()).decode()
    headers     = {"Authorization": f"Basic {encoded}"}
    data        = {"grant_type": "client_credentials"}
    response    = requests.post(SPOTIFY_TOKEN_URL, headers=headers, data=data, timeout=10)
    response.raise_for_status()
    token = response.json()["access_token"]
    logger.info("Spotify access token obtained")
    return token


def fetch_playlist_tracks(token: str, playlist_id: str, market: str) -> list[dict]:
    headers  = {"Authorization": f"Bearer {token}"}
    url      = f"{SPOTIFY_API_URL}/playlists/{playlist_id}/tracks"
    params   = {
        "limit":  50,
        "fields": "items(track(id,name,artists,album,popularity,duration_ms,explicit))"
    }
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    items    = response.json().get("items", [])
    tracks   = []

    for item in items:
        track = item.get("track")
        if not track or not track.get("id"):
            continue
        tracks.append({
            "track_id":    track["id"],
            "playlist_id": playlist_id,
            "market":      market,
            "source":      "playlist",
            "title":       track["name"],
            "artist":      track["artists"][0]["name"] if track["artists"] else None,
            "album":       track["album"]["name"] if track.get("album") else None,
            "popularity":  track.get("popularity"),
            "duration_ms": track.get("duration_ms"),
            "explicit":    track.get("explicit"),
        })

    logger.info(f"Fetched {len(tracks)} tracks from playlist {playlist_id}")
    return tracks


def fetch_top_tracks_by_search(token: str, query: str, market: str) -> list[dict]:
    headers  = {"Authorization": f"Bearer {token}"}
    url      = f"{SPOTIFY_API_URL}/search"
    params   = {
        "q":     query,
        "type":  "track",
        "limit": 10,
    }
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    items    = response.json().get("tracks", {}).get("items", [])
    tracks   = []

    for track in items:
        if not track or not track.get("id"):
            continue
        tracks.append({
            "track_id":    track["id"],
            "playlist_id": "search",
            "market":      market,
            "source":      "search",
            "title":       track["name"],
            "artist":      track["artists"][0]["name"] if track["artists"] else None,
            "album":       track["album"]["name"] if track.get("album") else None,
            "popularity":  track.get("popularity"),
            "duration_ms": track.get("duration_ms"),
            "explicit":    track.get("explicit"),
        })

    logger.info(f"Fetched {len(tracks)} tracks via search: {query}")
    return tracks


def fetch_audio_features_from_api(token: str, track_ids: list[str]) -> dict:
    """
    Attempts to fetch audio features from Spotify API.
    Returns empty dict if access is denied (403) — triggers Kaggle fallback.
    """
    headers      = {"Authorization": f"Bearer {token}"}
    features_map = {}

    for i in range(0, len(track_ids), 100):
        batch    = track_ids[i:i + 100]
        params   = {"ids": ",".join(batch)}
        url      = f"{SPOTIFY_API_URL}/audio-features"
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 403:
            logger.warning("Audio features API denied (403) — will use Kaggle fallback")
            return {}

        response.raise_for_status()

        for f in response.json().get("audio_features") or []:
            if f and f.get("id"):
                features_map[f["id"]] = {col: f.get(col) for col in AUDIO_FEATURE_COLS}

    return features_map


def fetch_audio_features_from_kaggle(track_ids: list[str]) -> dict:
    """
    Downloads the Spotify tracks dataset from Kaggle and looks up
    audio features by track_id for all tracks we fetched.
    Uses local cache after first download — never downloads twice.
    """
    logger.info("Loading audio features from Kaggle dataset")

    try:
        path     = kagglehub.dataset_download(KAGGLE_DATASET)
        csv_file = os.path.join(path, "dataset.csv")
        kaggle_df = pd.read_csv(csv_file)

        kaggle_df = kaggle_df.rename(columns={"track_id": "track_id"})

        if "track_id" not in kaggle_df.columns:
            logger.error("Kaggle dataset has no track_id column")
            return {}

        matched = kaggle_df[kaggle_df["track_id"].isin(track_ids)]
        logger.info(f"Kaggle matched {len(matched)} of {len(track_ids)} tracks")

        features_map = {}
        for _, row in matched.iterrows():
            features_map[row["track_id"]] = {
                col: row.get(col) for col in AUDIO_FEATURE_COLS
                if col in kaggle_df.columns
            }

        return features_map

    except Exception as e:
        logger.error(f"Kaggle fallback failed: {e}")
        return {}


def fetch_all_tracks(token: str) -> list[dict]:
    all_tracks     = []
    use_search_for = set()

    for playlist_id, market in SPOTIFY_TOP_PLAYLISTS.items():
        try:
            tracks = fetch_playlist_tracks(token, playlist_id, market)
            all_tracks.extend(tracks)
        except Exception as e:
            if "403" in str(e):
                logger.warning(f"Playlist {playlist_id} denied — fallback for {market}")
                use_search_for.add(market)
            else:
                logger.error(f"Playlist {playlist_id} failed: {e}")

    if use_search_for:
        logger.info(f"Using search fallback for: {use_search_for}")
        for item in CHART_SEARCHES:
            if item["market"] in use_search_for:
                try:
                    tracks = fetch_top_tracks_by_search(
                        token, item["query"], item["market"]
                    )
                    all_tracks.extend(tracks)
                except Exception as e:
                    logger.error(f"Search failed for '{item['query']}': {e}")

    return all_tracks


def enrich_with_audio_features(df: pd.DataFrame, token: str) -> pd.DataFrame:
    """
    Enriches the tracks DataFrame with audio features.
    Priority order:
    1. Spotify API (best — live, accurate)
    2. Kaggle dataset (fallback — historical, may not have all tracks)
    3. No features (last resort — saves tracks without audio data)
    """
    track_ids    = df["track_id"].unique().tolist()
    features_map = fetch_audio_features_from_api(token, track_ids)

    if not features_map:
        logger.info("Trying Kaggle fallback for audio features")
        features_map = fetch_audio_features_from_kaggle(track_ids)

    if features_map:
        df["audio_features"] = df["track_id"].map(features_map)
        features_df          = pd.json_normalize(df["audio_features"].dropna())

        if not features_df.empty:
            df = df.drop(columns=["audio_features"])
            df = df.merge(
                pd.concat([
                    df[["track_id"]].reset_index(drop=True),
                    features_df.reset_index(drop=True)
                ], axis=1),
                on="track_id",
                how="left"
            )
            logger.info("Audio features merged successfully")
        else:
            df = df.drop(columns=["audio_features"])
            logger.warning("Audio features empty after normalization")
    else:
        logger.warning("No audio features available from any source")

    return df


def run_spotify_ingestion() -> pd.DataFrame:
    logger.info("Starting Spotify ingestion run")
    token      = get_access_token()
    all_tracks = fetch_all_tracks(token)
    df         = pd.DataFrame(all_tracks)

    if df.empty:
        logger.warning("No tracks fetched — DataFrame is empty")
        return df

    logger.info(f"Enriching {len(df)} tracks with audio features")
    df = enrich_with_audio_features(df, token)

    df["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()
    logger.info(f"Spotify ingestion complete. Total tracks: {len(df)}")
    return df


def save_to_local(df: pd.DataFrame) -> str:
    output_dir = "data/raw/spotify"
    os.makedirs(output_dir, exist_ok=True)
    date_str   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    filename   = f"{output_dir}/spotify_{date_str}.json"
    df.to_json(filename, orient="records", indent=2)
    logger.info(f"Saved {len(df)} tracks to {filename}")
    return filename


if __name__ == "__main__":
    df = run_spotify_ingestion()
    save_to_local(df)
    from upload_helper import upload_to_gcs
    upload_to_gcs(df.to_dict('records'), 'spotify')

import sys
sys.path.append('..')
from upload_helper import upload_to_gcs
upload_to_gcs(df.to_dict('records'), 'spotify')