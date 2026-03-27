import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY")
GUARDIAN_KEY   = os.getenv("GUARDIAN_API_KEY")
MEDIASTACK_KEY = os.getenv("MEDIASTACK_API_KEY")

NEWSAPI_URL    = "https://newsapi.org/v2/top-headlines"
GUARDIAN_URL   = "https://content.guardianapis.com/search"
MEDIASTACK_URL = "http://api.mediastack.com/v1/news"

NEWSAPI_CATEGORIES = [
    "general",
    "health",
    "science",
    "technology",
    "business",
    "entertainment",
    "sports",
]

GUARDIAN_QUERIES = [
    {"query": "war conflict violence",        "topic": "conflict"},
    {"query": "mental health anxiety",        "topic": "mental_health"},
    {"query": "election politics government", "topic": "politics"},
    {"query": "economic crisis recession",    "topic": "economy"},
    {"query": "natural disaster climate",     "topic": "disaster"},
]

MEDIASTACK_QUERIES = [
    {"keywords": "guerra conflicto",     "languages": "es", "topic": "conflict"},
    {"keywords": "crisis economica",     "languages": "es", "topic": "economy"},
    {"keywords": "elecciones gobierno",  "languages": "es", "topic": "politics"},
    {"keywords": "desastre natural",     "languages": "es", "topic": "disaster"},
    {"keywords": "salud mental ansiedad","languages": "es", "topic": "mental_health"},
]

TOPIC_KEYWORDS = {
    "conflict":      ["war", "attack", "shooting", "bomb", "conflict",
                      "violence", "terror", "guerra", "conflicto"],
    "economy":       ["inflation", "recession", "unemployment", "economy",
                      "market", "crisis", "economica"],
    "politics":      ["election", "president", "congress", "government",
                      "vote", "senate", "elecciones", "gobierno"],
    "disaster":      ["earthquake", "hurricane", "flood", "wildfire",
                      "tornado", "tsunami", "desastre"],
    "mental_health": ["anxiety", "depression", "suicide", "mental health",
                      "stress", "grief", "loneliness"],
    "celebration":   ["celebration", "festival", "holiday", "victory",
                      "award", "wedding", "fiesta"],
    "health":        ["covid", "pandemic", "vaccine", "cancer",
                      "disease", "outbreak", "hospital"],
    "technology":    ["ai", "artificial intelligence", "tech",
                      "software", "cyber", "robot"],
    "sports":        ["championship", "world cup", "olympics",
                      "tournament", "final", "gold medal"],
    "crime":         ["murder", "arrest", "prison", "drug",
                      "trafficking", "corruption", "crimen"],
    "travel":        ["travel", "tourism", "flight", "visa",
                      "border", "airport"],
}


def tag_topic(title: str) -> str:
    if not title:
        return "general"
    title_lower = title.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return topic
    return "general"


def fetch_newsapi(category: str) -> list[dict]:
    if not NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY not set — skipping NewsAPI")
        return []

    params = {
        "category": category,
        "language": "en",
        "pageSize": 20,
        "apiKey":   NEWSAPI_KEY,
    }

    response = requests.get(NEWSAPI_URL, params=params, timeout=10)
    response.raise_for_status()
    articles = response.json().get("articles", [])

    return [
        {
            "source":       "newsapi",
            "category":     category,
            "topic":        tag_topic(a.get("title")),
            "title":        a.get("title"),
            "description":  a.get("description"),
            "url":          a.get("url"),
            "published_at": a.get("publishedAt"),
            "source_name":  a.get("source", {}).get("name"),
        }
        for a in articles
        if a.get("title")
    ]


def fetch_guardian(query: str, topic: str) -> list[dict]:
    if not GUARDIAN_KEY:
        logger.warning("GUARDIAN_API_KEY not set — skipping Guardian")
        return []

    params = {
        "q":          query,
        "api-key":    GUARDIAN_KEY,
        "page-size":  20,
        "show-fields":"headline,bodyText",
        "order-by":   "newest",
    }

    response = requests.get(GUARDIAN_URL, params=params, timeout=10)
    response.raise_for_status()
    results  = response.json().get("response", {}).get("results", [])

    return [
        {
            "source":       "guardian",
            "category":     "general",
            "topic":        topic,
            "title":        r.get("webTitle"),
            "description":  r.get("fields", {}).get("bodyText", "")[:200],
            "url":          r.get("webUrl"),
            "published_at": r.get("webPublicationDate"),
            "source_name":  "The Guardian",
        }
        for r in results
        if r.get("webTitle")
    ]


def fetch_mediastack(keywords: str, languages: str, topic: str) -> list[dict]:
    if not MEDIASTACK_KEY:
        logger.warning("MEDIASTACK_API_KEY not set — skipping MediaStack")
        return []

    params = {
        "access_key": MEDIASTACK_KEY,
        "keywords":   keywords,
        "languages":  languages,
        "limit":      25,
        "sort":       "published_desc",
    }

    response = requests.get(MEDIASTACK_URL, params=params, timeout=10)
    response.raise_for_status()
    articles = response.json().get("data", [])

    return [
        {
            "source":       "mediastack",
            "category":     "general",
            "topic":        topic,
            "title":        a.get("title"),
            "description":  a.get("description"),
            "url":          a.get("url"),
            "published_at": a.get("published_at"),
            "source_name":  a.get("source"),
        }
        for a in articles
        if a.get("title")
    ]


def run_news_ingestion() -> pd.DataFrame:
    logger.info("Starting news ingestion run")
    all_articles = []

    for category in NEWSAPI_CATEGORIES:
        try:
            articles = fetch_newsapi(category)
            all_articles.extend(articles)
            logger.info(f"NewsAPI '{category}' → {len(articles)} articles")
        except Exception as e:
            logger.error(f"NewsAPI failed for '{category}': {e}")

    for item in GUARDIAN_QUERIES:
        try:
            articles = fetch_guardian(item["query"], item["topic"])
            all_articles.extend(articles)
            logger.info(f"Guardian '{item['topic']}' → {len(articles)} articles")
        except Exception as e:
            logger.error(f"Guardian failed for '{item['query']}': {e}")
        finally:
            time.sleep(1)

    for item in MEDIASTACK_QUERIES:
        try:
            articles = fetch_mediastack(item["keywords"], item["languages"], item["topic"])
            all_articles.extend(articles)
            logger.info(f"MediaStack '{item['topic']}' ({item['languages']}) → {len(articles)} articles")
        except Exception as e:
            logger.error(f"MediaStack failed for '{item['keywords']}': {e}")
        finally:
            time.sleep(1)

    df = pd.DataFrame(all_articles)

    if df.empty:
        logger.warning("No articles fetched")
        return df

    df = df.drop_duplicates(subset=["url"])
    df["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()
    logger.info(f"News ingestion complete. Total articles: {len(df)}")
    return df


def save_to_local(df: pd.DataFrame) -> str:
    output_dir = "data/raw/news"
    os.makedirs(output_dir, exist_ok=True)
    date_str   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    filename   = f"{output_dir}/news_{date_str}.json"
    df.to_json(filename, orient="records", indent=2)
    logger.info(f"Saved {len(df)} articles to {filename}")
    return filename


if __name__ == "__main__":
    df = run_news_ingestion()
    save_to_local(df)
    if not df.empty:
        print(df[["source", "topic", "title"]].head(10))
    else:
        print("No articles fetched")