import requests
import time
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

HEADERS = {"User-Agent": "soundpulse-bot/1.0"}

SUBREDDITS = [
    "worldnews",
    "news",
    "AskReddit",
    "offmychest",
    "anxiety",
    "depression",
    "happy",
    "UpliftingNews",
    "rant",
    "TrueOffMyChest",
]

POSTS_LIMIT = 50
REQUEST_DELAY = 2

def fetch_subreddit_posts(subreddit_name: str) -> list[dict]:
    url = f"https://www.reddit.com/r/{subreddit_name}/hot.json"
    params = {"limit": POSTS_LIMIT}

    logger.info(f"Fetching r/{subreddit_name}")

    response = requests.get(url, headers=HEADERS, params=params, timeout=10)
    response.raise_for_status()

    children = response.json()["data"]["children"]

    posts = []
    for child in children:
        post = child["data"]
        posts.append({
            "post_id":      post.get("id"),
            "subreddit":    subreddit_name,
            "title":        post.get("title"),
            "body":         post.get("selftext"),
            "score":        post.get("score"),
            "upvote_ratio": post.get("upvote_ratio"),
            "num_comments": post.get("num_comments"),
            "created_utc":  datetime.fromtimestamp(
                                post.get("created_utc", 0), tz=timezone.utc
                            ).isoformat(),
        })

    logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name}")
    return posts

def run_reddit_ingestion() -> list[dict]:
    logger.info("Starting Reddit ingestion")
    all_posts = []

    for subreddit_name in SUBREDDITS:
        try:
            posts = fetch_subreddit_posts(subreddit_name)
            all_posts.extend(posts)
        except Exception as e:
            logger.error(f"Failed r/{subreddit_name}: {e}")
        finally:
            time.sleep(REQUEST_DELAY)

    logger.info(f"Ingestion complete. Total posts: {len(all_posts)}")
    return all_posts


def save_to_local(posts: list[dict]) -> str:
    output_dir = "data/raw/reddit"
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    filename = f"{output_dir}/reddit_{date_str}.json"

    with open(filename, "w") as f:
        json.dump(posts, f, indent=2)

    logger.info(f"Saved {len(posts)} posts to {filename}")
    return filename


if __name__ == "__main__":
    posts = run_reddit_ingestion()
    save_to_local(posts)
    from upload_helper import upload_to_gcs
    upload_to_gcs(posts, 'reddit')