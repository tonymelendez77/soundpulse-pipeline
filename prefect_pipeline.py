from prefect import flow, task
from datetime import datetime
import subprocess
import os
import sys

PYTHON_EXE = sys.executable

@task(name="Ingest Reddit Data", retries=2, retry_delay_seconds=60)
def ingest_reddit():
    result = subprocess.run([PYTHON_EXE, "ingestion/reddit_ingestion.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Reddit ingestion failed: {result.stderr}")

@task(name="Ingest News Data", retries=2, retry_delay_seconds=60)
def ingest_news():
    result = subprocess.run([PYTHON_EXE, "ingestion/news_ingestion.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"News ingestion failed: {result.stderr}")

@task(name="Ingest Spotify Data", retries=2, retry_delay_seconds=60)
def ingest_spotify():
    result = subprocess.run([PYTHON_EXE, "ingestion/spotify_ingestion.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Spotify ingestion failed: {result.stderr}")

@task(name="Ingest YouTube Data", retries=2, retry_delay_seconds=60)
def ingest_youtube():
    result = subprocess.run([PYTHON_EXE, "ingestion/youtube_ingestion.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"YouTube ingestion failed: {result.stderr}")

@task(name="Ingest Billboard Data", retries=2, retry_delay_seconds=60)
def ingest_billboard():
    result = subprocess.run([PYTHON_EXE, "ingestion/billboard_ingestion.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Billboard ingestion failed: {result.stderr}")

@task(name="Upload to GCS", retries=3, retry_delay_seconds=30)
def upload_to_gcs():
    result = subprocess.run([PYTHON_EXE, "gcs_upload.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"GCS upload failed: {result.stderr}")

@task(name="Load to BigQuery", retries=3, retry_delay_seconds=30)
def load_to_bigquery():
    load_date = datetime.now().strftime('%Y-%m-%d')
    env = os.environ.copy()
    env['LOAD_DATE'] = load_date
    
    result = subprocess.run(
        [PYTHON_EXE, "bigquery_setup.py"],
        capture_output=True,
        text=True,
        env=env
    )
    if result.returncode != 0:
        raise Exception(f"BigQuery load failed: {result.stderr}")

@task(name="Run dbt", retries=2, retry_delay_seconds=60)
def run_dbt():
    result = subprocess.run(
        ["dbt", "run"],
        cwd="soundpulse_dbt",
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise Exception(f"dbt run failed: {result.stderr}")
    
    result = subprocess.run(
        ["dbt", "test"],
        cwd="soundpulse_dbt",
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise Exception(f"dbt test failed: {result.stderr}")

@flow(name="SoundPulse Daily Pipeline", log_prints=True)
def soundpulse_pipeline():
    reddit_future = ingest_reddit.submit()
    news_future = ingest_news.submit()
    spotify_future = ingest_spotify.submit()
    youtube_future = ingest_youtube.submit()
    billboard_future = ingest_billboard.submit()
    
    reddit_future.result()
    news_future.result()
    spotify_future.result()
    youtube_future.result()
    billboard_future.result()
    
    upload_to_gcs()
    load_to_bigquery()
    run_dbt()

if __name__ == "__main__":
    soundpulse_pipeline()