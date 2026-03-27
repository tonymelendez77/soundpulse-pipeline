from google.cloud import bigquery
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

client = bigquery.Client(project='soundpulse-production')
dataset_id = 'music_analytics'
bucket_name = 'soundpulse-raw-lake_ojmo'

# Reddit posts table
reddit_schema = [
    bigquery.SchemaField('post_id', 'STRING', mode='REQUIRED'),
    bigquery.SchemaField('subreddit', 'STRING'),
    bigquery.SchemaField('title', 'STRING'),
    bigquery.SchemaField('body', 'STRING'),
    bigquery.SchemaField('score', 'INTEGER'),
    bigquery.SchemaField('upvote_ratio', 'FLOAT'),
    bigquery.SchemaField('num_comments', 'INTEGER'),
    bigquery.SchemaField('created_utc', 'STRING'),
]

# News articles table
news_schema = [
    bigquery.SchemaField('source', 'STRING'),
    bigquery.SchemaField('category', 'STRING'),
    bigquery.SchemaField('topic', 'STRING'),
    bigquery.SchemaField('title', 'STRING'),
    bigquery.SchemaField('description', 'STRING'),
    bigquery.SchemaField('url', 'STRING', mode='REQUIRED'),
    bigquery.SchemaField('published_at', 'STRING'),
    bigquery.SchemaField('source_name', 'STRING'),
    bigquery.SchemaField('ingested_at', 'STRING'),
]

# Spotify tracks table
spotify_schema = [
    bigquery.SchemaField('track_id', 'STRING', mode='REQUIRED'),
    bigquery.SchemaField('playlist_id', 'STRING'),
    bigquery.SchemaField('market', 'STRING'),
    bigquery.SchemaField('source', 'STRING'),
    bigquery.SchemaField('title', 'STRING'),
    bigquery.SchemaField('artist', 'STRING'),
    bigquery.SchemaField('album', 'STRING'),
    bigquery.SchemaField('popularity', 'INTEGER'),
    bigquery.SchemaField('duration_ms', 'INTEGER'),
    bigquery.SchemaField('explicit', 'BOOLEAN'),
    bigquery.SchemaField('danceability', 'FLOAT'),
    bigquery.SchemaField('energy', 'FLOAT'),
    bigquery.SchemaField('valence', 'FLOAT'),
    bigquery.SchemaField('tempo', 'FLOAT'),
    bigquery.SchemaField('acousticness', 'FLOAT'),
    bigquery.SchemaField('instrumentalness', 'FLOAT'),
    bigquery.SchemaField('liveness', 'FLOAT'),
    bigquery.SchemaField('loudness', 'FLOAT'),
    bigquery.SchemaField('speechiness', 'FLOAT'),
    bigquery.SchemaField('key', 'FLOAT'),
    bigquery.SchemaField('mode', 'FLOAT'),
    bigquery.SchemaField('time_signature', 'FLOAT'),
    bigquery.SchemaField('ingested_at', 'STRING'),
]

# YouTube videos table
youtube_schema = [
    bigquery.SchemaField('video_id', 'STRING', mode='REQUIRED'),
    bigquery.SchemaField('country_code', 'STRING'),
    bigquery.SchemaField('market', 'STRING'),
    bigquery.SchemaField('title', 'STRING'),
    bigquery.SchemaField('channel', 'STRING'),
    bigquery.SchemaField('published_at', 'STRING'),
    bigquery.SchemaField('view_count', 'INTEGER'),
    bigquery.SchemaField('like_count', 'INTEGER'),
    bigquery.SchemaField('comment_count', 'INTEGER'),
    bigquery.SchemaField('tags', 'STRING', mode='REPEATED'),
    bigquery.SchemaField('ingested_at', 'STRING'),
]

# Billboard chart table
billboard_schema = [
    bigquery.SchemaField('chart_slug', 'STRING'),
    bigquery.SchemaField('chart_name', 'STRING'),
    bigquery.SchemaField('rank', 'INTEGER', mode='REQUIRED'),
    bigquery.SchemaField('title', 'STRING'),
    bigquery.SchemaField('artist', 'STRING'),
    bigquery.SchemaField('chart_date', 'STRING', mode='REQUIRED'),
    bigquery.SchemaField('ingested_at', 'STRING'),
]

def create_table(table_name, schema):
    """Create a BigQuery table"""
    table_id = f"{client.project}.{dataset_id}.{table_name}"
    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table, exists_ok=True)
    print(f"[OK] Created table {table_name}")

def load_from_gcs(table_name, gcs_uri):
    """Load JSON data from GCS into BigQuery table"""
    table_id = f"{client.project}.{dataset_id}.{table_name}"
    
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ignore_unknown_values=True,
    )
    
    load_job = client.load_table_from_uri(gcs_uri, table_id, job_config=job_config)
    load_job.result()
    
    table = client.get_table(table_id)
    print(f"[OK] Loaded {table.num_rows} rows into {table_name}")

if __name__ == "__main__":
    # Get date from environment variable or use today
    date_str = os.getenv('LOAD_DATE')
    if date_str:
        # Parse date like "2026-03-24" to "2026/03/24"
        date_path = date_str.replace('-', '/')
    else:
        # Use today's date
        date_path = datetime.now().strftime('%Y/%m/%d')
    
    print(f"Loading data for date: {date_path}")
    
    print("\nCreating tables...")
    create_table('reddit_posts', reddit_schema)
    create_table('news_articles', news_schema)
    create_table('spotify_tracks', spotify_schema)
    create_table('youtube_videos', youtube_schema)
    create_table('billboard_hot100', billboard_schema)
    
    print("\nLoading data from GCS...")
    load_from_gcs('reddit_posts', f'gs://{bucket_name}/reddit/{date_path}/*.jsonl')
    load_from_gcs('news_articles', f'gs://{bucket_name}/news/{date_path}/*.jsonl')
    load_from_gcs('spotify_tracks', f'gs://{bucket_name}/spotify/{date_path}/*.jsonl')
    load_from_gcs('youtube_videos', f'gs://{bucket_name}/youtube/{date_path}/*.jsonl')
    load_from_gcs('billboard_hot100', f'gs://{bucket_name}/billboard/{date_path}/*.jsonl')
    
    print("\n[OK] BigQuery setup complete!")