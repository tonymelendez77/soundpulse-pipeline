from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from prefect_gcp import GcpCredentials
    gcp_credentials = GcpCredentials.load("gcp-credentials")
    client = bigquery.Client(credentials=gcp_credentials.get_credentials_from_service_account(), project='soundpulse-490820')
except:
    client = bigquery.Client(project='soundpulse-490820')

dataset_id = 'music_analytics'

load_date = os.environ.get('LOAD_DATE', datetime.now().strftime('%Y-%m-%d'))
year, month, day = load_date.split('-')

reddit_schema = [
    bigquery.SchemaField('id', 'STRING'),
    bigquery.SchemaField('title', 'STRING'),
    bigquery.SchemaField('body', 'STRING'),
    bigquery.SchemaField('score', 'INTEGER'),
    bigquery.SchemaField('num_comments', 'INTEGER'),
    bigquery.SchemaField('created_utc', 'STRING'),
    bigquery.SchemaField('author', 'STRING'),
    bigquery.SchemaField('subreddit', 'STRING'),
    bigquery.SchemaField('url', 'STRING'),
    bigquery.SchemaField('ingested_at', 'STRING'),
]

news_schema = [
    bigquery.SchemaField('title', 'STRING'),
    bigquery.SchemaField('description', 'STRING'),
    bigquery.SchemaField('url', 'STRING'),
    bigquery.SchemaField('source', 'STRING'),
    bigquery.SchemaField('published_at', 'STRING'),
    bigquery.SchemaField('author', 'STRING'),
    bigquery.SchemaField('content', 'STRING'),
    bigquery.SchemaField('ingested_at', 'STRING'),
]

# UPDATED SPOTIFY SCHEMA WITH 30 AUDIO FEATURES
spotify_schema = [
    # Metadata
    bigquery.SchemaField('title', 'STRING'),
    bigquery.SchemaField('artist', 'STRING'),
    bigquery.SchemaField('album', 'STRING'),
    bigquery.SchemaField('popularity', 'INTEGER'),
    bigquery.SchemaField('duration_ms', 'INTEGER'),
    bigquery.SchemaField('explicit', 'BOOLEAN'),
    bigquery.SchemaField('release_date', 'STRING'),
    bigquery.SchemaField('source', 'STRING'),
    
    # Spotify Baseline Features (12)
    bigquery.SchemaField('tempo', 'FLOAT'),
    bigquery.SchemaField('energy', 'FLOAT'),
    bigquery.SchemaField('danceability', 'FLOAT'),
    bigquery.SchemaField('valence', 'FLOAT'),
    bigquery.SchemaField('acousticness', 'FLOAT'),
    bigquery.SchemaField('instrumentalness', 'FLOAT'),
    bigquery.SchemaField('liveness', 'FLOAT'),
    bigquery.SchemaField('loudness', 'FLOAT'),
    bigquery.SchemaField('speechiness', 'FLOAT'),
    bigquery.SchemaField('key', 'INTEGER'),
    bigquery.SchemaField('mode', 'INTEGER'),
    bigquery.SchemaField('time_signature', 'INTEGER'),
    
    # MFCCs (4)
    bigquery.SchemaField('mfcc_1', 'FLOAT'),
    bigquery.SchemaField('mfcc_2', 'FLOAT'),
    bigquery.SchemaField('mfcc_5', 'FLOAT'),
    bigquery.SchemaField('mfcc_13', 'FLOAT'),
    
    # Chroma Vector (12)
    bigquery.SchemaField('chroma_C', 'FLOAT'),
    bigquery.SchemaField('chroma_C_sharp', 'FLOAT'),
    bigquery.SchemaField('chroma_D', 'FLOAT'),
    bigquery.SchemaField('chroma_D_sharp', 'FLOAT'),
    bigquery.SchemaField('chroma_E', 'FLOAT'),
    bigquery.SchemaField('chroma_F', 'FLOAT'),
    bigquery.SchemaField('chroma_F_sharp', 'FLOAT'),
    bigquery.SchemaField('chroma_G', 'FLOAT'),
    bigquery.SchemaField('chroma_G_sharp', 'FLOAT'),
    bigquery.SchemaField('chroma_A', 'FLOAT'),
    bigquery.SchemaField('chroma_A_sharp', 'FLOAT'),
    bigquery.SchemaField('chroma_B', 'FLOAT'),
    
    # Advanced Features (2)
    bigquery.SchemaField('spectral_centroid', 'FLOAT'),
    bigquery.SchemaField('harmonic_percussive_ratio', 'FLOAT'),
    
    # iTunes metadata
    bigquery.SchemaField('preview_url', 'STRING'),
    bigquery.SchemaField('itunes_track_id', 'STRING'),
    
    # Ingestion timestamp
    bigquery.SchemaField('ingested_at', 'TIMESTAMP'),
]

youtube_schema = [
    bigquery.SchemaField('video_id', 'STRING'),
    bigquery.SchemaField('title', 'STRING'),
    bigquery.SchemaField('channel_title', 'STRING'),
    bigquery.SchemaField('view_count', 'INTEGER'),
    bigquery.SchemaField('like_count', 'INTEGER'),
    bigquery.SchemaField('comment_count', 'INTEGER'),
    bigquery.SchemaField('published_at', 'STRING'),
    bigquery.SchemaField('description', 'STRING'),
    bigquery.SchemaField('duration', 'STRING'),
    bigquery.SchemaField('ingested_at', 'STRING'),
]

billboard_schema = [
    bigquery.SchemaField('rank', 'INTEGER'),
    bigquery.SchemaField('title', 'STRING'),
    bigquery.SchemaField('artist', 'STRING'),
    bigquery.SchemaField('chart', 'STRING'),
    bigquery.SchemaField('source', 'STRING'),
    bigquery.SchemaField('ingested_at', 'STRING'),
]

def create_table(table_name, schema):
    table_ref = client.dataset(dataset_id).table(table_name)

    try:
        client.get_table(table_ref)
        print(f"[OK] Table {table_name} already exists")
    except Exception as get_error:
        # Can't read table, try to create
        try:
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
            print(f"[OK] Created table {table_name}")
        except Exception as create_error:
            if "Already Exists" in str(create_error):
                print(f"[OK] Table {table_name} exists (permission issue on read, but can write)")
            else:
                print(f"[ERROR] Failed to create {table_name}: {create_error}")
                raise

def load_table_from_gcs(table_name, gcs_uri):
    table_ref = client.dataset(dataset_id).table(table_name)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,  # CHANGED TO APPEND
        ignore_unknown_values=True,
        autodetect=False,
    )

    load_job = client.load_table_from_uri(
        gcs_uri,
        table_ref,
        job_config=job_config
    )

    load_job.result()
    print(f"[OK] Loaded {load_job.output_rows} rows into {table_name}")


# Create dataset if not exists (handle permission issues)
dataset_ref = client.dataset(dataset_id)
try:
    client.get_dataset(dataset_ref)
    print(f"[OK] Dataset {dataset_id} already exists")
except Exception as get_error:
    # Can't read dataset, try to create (will succeed if exists)
    try:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"[OK] Created dataset {dataset_id}")
    except Exception as create_error:
        if "Already Exists" in str(create_error):
            print(f"[OK] Dataset {dataset_id} exists (permission issue on read, but can write)")
        else:
            print(f"[ERROR] Failed: {create_error}")
            raise

# Create tables
create_table('reddit_posts', reddit_schema)
create_table('news_articles', news_schema)
create_table('spotify_tracks', spotify_schema)
create_table('youtube_videos', youtube_schema)
create_table('billboard_hot100', billboard_schema)

# Load data
load_table_from_gcs('reddit_posts', f'gs://soundpulse-raw-lake_ojmo/reddit/{year}/{month}/{day}/*.jsonl')
load_table_from_gcs('news_articles', f'gs://soundpulse-raw-lake_ojmo/news/{year}/{month}/{day}/*.jsonl')
load_table_from_gcs('spotify_tracks', f'gs://soundpulse-raw-lake_ojmo/spotify/{year}/{month}/{day}/*.jsonl')
load_table_from_gcs('youtube_videos', f'gs://soundpulse-raw-lake_ojmo/youtube/{year}/{month}/{day}/*.jsonl')
load_table_from_gcs('billboard_hot100', f'gs://soundpulse-raw-lake_ojmo/billboard/{year}/{month}/{day}/*.jsonl')

print(f"\n[OK] All tables loaded for date: {load_date}")