"""
SoundPulse - BigQuery Schema Migration + GCS Loader
Table: trending_tracks (renamed from spotify_tracks)
"""

from google.cloud import bigquery, storage
from datetime import datetime

PROJECT = "soundpulse-production"
DATASET = "music_analytics"
TABLE = "trending_tracks"
BUCKET = "soundpulse-prod-raw-lake"
FULL_TABLE = f"{PROJECT}.{DATASET}.{TABLE}"

SCHEMA = [
    bigquery.SchemaField("title",                    "STRING"),
    bigquery.SchemaField("artist",                   "STRING"),
    bigquery.SchemaField("release_date",             "STRING"),
    bigquery.SchemaField("genre",                    "STRING"),
    bigquery.SchemaField("source",                   "STRING"),
    bigquery.SchemaField("country",                  "STRING"),
    bigquery.SchemaField("chart_rank",               "INTEGER"),
    bigquery.SchemaField("listeners",                "FLOAT"),
    bigquery.SchemaField("playcount",                "FLOAT"),
    bigquery.SchemaField("chart_date",               "STRING"),
    bigquery.SchemaField("source_count",             "INTEGER"),
    bigquery.SchemaField("itunes_track_id",          "INTEGER"),
    bigquery.SchemaField("itunes_title",             "STRING"),
    bigquery.SchemaField("itunes_artist",            "STRING"),
    bigquery.SchemaField("preview_url",              "STRING"),
    bigquery.SchemaField("itunes_duration_ms",       "FLOAT"),
    bigquery.SchemaField("itunes_genre",             "STRING"),
    bigquery.SchemaField("itunes_album",             "STRING"),
    bigquery.SchemaField("itunes_release_date",      "STRING"),
    bigquery.SchemaField("match_layer",              "INTEGER"),
    bigquery.SchemaField("spotify_track_id",         "STRING"),
    bigquery.SchemaField("explicit",                 "BOOLEAN"),
    bigquery.SchemaField("duration_ms",              "FLOAT"),
    bigquery.SchemaField("spotify_url",              "STRING"),
    bigquery.SchemaField("ingested_at",              "STRING"),
    bigquery.SchemaField("tempo",                    "FLOAT"),
    bigquery.SchemaField("energy",                   "FLOAT"),
    bigquery.SchemaField("danceability",             "FLOAT"),
    bigquery.SchemaField("valence",                  "FLOAT"),
    bigquery.SchemaField("acousticness",             "FLOAT"),
    bigquery.SchemaField("instrumentalness",         "FLOAT"),
    bigquery.SchemaField("liveness",                 "FLOAT"),
    bigquery.SchemaField("loudness",                 "FLOAT"),
    bigquery.SchemaField("speechiness",              "FLOAT"),
    bigquery.SchemaField("key",                      "FLOAT"),
    bigquery.SchemaField("mode",                     "FLOAT"),
    bigquery.SchemaField("time_signature",           "FLOAT"),
    bigquery.SchemaField("mfcc_1",                   "FLOAT"),
    bigquery.SchemaField("mfcc_2",                   "FLOAT"),
    bigquery.SchemaField("mfcc_5",                   "FLOAT"),
    bigquery.SchemaField("mfcc_13",                  "FLOAT"),
    bigquery.SchemaField("chroma_C",                 "FLOAT"),
    bigquery.SchemaField("chroma_C_sharp",           "FLOAT"),
    bigquery.SchemaField("chroma_D",                 "FLOAT"),
    bigquery.SchemaField("chroma_D_sharp",           "FLOAT"),
    bigquery.SchemaField("chroma_E",                 "FLOAT"),
    bigquery.SchemaField("chroma_F",                 "FLOAT"),
    bigquery.SchemaField("chroma_F_sharp",           "FLOAT"),
    bigquery.SchemaField("chroma_G",                 "FLOAT"),
    bigquery.SchemaField("chroma_G_sharp",           "FLOAT"),
    bigquery.SchemaField("chroma_A",                 "FLOAT"),
    bigquery.SchemaField("chroma_A_sharp",           "FLOAT"),
    bigquery.SchemaField("chroma_B",                 "FLOAT"),
    bigquery.SchemaField("spectral_centroid",        "FLOAT"),
    bigquery.SchemaField("harmonic_percussive_ratio","FLOAT"),
]

def migrate_table(client):
    # Drop old spotify_tracks if exists
    old_table = f"{PROJECT}.{DATASET}.spotify_tracks"
    client.delete_table(old_table, not_found_ok=True)
    print("[OK] Old spotify_tracks table dropped")

    # Drop and recreate trending_tracks
    client.delete_table(FULL_TABLE, not_found_ok=True)
    print("[OK] Old trending_tracks table dropped")

    table = bigquery.Table(FULL_TABLE, schema=SCHEMA)
    client.create_table(table)
    print(f"[OK] New trending_tracks table created with {len(SCHEMA)} columns")


def get_latest_gcs_file(storage_client):
    bucket = storage_client.bucket(BUCKET)
    blobs = list(bucket.list_blobs(prefix="raw/trending_tracks_"))
    if not blobs:
        raise FileNotFoundError("No trending_tracks JSONL files found in GCS")
    latest = sorted(blobs, key=lambda b: b.name)[-1]
    print(f"[OK] Latest GCS file: {latest.name}")
    return latest.name


def load_gcs_to_bigquery(client, gcs_file):
    print(f"Loading {gcs_file} into BigQuery...")
    uri = f"gs://{BUCKET}/{gcs_file}"
    job_config = bigquery.LoadJobConfig(
        schema=SCHEMA,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        ignore_unknown_values=True,
    )
    load_job = client.load_table_from_uri(uri, FULL_TABLE, job_config=job_config)
    load_job.result()
    table = client.get_table(FULL_TABLE)
    print(f"[OK] Loaded {table.num_rows} rows into {FULL_TABLE}")


def main():
    print("=" * 60)
    print("SOUNDPULSE - BigQuery Migration + GCS Load")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    bq_client = bigquery.Client(project=PROJECT)
    gcs_client = storage.Client(project=PROJECT)
    migrate_table(bq_client)
    latest_file = get_latest_gcs_file(gcs_client)
    load_gcs_to_bigquery(bq_client, latest_file)
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()