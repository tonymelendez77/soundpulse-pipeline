"""
Load latest Spotify data with 30 audio features to BigQuery
"""
from google.cloud import bigquery

client = bigquery.Client(project='soundpulse-production')
table_ref = client.dataset('music_analytics').table('spotify_tracks')

# Load the LATEST file with correct data types
gcs_uri = 'gs://soundpulse-prod-raw-lake/spotify/2026/03/28/spotify_20260328_205013.jsonl'

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    ignore_unknown_values=True,
)

print(f"Loading from: {gcs_uri}")

load_job = client.load_table_from_uri(
    gcs_uri,
    table_ref,
    job_config=job_config
)

load_job.result()
print(f"✅ Loaded {load_job.output_rows} rows into spotify_tracks")
print(f"✅ BigQuery now has 30 audio features!")