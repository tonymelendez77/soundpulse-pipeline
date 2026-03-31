"""
Update spotify_tracks table to add 18 new audio feature columns
CORRECT PROJECT: soundpulse-production
"""
from google.cloud import bigquery

# CORRECT PROJECT
client = bigquery.Client(project='soundpulse-production')
table_ref = client.dataset('music_analytics').table('spotify_tracks')

# Get current table
table = client.get_table(table_ref)
print(f"Current schema has {len(table.schema)} columns")

# Add 18 new columns
new_fields = [
    bigquery.SchemaField('mfcc_1', 'FLOAT'),
    bigquery.SchemaField('mfcc_2', 'FLOAT'),
    bigquery.SchemaField('mfcc_5', 'FLOAT'),
    bigquery.SchemaField('mfcc_13', 'FLOAT'),
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
    bigquery.SchemaField('spectral_centroid', 'FLOAT'),
    bigquery.SchemaField('harmonic_percussive_ratio', 'FLOAT'),
    bigquery.SchemaField('preview_url', 'STRING'),
    bigquery.SchemaField('itunes_track_id', 'STRING'),
]

table.schema = list(table.schema) + new_fields
table = client.update_table(table, ['schema'])

print(f"✅ Updated! New schema has {len(table.schema)} columns")