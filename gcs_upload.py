from google.cloud import storage
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from prefect.blocks.system import Secret
    from prefect_gcp import GcpCredentials
    gcp_credentials = GcpCredentials.load("gcp-credentials")
    client = storage.Client(credentials=gcp_credentials.get_credentials_from_service_account(), project='soundpulse-production')
except:
    client = storage.Client(project='soundpulse-production')

bucket = client.bucket('soundpulse-prod-raw-lake')

date_partition = datetime.now().strftime('%Y/%m/%d')

data_dir = Path('data/raw')
jsonl_files = list(data_dir.glob('*/*.jsonl'))

for local_file in jsonl_files:
    source_folder = local_file.parent.name
    gcs_path = f'{source_folder}/{date_partition}/{local_file.name}'
    
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_file))
    print(f"[OK] Uploaded {gcs_path}")

print(f"\n[OK] Uploaded {len(jsonl_files)} files to GCS")