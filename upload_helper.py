from google.cloud import storage
from datetime import datetime
import json

def upload_to_gcs(data, source_name, bucket_name='soundpulse-prod-raw-lake'):
    """Upload data directly to GCS"""
    try:
        from prefect_gcp import GcpCredentials
        gcp_credentials = GcpCredentials.load("gcp-credentials")
        client = storage.Client(
            credentials=gcp_credentials.get_credentials_from_service_account(),
            project='soundpulse-production'
        )
    except:
        client = storage.Client(project='soundpulse-production')
    
    bucket = client.bucket(bucket_name)
    date_partition = datetime.now().strftime('%Y/%m/%d')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    filename = f"{source_name}/{date_partition}/{source_name}_{timestamp}.jsonl"
    blob = bucket.blob(filename)
    
    jsonl_content = '\n'.join([json.dumps(record) for record in data])
    blob.upload_from_string(jsonl_content)
    
    print(f"[OK] Uploaded {len(data)} records to gs://{bucket_name}/{filename}")
    return filename