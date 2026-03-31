"""
GCS Upload Helper
Uploads DataFrame to Google Cloud Storage as JSONL
"""

import os
import json
from datetime import datetime
from google.cloud import storage
from loguru import logger
import pandas as pd


def upload_to_gcs(df, source_name='spotify'):
    """
    Upload DataFrame to GCS as JSONL file
    
    Args:
        df: DataFrame with tracks and audio features
        source_name: Source identifier (default: 'spotify')
    
    Returns:
        str: GCS path (gs://bucket/path/to/file.jsonl)
    """
    
    # GCS configuration
    PROJECT_ID = 'soundpulse-490820'
    BUCKET_NAME = 'soundpulse-prod-raw-lake'  
    
    # Generate file path with timestamp
    now = datetime.utcnow()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    
    blob_path = f"{source_name}/{year}/{month}/{day}/{source_name}_{timestamp}.jsonl"
    gcs_path = f"gs://{BUCKET_NAME}/{blob_path}"
    
    logger.info(f"Uploading {len(df)} tracks to GCS...")
    logger.info(f"Destination: {gcs_path}")
    
    # Convert DataFrame to JSONL string
    # Convert NaN to null for valid JSON
    jsonl_content = df.to_json(orient='records', lines=True, date_format='iso', force_ascii=False)
    jsonl_content = jsonl_content.replace(': NaN', ': null')
    
    # Initialize GCS client (uses Application Default Credentials)
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_path)
        
        # Upload
        blob.upload_from_string(
            jsonl_content,
            content_type='application/jsonl'
        )
        
        logger.info(f"✅ Upload successful: {len(df)} tracks")
        logger.info(f"   Size: {len(jsonl_content) / 1024:.2f} KB")
        logger.info(f"   Path: {gcs_path}")
        
        return gcs_path
        
    except Exception as e:
        logger.error(f"❌ GCS upload failed: {e}")
        
        # Fallback: Save locally
        local_dir = f"data/raw/{source_name}/{year}/{month}/{day}"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f"{source_name}_{timestamp}.jsonl")
        
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(jsonl_content)
        
        logger.warning(f"⚠️  Saved locally instead: {local_path}")
        return local_path


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    test_df = pd.DataFrame([
        {
            'title': 'Test Song',
            'artist': 'Test Artist',
            'album': 'Test Album',
            'popularity': 80,
            'duration_ms': 180000,
            'tempo': 120.0,
            'energy': 0.8
        }
    ])
    
    gcs_path = upload_to_gcs(test_df, source_name='test')
    print(f"Test upload: {gcs_path}")