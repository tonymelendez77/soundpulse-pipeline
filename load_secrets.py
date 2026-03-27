from prefect.blocks.system import Secret
import os

def load_secrets_to_env():
    """Load Prefect Cloud secrets into environment variables"""
    secrets_map = {
        'NEWSAPI_KEY': 'newsapi-key',
        'SPOTIFY_CLIENT_ID': 'spotify-client-id',
        'SPOTIFY_CLIENT_SECRET': 'spotify-client-secret',
        'YOUTUBE_API_KEY': 'youtube-api-key',
        'GUARDIAN_API_KEY': 'guardian-api-key',
        'MEDIASTACK_API_KEY': 'mediastack-api-key',
        'KAGGLE_TOKEN': 'kaggle-token',
        'KAGGLE_USERNAME': 'kaggle-username',
    }
    
    for env_var, block_name in secrets_map.items():
        try:
            secret_block = Secret.load(block_name)
            os.environ[env_var] = secret_block.get()
        except Exception as e:
            print(f"Warning: Could not load {block_name}: {e}")

if __name__ == "__main__":
    load_secrets_to_env()