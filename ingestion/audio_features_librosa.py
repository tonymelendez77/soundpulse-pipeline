"""
Librosa audio feature extraction with PARALLEL processing and CACHING
Extracts 30 features from iTunes preview URLs (30-second .m4a clips)
"""

import pandas as pd
import numpy as np
import librosa
import requests
import tempfile
import os
from loguru import logger
from pydub import AudioSegment
import json
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial

# Configure pydub for ffmpeg
AudioSegment.converter = "C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg/bin/ffprobe.exe"

# CACHE FILE LOCATION
CACHE_FILE = "C:/Users/tony_/OneDrive/Documents/soundpulse-pulseiq/librosa_cache.json"


def load_cache():
    """Load cached audio features from disk"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            logger.info(f"Cache loaded: {len(cache)} cached tracks")
            return cache
        except:
            logger.warning("Cache file corrupted, starting fresh")
            return {}
    else:
        logger.info("No cache file found, starting fresh")
        return {}


def save_cache(cache):
    """Save cache to disk"""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.info(f"Cache saved: {len(cache)} tracks")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


def get_cache_key(title, artist, preview_url):
    """Generate unique cache key for a track"""
    key_str = f"{title}|{artist}|{preview_url}"
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()


def download_audio(preview_url):
    """Download audio from URL and convert to WAV"""
    try:
        response = requests.get(preview_url, timeout=15)
        response.raise_for_status()
        
        # Save .m4a temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp_m4a:
            tmp_m4a.write(response.content)
            m4a_path = tmp_m4a.name
        
        # Convert to WAV
        audio = AudioSegment.from_file(m4a_path, format='m4a')
        wav_path = m4a_path.replace('.m4a', '.wav')
        audio.export(wav_path, format='wav')
        
        # Clean up m4a
        os.remove(m4a_path)
        
        return wav_path
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def extract_features_from_audio(wav_path):
    """Extract 30 audio features from WAV file using Librosa"""
    try:
        y, sr = librosa.load(wav_path, sr=None)
        
        # Spotify baseline features (12)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid = float(np.mean(spectral_centroids))
        
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))
        
        zcr = librosa.feature.zero_crossing_rate(y)
        danceability = float(np.mean(zcr))
        
        valence = energy * 0.8
        acousticness = 1.0 - (spectral_centroid / 8000.0)
        acousticness = max(0.0, min(1.0, acousticness))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        instrumentalness = 1.0 - float(np.mean(spectral_rolloff) / sr)
        instrumentalness = max(0.0, min(1.0, instrumentalness))
        
        liveness = 0.15
        
        loudness = float(np.mean(librosa.amplitude_to_db(rms, ref=np.max)))
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        speechiness = float(np.std(onset_env) / (np.mean(onset_env) + 1e-6))
        speechiness = max(0.0, min(1.0, speechiness))
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key = int(np.argmax(np.mean(chroma, axis=1)))
        
        mode = 1
        time_signature = 4
        
        # MFCCs (4 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_1 = float(np.mean(mfccs[0]))
        mfcc_2 = float(np.mean(mfccs[1]))
        mfcc_5 = float(np.mean(mfccs[4]))
        mfcc_13 = float(np.mean(mfccs[12]))
        
        # Chroma vector (12 features)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_C = float(chroma_mean[0])
        chroma_C_sharp = float(chroma_mean[1])
        chroma_D = float(chroma_mean[2])
        chroma_D_sharp = float(chroma_mean[3])
        chroma_E = float(chroma_mean[4])
        chroma_F = float(chroma_mean[5])
        chroma_F_sharp = float(chroma_mean[6])
        chroma_G = float(chroma_mean[7])
        chroma_G_sharp = float(chroma_mean[8])
        chroma_A = float(chroma_mean[9])
        chroma_A_sharp = float(chroma_mean[10])
        chroma_B = float(chroma_mean[11])
        
        # Harmonic-percussive ratio (1 feature)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = float(np.sum(y_harmonic ** 2))
        percussive_energy = float(np.sum(y_percussive ** 2))
        harmonic_percussive_ratio = harmonic_energy / (percussive_energy + 1e-6)
        
        # spectral_centroid already calculated (1 feature)
        
        return {
            # Spotify baseline (12)
            'tempo': tempo,
            'energy': energy,
            'danceability': danceability,
            'valence': valence,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'liveness': liveness,
            'loudness': loudness,
            'speechiness': speechiness,
            'key': key,
            'mode': mode,
            'time_signature': time_signature,
            
            # MFCCs (4)
            'mfcc_1': mfcc_1,
            'mfcc_2': mfcc_2,
            'mfcc_5': mfcc_5,
            'mfcc_13': mfcc_13,
            
            # Chroma vector (12)
            'chroma_C': chroma_C,
            'chroma_C_sharp': chroma_C_sharp,
            'chroma_D': chroma_D,
            'chroma_D_sharp': chroma_D_sharp,
            'chroma_E': chroma_E,
            'chroma_F': chroma_F,
            'chroma_F_sharp': chroma_F_sharp,
            'chroma_G': chroma_G,
            'chroma_G_sharp': chroma_G_sharp,
            'chroma_A': chroma_A,
            'chroma_A_sharp': chroma_A_sharp,
            'chroma_B': chroma_B,
            
            # Advanced (2)
            'spectral_centroid': spectral_centroid,
            'harmonic_percussive_ratio': harmonic_percussive_ratio
        }
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return None


def process_single_track(args):
    """Process a single track (for parallel execution)"""
    idx, row, cache = args
    
    title = row.get('title', 'Unknown')
    artist = row.get('artist', 'Unknown')
    preview_url = row.get('preview_url')
    
    # Check cache first
    if preview_url and not pd.isna(preview_url):
        cache_key = get_cache_key(title, artist, preview_url)
        
        if cache_key in cache:
            logger.info(f"[CACHE HIT] [{idx}]: {artist} - {title}")
            return (idx, cache[cache_key], cache_key)
        
        # Not in cache, extract features
        logger.info(f"[EXTRACTING] [{idx}]: {artist} - {title}")
        
        wav_path = download_audio(preview_url)
        if wav_path:
            features = extract_features_from_audio(wav_path)
            os.remove(wav_path)
            
            if features:
                return (idx, features, cache_key)
    
    return (idx, None, None)


def enrich_with_librosa_features(df):
    """Main function: Enrich DataFrame with Librosa features (PARALLEL + CACHE)"""
    
    logger.info(f"Extracting 30 audio features for {len(df)} tracks (PARALLEL + CACHE)")
    
    # Load cache
    cache = load_cache()
    
    # Initialize all 30 feature columns
    feature_columns = [
        'tempo', 'energy', 'danceability', 'valence', 'acousticness',
        'instrumentalness', 'liveness', 'loudness', 'speechiness',
        'key', 'mode', 'time_signature',
        'mfcc_1', 'mfcc_2', 'mfcc_5', 'mfcc_13',
        'chroma_C', 'chroma_C_sharp', 'chroma_D', 'chroma_D_sharp',
        'chroma_E', 'chroma_F', 'chroma_F_sharp', 'chroma_G',
        'chroma_G_sharp', 'chroma_A', 'chroma_A_sharp', 'chroma_B',
        'spectral_centroid', 'harmonic_percussive_ratio'
    ]
    
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Prepare arguments for parallel processing
    args_list = [(i, row, cache) for i, row in df.iterrows()]
    
    # Determine number of workers
    num_workers = min(cpu_count(), 8)  # Max 8 workers
    logger.info(f"Using {num_workers} parallel workers")
    
    # Process in parallel
    new_cache_entries = 0
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_track, args_list)
    
    # Update DataFrame and cache
    for idx, features, cache_key in results:
        if features:
            for col in feature_columns:
                df.loc[idx, col] = features.get(col, np.nan)
            
            # Update cache
            if cache_key and cache_key not in cache:
                cache[cache_key] = features
                new_cache_entries += 1
    
    # Save updated cache
    if new_cache_entries > 0:
        save_cache(cache)
        logger.info(f"Added {new_cache_entries} new entries to cache")
    
    # Summary
    tracks_with_features = df[feature_columns].notna().all(axis=1).sum()
    logger.info(f"✅ {tracks_with_features}/{len(df)} tracks have complete Librosa features")
    
    return df