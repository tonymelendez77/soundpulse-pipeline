"""
SoundPulse — Prefect Orchestration
Daily 2 AM run of the full 14-module pipeline in dependency order.

Setup & run instructions are in orchestration/README_PREFECT.md

Schedule: 0 2 * * *  (daily at 02:00 UTC)
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

# ── Repo root (one level up from this file) ──────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
VENV_PYTHON = sys.executable   # uses whichever Python runs Prefect (venv)


# ── Helper: run a Python module / script ────────────────────────────────────
def _run(script_path: str, label: str) -> None:
    logger = get_run_logger()
    abs_path = REPO_ROOT / script_path
    cmd = [VENV_PYTHON, str(abs_path)]
    logger.info(f"[{label}] Starting: {abs_path.name}")
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.info(result.stdout[-4000:])   # tail last 4 KB to avoid log spam
    if result.returncode != 0:
        logger.error(result.stderr[-3000:])
        raise RuntimeError(f"[{label}] exited with code {result.returncode}")
    logger.info(f"[{label}] ✓ Done")


def _run_dbt(command: str, label: str) -> None:
    """Run a dbt command inside soundpulse_dbt/."""
    logger = get_run_logger()
    dbt_dir = REPO_ROOT / "soundpulse_dbt"
    full_cmd = f"dbt {command}"
    logger.info(f"[{label}] {full_cmd}")
    result = subprocess.run(
        full_cmd,
        cwd=str(dbt_dir),
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.info(result.stdout[-4000:])
    if result.returncode != 0:
        logger.error(result.stderr[-3000:])
        raise RuntimeError(f"[{label}] dbt {command} failed (exit {result.returncode})")
    logger.info(f"[{label}] ✓ Done")


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Ingestion  (Modules 1–6)
# Each source is independent — they run in the same Prefect task sequentially.
# (Split into separate @task if you want per-step retry / observability.)
# ════════════════════════════════════════════════════════════════════════════

@task(name="M1 · News Ingestion", retries=2, retry_delay_seconds=60)
def ingest_news():
    _run("ingestion/news_ingestion.py", "M1-news")

@task(name="M1 · Reddit Ingestion", retries=2, retry_delay_seconds=60)
def ingest_reddit():
    _run("ingestion/reddit_ingestion.py", "M1-reddit")

@task(name="M1 · YouTube Ingestion", retries=2, retry_delay_seconds=60)
def ingest_youtube():
    _run("ingestion/youtube_ingestion.py", "M1-youtube")

@task(name="M2 · Spotify Audio Features", retries=2, retry_delay_seconds=90)
def ingest_spotify():
    _run("ingestion/spotify_ingestion.py", "M2-spotify")

@task(name="M3 · iTunes Trending", retries=2, retry_delay_seconds=60)
def ingest_itunes():
    _run("ingestion/itunes_ingestion.py", "M3-itunes")

@task(name="M4 · Last.fm Trending", retries=2, retry_delay_seconds=60)
def ingest_lastfm():
    _run("ingestion/lastfm_ingestion.py", "M4-lastfm")

@task(name="M5 · Billboard Charts", retries=2, retry_delay_seconds=60)
def ingest_billboard():
    _run("ingestion/historical_backfill.py", "M5-billboard")

@task(name="M6 · Librosa Audio Features", retries=1, retry_delay_seconds=120)
def ingest_librosa():
    _run("ingestion/audio_features_librosa.py", "M6-librosa")

# ════════════════════════════════════════════════════════════════════════════
# LAYER 2 — dbt Transformations  (Modules 7–8)
# ════════════════════════════════════════════════════════════════════════════

@task(name="M7–8 · dbt run + test", retries=1, retry_delay_seconds=120)
def run_dbt():
    _run_dbt("run", "M7-dbt-run")
    _run_dbt("test", "M8-dbt-test")

# ════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Emotion Classification  (Module 9)
# ════════════════════════════════════════════════════════════════════════════

@task(name="M9 · DistilRoBERTa Emotion NLP", retries=1, retry_delay_seconds=180)
def run_emotion_nlp():
    _run("ingestion/emotion_classification.py", "M9-emotion")

# ════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Clustering + Correlation  (Modules 10–11)
# ════════════════════════════════════════════════════════════════════════════

@task(name="M10 · KMeans Audio Clustering", retries=1, retry_delay_seconds=60)
def run_clustering():
    _run("ingestion/audio_clustering.py", "M10-kmeans")

@task(name="M11 · Pearson Correlation", retries=1, retry_delay_seconds=60)
def run_correlation():
    _run("ingestion/emotion_music_correlation.py", "M11-correlation")

# ════════════════════════════════════════════════════════════════════════════
# LAYER 5 — ML Prediction  (Module 12)
# ════════════════════════════════════════════════════════════════════════════

@task(name="M12 · XGBoost + SHAP Prediction", retries=1, retry_delay_seconds=120)
def run_ml_predictions():
    _run("ingestion/ml_predictions.py", "M12-xgboost")

# ════════════════════════════════════════════════════════════════════════════
# LAYER 6 — dbt (second pass — picks up ML output tables)
# ════════════════════════════════════════════════════════════════════════════

@task(name="M12b · dbt re-run for ML tables", retries=1, retry_delay_seconds=120)
def run_dbt_post_ml():
    _run_dbt("run --select stg_shap_importance stg_ml_predictions fct_mood_prediction_summary", "M12b-dbt-ml")

# ════════════════════════════════════════════════════════════════════════════
# LAYER 7 — GenAI  (Module 13)
# ════════════════════════════════════════════════════════════════════════════

@task(name="M13a · Pinecone Vector Upsert", retries=2, retry_delay_seconds=60)
def run_pinecone():
    _run("ingestion/pinecone_index.py", "M13a-pinecone")

@task(name="M13b · MusicGen Audio Generation", retries=1, retry_delay_seconds=300)
def run_musicgen():
    _run("ingestion/music_generation.py", "M13b-musicgen")

# ════════════════════════════════════════════════════════════════════════════
# LAYER 8 — Static export  (Module 14)
# ════════════════════════════════════════════════════════════════════════════

@task(name="M14 · Export static data to docs/", retries=2, retry_delay_seconds=60)
def run_export():
    _run("serving/export_static.py", "M14-export")

@task(name="M14 · Git commit + push docs/data/", retries=1, retry_delay_seconds=30)
def git_push_docs():
    logger = get_run_logger()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cmds = [
        ["git", "add", "docs/data/", "docs/audio/"],
        ["git", "commit", "--allow-empty", "-m", f"chore: daily data refresh [{now}]"],
        ["git", "push"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode not in (0, 1):   # 1 = nothing to commit, ok
            logger.warning(result.stderr)


# ════════════════════════════════════════════════════════════════════════════
# MAIN FLOW
# ════════════════════════════════════════════════════════════════════════════

@flow(
    name="SoundPulse — Daily Pipeline",
    description="Full 14-module pipeline: ingest → dbt → NLP → cluster → ML → GenAI → export",
    log_prints=True,
)
def soundpulse_daily():
    logger = get_run_logger()
    logger.info(f"Pipeline started at {datetime.now(timezone.utc).isoformat()}")

    # ── Layer 1: Ingestion (parallel sources) ────────────────────────────────
    news_f      = ingest_news.submit()
    reddit_f    = ingest_reddit.submit()
    youtube_f   = ingest_youtube.submit()
    spotify_f   = ingest_spotify.submit()
    itunes_f    = ingest_itunes.submit()
    lastfm_f    = ingest_lastfm.submit()
    billboard_f = ingest_billboard.submit()
    librosa_f   = ingest_librosa.submit()

    # Wait for all ingestion to finish before dbt
    for f in [news_f, reddit_f, youtube_f, spotify_f, itunes_f, lastfm_f, billboard_f, librosa_f]:
        f.result()

    # ── Layer 2: dbt (first run — raw → staging → marts) ─────────────────────
    dbt_f = run_dbt.submit()
    dbt_f.result()

    # ── Layer 3: Emotion NLP ──────────────────────────────────────────────────
    nlp_f = run_emotion_nlp.submit()
    nlp_f.result()

    # ── Layer 4: Clustering + Correlation (parallel) ──────────────────────────
    cluster_f = run_clustering.submit()
    corr_f    = run_correlation.submit()
    cluster_f.result()
    corr_f.result()

    # ── Layer 5: ML Prediction ────────────────────────────────────────────────
    ml_f = run_ml_predictions.submit()
    ml_f.result()

    # ── Layer 6: dbt second pass (picks up ML output tables) ─────────────────
    dbt2_f = run_dbt_post_ml.submit()
    dbt2_f.result()

    # ── Layer 7: GenAI ────────────────────────────────────────────────────────
    pinecone_f = run_pinecone.submit()
    pinecone_f.result()
    music_f = run_musicgen.submit()
    music_f.result()

    # ── Layer 8: Export + push ────────────────────────────────────────────────
    export_f = run_export.submit()
    export_f.result()
    git_push_docs()

    logger.info("Pipeline complete.")


# ────────────────────────────────────────────────────────────────────────────
# Entry point — allows running ad-hoc: python orchestration/prefect_pipeline.py
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    soundpulse_daily()
