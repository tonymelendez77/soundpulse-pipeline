"""
SoundPulse — Prefect Orchestration
Daily 2 AM run of the full 14-module pipeline in dependency order.

Setup & run instructions are in orchestration/README_PREFECT.md

Schedule: 0 2 * * *  (daily at 02:00 UTC)
"""

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task, get_run_logger

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
    _run("ingestion/billboard_ingestion.py", "M5-billboard")

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
    _run("ingestion/news_sentiment.py", "M9-emotion")

# ════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Clustering + Correlation  (Modules 10–11)
# ════════════════════════════════════════════════════════════════════════════

@task(name="M10 · KMeans Audio Clustering", retries=1, retry_delay_seconds=60)
def run_clustering():
    _run("ingestion/audio_mood_clusters.py", "M10-kmeans")

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
    _run("ingestion/vector_index.py", "M13a-pinecone")

@task(name="M13b · MusicGen Audio Generation", retries=1, retry_delay_seconds=300)
def run_musicgen():
    _run("ingestion/music_generation.py", "M13b-musicgen")

# ════════════════════════════════════════════════════════════════════════════
# LAYER 8 — Static export  (Module 14)
# ════════════════════════════════════════════════════════════════════════════

@task(name="M14 · Export static data to docs/", retries=2, retry_delay_seconds=60)
def run_export():
    _run("serving/export_static.py", "M14-export")

def _write_run_log(run_record: dict) -> None:
    """Append/update a run record in docs/data/pipeline_runs.json."""
    runs_path = REPO_ROOT / "docs" / "data" / "pipeline_runs.json"
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if runs_path.exists():
        try:
            existing = json.loads(runs_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    # Update in-place if run_id already exists (e.g. final status update), else append
    ids = {r.get("run_id") for r in existing}
    if run_record["run_id"] in ids:
        existing = [run_record if r.get("run_id") == run_record["run_id"] else r for r in existing]
    else:
        existing.insert(0, run_record)   # newest first

    # Keep last 90 days / 90 runs
    existing = existing[:90]
    runs_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


# NOTE: git commit + push to GitHub Pages is handled by GitHub Actions
# after this Prefect flow exits. No git task needed here.


# ════════════════════════════════════════════════════════════════════════════
# MAIN FLOW
# ════════════════════════════════════════════════════════════════════════════

MODULE_NAMES = [
    "M1-news", "M1-reddit", "M1-youtube",
    "M2-spotify", "M3-itunes", "M4-lastfm", "M5-billboard", "M6-librosa",
    "M7-dbt", "M9-emotion", "M10-kmeans", "M11-correlation",
    "M12-xgboost", "M13a-pinecone", "M13b-musicgen", "M14-export",
]


@flow(
    name="SoundPulse — Daily Pipeline",
    description="Full 14-module pipeline: ingest → dbt → NLP → cluster → ML → GenAI → export",
    log_prints=True,
)
def soundpulse_daily():
    logger = get_run_logger()
    run_id      = str(uuid.uuid4())[:8]
    started_at  = datetime.now(timezone.utc)
    t0          = time.time()
    modules_ok  = []
    tasks_ok    = 0
    tasks_total = len(MODULE_NAMES)

    logger.info(f"Pipeline started at {started_at.isoformat()}  run_id={run_id}")

    # Write "running" record immediately so the website can show it in progress
    _write_run_log({
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": None,
        "status": "running",
        "duration_seconds": 0,
        "modules_completed": [],
        "tasks_ok": 0,
        "tasks_total": tasks_total,
    })

    def _done(name):
        nonlocal tasks_ok
        modules_ok.append(name)
        tasks_ok += 1

    try:
        # ── Layer 1: Ingestion (parallel sources) ─────────────────────────────
        news_f      = ingest_news.submit()
        reddit_f    = ingest_reddit.submit()
        youtube_f   = ingest_youtube.submit()
        spotify_f   = ingest_spotify.submit()
        itunes_f    = ingest_itunes.submit()
        lastfm_f    = ingest_lastfm.submit()
        billboard_f = ingest_billboard.submit()
        librosa_f   = ingest_librosa.submit()
        for f, n in [(news_f,"M1-news"),(reddit_f,"M1-reddit"),(youtube_f,"M1-youtube"),
                     (spotify_f,"M2-spotify"),(itunes_f,"M3-itunes"),(lastfm_f,"M4-lastfm"),
                     (billboard_f,"M5-billboard"),(librosa_f,"M6-librosa")]:
            f.result(); _done(n)

        # ── Layer 2: dbt ──────────────────────────────────────────────────────
        run_dbt.submit().result(); _done("M7-dbt")

        # ── Layer 3: Emotion NLP ──────────────────────────────────────────────
        run_emotion_nlp.submit().result(); _done("M9-emotion")

        # ── Layer 4: Clustering + Correlation (parallel) ──────────────────────
        cluster_f = run_clustering.submit()
        corr_f    = run_correlation.submit()
        cluster_f.result(); _done("M10-kmeans")
        corr_f.result();    _done("M11-correlation")

        # ── Layer 5: ML Prediction ────────────────────────────────────────────
        run_ml_predictions.submit().result(); _done("M12-xgboost")

        # ── Layer 6: dbt second pass ──────────────────────────────────────────
        run_dbt_post_ml.submit().result(); _done("M12b-dbt-ml")

        # ── Layer 7: GenAI ────────────────────────────────────────────────────
        run_pinecone.submit().result(); _done("M13a-pinecone")
        run_musicgen.submit().result(); _done("M13b-musicgen")

        # ── Layer 8: Export ───────────────────────────────────────────────────
        run_export.submit().result(); _done("M14-export")

        status = "success"
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        status = "failed"

    duration = int(time.time() - t0)
    completed_at = datetime.now(timezone.utc)

    # Write final run record (will be picked up by git push below)
    _write_run_log({
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "status": status,
        "duration_seconds": duration,
        "modules_completed": modules_ok,
        "tasks_ok": tasks_ok,
        "tasks_total": tasks_total,
    })

    # Add Prefect Cloud run URL if available (injected by Prefect at runtime)
    prefect_ui_url = os.getenv("PREFECT_UI_URL", "")

    # Re-write run log with final state + Prefect URL
    _write_run_log({
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "status": status,
        "duration_seconds": duration,
        "modules_completed": modules_ok,
        "tasks_ok": tasks_ok,
        "tasks_total": tasks_total,
        "run_url": prefect_ui_url or f"https://app.prefect.cloud/",
    })

    # NOTE: git commit + push is handled by GitHub Actions after this script exits
    logger.info(f"Pipeline {status} — {duration}s  ({tasks_ok}/{tasks_total} tasks)")


# ────────────────────────────────────────────────────────────────────────────
# Entry point — allows running ad-hoc: python orchestration/prefect_pipeline.py
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    soundpulse_daily()
