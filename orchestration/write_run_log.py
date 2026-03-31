"""
Called by GitHub Actions at the end of every pipeline run.
Usage: python orchestration/write_run_log.py success|failure
Writes/updates docs/data/pipeline_runs.json
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RUNS_PATH = REPO_ROOT / "docs" / "data" / "pipeline_runs.json"

# GitHub Actions injects these automatically
RUN_ID       = os.getenv("GITHUB_RUN_ID", str(uuid.uuid4())[:8])
RUN_NUMBER   = os.getenv("GITHUB_RUN_NUMBER", "0")
RUN_URL      = f"https://github.com/{os.getenv('GITHUB_REPOSITORY','')}/actions/runs/{RUN_ID}"
STARTED_AT   = os.getenv("GITHUB_RUN_STARTED_AT", datetime.now(timezone.utc).isoformat())

status_arg = (sys.argv[1] if len(sys.argv) > 1 else "success").lower()
# GitHub passes "success" or "failure" (or "cancelled")
status = "success" if "success" in status_arg else "failed"

now = datetime.now(timezone.utc)

# Approximate duration: GH Actions doesn't expose it directly in env,
# so we compute from GITHUB_RUN_STARTED_AT if available
try:
    started = datetime.fromisoformat(STARTED_AT.replace("Z", "+00:00"))
    duration_seconds = int((now - started).total_seconds())
except Exception:
    duration_seconds = 0

MODULES_ALL = [
    "M1-news", "M1-reddit", "M1-youtube",
    "M2-spotify", "M3-itunes", "M4-lastfm", "M5-billboard", "M6-librosa",
    "M7-dbt", "M9-emotion", "M10-kmeans", "M11-correlation",
    "M12-xgboost", "M12b-dbt-ml", "M13a-pinecone", "M13b-musicgen", "M14-export",
]

record = {
    "run_id":            RUN_ID,
    "run_number":        int(RUN_NUMBER) if RUN_NUMBER.isdigit() else 0,
    "run_url":           RUN_URL,
    "started_at":        STARTED_AT,
    "completed_at":      now.isoformat(),
    "status":            status,
    "duration_seconds":  duration_seconds,
    # On success assume all modules ran; on failure approximate from duration
    "modules_completed": MODULES_ALL if status == "success" else MODULES_ALL[:max(1, min(len(MODULES_ALL), duration_seconds // 120))],
    "tasks_ok":          len(MODULES_ALL) if status == "success" else 0,
    "tasks_total":       len(MODULES_ALL),
}

# Load existing
RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
existing = []
if RUNS_PATH.exists():
    try:
        existing = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
    except Exception:
        existing = []

# Deduplicate by run_id
existing = [r for r in existing if r.get("run_id") != RUN_ID]
existing.insert(0, record)
existing = existing[:90]  # keep last 90

RUNS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
print(f"Run log updated: {record['status']} in {duration_seconds}s → {RUNS_PATH}")
