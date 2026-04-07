# SoundPulse — Prefect Orchestration

Daily 2 AM pipeline run covering all 14 modules in dependency order.

---

## 1. Install Prefect

```bash
pip install prefect
```

---

## 2. Start a Prefect server (local self-hosted)

Open a dedicated terminal and leave it running:

```bash
prefect server start
```

This starts the Prefect UI at http://127.0.0.1:4200 — you can watch every run there.

---

## 3. Create a work pool

```bash
prefect work-pool create --type process soundpulse-local
```

---

## 4. Start a worker (separate terminal, leave running)

```bash
prefect worker start --pool soundpulse-local
```

---

## 5. Deploy the flow with a daily 2 AM schedule

Run this once from the repo root (with your venv active):

```bash
prefect deploy \
  orchestration/prefect_pipeline.py:soundpulse_daily \
  --name "soundpulse-daily" \
  --pool soundpulse-local \
  --cron "0 2 * * *" \
  --timezone "UTC"
```

To use a different timezone (e.g. London):
```bash
  --timezone "Europe/London"
```

---

## 6. Verify the deployment

```bash
prefect deployment ls
```

You should see `SoundPulse — Daily Pipeline/soundpulse-daily` with the next scheduled run.

---

## 7. Trigger a manual run (test)

```bash
prefect deployment run "SoundPulse — Daily Pipeline/soundpulse-daily"
```

Or run the flow directly (no Prefect server needed):

```bash
python orchestration/prefect_pipeline.py
```

---

## 8. Monitor runs

Open http://127.0.0.1:4200 in your browser.

- **Flow Runs** tab → see each daily run, status, duration
- Click any run → see per-task logs with the module name
- Failed tasks show the error + stderr tail

---

## Flow structure

```
Layer 1  -- Ingestion (parallel)
             M1 news / reddit / youtube
             M2 Spotify audio features
             M3 iTunes · M4 Last.fm · M5 Billboard
             M6 Librosa

Layer 2  -- dbt run + test (raw → staging → marts)

Layer 3  -- M9 DistilRoBERTa emotion NLP

Layer 4  -- M10 KMeans clustering  (parallel)
             M11 Pearson correlation

Layer 5  -- M12 XGBoost + SHAP predictions

Layer 6  -- dbt re-run (picks up ML output tables)

Layer 7  -- M13a Pinecone upsert
             M13b MusicGen generation

Layer 8  -- M14 export_static.py → docs/data/
             git add docs/ && git commit && git push
```

---

## Retry policy

| Task | Retries | Delay |
|---|---|---|
| Ingestion (all) | 2 | 60–90 s |
| dbt | 1 | 120 s |
| Emotion NLP | 1 | 180 s |
| XGBoost | 1 | 120 s |
| MusicGen | 1 | 300 s |
| Export | 2 | 60 s |

---

## Adding email/Slack alerts on failure

```bash
pip install prefect-slack
```

Then in `prefect_pipeline.py`, add to the `@flow` decorator:

```python
from prefect.blocks.notifications import SlackWebhook

@flow(on_failure=[lambda flow, run, state: SlackWebhook.load("soundpulse").notify(str(state))])
```

Or use the Prefect UI: **Notifications** → add a webhook for failed runs.

---

## Keeping the server alive (Windows)

To auto-start both `prefect server` and `prefect worker` on login, create two scheduled tasks in Windows Task Scheduler pointing to:

```
prefect server start
prefect worker start --pool soundpulse-local
```

Or run them inside a `screen`/`tmux` equivalent like Windows Terminal with auto-restore.
