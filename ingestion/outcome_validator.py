"""Validates past predictions against actual mood outcomes."""

import json
from datetime import date, datetime, timedelta, timezone

from google.cloud import bigquery
from loguru import logger

PROJECT     = "soundpulse-production"
DATASET     = "music_analytics"
PRED_TABLE  = f"{PROJECT}.{DATASET}.ml_predictions"
MOOD_TABLE  = f"{PROJECT}.{DATASET}.audio_mood_weekly"
ACC_TABLE   = f"{PROJECT}.{DATASET}.prediction_accuracy"

ACC_SCHEMA = [
    bigquery.SchemaField("week_start",        "DATE"),
    bigquery.SchemaField("period",            "STRING"),
    bigquery.SchemaField("predicted_mood",    "STRING"),
    bigquery.SchemaField("actual_mood",       "STRING"),
    bigquery.SchemaField("correct",           "BOOLEAN"),
    bigquery.SchemaField("confidence",        "FLOAT64"),
    bigquery.SchemaField("rolling_8w_acc",    "FLOAT64"),   # rolling accuracy last 8 weeks
    bigquery.SchemaField("rolling_8w_n",      "INTEGER"),   # sample count for that window
    bigquery.SchemaField("validated_at",      "TIMESTAMP"),
]


def ensure_acc_table(client: bigquery.Client) -> None:
    table = bigquery.Table(ACC_TABLE, schema=ACC_SCHEMA)
    client.create_table(table, exists_ok=True)
    logger.info(f"Table ready: {ACC_TABLE}")


def fetch_unvalidated_predictions(client: bigquery.Client) -> list[dict]:
    """Get predictions older than 7 days that haven't been validated."""
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    query = f"""
        SELECT
            week_start,
            period,
            target_date,
            predicted_mood,
            confidence,
            ingested_at
        FROM `{PRED_TABLE}`
        WHERE period IS NOT NULL
          AND correct IS NULL
          AND target_date <= '{cutoff}'
        ORDER BY target_date
    """
    rows = list(client.query(query).result())
    logger.info(f"Found {len(rows)} unvalidated predictions")
    return [dict(r) for r in rows]


def fetch_actual_mood(client: bigquery.Client, week_start: str) -> str | None:
    """Look up the dominant mood for a given week."""
    query = f"""
        SELECT dominant_mood, COUNT(*) AS n
        FROM `{MOOD_TABLE}`
        WHERE CAST(week_start AS STRING) = '{week_start}'
        GROUP BY dominant_mood
        ORDER BY n DESC
        LIMIT 1
    """
    rows = list(client.query(query).result())
    if rows:
        return str(rows[0]["dominant_mood"])
    return None


def update_correct_in_predictions(client: bigquery.Client,
                                   week_start: str,
                                   period: str,
                                   predicted_mood: str,
                                   actual_mood: str) -> None:
    """Mark a prediction as correct or incorrect."""
    correct = predicted_mood == actual_mood
    query = f"""
        UPDATE `{PRED_TABLE}`
        SET
            actual_mood = '{actual_mood}',
            correct     = {str(correct).upper()}
        WHERE period     = '{period}'
          AND CAST(week_start AS STRING) = '{week_start}'
          AND correct IS NULL
    """
    client.query(query).result()
    logger.info(
        f"  Validated {period} week {week_start}: "
        f"predicted={predicted_mood} actual={actual_mood} correct={correct}"
    )


def compute_rolling_accuracy(client: bigquery.Client, weeks: int = 8) -> dict:
    """Compute rolling accuracy over the last N weeks."""
    cutoff = (date.today() - timedelta(weeks=weeks)).isoformat()
    query = f"""
        SELECT
            COUNT(*)                                          AS n,
            COUNTIF(correct = TRUE)                          AS n_correct,
            SAFE_DIVIDE(COUNTIF(correct = TRUE), COUNT(*))   AS accuracy
        FROM `{PRED_TABLE}`
        WHERE period IS NOT NULL
          AND correct IS NOT NULL
          AND target_date >= '{cutoff}'
    """
    rows = list(client.query(query).result())
    if not rows or rows[0]["n"] == 0:
        return {"n": 0, "n_correct": 0, "accuracy": None}
    r = rows[0]
    return {
        "n":         int(r["n"]),
        "n_correct": int(r["n_correct"]),
        "accuracy":  round(float(r["accuracy"]), 4),
    }


def write_accuracy_rows(client: bigquery.Client,
                         validated: list[dict],
                         rolling: dict) -> None:
    """Write validated outcomes to prediction_accuracy."""
    if not validated:
        return
    now_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for v in validated:
        rows.append({
            "week_start":     str(v["week_start"]),
            "period":         v["period"],
            "predicted_mood": v["predicted_mood"],
            "actual_mood":    v["actual_mood"],
            "correct":        v["correct"],
            "confidence":     round(float(v["confidence"]), 4),
            "rolling_8w_acc": rolling.get("accuracy"),
            "rolling_8w_n":   rolling.get("n", 0),
            "validated_at":   now_ts,
        })
    errors = client.insert_rows_json(ACC_TABLE, rows)
    if errors:
        logger.error(f"BQ insert errors: {errors[:2]}")
    else:
        logger.info(f"Wrote {len(rows)} rows → {ACC_TABLE}")


def run_outcome_validation() -> dict:
    """Run prediction validation loop."""
    client = bigquery.Client(project=PROJECT)
    ensure_acc_table(client)

    # Find predictions that need validation
    unvalidated = fetch_unvalidated_predictions(client)
    if not unvalidated:
        logger.info("All predictions already validated — nothing to do")
        rolling = compute_rolling_accuracy(client)
        logger.info(f"Rolling 8-week accuracy: {rolling}")
        return {"validated": 0, "rolling": rolling}

    # For each, look up actual mood and update
    validated_rows = []
    for pred in unvalidated:
        week_str     = str(pred["week_start"])
        actual_mood  = fetch_actual_mood(client, week_str)
        if actual_mood is None:
            logger.warning(f"  No actual mood found for week {week_str} — skipping")
            continue
        correct = pred["predicted_mood"] == actual_mood
        update_correct_in_predictions(
            client, week_str, pred["period"],
            pred["predicted_mood"], actual_mood
        )
        validated_rows.append({**pred, "actual_mood": actual_mood, "correct": correct})

    # Compute rolling accuracy after updates
    rolling = compute_rolling_accuracy(client)
    logger.info(
        f"Rolling 8-week accuracy: {rolling['accuracy']:.1%} "
        f"({rolling['n_correct']}/{rolling['n']} predictions)"
        if rolling["accuracy"] is not None else "Rolling accuracy: insufficient data"
    )

    # Persist to accuracy table
    write_accuracy_rows(client, validated_rows, rolling)

    logger.info(f"OUTCOME VALIDATION COMPLETE ")
    logger.info(f"  Validated: {len(validated_rows)}")
    logger.info(f"  Rolling 8w accuracy: {rolling}")
    return {"validated": len(validated_rows), "rolling": rolling}


if __name__ == "__main__":
    result = run_outcome_validation()
