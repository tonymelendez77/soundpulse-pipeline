"""
SoundPulse — Module 12, Layer 4
XGBoost + SHAP: Predict next week's dominant audio mood from news emotion scores

Input:  music_analytics.weekly_features  (Layer 3 joined table)
        music_analytics.audio_mood_weekly (for mood pct features)
Output: music_analytics.ml_predictions   (predictions + mood_blend_json per period)
        music_analytics.shap_importance  (global feature importance)

Model: XGBoost multiclass classifier, 23 features:
  - 10 emotion scores  (from DistilRoBERTa news sentiment)
  - 5 mood percentages (from KMeans audio clustering)
  - 8 temporal/seasonal features (cyclical encoding + season flags)

Three inference periods per run:
  - "today"   → target_date = today,         emotion base = latest week
  - "weekly"  → target_date = this Monday,   emotion base = latest week
  - "monthly" → target_date = 1st of month,  emotion base = avg across current month

Install first (if not already):
    pip install xgboost shap
"""

import json
import math
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from google.cloud import bigquery
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# ── Config ──────────────────────────────────────────────────────────────────────
PROJECT      = "soundpulse-production"
DATASET      = "music_analytics"
SRC_TABLE    = f"{PROJECT}.{DATASET}.weekly_features"
MOOD_TABLE   = f"{PROJECT}.{DATASET}.audio_mood_weekly"
PRED_TABLE   = f"{PROJECT}.{DATASET}.ml_predictions"
SHAP_TABLE   = f"{PROJECT}.{DATASET}.shap_importance"

EMOTION_FEATURES = [
    "avg_fear", "avg_anger", "avg_joy", "avg_sadness",
    "avg_surprise", "avg_disgust", "avg_neutral",
    "anxiety_index", "tension_index", "positivity_index",
]

MOOD_PCT_FEATURES = [
    "euphoric_pct", "melancholic_pct", "aggressive_pct", "peaceful_pct", "groovy_pct",
]

TEMPORAL_FEATURES = [
    "month_sin", "month_cos",
    "week_of_yr_sin", "week_of_yr_cos",
    "season",
    "is_holiday_season",
    "is_summer",
    "is_new_year_period",
]

ALL_FEATURES = EMOTION_FEATURES + MOOD_PCT_FEATURES + TEMPORAL_FEATURES   # 23 total

TARGET_COL = "dominant_mood"

PRED_SCHEMA = [
    bigquery.SchemaField("week_start",           "DATE"),
    bigquery.SchemaField("period",               "STRING"),   # "today"|"weekly"|"monthly"|null
    bigquery.SchemaField("target_date",          "DATE"),
    bigquery.SchemaField("actual_mood",          "STRING"),
    bigquery.SchemaField("predicted_mood",       "STRING"),
    bigquery.SchemaField("correct",              "BOOLEAN"),
    bigquery.SchemaField("confidence",           "FLOAT64"),
    bigquery.SchemaField("mood_blend_json",      "STRING"),   # {"mood": probability, ...}
    bigquery.SchemaField("avg_fear",             "FLOAT64"),
    bigquery.SchemaField("avg_anger",            "FLOAT64"),
    bigquery.SchemaField("avg_joy",              "FLOAT64"),
    bigquery.SchemaField("avg_sadness",          "FLOAT64"),
    bigquery.SchemaField("anxiety_index",        "FLOAT64"),
    bigquery.SchemaField("tension_index",        "FLOAT64"),
    bigquery.SchemaField("positivity_index",     "FLOAT64"),
    bigquery.SchemaField("ingested_at",          "TIMESTAMP"),
]

SHAP_SCHEMA = [
    bigquery.SchemaField("feature",              "STRING"),
    bigquery.SchemaField("mood_archetype",       "STRING"),
    bigquery.SchemaField("mean_shap_value",      "FLOAT64"),
    bigquery.SchemaField("mean_abs_shap",        "FLOAT64"),
    bigquery.SchemaField("rank",                 "INTEGER"),
    bigquery.SchemaField("ingested_at",          "TIMESTAMP"),
]


# ── Temporal features ────────────────────────────────────────────────────────────

def extract_temporal_features(d: date) -> dict:
    """8 seasonal/calendar features computed from a date.
    Uses cyclical (sin/cos) encoding for month and week-of-year so the model
    treats Dec and Jan as adjacent rather than 11 months apart.
    """
    month = d.month
    woy   = d.isocalendar()[1]
    season_map = {12: 0, 1: 0, 2: 0,   # Winter
                  3: 1, 4: 1, 5: 1,    # Spring
                  6: 2, 7: 2, 8: 2,    # Summer
                  9: 3, 10: 3, 11: 3}  # Autumn
    season = season_map[month]
    return {
        "month_sin":          math.sin(2 * math.pi * month / 12),
        "month_cos":          math.cos(2 * math.pi * month / 12),
        "week_of_yr_sin":     math.sin(2 * math.pi * woy / 52),
        "week_of_yr_cos":     math.cos(2 * math.pi * woy / 52),
        "season":             float(season),                    # 0=Winter 1=Spring 2=Summer 3=Autumn
        "is_holiday_season":  float(month in (11, 12)),
        "is_summer":          float(season == 2),
        "is_new_year_period": float(month == 1 and d.day <= 15),
    }


# ── BQ helpers ──────────────────────────────────────────────────────────────────

def ensure_table(client, table_id, schema):
    client.delete_table(table_id, not_found_ok=True)
    client.create_table(bigquery.Table(table_id, schema=schema))
    time.sleep(10)
    logger.info(f"Table ready: {table_id}")


def streaming_insert(client, table_id, rows):
    """Batch load rows via NDJSON (reliable after table create)."""
    import io
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    ndjson = "\n".join(json.dumps(r) for r in rows)
    job = client.load_table_from_file(
        io.BytesIO(ndjson.encode()),
        table_id,
        job_config=job_config,
    )
    job.result()
    logger.info(f"Loaded {len(rows):,} rows → {table_id}")


# ── Data loading ─────────────────────────────────────────────────────────────────

def load_training_data(client: bigquery.Client) -> pd.DataFrame:
    """Load weekly_features joined with avg mood percentages from audio_mood_weekly."""
    query = f"""
        SELECT
            wf.week_start,
            wf.dominant_mood,
            wf.avg_fear,    wf.avg_anger,   wf.avg_joy,     wf.avg_sadness,
            wf.avg_surprise,wf.avg_disgust, wf.avg_neutral,
            wf.anxiety_index, wf.tension_index, wf.positivity_index,
            COALESCE(mw.euphoric_pct,    0.0) AS euphoric_pct,
            COALESCE(mw.melancholic_pct, 0.0) AS melancholic_pct,
            COALESCE(mw.aggressive_pct,  0.0) AS aggressive_pct,
            COALESCE(mw.peaceful_pct,    0.0) AS peaceful_pct,
            COALESCE(mw.groovy_pct,      0.0) AS groovy_pct
        FROM `{SRC_TABLE}` wf
        LEFT JOIN (
            SELECT week_start,
                   AVG(euphoric_pct)    AS euphoric_pct,
                   AVG(melancholic_pct) AS melancholic_pct,
                   AVG(aggressive_pct)  AS aggressive_pct,
                   AVG(peaceful_pct)    AS peaceful_pct,
                   AVG(groovy_pct)      AS groovy_pct
            FROM `{MOOD_TABLE}`
            GROUP BY week_start
        ) mw USING (week_start)
        ORDER BY wf.week_start
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df):,} weeks (weekly_features + mood_weekly join)")
    return df


def fetch_latest_emotions(client: bigquery.Client) -> tuple[dict, dict]:
    """Return (emotion_dict, mood_pct_dict) for the most recent week."""
    query = f"""
        SELECT
            wf.avg_fear, wf.avg_anger, wf.avg_joy, wf.avg_sadness,
            wf.avg_surprise, wf.avg_disgust, wf.avg_neutral,
            wf.anxiety_index, wf.tension_index, wf.positivity_index,
            wf.week_start,
            COALESCE(mw.euphoric_pct,    0.0) AS euphoric_pct,
            COALESCE(mw.melancholic_pct, 0.0) AS melancholic_pct,
            COALESCE(mw.aggressive_pct,  0.0) AS aggressive_pct,
            COALESCE(mw.peaceful_pct,    0.0) AS peaceful_pct,
            COALESCE(mw.groovy_pct,      0.0) AS groovy_pct
        FROM `{SRC_TABLE}` wf
        LEFT JOIN (
            SELECT week_start,
                   AVG(euphoric_pct)    AS euphoric_pct,
                   AVG(melancholic_pct) AS melancholic_pct,
                   AVG(aggressive_pct)  AS aggressive_pct,
                   AVG(peaceful_pct)    AS peaceful_pct,
                   AVG(groovy_pct)      AS groovy_pct
            FROM `{MOOD_TABLE}`
            GROUP BY week_start
        ) mw USING (week_start)
        ORDER BY wf.week_start DESC
        LIMIT 1
    """
    rows = list(client.query(query).result())
    if not rows:
        raise RuntimeError("No rows in weekly_features")
    row = rows[0]
    emotion_dict = {f: float(row[f] or 0.0) for f in EMOTION_FEATURES}
    mood_pct_dict = {f: float(row[f] or 0.0) for f in MOOD_PCT_FEATURES}
    logger.info(f"Latest emotions from week: {row['week_start']}")
    return emotion_dict, mood_pct_dict


def fetch_month_emotions(client: bigquery.Client, year: int, month: int) -> tuple[dict, dict]:
    """Return avg (emotion_dict, mood_pct_dict) across all weeks in the given month."""
    query = f"""
        SELECT
            AVG(wf.avg_fear)          AS avg_fear,
            AVG(wf.avg_anger)         AS avg_anger,
            AVG(wf.avg_joy)           AS avg_joy,
            AVG(wf.avg_sadness)       AS avg_sadness,
            AVG(wf.avg_surprise)      AS avg_surprise,
            AVG(wf.avg_disgust)       AS avg_disgust,
            AVG(wf.avg_neutral)       AS avg_neutral,
            AVG(wf.anxiety_index)     AS anxiety_index,
            AVG(wf.tension_index)     AS tension_index,
            AVG(wf.positivity_index)  AS positivity_index,
            AVG(COALESCE(mw.euphoric_pct,    0.0)) AS euphoric_pct,
            AVG(COALESCE(mw.melancholic_pct, 0.0)) AS melancholic_pct,
            AVG(COALESCE(mw.aggressive_pct,  0.0)) AS aggressive_pct,
            AVG(COALESCE(mw.peaceful_pct,    0.0)) AS peaceful_pct,
            AVG(COALESCE(mw.groovy_pct,      0.0)) AS groovy_pct
        FROM `{SRC_TABLE}` wf
        LEFT JOIN (
            SELECT week_start,
                   AVG(euphoric_pct)    AS euphoric_pct,
                   AVG(melancholic_pct) AS melancholic_pct,
                   AVG(aggressive_pct)  AS aggressive_pct,
                   AVG(peaceful_pct)    AS peaceful_pct,
                   AVG(groovy_pct)      AS groovy_pct
            FROM `{MOOD_TABLE}`
            GROUP BY week_start
        ) mw USING (week_start)
        WHERE EXTRACT(YEAR FROM wf.week_start) = {year}
          AND EXTRACT(MONTH FROM wf.week_start) = {month}
    """
    rows = list(client.query(query).result())
    if not rows or rows[0]["avg_fear"] is None:
        logger.warning(f"No data for {year}-{month:02d} — falling back to latest week emotions")
        return fetch_latest_emotions(client)
    row = rows[0]
    emotion_dict = {f: float(row[f] or 0.0) for f in EMOTION_FEATURES}
    mood_pct_dict = {f: float(row[f] or 0.0) for f in MOOD_PCT_FEATURES}
    logger.info(f"Monthly emotions: {year}-{month:02d} averaged")
    return emotion_dict, mood_pct_dict


# ── Inference ────────────────────────────────────────────────────────────────────

def predict_for_period(
    model: xgb.XGBClassifier,
    le: LabelEncoder,
    emotion_row: dict,
    mood_pcts: dict,
    target_date: date,
) -> dict:
    """Run inference for a single period with its own temporal context."""
    temp = extract_temporal_features(target_date)
    feat_vec = (
        [emotion_row.get(f, 0.0) for f in EMOTION_FEATURES]
        + [mood_pcts.get(f, 0.0) for f in MOOD_PCT_FEATURES]
        + [temp[f] for f in TEMPORAL_FEATURES]
    )
    X_inf = pd.DataFrame([feat_vec], columns=ALL_FEATURES)
    proba = model.predict_proba(X_inf)[0]
    pred_idx = int(proba.argmax())

    # Full blend: all moods that have >10% probability (richer prompt material)
    mood_blend = {
        le.classes_[i]: round(float(p), 4)
        for i, p in enumerate(proba)
        if p > 0.10
    }
    return {
        "predicted_mood": str(le.classes_[pred_idx]),
        "confidence":     round(float(proba[pred_idx]), 4),
        "mood_blend":     mood_blend,
        "target_date":    str(target_date),
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    client = bigquery.Client(project=PROJECT)
    now_ts = datetime.now(timezone.utc).isoformat()

    # 1. Load training data (weekly_features + mood pcts)
    logger.info("Reading training data …")
    df = load_training_data(client)

    if len(df) < 6:
        logger.error("Need at least 6 weeks of data. Run Layers 1–3 first.")
        return

    # 2. Build lag-1 target: next week's dominant_mood
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)
    df["target_mood"] = df[TARGET_COL].shift(-1)
    df = df.dropna(subset=["target_mood"] + EMOTION_FEATURES)

    logger.info(f"  {len(df):,} rows after lag alignment")

    # 3. Add temporal features to training rows
    for idx, row in df.iterrows():
        feats = extract_temporal_features(row["week_start"].date())
        for k, v in feats.items():
            df.at[idx, k] = v

    X = df[ALL_FEATURES].values
    y_raw = df["target_mood"].values

    # 4. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()
    n_classes = len(class_names)
    logger.info(f"  Target classes ({n_classes}): {class_names}")

    if n_classes < 2:
        logger.warning("Only 1 mood class — need more weeks. Skipping ML training.")
        return

    # 5. Class-balanced sample weights (fix majority-class bias)
    class_counts = Counter(y_raw)
    total = len(y_raw)
    sample_weights = np.array([
        total / (len(class_counts) * class_counts[raw]) for raw in y_raw
    ])
    logger.info(f"  Class distribution: {dict(class_counts)}")
    logger.info(f"  Sample weights range: {sample_weights.min():.3f} – {sample_weights.max():.3f}")

    # 6. Train XGBoost
    params = {
        "objective":        "multi:softprob",
        "num_class":        n_classes,
        "max_depth":        4,
        "learning_rate":    0.1,
        "n_estimators":     200,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "eval_metric":      "mlogloss",
        "random_state":     42,
    }
    model = xgb.XGBClassifier(**params)

    n_splits = min(3, len(df) // 2)
    if n_splits >= 2:
        cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring="accuracy",
                                    fit_params={"sample_weight": sample_weights})
        logger.info(f"  CV accuracy ({n_splits}-fold, weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X, y, sample_weight=sample_weights)
    proba_train = model.predict_proba(X)
    y_pred = np.argmax(proba_train, axis=1)
    confidence_train = proba_train.max(axis=1)

    train_acc = accuracy_score(y, y_pred)
    logger.info(f"  Train accuracy (weighted): {train_acc:.3f}")
    logger.info("\n" + classification_report(y, y_pred, labels=list(range(n_classes)), target_names=class_names))

    # 7. SHAP values
    logger.info("Computing SHAP values …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=0)
    else:
        shap_arr = np.array(shap_values)

    if shap_arr.ndim == 3 and shap_arr.shape[0] == len(X):
        shap_arr = shap_arr.transpose(2, 0, 1)

    max_abs_shap = float(np.abs(shap_arr).max()) if shap_arr.size > 0 else 0.0
    use_gain_fallback = max_abs_shap < 1e-6

    if use_gain_fallback:
        logger.warning(
            f"SHAP near-zero (max={max_abs_shap:.2e}). "
            "Falling back to point-biserial correlation."
        )
        shap_rows = []
        for cls_idx, cls_name in enumerate(class_names):
            y_binary = (y == cls_idx).astype(float)
            for feat_name in ALL_FEATURES:
                feat_vals = X[:, ALL_FEATURES.index(feat_name)]
                feat_std   = float(feat_vals.std())
                target_std = float(y_binary.std())
                if feat_std > 1e-9 and target_std > 1e-9:
                    r = float(np.corrcoef(feat_vals, y_binary)[0, 1])
                    r = 0.0 if np.isnan(r) else r
                else:
                    r = 0.0
                shap_rows.append({
                    "feature":         feat_name,
                    "mood_archetype":  cls_name,
                    "mean_shap_value": round(r, 6),
                    "mean_abs_shap":   round(abs(r), 6),
                    "rank":            0,
                    "ingested_at":     now_ts,
                })
    else:
        shap_rows = []
        for cls_idx, cls_name in enumerate(class_names):
            sv = shap_arr[cls_idx]
            for feat_idx, feat_name in enumerate(ALL_FEATURES):
                col = sv[:, feat_idx]
                shap_rows.append({
                    "feature":         feat_name,
                    "mood_archetype":  cls_name,
                    "mean_shap_value": round(float(col.mean()), 6),
                    "mean_abs_shap":   round(float(np.abs(col).mean()), 6),
                    "rank":            0,
                    "ingested_at":     now_ts,
                })

    shap_df = pd.DataFrame(shap_rows)
    shap_df["rank"] = (
        shap_df.groupby("mood_archetype")["mean_abs_shap"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    shap_rows = shap_df.to_dict("records")

    logger.info(f"  Importance source: {'fallback correlation' if use_gain_fallback else 'SHAP TreeExplainer'}")

    # 8. Write per-week historical predictions (backward compatibility)
    pred_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        blend = {
            class_names[j]: round(float(proba_train[i][j]), 4)
            for j in range(n_classes)
            if proba_train[i][j] > 0.10
        }
        pred_rows.append({
            "week_start":       str(row["week_start"].date()),
            "period":           None,
            "target_date":      str(row["week_start"].date()),
            "actual_mood":      str(y_raw[i]),
            "predicted_mood":   class_names[y_pred[i]],
            "correct":          bool(y[i] == y_pred[i]),
            "confidence":       round(float(confidence_train[i]), 6),
            "mood_blend_json":  json.dumps(blend),
            "avg_fear":         round(float(row["avg_fear"]),         6),
            "avg_anger":        round(float(row["avg_anger"]),        6),
            "avg_joy":          round(float(row["avg_joy"]),          6),
            "avg_sadness":      round(float(row["avg_sadness"]),      6),
            "anxiety_index":    round(float(row["anxiety_index"]),    6),
            "tension_index":    round(float(row["tension_index"]),    6),
            "positivity_index": round(float(row["positivity_index"]), 6),
            "ingested_at":      now_ts,
        })

    # 9. Three-period forward inference (today / weekly / monthly)
    today_d      = date.today()
    this_monday  = today_d - timedelta(days=today_d.weekday())
    this_month1  = today_d.replace(day=1)

    latest_emo, latest_pct = fetch_latest_emotions(client)
    month_emo,  month_pct  = fetch_month_emotions(client, today_d.year, today_d.month)

    period_configs = [
        ("today",   latest_emo, latest_pct, today_d),
        ("weekly",  latest_emo, latest_pct, this_monday),
        ("monthly", month_emo,  month_pct,  this_month1),
    ]

    # Use the most recent actual_mood as proxy for forward-inference actual_mood
    latest_actual = str(df.iloc[-1][TARGET_COL]) if len(df) else "unknown"

    logger.info("Running 3-period forward inference …")
    for period_name, emo, pcts, tgt_date in period_configs:
        result = predict_for_period(model, le, emo, pcts, tgt_date)
        # For forward inference we don't know the actual mood yet
        blend_json = json.dumps(result["mood_blend"])
        pred_rows.append({
            "week_start":       str(this_monday),
            "period":           period_name,
            "target_date":      result["target_date"],
            "actual_mood":      latest_actual,
            "predicted_mood":   result["predicted_mood"],
            "correct":          None,            # unknown until the week passes
            "confidence":       result["confidence"],
            "mood_blend_json":  blend_json,
            "avg_fear":         round(emo.get("avg_fear", 0.0), 6),
            "avg_anger":        round(emo.get("avg_anger", 0.0), 6),
            "avg_joy":          round(emo.get("avg_joy", 0.0), 6),
            "avg_sadness":      round(emo.get("avg_sadness", 0.0), 6),
            "anxiety_index":    round(emo.get("anxiety_index", 0.0), 6),
            "tension_index":    round(emo.get("tension_index", 0.0), 6),
            "positivity_index": round(emo.get("positivity_index", 0.0), 6),
            "ingested_at":      now_ts,
        })
        logger.info(
            f"  {period_name:8s} → {result['predicted_mood']:12s} "
            f"({result['confidence']:.1%} conf)  blend={result['mood_blend']}"
        )

    # 10. Persist
    ensure_table(client, PRED_TABLE, PRED_SCHEMA)
    streaming_insert(client, PRED_TABLE, pred_rows)

    ensure_table(client, SHAP_TABLE, SHAP_SCHEMA)
    streaming_insert(client, SHAP_TABLE, shap_rows)

    logger.info("─── LAYER 4 COMPLETE ───")
    logger.info(f"  Weeks in training   : {len(df)}")
    logger.info(f"  Train accuracy      : {train_acc:.3f}")
    logger.info(f"  Total pred rows     : {len(pred_rows)} ({len(df)} historical + 3 forward)")
    logger.info(f"  SHAP rows written   : {len(shap_rows)}")
    logger.info(f"  Features used       : {len(ALL_FEATURES)} ({ALL_FEATURES})")


if __name__ == "__main__":
    main()
