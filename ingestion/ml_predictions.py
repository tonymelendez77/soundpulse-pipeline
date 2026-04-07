"""
XGBoost + SHAP mood predictor. Trains per-region models on weekly_features
to predict next week's dominant audio mood from news emotion scores,
mood percentages, and temporal features. Outputs predictions and SHAP importance.
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

PROJECT = "soundpulse-production"
DATASET = "music_analytics"
SRC_TABLE = f"{PROJECT}.{DATASET}.weekly_features"
PRED_TABLE = f"{PROJECT}.{DATASET}.ml_predictions"
SHAP_TABLE = f"{PROJECT}.{DATASET}.shap_importance"

REGIONS = ["north_america", "latin_america", "europe", "global"]

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

ALL_FEATURES = EMOTION_FEATURES + MOOD_PCT_FEATURES + TEMPORAL_FEATURES

TARGET_COL = "dominant_mood"

PRED_SCHEMA = [
    bigquery.SchemaField("week_start", "DATE"),
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("period", "STRING"),
    bigquery.SchemaField("target_date", "DATE"),
    bigquery.SchemaField("actual_mood", "STRING"),
    bigquery.SchemaField("predicted_mood", "STRING"),
    bigquery.SchemaField("correct", "BOOLEAN"),
    bigquery.SchemaField("confidence", "FLOAT64"),
    bigquery.SchemaField("mood_blend_json", "STRING"),
    bigquery.SchemaField("avg_fear", "FLOAT64"),
    bigquery.SchemaField("avg_anger", "FLOAT64"),
    bigquery.SchemaField("avg_joy", "FLOAT64"),
    bigquery.SchemaField("avg_sadness", "FLOAT64"),
    bigquery.SchemaField("anxiety_index", "FLOAT64"),
    bigquery.SchemaField("tension_index", "FLOAT64"),
    bigquery.SchemaField("positivity_index", "FLOAT64"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP"),
]

SHAP_SCHEMA = [
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("feature", "STRING"),
    bigquery.SchemaField("mood_archetype", "STRING"),
    bigquery.SchemaField("mean_shap_value", "FLOAT64"),
    bigquery.SchemaField("mean_abs_shap", "FLOAT64"),
    bigquery.SchemaField("rank", "INTEGER"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP"),
]


def extract_temporal_features(d: date) -> dict:
    month = d.month
    woy = d.isocalendar()[1]
    season_map = {12: 0, 1: 0, 2: 0,
                  3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2,
                  9: 3, 10: 3, 11: 3}
    season = season_map[month]
    return {
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "week_of_yr_sin": math.sin(2 * math.pi * woy / 52),
        "week_of_yr_cos": math.cos(2 * math.pi * woy / 52),
        "season": float(season),
        "is_holiday_season": float(month in (11, 12)),
        "is_summer": float(season == 2),
        "is_new_year_period": float(month == 1 and d.day <= 15),
    }


def ensure_table(client, table_id, schema):
    client.delete_table(table_id, not_found_ok=True)
    client.create_table(bigquery.Table(table_id, schema=schema))
    time.sleep(10)
    logger.info(f"Table ready: {table_id}")


def streaming_insert(client, table_id, rows):
    """Batch load rows via NDJSON."""
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
    logger.info(f"Loaded {len(rows):,} rows -> {table_id}")


def load_training_data(client: bigquery.Client, region: str) -> pd.DataFrame:
    """Load weekly_features rows for a region."""
    query = f"""
        SELECT
            week_start, dominant_mood,
            avg_fear, avg_anger, avg_joy, avg_sadness,
            avg_surprise, avg_disgust, avg_neutral,
            anxiety_index, tension_index, positivity_index,
            COALESCE(euphoric_pct,    0.0) AS euphoric_pct,
            COALESCE(melancholic_pct, 0.0) AS melancholic_pct,
            COALESCE(aggressive_pct,  0.0) AS aggressive_pct,
            COALESCE(peaceful_pct,    0.0) AS peaceful_pct,
            COALESCE(groovy_pct,      0.0) AS groovy_pct
        FROM `{SRC_TABLE}`
        WHERE region = '{region}'
        ORDER BY week_start
    """
    df = client.query(query).to_dataframe()
    logger.info(f"  [{region}] Loaded {len(df):,} weeks from weekly_features")
    return df


def fetch_latest_emotions(client: bigquery.Client, region: str) -> tuple[dict, dict]:
    """Get emotions and mood pcts for the most recent week."""
    query = f"""
        SELECT
            avg_fear, avg_anger, avg_joy, avg_sadness,
            avg_surprise, avg_disgust, avg_neutral,
            anxiety_index, tension_index, positivity_index,
            week_start,
            COALESCE(euphoric_pct,    0.0) AS euphoric_pct,
            COALESCE(melancholic_pct, 0.0) AS melancholic_pct,
            COALESCE(aggressive_pct,  0.0) AS aggressive_pct,
            COALESCE(peaceful_pct,    0.0) AS peaceful_pct,
            COALESCE(groovy_pct,      0.0) AS groovy_pct
        FROM `{SRC_TABLE}`
        WHERE region = '{region}'
        ORDER BY week_start DESC
        LIMIT 1
    """
    rows = list(client.query(query).result())
    if not rows:
        raise RuntimeError(f"No rows in weekly_features for region={region}")
    row = rows[0]
    emotion_dict = {f: float(row[f] or 0.0) for f in EMOTION_FEATURES}
    mood_pct_dict = {f: float(row[f] or 0.0) for f in MOOD_PCT_FEATURES}
    logger.info(f"  [{region}] Latest emotions from week: {row['week_start']}")
    return emotion_dict, mood_pct_dict


def fetch_month_emotions(client: bigquery.Client, region: str, year: int, month: int) -> tuple[dict, dict]:
    """Get averaged emotions and mood pcts for a given month."""
    query = f"""
        SELECT
            AVG(avg_fear)          AS avg_fear,
            AVG(avg_anger)         AS avg_anger,
            AVG(avg_joy)           AS avg_joy,
            AVG(avg_sadness)       AS avg_sadness,
            AVG(avg_surprise)      AS avg_surprise,
            AVG(avg_disgust)       AS avg_disgust,
            AVG(avg_neutral)       AS avg_neutral,
            AVG(anxiety_index)     AS anxiety_index,
            AVG(tension_index)     AS tension_index,
            AVG(positivity_index)  AS positivity_index,
            AVG(COALESCE(euphoric_pct,    0.0)) AS euphoric_pct,
            AVG(COALESCE(melancholic_pct, 0.0)) AS melancholic_pct,
            AVG(COALESCE(aggressive_pct,  0.0)) AS aggressive_pct,
            AVG(COALESCE(peaceful_pct,    0.0)) AS peaceful_pct,
            AVG(COALESCE(groovy_pct,      0.0)) AS groovy_pct
        FROM `{SRC_TABLE}`
        WHERE region = '{region}'
          AND EXTRACT(YEAR  FROM week_start) = {year}
          AND EXTRACT(MONTH FROM week_start) = {month}
    """
    rows = list(client.query(query).result())
    if not rows or rows[0]["avg_fear"] is None:
        logger.warning(f"  [{region}] No data for {year}-{month:02d}, falling back to latest week")
        return fetch_latest_emotions(client, region)
    row = rows[0]
    emotion_dict = {f: float(row[f] or 0.0) for f in EMOTION_FEATURES}
    mood_pct_dict = {f: float(row[f] or 0.0) for f in MOOD_PCT_FEATURES}
    logger.info(f"  [{region}] Monthly emotions: {year}-{month:02d} averaged")
    return emotion_dict, mood_pct_dict


def predict_for_period(
    model: xgb.XGBClassifier,
    le: LabelEncoder,
    emotion_row: dict,
    mood_pcts: dict,
    target_date: date,
) -> dict:
    """Run inference for a single period."""
    temp = extract_temporal_features(target_date)
    feat_vec = (
        [emotion_row.get(f, 0.0) for f in EMOTION_FEATURES]
        + [mood_pcts.get(f, 0.0) for f in MOOD_PCT_FEATURES]
        + [temp[f] for f in TEMPORAL_FEATURES]
    )
    X_inf = pd.DataFrame([feat_vec], columns=ALL_FEATURES)
    proba = model.predict_proba(X_inf)[0]
    pred_idx = int(proba.argmax())

    mood_blend = {
        le.classes_[i]: round(float(p), 4)
        for i, p in enumerate(proba)
        if p > 0.10
    }
    return {
        "predicted_mood": str(le.classes_[pred_idx]),
        "confidence": round(float(proba[pred_idx]), 4),
        "mood_blend": mood_blend,
        "target_date": str(target_date),
    }


def train_region(client, region, now_ts, today_d, this_monday, this_month1):
    """Train one XGBoost model for a region. Returns (pred_rows, shap_rows)."""
    pred_rows = []
    shap_rows = []

    df = load_training_data(client, region)
    if len(df) < 6:
        logger.warning(f"  [{region}] Only {len(df)} weeks, need 6+. Skipping.")
        return pred_rows, shap_rows

    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)
    df["target_mood"] = df[TARGET_COL].shift(-1)
    df = df.dropna(subset=["target_mood"] + EMOTION_FEATURES)
    logger.info(f"  [{region}] {len(df)} rows after lag alignment")

    for idx, row in df.iterrows():
        feats = extract_temporal_features(row["week_start"].date())
        for k, v in feats.items():
            df.at[idx, k] = v

    X = df[ALL_FEATURES].values
    y_raw = df["target_mood"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()
    n_classes = len(class_names)
    logger.info(f"  [{region}] Target classes ({n_classes}): {class_names}")

    if n_classes < 2:
        dominant_mood = class_names[0]
        logger.warning(f"  [{region}] Only 1 class ('{dominant_mood}'), using fallback")
        try:
            latest_emo, latest_pct = fetch_latest_emotions(client, region)
        except Exception:
            latest_emo = {f: 0.0 for f in EMOTION_FEATURES}
            latest_pct = {f: 0.0 for f in MOOD_PCT_FEATURES}
        for period_name, tgt_date in [("today", today_d), ("weekly", this_monday), ("monthly", this_month1)]:
            pred_rows.append({
                "week_start": str(this_monday), "region": region,
                "period": period_name, "target_date": str(tgt_date),
                "actual_mood": dominant_mood, "predicted_mood": dominant_mood,
                "correct": None, "confidence": 1.0,
                "mood_blend_json": json.dumps({dominant_mood: 1.0}),
                "avg_fear": round(latest_emo.get("avg_fear", 0.0), 6),
                "avg_anger": round(latest_emo.get("avg_anger", 0.0), 6),
                "avg_joy": round(latest_emo.get("avg_joy", 0.0), 6),
                "avg_sadness": round(latest_emo.get("avg_sadness", 0.0), 6),
                "anxiety_index": round(latest_emo.get("anxiety_index", 0.0), 6),
                "tension_index": round(latest_emo.get("tension_index", 0.0), 6),
                "positivity_index": round(latest_emo.get("positivity_index", 0.0), 6),
                "ingested_at": now_ts,
            })
        return pred_rows, shap_rows

    # Sample weights combining recency decay and class balance
    class_counts = Counter(y_raw)
    total = len(y_raw)
    half_life = 26.0
    decay = np.array([np.exp(-np.log(2) * (total - 1 - i) / half_life) for i in range(total)])
    class_balance = np.array([total / (len(class_counts) * class_counts[raw]) for raw in y_raw])
    sample_weights = decay * class_balance
    sample_weights = sample_weights / sample_weights.mean()

    logger.info(f"  [{region}] Class distribution: {dict(class_counts)}")

    params = {
        "objective": "multi:softprob", "num_class": n_classes,
        "max_depth": 4, "learning_rate": 0.1, "n_estimators": 200,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "eval_metric": "mlogloss", "random_state": 42,
    }
    model = xgb.XGBClassifier(**params)

    n_splits = min(3, len(df) // 2)
    if n_splits >= 2:
        try:
            cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring="accuracy",
                                        fit_params={"sample_weight": sample_weights})
        except TypeError:
            cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring="accuracy")
        logger.info(f"  [{region}] CV accuracy ({n_splits}-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    model.fit(X, y, sample_weight=sample_weights)
    proba_train = model.predict_proba(X)
    y_pred = np.argmax(proba_train, axis=1)
    confidence_train = proba_train.max(axis=1)

    train_acc = accuracy_score(y, y_pred)
    logger.info(f"  [{region}] Train accuracy: {train_acc:.3f}")
    logger.info("\n" + classification_report(y, y_pred, labels=list(range(n_classes)), target_names=class_names))

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
        for cls_idx, cls_name in enumerate(class_names):
            y_binary = (y == cls_idx).astype(float)
            for feat_name in ALL_FEATURES:
                feat_vals = X[:, ALL_FEATURES.index(feat_name)]
                feat_std = float(feat_vals.std())
                target_std = float(y_binary.std())
                r = 0.0
                if feat_std > 1e-9 and target_std > 1e-9:
                    r = float(np.corrcoef(feat_vals, y_binary)[0, 1])
                    r = 0.0 if np.isnan(r) else r
                shap_rows.append({
                    "region": region, "feature": feat_name,
                    "mood_archetype": cls_name,
                    "mean_shap_value": round(r, 6), "mean_abs_shap": round(abs(r), 6),
                    "rank": 0, "ingested_at": now_ts,
                })
    else:
        for cls_idx, cls_name in enumerate(class_names):
            sv = shap_arr[cls_idx]
            for feat_idx, feat_name in enumerate(ALL_FEATURES):
                col = sv[:, feat_idx]
                shap_rows.append({
                    "region": region, "feature": feat_name,
                    "mood_archetype": cls_name,
                    "mean_shap_value": round(float(col.mean()), 6),
                    "mean_abs_shap": round(float(np.abs(col).mean()), 6),
                    "rank": 0, "ingested_at": now_ts,
                })

    shap_df = pd.DataFrame(shap_rows)
    shap_df["rank"] = (
        shap_df.groupby(["region", "mood_archetype"])["mean_abs_shap"]
        .rank(ascending=False, method="dense").astype(int)
    )
    shap_rows = shap_df.to_dict("records")

    latest_actual = str(df.iloc[-1][TARGET_COL]) if len(df) else "unknown"
    for i, (_, row) in enumerate(df.iterrows()):
        blend = {class_names[j]: round(float(proba_train[i][j]), 4) for j in range(n_classes) if proba_train[i][j] > 0.10}
        pred_rows.append({
            "week_start": str(row["week_start"].date()),
            "region": region,
            "period": None,
            "target_date": str(row["week_start"].date()),
            "actual_mood": str(y_raw[i]),
            "predicted_mood": class_names[y_pred[i]],
            "correct": bool(y[i] == y_pred[i]),
            "confidence": round(float(confidence_train[i]), 6),
            "mood_blend_json": json.dumps(blend),
            "avg_fear": round(float(row["avg_fear"]), 6),
            "avg_anger": round(float(row["avg_anger"]), 6),
            "avg_joy": round(float(row["avg_joy"]), 6),
            "avg_sadness": round(float(row["avg_sadness"]), 6),
            "anxiety_index": round(float(row["anxiety_index"]), 6),
            "tension_index": round(float(row["tension_index"]), 6),
            "positivity_index": round(float(row["positivity_index"]), 6),
            "ingested_at": now_ts,
        })

    # Forward inference for today/weekly/monthly
    try:
        latest_emo, latest_pct = fetch_latest_emotions(client, region)
        month_emo, month_pct = fetch_month_emotions(client, region, today_d.year, today_d.month)
    except Exception as exc:
        logger.warning(f"  [{region}] Could not fetch latest emotions ({exc}), using zero features")
        latest_emo = month_emo = {f: 0.0 for f in EMOTION_FEATURES}
        latest_pct = month_pct = {f: 0.0 for f in MOOD_PCT_FEATURES}

    period_configs = [
        ("today", latest_emo, latest_pct, today_d),
        ("weekly", latest_emo, latest_pct, this_monday),
        ("monthly", month_emo, month_pct, this_month1),
    ]
    for period_name, emo, pcts, tgt_date in period_configs:
        result = predict_for_period(model, le, emo, pcts, tgt_date)
        pred_rows.append({
            "week_start": str(this_monday),
            "region": region,
            "period": period_name,
            "target_date": result["target_date"],
            "actual_mood": latest_actual,
            "predicted_mood": result["predicted_mood"],
            "correct": None,
            "confidence": result["confidence"],
            "mood_blend_json": json.dumps(result["mood_blend"]),
            "avg_fear": round(emo.get("avg_fear", 0.0), 6),
            "avg_anger": round(emo.get("avg_anger", 0.0), 6),
            "avg_joy": round(emo.get("avg_joy", 0.0), 6),
            "avg_sadness": round(emo.get("avg_sadness", 0.0), 6),
            "anxiety_index": round(emo.get("anxiety_index", 0.0), 6),
            "tension_index": round(emo.get("tension_index", 0.0), 6),
            "positivity_index": round(emo.get("positivity_index", 0.0), 6),
            "ingested_at": now_ts,
        })
        logger.info(
            f"  [{region}] {period_name:8s} -> {result['predicted_mood']:12s} "
            f"({result['confidence']:.1%} conf)"
        )

    return pred_rows, shap_rows


def main():
    client = bigquery.Client(project=PROJECT)
    now_ts = datetime.now(timezone.utc).isoformat()

    logger.info("Running outcome validation ...")
    try:
        from outcome_validator import run_outcome_validation
        val_result = run_outcome_validation()
        if val_result["rolling"]["accuracy"] is not None:
            logger.info(
                f"  Rolling 8w accuracy: {val_result['rolling']['accuracy']:.1%} "
                f"({val_result['rolling']['n_correct']}/{val_result['rolling']['n']} validated)"
            )
    except Exception as e:
        logger.warning(f"Outcome validation failed ({e}), continuing with training")

    today_d = date.today()
    this_monday = today_d - timedelta(days=today_d.weekday())
    this_month1 = today_d.replace(day=1)

    all_pred_rows = []
    all_shap_rows = []

    for region in REGIONS:
        logger.info(f"Region: {region}")
        pred_rows, shap_rows = train_region(
            client, region, now_ts, today_d, this_monday, this_month1
        )
        all_pred_rows.extend(pred_rows)
        all_shap_rows.extend(shap_rows)

    if not all_pred_rows:
        logger.error("No predictions produced for any region.")
        return

    ensure_table(client, PRED_TABLE, PRED_SCHEMA)
    streaming_insert(client, PRED_TABLE, all_pred_rows)

    ensure_table(client, SHAP_TABLE, SHAP_SCHEMA)
    streaming_insert(client, SHAP_TABLE, all_shap_rows)

    fwd_rows = [r for r in all_pred_rows if r["period"] is not None]
    logger.info("Done. %d regions, %d pred rows (%d forward), %d SHAP rows",
                len(REGIONS), len(all_pred_rows), len(fwd_rows), len(all_shap_rows))


if __name__ == "__main__":
    main()
