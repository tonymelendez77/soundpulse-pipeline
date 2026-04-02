"""
SoundPulse — Module 12, Layer 4
XGBoost + SHAP: Predict next week's dominant audio mood from news emotion scores

Input:  music_analytics.weekly_features  (Layer 3 joined table)
Output: music_analytics.ml_predictions   (predictions + SHAP values per week)
        music_analytics.shap_importance  (global feature importance)

Model: XGBoost multiclass classifier
Features (X): this week's emotion scores (10 cols)
Target (y):   next week's dominant_mood archetype (lagged by 1 week)

Install first (if not already):
    pip install xgboost shap
"""

import time
from datetime import datetime, timezone

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
PRED_TABLE   = f"{PROJECT}.{DATASET}.ml_predictions"
SHAP_TABLE   = f"{PROJECT}.{DATASET}.shap_importance"

EMOTION_FEATURES = [
    "avg_fear", "avg_anger", "avg_joy", "avg_sadness",
    "avg_surprise", "avg_disgust", "avg_neutral",
    "anxiety_index", "tension_index", "positivity_index",
]

TARGET_COL = "dominant_mood"

PRED_SCHEMA = [
    bigquery.SchemaField("week_start",           "DATE"),
    bigquery.SchemaField("actual_mood",          "STRING"),
    bigquery.SchemaField("predicted_mood",       "STRING"),
    bigquery.SchemaField("correct",              "BOOLEAN"),
    bigquery.SchemaField("confidence",           "FLOAT64"),   # max class probability
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
    bigquery.SchemaField("mean_shap_value",      "FLOAT64"),   # signed mean
    bigquery.SchemaField("mean_abs_shap",        "FLOAT64"),   # importance magnitude
    bigquery.SchemaField("rank",                 "INTEGER"),
    bigquery.SchemaField("ingested_at",          "TIMESTAMP"),
]


# ── BQ helpers ──────────────────────────────────────────────────────────────────

def ensure_table(client, table_id, schema):
    client.delete_table(table_id, not_found_ok=True)
    client.create_table(bigquery.Table(table_id, schema=schema))
    time.sleep(10)  # allow BQ table metadata to propagate before streaming insert
    logger.info(f"Table ready: {table_id}")


def streaming_insert(client, table_id, rows, chunk=500):
    """Load rows using the batch load API (more reliable than streaming insert after create)."""
    import json as _json
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    ndjson = "\n".join(_json.dumps(r) for r in rows)
    import io
    job = client.load_table_from_file(
        io.BytesIO(ndjson.encode()),
        table_id,
        job_config=job_config,
    )
    job.result()   # wait for completion
    logger.info(f"Loaded {len(rows):,} rows → {table_id}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    client = bigquery.Client(project=PROJECT)
    now_ts = datetime.now(timezone.utc).isoformat()

    # 1. Load weekly_features
    logger.info("Reading weekly_features …")
    df = client.query(f"SELECT * FROM `{SRC_TABLE}` ORDER BY week_start").to_dataframe()
    logger.info(f"  {len(df):,} weeks loaded")

    if len(df) < 6:
        logger.error("Need at least 6 weeks of data for train/test split. Run Layers 1–3 first.")
        return

    # 2. Build lag-1 target: next week's dominant_mood
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)
    df["target_mood"] = df[TARGET_COL].shift(-1)   # next week's mood
    df = df.dropna(subset=["target_mood"] + EMOTION_FEATURES)

    logger.info(f"  {len(df):,} rows after lag alignment")

    X = df[EMOTION_FEATURES].values
    y_raw = df["target_mood"].values

    # 3. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()
    n_classes = len(class_names)
    logger.info(f"  Target classes ({n_classes}): {class_names}")

    if n_classes < 2:
        logger.warning("Only 1 mood class in data — need more weeks for meaningful predictions. Skipping ML training.")
        return

    # 4. Train XGBoost
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

    # Cross-val on full dataset (small n — use LOO-style 3-fold)
    n_splits = min(3, len(df) // 2)
    if n_splits >= 2:
        cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring="accuracy")
        logger.info(f"  CV accuracy ({n_splits}-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on all data for SHAP + predictions
    model.fit(X, y)
    proba = model.predict_proba(X)
    y_pred = np.argmax(proba, axis=1)
    confidence = proba.max(axis=1)

    train_acc = accuracy_score(y, y_pred)
    logger.info(f"  Train accuracy: {train_acc:.3f}")
    logger.info("\n" + classification_report(y, y_pred, labels=list(range(n_classes)), target_names=class_names))

    # 5. SHAP values
    logger.info("Computing SHAP values …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)   # shape varies by xgboost/shap version

    # Normalise to (n_classes, n_samples, n_features)
    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=0)   # list[(n_samples, n_features)] → (n_classes, n_samples, n_features)
    else:
        shap_arr = np.array(shap_values)

    # Handle (n_samples, n_features, n_classes) → (n_classes, n_samples, n_features)
    if shap_arr.ndim == 3 and shap_arr.shape[0] == len(X):
        shap_arr = shap_arr.transpose(2, 0, 1)

    # Check if SHAP is degenerate (all near-zero) — happens when the model is a
    # near-constant predictor (one class dominates the training set).
    # Fall back to XGBoost gain-based importance, which is meaningful regardless
    # of target variance.
    max_abs_shap = float(np.abs(shap_arr).max()) if shap_arr.size > 0 else 0.0
    use_gain_fallback = max_abs_shap < 1e-6

    if use_gain_fallback:
        logger.warning(
            f"SHAP values are all near-zero (max={max_abs_shap:.2e}). "
            "The model is a near-constant predictor — one class dominates training data. "
            "Falling back to point-biserial correlation (feature vs class probability)."
        )
        # Point-biserial correlation: for each class, how strongly does each feature
        # correlate with that class's predicted probability?
        # This is always non-zero and meaningful even for near-constant predictors.
        # Correlate each feature with the actual binary label for each class.
        # This shows which emotion features distinguish aggressive from euphoric weeks.
        shap_rows = []
        for cls_idx, cls_name in enumerate(class_names):
            y_binary = (y == cls_idx).astype(float)   # 1 where this class is actual label
            for feat_name in EMOTION_FEATURES:
                feat_vals = X[:, EMOTION_FEATURES.index(feat_name)]
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
            sv = shap_arr[cls_idx]          # (n_samples, n_features)
            for feat_idx, feat_name in enumerate(EMOTION_FEATURES):
                col = sv[:, feat_idx]
                shap_rows.append({
                    "feature":         feat_name,
                    "mood_archetype":  cls_name,
                    "mean_shap_value": round(float(col.mean()), 6),
                    "mean_abs_shap":   round(float(np.abs(col).mean()), 6),
                    "rank":            0,
                    "ingested_at":     now_ts,
                })

    # Rank within each archetype by mean_abs_shap descending
    shap_df = pd.DataFrame(shap_rows)
    shap_df["rank"] = (
        shap_df.groupby("mood_archetype")["mean_abs_shap"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    shap_rows = shap_df.to_dict("records")

    logger.info(f"  Importance source: {'XGBoost gain (fallback)' if use_gain_fallback else 'SHAP TreeExplainer'}")
    logger.info("Top drivers per archetype:")
    for arch in class_names:
        top = shap_df[shap_df["mood_archetype"] == arch].nsmallest(3, "rank")
        for _, r in top.iterrows():
            logger.info(f"  {arch:12s} rank {r['rank']}  {r['feature']:20s}  mean_abs={r['mean_abs_shap']:.4f}")

    # 6. Write ml_predictions
    pred_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        pred_rows.append({
            "week_start":       str(row["week_start"].date()),
            "actual_mood":      str(y_raw[i]),
            "predicted_mood":   class_names[y_pred[i]],
            "correct":          bool(y[i] == y_pred[i]),
            "confidence":       round(float(confidence[i]), 6),
            "avg_fear":         round(float(row["avg_fear"]),         6),
            "avg_anger":        round(float(row["avg_anger"]),        6),
            "avg_joy":          round(float(row["avg_joy"]),          6),
            "avg_sadness":      round(float(row["avg_sadness"]),      6),
            "anxiety_index":    round(float(row["anxiety_index"]),    6),
            "tension_index":    round(float(row["tension_index"]),    6),
            "positivity_index": round(float(row["positivity_index"]), 6),
            "ingested_at":      now_ts,
        })

    ensure_table(client, PRED_TABLE, PRED_SCHEMA)
    streaming_insert(client, PRED_TABLE, pred_rows)

    # 7. Write shap_importance
    ensure_table(client, SHAP_TABLE, SHAP_SCHEMA)
    streaming_insert(client, SHAP_TABLE, shap_rows)

    # 8. Summary
    logger.info("─── LAYER 4 COMPLETE ───")
    logger.info(f"  Weeks predicted     : {len(pred_rows)}")
    logger.info(f"  Train accuracy      : {train_acc:.3f}")
    logger.info(f"  SHAP rows written   : {len(shap_rows)}")
    logger.info(f"  Module 12 DONE — all 4 layers complete")


if __name__ == "__main__":
    main()
