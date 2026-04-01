"""
SoundPulse — Module 14
Dashboard 1: "Mood Intelligence"

Story: World events predict music mood — here's the proof.

Run:
    streamlit run serving/dashboard_mood.py --server.port 8501
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from datetime import datetime

API_BASE = "https://soundpulse-pipeline.onrender.com"

st.set_page_config(
    page_title="SoundPulse — Mood Intelligence",
    page_icon="🧠",
    layout="wide",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch(endpoint: str, params: dict = None) -> list[dict]:
    resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def df(endpoint: str, params: dict = None) -> pd.DataFrame:
    return pd.DataFrame(fetch(endpoint, params))


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎵 SoundPulse")
    st.caption("Module 14 — Mood Intelligence")
    st.divider()

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    # SHAP mood filter (used in Panel C)
    mood_options = ["All", "aggressive", "euphoric", "melancholic", "peaceful", "groovy"]
    selected_mood = st.selectbox("SHAP: filter by mood", mood_options)

    st.divider()
    st.markdown(
        "**Data sources**\n"
        "- `fct_emotion_music_correlation`\n"
        "- `stg_weekly_features`\n"
        "- `stg_shap_importance`\n"
        "- `fct_mood_prediction_summary`"
    )


# ── Page header ───────────────────────────────────────────────────────────────

st.title("🧠 Mood Intelligence")
st.markdown(
    "How world events shape the music people listen to — "
    "from news sentiment to XGBoost-predicted audio mood archetypes."
)
st.divider()

tab_corr, tab_timeline, tab_shap, tab_pred = st.tabs([
    "📊 Correlation Heatmap",
    "📈 World Events Timeline",
    "🔬 SHAP Importance",
    "🎯 Prediction Scorecard",
])


# ── Panel A — Correlation Heatmap ─────────────────────────────────────────────

with tab_corr:
    st.subheader("News Emotion → Music Mood Correlation (Pearson r)")
    st.caption(
        "Pearson r between weekly news emotion scores and audio mood archetype percentages. "
        "Notable cells (|r| ≥ 0.3) are highlighted with ★."
    )
    try:
        corr_df = df("/correlation")
        corr_df["pearson_r"] = pd.to_numeric(corr_df["pearson_r"], errors="coerce")

        pivot = corr_df.pivot(index="emotion", columns="mood_archetype", values="pearson_r")

        # Clean up axis labels
        pivot.index = pivot.index.str.replace("avg_", "").str.replace("_", " ").str.title()
        pivot.columns = pivot.columns.str.title()

        fig = px.imshow(
            pivot,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
            labels={"color": "Pearson r"},
        )

        # Overlay ★ on notable cells
        notable_df = corr_df[corr_df["notable"] == True]
        for _, row in notable_df.iterrows():
            emotion_label = row["emotion"].replace("avg_", "").replace("_", " ").title()
            mood_label = row["mood_archetype"].title()
            if emotion_label in pivot.index and mood_label in pivot.columns:
                col_idx = list(pivot.columns).index(mood_label)
                row_idx = list(pivot.index).index(emotion_label)
                fig.add_annotation(
                    x=col_idx, y=row_idx, text="★",
                    showarrow=False, font=dict(size=10, color="gold"),
                    xshift=12, yshift=12,
                )

        fig.update_layout(height=480, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            notable_count = corr_df["notable"].sum()
            st.metric("Notable correlations (|r| ≥ 0.3)", int(notable_count))
        with col2:
            strongest = corr_df.loc[corr_df["pearson_r"].abs().idxmax()]
            st.metric(
                "Strongest pair",
                f"{strongest['emotion']} → {strongest['mood_archetype']}",
                f"r = {float(strongest['pearson_r']):.3f}",
            )

        with st.expander("Raw data"):
            st.dataframe(corr_df.sort_values("pearson_r", ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"Could not load correlation data: {e}")


# ── Panel B — Timeline ────────────────────────────────────────────────────────

with tab_timeline:
    st.subheader("World Sentiment vs. Music Mood Over Time")
    st.caption(
        "Left axis: composite sentiment indices derived from DistilRoBERTa emotion scores. "
        "Right axis: dominant audio mood archetype (KMeans). "
        "Hover to compare."
    )
    try:
        tl_df = df("/timeline")
        tl_df["week_start"] = pd.to_datetime(tl_df["week_start"])
        for col in ["anxiety_index", "tension_index", "positivity_index"]:
            tl_df[col] = pd.to_numeric(tl_df[col], errors="coerce")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        colors = {"anxiety_index": "#e63946", "tension_index": "#f4a261", "positivity_index": "#2a9d8f"}
        for idx_col, color in colors.items():
            label = idx_col.replace("_index", "").title() + " Index"
            fig.add_trace(
                go.Scatter(
                    x=tl_df["week_start"], y=tl_df[idx_col],
                    name=label, line=dict(color=color, width=2),
                    mode="lines+markers", marker=dict(size=4),
                ),
                secondary_y=False,
            )

        # Dominant mood as colored scatter on right axis
        mood_map = {"euphoric": 4, "melancholic": 3, "aggressive": 2, "peaceful": 1, "groovy": 0}
        mood_colors = {"euphoric": "#ffd166", "melancholic": "#118ab2", "aggressive": "#ef476f",
                       "peaceful": "#06d6a0", "groovy": "#8338ec"}
        tl_df["mood_numeric"] = tl_df["dominant_mood"].map(mood_map)

        fig.add_trace(
            go.Scatter(
                x=tl_df["week_start"], y=tl_df["mood_numeric"],
                name="Dominant Mood", mode="markers+text",
                marker=dict(
                    size=10,
                    color=[mood_colors.get(m, "#aaa") for m in tl_df["dominant_mood"]],
                    line=dict(width=1, color="white"),
                ),
                text=tl_df["dominant_mood"],
                textposition="top center",
                textfont=dict(size=9),
            ),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="Sentiment Index (0–1)", secondary_y=False)
        fig.update_yaxes(
            title_text="Mood Archetype",
            secondary_y=True,
            tickvals=list(mood_map.values()),
            ticktext=list(mood_map.keys()),
            range=[-0.5, 4.5],
        )
        fig.update_xaxes(title_text="Week")
        fig.update_layout(height=480, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load timeline data: {e}")


# ── Panel C — SHAP ────────────────────────────────────────────────────────────

with tab_shap:
    st.subheader("What Drives the Mood Prediction? (SHAP Values)")
    st.caption(
        "Mean absolute SHAP values from the XGBoost model. "
        "Higher = stronger influence on predicting that mood archetype."
    )
    try:
        params = {"mood": selected_mood} if selected_mood != "All" else None
        shap_df = df("/shap", params)
        shap_df["mean_abs_shap"] = pd.to_numeric(shap_df["mean_abs_shap"], errors="coerce")
        shap_df["feature_label"] = (
            shap_df["feature"]
            .str.replace("avg_", "")
            .str.replace("_", " ")
            .str.title()
        )
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=True)

        fig = px.bar(
            shap_df,
            x="mean_abs_shap",
            y="feature_label",
            color="mood_archetype",
            orientation="h",
            barmode="group",
            labels={"mean_abs_shap": "Mean |SHAP|", "feature_label": "Feature", "mood_archetype": "Mood"},
            color_discrete_map={
                "euphoric": "#ffd166", "melancholic": "#118ab2",
                "aggressive": "#ef476f", "peaceful": "#06d6a0", "groovy": "#8338ec",
            },
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), legend_title="Mood Archetype")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw data"):
            st.dataframe(shap_df.drop(columns=["feature_label"]), use_container_width=True)

    except Exception as e:
        st.error(f"Could not load SHAP data: {e}")


# ── Panel D — Prediction Scorecard ────────────────────────────────────────────

with tab_pred:
    st.subheader("XGBoost Prediction History")
    st.caption("Lag-1 predictions: this week's news emotion → next week's dominant audio mood.")
    try:
        pred_df = df("/predictions")
        pred_df["confidence"] = pd.to_numeric(pred_df["confidence"], errors="coerce")
        pred_df["overall_accuracy"] = pd.to_numeric(pred_df["overall_accuracy"], errors="coerce")
        pred_df["avg_confidence"] = pd.to_numeric(pred_df["avg_confidence"], errors="coerce")

        if not pred_df.empty:
            summary = pred_df.iloc[-1]  # most recent row has model-level stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Accuracy", f"{float(summary['overall_accuracy']):.0%}")
            c2.metric("Avg Confidence", f"{float(summary['avg_confidence']):.0%}")
            c3.metric("Total Weeks", int(summary["total_weeks"]))
            c4.metric("Correct Predictions", int(summary["correct_predictions"]))

        st.divider()

        # Per-week table with colour
        display = pred_df[["week_start", "predicted_mood", "actual_mood", "correct", "confidence"]].copy()
        display["confidence"] = display["confidence"].map(lambda x: f"{float(x):.1%}" if pd.notna(x) else "—")

        def _style_correct(row):
            color = "#d4edda" if row["correct"] == True else "#f8d7da"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            display.style.apply(_style_correct, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.error(f"Could not load prediction data: {e}")
