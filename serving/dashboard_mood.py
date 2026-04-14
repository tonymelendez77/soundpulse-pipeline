"""
SoundPulse Dashboard 1: Mood Intelligence
Correlation heatmap, sentiment timeline, SHAP importance, prediction scorecard.
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

_APP_VERSION = "2026-04-14-v2"

REGIONS = ["global", "north_america", "latin_america"]
REGION_LABELS = {
    "global": "🌐 Global",
    "north_america": "🇺🇸 N. America",
    "latin_america": "🌎 Latin America",
    "europe": "🌍 Europe",
}


@st.cache_data(ttl=300)
def fetch(endpoint: str, params: dict = None) -> list[dict]:
    resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def load_df(endpoint: str, params: dict = None) -> pd.DataFrame:
    return pd.DataFrame(fetch(endpoint, params))


with st.sidebar:
    st.title("🎵 SoundPulse")
    st.caption("Mood Intelligence")
    st.divider()

    selected_region = st.selectbox(
        "Region",
        options=REGIONS,
        format_func=lambda r: REGION_LABELS.get(r, r),
    )

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}  ·  {_APP_VERSION}")
    st.divider()

    mood_options = ["All", "aggressive", "euphoric", "melancholic", "peaceful", "groovy"]
    selected_mood = st.selectbox("SHAP: filter by mood", mood_options)

    st.divider()
    st.markdown(
        "**Data sources**\n"
        "- `emotion_music_correlation`\n"
        "- `weekly_features`\n"
        "- `shap_importance`\n"
        "- `fct_mood_prediction_summary`"
    )


st.title("🧠 Mood Intelligence")
st.markdown(
    f"Region: **{REGION_LABELS.get(selected_region, selected_region)}** — "
    "How world events shape the music people listen to."
)
st.divider()

tab_corr, tab_timeline, tab_shap, tab_pred = st.tabs([
    "📊 Correlation",
    "📈 Timeline",
    "🔬 SHAP",
    "🎯 Predictions",
])


with tab_corr:
    st.subheader("News Emotion → Music Mood Correlation")
    try:
        corr_df = load_df("/correlation", {"region": selected_region})
        corr_df["pearson_r"] = pd.to_numeric(corr_df["pearson_r"], errors="coerce")

        pivot = corr_df.pivot(index="emotion", columns="mood_archetype", values="pearson_r")
        pivot.index = pivot.index.str.replace("avg_", "").str.replace("_", " ").str.title()
        pivot.columns = pivot.columns.str.title()

        fig = px.imshow(
            pivot, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=".2f", aspect="auto", labels={"color": "Pearson r"},
        )
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            notable_count = corr_df["notable"].sum() if "notable" in corr_df.columns else 0
            st.metric("Notable (|r| >= 0.3)", int(notable_count))
        with col2:
            if not corr_df.empty:
                strongest = corr_df.loc[corr_df["pearson_r"].abs().idxmax()]
                st.metric("Strongest", f"{strongest['emotion']} -> {strongest['mood_archetype']}",
                          f"r = {float(strongest['pearson_r']):.3f}")

        with st.expander("Raw data"):
            st.dataframe(corr_df.sort_values("pearson_r", ascending=False), use_container_width=True)
    except Exception as e:
        st.error(f"Could not load correlation data: {e}")


with tab_timeline:
    st.subheader("Sentiment vs. Music Mood Over Time")
    try:
        tl_df = load_df("/timeline", {"region": selected_region})
        tl_df["week_start"] = pd.to_datetime(tl_df["week_start"])
        for col in ["anxiety_index", "tension_index", "positivity_index"]:
            tl_df[col] = pd.to_numeric(tl_df[col], errors="coerce")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        colors = {"anxiety_index": "#e63946", "tension_index": "#f4a261", "positivity_index": "#2a9d8f"}
        for idx_col, color in colors.items():
            label = idx_col.replace("_index", "").title()
            fig.add_trace(
                go.Scatter(x=tl_df["week_start"], y=tl_df[idx_col],
                           name=label, line=dict(color=color, width=2),
                           mode="lines+markers", marker=dict(size=4)),
                secondary_y=False,
            )

        mood_map = {"euphoric": 4, "melancholic": 3, "aggressive": 2, "peaceful": 1, "groovy": 0}
        mood_colors = {"euphoric": "#ffd166", "melancholic": "#118ab2", "aggressive": "#ef476f",
                       "peaceful": "#06d6a0", "groovy": "#8338ec"}
        tl_df["mood_numeric"] = tl_df["dominant_mood"].map(mood_map)

        fig.add_trace(
            go.Scatter(x=tl_df["week_start"], y=tl_df["mood_numeric"],
                       name="Dominant Mood", mode="markers",
                       marker=dict(size=10,
                                   color=[mood_colors.get(m, "#aaa") for m in tl_df["dominant_mood"]],
                                   line=dict(width=1, color="white")),
                       text=tl_df["dominant_mood"]),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="Sentiment Index", secondary_y=False)
        fig.update_yaxes(title_text="Mood", secondary_y=True,
                         tickvals=list(mood_map.values()), ticktext=list(mood_map.keys()),
                         range=[-0.5, 4.5])
        fig.update_layout(height=480, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load timeline: {e}")


with tab_shap:
    st.subheader("Feature Importance (SHAP)")
    try:
        params = {"region": selected_region}
        if selected_mood != "All":
            params["mood"] = selected_mood
        shap_df = load_df("/shap", params)
        shap_df["mean_abs_shap"] = pd.to_numeric(shap_df["mean_abs_shap"], errors="coerce")
        shap_df["feature_label"] = (
            shap_df["feature"].str.replace("avg_", "").str.replace("_", " ").str.title()
        )

        # top 10 features
        top_features = (shap_df.groupby("feature")["mean_abs_shap"].max()
                        .nlargest(10).index.tolist())
        shap_df = shap_df[shap_df["feature"].isin(top_features)]
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=True)

        fig = px.bar(
            shap_df, x="mean_abs_shap", y="feature_label",
            color="mood_archetype", orientation="h", barmode="group",
            labels={"mean_abs_shap": "Mean |SHAP|", "feature_label": "Feature", "mood_archetype": "Mood"},
            color_discrete_map={
                "euphoric": "#ffd166", "melancholic": "#118ab2",
                "aggressive": "#ef476f", "peaceful": "#06d6a0", "groovy": "#8338ec",
            },
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw data"):
            st.dataframe(shap_df.drop(columns=["feature_label"]), use_container_width=True)
    except Exception as e:
        st.error(f"Could not load SHAP data: {e}")


with tab_pred:
    st.subheader("Prediction History")
    try:
        pred_df = load_df("/predictions", {"region": selected_region})
        pred_df["confidence"] = pd.to_numeric(pred_df["confidence"], errors="coerce")
        pred_df["overall_accuracy"] = pd.to_numeric(pred_df["overall_accuracy"], errors="coerce")
        pred_df["avg_confidence"] = pd.to_numeric(pred_df["avg_confidence"], errors="coerce")

        # Separate validated history from forward (unvalidated) predictions
        if "is_forward" in pred_df.columns:
            history_df = pred_df[pred_df["is_forward"] != True].copy()
            forward_df = pred_df[pred_df["is_forward"] == True].copy()
        else:
            history_df = pred_df[pred_df["correct"].notna()].copy()
            forward_df = pred_df[pred_df["correct"].isna()].copy()

        if not pred_df.empty:
            summary = pred_df.iloc[-1]
            c1, c2 = st.columns(2)
            c1.metric("Weeks Evaluated", int(summary["total_weeks"]))
            c2.metric("Correct Predictions", int(summary["correct_predictions"]))

        st.divider()
        st.markdown("**Per-Week Prediction History**")
        display = history_df[["week_start", "predicted_mood", "actual_mood", "correct", "confidence"]].copy()
        display["confidence"] = display["confidence"].map(lambda x: f"{float(x):.1%}" if pd.notna(x) else "")

        def _style(row):
            color = "#d4edda" if row["correct"] == True else "#f8d7da" if row["correct"] == False else ""
            return [f"background-color: {color}"] * len(row)

        st.dataframe(display.style.apply(_style, axis=1), use_container_width=True, hide_index=True)

        if not forward_df.empty:
            st.divider()
            st.markdown("**Current Predictions (awaiting validation)**")
            fwd_display = forward_df[["week_start", "predicted_mood", "confidence"]].copy()
            fwd_display["confidence"] = fwd_display["confidence"].map(lambda x: f"{float(x):.1%}" if pd.notna(x) else "")
            st.dataframe(fwd_display, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
