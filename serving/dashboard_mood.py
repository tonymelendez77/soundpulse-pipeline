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
    page_icon="SP",
    layout="wide",
)

# ── custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}
code, pre, .stCode, .stDataFrame th {
    font-family: 'IBM Plex Mono', monospace !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
.stTabs [data-baseweb="tab-list"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    border-bottom: 2px solid #e85d26 !important;
    color: #e85d26 !important;
}
div.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    border: 1px solid #e85d26 !important;
    color: #e85d26 !important;
    background: transparent !important;
    border-radius: 2px !important;
    padding: 6px 18px !important;
    transition: all 0.15s ease !important;
}
div.stButton > button:hover {
    background: #e85d26 !important;
    color: #0f1117 !important;
}
[data-testid="stMetric"] {
    background: #1a1d27;
    padding: 16px;
    border-left: 3px solid #e85d26;
    border-radius: 0;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.stExpander {
    border: 1px solid #2a2d37 !important;
    border-radius: 0 !important;
}
div[data-testid="stSidebar"] {
    border-right: 1px solid #2a2d37;
}
</style>
""", unsafe_allow_html=True)

_APP_VERSION = "2026-04-14-v2"

REGIONS = ["global", "north_america", "latin_america"]
REGION_LABELS = {
    "global": "Global",
    "north_america": "N. America",
    "latin_america": "Latin America",
    "europe": "Europe",
}

# ── palette: warm earth + neon ────────────────────────────────────────
MOOD_COLORS = {
    "euphoric": "#f0c246",
    "melancholic": "#4a7fb5",
    "aggressive": "#d64045",
    "peaceful": "#3cba8a",
    "groovy": "#a855f7",
}

SENTIMENT_COLORS = {
    "anxiety_index": "#d64045",
    "tension_index": "#e8973e",
    "positivity_index": "#3cba8a",
}


@st.cache_data(ttl=300)
def fetch(endpoint: str, params: dict = None) -> list[dict]:
    resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def load_df(endpoint: str, params: dict = None) -> pd.DataFrame:
    return pd.DataFrame(fetch(endpoint, params))


# ── sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("SOUNDPULSE")
    st.caption("Mood Intelligence")
    st.divider()

    selected_region = st.selectbox(
        "Region",
        options=REGIONS,
        format_func=lambda r: REGION_LABELS.get(r, r),
    )

    if st.button("Reload", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Synced {datetime.now().strftime('%H:%M:%S')}  //  {_APP_VERSION}")
    st.divider()

    mood_options = ["All", "aggressive", "euphoric", "melancholic", "peaceful", "groovy"]
    selected_mood = st.selectbox("SHAP mood filter", mood_options)

    st.divider()
    st.markdown(
        "**Sources**\n"
        "- `emotion_music_correlation`\n"
        "- `weekly_features`\n"
        "- `shap_importance`\n"
        "- `fct_mood_prediction_summary`"
    )


# ── main content ──────────────────────────────────────────────────────
st.title("Mood Intelligence")
st.markdown(
    f"**{REGION_LABELS.get(selected_region, selected_region)}** / "
    "Mapping event sentiment to listener behavior."
)
st.divider()

tab_corr, tab_timeline, tab_shap, tab_pred = st.tabs([
    "CORRELATION",
    "TIMELINE",
    "SHAP",
    "PREDICTIONS",
])


with tab_corr:
    st.subheader("Event Emotion vs. Music Mood")
    try:
        corr_df = load_df("/correlation", {"region": selected_region})
        corr_df["pearson_r"] = pd.to_numeric(corr_df["pearson_r"], errors="coerce")

        pivot = corr_df.pivot(index="emotion", columns="mood_archetype", values="pearson_r")
        pivot.index = pivot.index.str.replace("avg_", "").str.replace("_", " ").str.title()
        pivot.columns = pivot.columns.str.title()

        fig = px.imshow(
            pivot, color_continuous_scale="Inferno", zmin=-1, zmax=1,
            text_auto=".2f", aspect="auto", labels={"color": "Pearson r"},
        )
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=30, b=0),
                          font=dict(family="Space Grotesk"))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            notable_count = corr_df["notable"].sum() if "notable" in corr_df.columns else 0
            st.metric("NOTABLE  (|r| >= 0.3)", int(notable_count))
        with col2:
            if not corr_df.empty:
                strongest = corr_df.loc[corr_df["pearson_r"].abs().idxmax()]
                st.metric("STRONGEST LINK", f"{strongest['emotion']} -> {strongest['mood_archetype']}",
                          f"r = {float(strongest['pearson_r']):.3f}")

        with st.expander("View raw data"):
            st.dataframe(corr_df.sort_values("pearson_r", ascending=False), use_container_width=True)
    except Exception as e:
        st.error(f"Could not load correlation data: {e}")


with tab_timeline:
    st.subheader("Sentiment + Mood Over Time")
    try:
        tl_df = load_df("/timeline", {"region": selected_region})
        tl_df["week_start"] = pd.to_datetime(tl_df["week_start"])
        for col in ["anxiety_index", "tension_index", "positivity_index"]:
            tl_df[col] = pd.to_numeric(tl_df[col], errors="coerce")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for idx_col, color in SENTIMENT_COLORS.items():
            label = idx_col.replace("_index", "").title()
            fig.add_trace(
                go.Scatter(x=tl_df["week_start"], y=tl_df[idx_col],
                           name=label, line=dict(color=color, width=2),
                           mode="lines+markers", marker=dict(size=4)),
                secondary_y=False,
            )

        mood_map = {"euphoric": 4, "melancholic": 3, "aggressive": 2, "peaceful": 1, "groovy": 0}
        tl_df["mood_numeric"] = tl_df["dominant_mood"].map(mood_map)

        fig.add_trace(
            go.Scatter(x=tl_df["week_start"], y=tl_df["mood_numeric"],
                       name="Dominant Mood", mode="markers",
                       marker=dict(size=10,
                                   color=[MOOD_COLORS.get(m, "#555") for m in tl_df["dominant_mood"]],
                                   line=dict(width=1, color="#0f1117")),
                       text=tl_df["dominant_mood"]),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="Sentiment Index", secondary_y=False)
        fig.update_yaxes(title_text="Mood", secondary_y=True,
                         tickvals=list(mood_map.values()), ticktext=list(mood_map.keys()),
                         range=[-0.5, 4.5])
        fig.update_layout(height=480, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
                          font=dict(family="Space Grotesk"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load timeline: {e}")


with tab_shap:
    st.subheader("Feature Importance  //  SHAP")
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
            color_discrete_map=MOOD_COLORS,
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0),
                          font=dict(family="Space Grotesk"))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View raw data"):
            st.dataframe(shap_df.drop(columns=["feature_label"]), use_container_width=True)
    except Exception as e:
        st.error(f"Could not load SHAP data: {e}")


with tab_pred:
    st.subheader("Prediction Scorecard")
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
            c1.metric("WEEKS EVALUATED", int(summary["total_weeks"]))
            c2.metric("CORRECT CALLS", int(summary["correct_predictions"]))

        st.divider()
        st.markdown("**Validated History**")
        display = history_df[["week_start", "predicted_mood", "actual_mood", "correct", "confidence"]].copy()
        display["confidence"] = display["confidence"].map(lambda x: f"{float(x):.1%}" if pd.notna(x) else "")

        def _style(row):
            if row["correct"] == True:
                color = "rgba(60, 186, 138, 0.18)"
            elif row["correct"] == False:
                color = "rgba(214, 64, 69, 0.18)"
            else:
                color = ""
            return [f"background-color: {color}"] * len(row)

        st.dataframe(display.style.apply(_style, axis=1), use_container_width=True, hide_index=True)

        if not forward_df.empty:
            st.divider()
            st.markdown("**Pending Validation**")
            fwd_display = forward_df[["week_start", "predicted_mood", "confidence"]].copy()
            fwd_display["confidence"] = fwd_display["confidence"].map(lambda x: f"{float(x):.1%}" if pd.notna(x) else "")
            st.dataframe(fwd_display, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
