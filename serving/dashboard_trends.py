"""
SoundPulse Dashboard 2: Music Trends
Mood radar by region, mood timeline, news sentiment heatmap, AI generation log.
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import datetime

API_BASE = "https://soundpulse-pipeline.onrender.com"

st.set_page_config(
    page_title="SoundPulse — Music Trends",
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

MOOD_COLORS = {
    "euphoric": "#f0c246",
    "melancholic": "#4a7fb5",
    "aggressive": "#d64045",
    "peaceful": "#3cba8a",
    "groovy": "#a855f7",
}

MOOD_COLS = ["euphoric_pct", "melancholic_pct", "aggressive_pct", "peaceful_pct", "groovy_pct"]
MOOD_LABELS = ["Euphoric", "Melancholic", "Aggressive", "Peaceful", "Groovy"]

REGIONS = ["global", "north_america", "latin_america"]
REGION_LABELS = {
    "global": "Global",
    "north_america": "N. America",
    "latin_america": "Latin America",
    "europe": "Europe",
}
REGION_COLORS = {
    "global": "#e85d26",
    "north_america": "#f0c246",
    "latin_america": "#d64045",
    "europe": "#3cba8a",
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
    st.caption("Music Trends")
    st.divider()

    selected_region = st.selectbox(
        "Region",
        options=REGIONS,
        format_func=lambda r: REGION_LABELS.get(r, r),
    )

    if st.button("Reload", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Synced {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    try:
        sent_raw = load_df("/news-sentiment")
        available_topics = sorted(sent_raw["topic"].dropna().unique().tolist()) if not sent_raw.empty else []
    except Exception:
        available_topics = []

    selected_topics = st.multiselect(
        "News topics",
        options=available_topics,
        default=available_topics[:5] if available_topics else [],
    )


# ── main content ──────────────────────────────────────────────────────
st.title("Music Trends")
st.markdown(
    f"**{REGION_LABELS.get(selected_region, selected_region)}** / "
    "Mood distributions, sentiment shifts, generated tracks."
)
st.divider()

tab_radar, tab_timeline, tab_sentiment, tab_gen = st.tabs([
    "MOOD RADAR",
    "MOOD TIMELINE",
    "NEWS SENTIMENT",
    "GENERATION LOG",
])


with tab_radar:
    st.subheader("Mood Distribution  //  Latest Week")
    st.caption("One polygon per region — cross-regional mood comparison.")
    try:
        mr_df = load_df("/mood-regional")
        for col in MOOD_COLS:
            mr_df[col] = pd.to_numeric(mr_df[col], errors="coerce")

        latest_week = mr_df["week_start"].max()
        latest_df = mr_df[mr_df["week_start"] == latest_week]

        fig = go.Figure()
        for _, row in latest_df.iterrows():
            region = row["region"]
            values = [float(row[c]) for c in MOOD_COLS] + [float(row[MOOD_COLS[0]])]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=MOOD_LABELS + [MOOD_LABELS[0]],
                fill="toself",
                name=REGION_LABELS.get(region, region),
                line=dict(color=REGION_COLORS.get(region, "#555")),
                opacity=0.7,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
            height=500, title=f"Week of {latest_week}",
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Mood % breakdown")
        display_cols = ["region", "dominant_mood", "track_count"] + MOOD_COLS
        st.dataframe(
            latest_df[display_cols].rename(columns={c: c.replace("_pct", "") for c in MOOD_COLS}),
            use_container_width=True, hide_index=True,
        )
    except Exception as e:
        st.error(f"Could not load mood radar: {e}")


with tab_timeline:
    st.subheader("Dominant Mood Over Time")
    try:
        mr_df2 = load_df("/mood-regional", {"region": selected_region})
        mr_df2["week_start"] = pd.to_datetime(mr_df2["week_start"])

        fig = px.line(
            mr_df2, x="week_start", y="dominant_mood",
            markers=True,
            labels={"week_start": "Week", "dominant_mood": "Dominant Mood"},
            category_orders={"dominant_mood": list(MOOD_COLORS.keys())},
            color_discrete_sequence=[REGION_COLORS.get(selected_region, "#e85d26")],
        )
        fig.update_layout(height=400, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
                          font=dict(family="Space Grotesk"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Energy Over Time")
        mr_df2["avg_energy"] = pd.to_numeric(mr_df2["avg_energy"], errors="coerce")
        fig2 = px.area(
            mr_df2, x="week_start", y="avg_energy",
            labels={"avg_energy": "Avg Energy (0-1)", "week_start": "Week"},
            color_discrete_sequence=[REGION_COLORS.get(selected_region, "#e85d26")],
        )
        fig2.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0),
                           font=dict(family="Space Grotesk"))
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load mood timeline: {e}")


with tab_sentiment:
    st.subheader("News Sentiment by Topic")
    st.caption("Global aggregate — no regional breakdown.")
    try:
        sent_df = load_df("/news-sentiment")
        sent_df["week_start"] = pd.to_datetime(sent_df["week_start"])
        for col in ["avg_joy", "avg_fear", "avg_anger", "anxiety_index", "positivity_index"]:
            sent_df[col] = pd.to_numeric(sent_df[col], errors="coerce")

        if selected_topics:
            sent_df = sent_df[sent_df["topic"].isin(selected_topics)]

        if sent_df.empty:
            st.info("No data for selected topics.")
        else:
            pivot_emotion = sent_df.pivot_table(
                index="topic", columns="week_start", values="avg_joy", aggfunc="mean"
            )
            fig_heat = px.imshow(
                pivot_emotion, color_continuous_scale="Turbo",
                aspect="auto", labels={"color": "Avg Joy"},
                title="Joy Score  //  Topic x Week",
            )
            fig_heat.update_layout(height=320, margin=dict(l=0, r=0, t=50, b=0),
                                   font=dict(family="Space Grotesk"))
            st.plotly_chart(fig_heat, use_container_width=True)

            fig_anx = px.line(
                sent_df, x="week_start", y="anxiety_index",
                color="topic", markers=True,
                labels={"anxiety_index": "Anxiety Index", "week_start": "Week"},
                title="Anxiety Index  //  Topic",
            )
            fig_anx.update_layout(height=360, margin=dict(l=0, r=0, t=50, b=0),
                                  font=dict(family="Space Grotesk"))
            st.plotly_chart(fig_anx, use_container_width=True)

            emotion_counts = sent_df.groupby(["topic", "dominant_emotion"]).size().reset_index(name="weeks")
            fig_bar = px.bar(
                emotion_counts, x="topic", y="weeks",
                color="dominant_emotion", barmode="stack",
                title="Dominant Emotion  //  Topic",
            )
            fig_bar.update_layout(height=360, margin=dict(l=0, r=0, t=50, b=0),
                                  font=dict(family="Space Grotesk"))
            st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load sentiment data: {e}")


with tab_gen:
    st.subheader("Generation Log")
    try:
        gen_data = fetch("/generated-tracks", {"region": selected_region})

        if not gen_data:
            st.info("No generated tracks for this region yet.")
        else:
            latest = gen_data[0]
            mood = latest.get("mood_archetype", "")
            mood_color = MOOD_COLORS.get(mood, "#555")

            st.markdown(f"""
            <div style="border-left: 3px solid {mood_color}; padding: 14px 20px; background: #1a1d27; border-radius: 0; margin-bottom: 18px;">
                <h3 style="margin:0; color:{mood_color}; font-family:'Space Grotesk',sans-serif; letter-spacing:-0.02em;">{mood.upper()}  /  {latest.get('period', '')}  /  Week of {latest.get('week_start','?')}</h3>
                <p style="color:#b0b0b0; margin:10px 0 4px; font-family:'IBM Plex Mono',monospace; font-size:0.82rem;"><strong>Prompt:</strong> {latest.get('prompt_text','')}</p>
                <p style="color:#6b6b6b; font-size:0.78rem; font-family:'IBM Plex Mono',monospace;">
                    {float(latest.get('duration_seconds',0)):.1f}s &nbsp;&middot;&nbsp;
                    {latest.get('generated_at','')[:16]}
                </p>
            </div>
            """, unsafe_allow_html=True)

            similar_json = latest.get("similar_tracks_json")
            if similar_json:
                try:
                    similar = json.loads(similar_json)
                    st.markdown("**Similar Tracks**")
                    sim_df = pd.DataFrame(similar)
                    if "score" in sim_df.columns:
                        sim_df["score"] = sim_df["score"].map(lambda x: f"{float(x):.3f}")
                    st.dataframe(sim_df, use_container_width=True, hide_index=True)
                except Exception:
                    pass

            st.divider()
            st.markdown("**Full Log**")
            log_df = pd.DataFrame(gen_data)
            display_cols = ["period", "week_start", "mood_archetype", "duration_seconds", "generated_at"]
            available = [c for c in display_cols if c in log_df.columns]
            st.dataframe(log_df[available], use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not load generation log: {e}")
