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
    page_icon="🎵",
    layout="wide",
)

MOOD_COLORS = {
    "euphoric": "#ffd166",
    "melancholic": "#118ab2",
    "aggressive": "#ef476f",
    "peaceful": "#06d6a0",
    "groovy": "#8338ec",
}

MOOD_COLS = ["euphoric_pct", "melancholic_pct", "aggressive_pct", "peaceful_pct", "groovy_pct"]
MOOD_LABELS = ["Euphoric", "Melancholic", "Aggressive", "Peaceful", "Groovy"]

REGIONS = ["global", "north_america", "latin_america"]
REGION_LABELS = {
    "global": "🌐 Global",
    "north_america": "🇺🇸 N. America",
    "latin_america": "🌎 Latin America",
    "europe": "🌍 Europe",
}
REGION_COLORS = {
    "global": "#58a6ff",
    "north_america": "#ffd166",
    "latin_america": "#ef476f",
    "europe": "#06d6a0",
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
    st.caption("Music Trends")
    st.divider()

    selected_region = st.selectbox(
        "Region",
        options=REGIONS,
        format_func=lambda r: REGION_LABELS.get(r, r),
    )

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")
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


st.title("🎵 Music Trends")
st.markdown(
    f"Region: **{REGION_LABELS.get(selected_region, selected_region)}** — "
    "Audio mood distributions, sentiment shifts, and AI-generated tracks."
)
st.divider()

tab_radar, tab_timeline, tab_sentiment, tab_gen = st.tabs([
    "🕸️ Mood Radar",
    "📈 Mood Timeline",
    "📰 News Sentiment",
    "🤖 AI Generation Log",
])


with tab_radar:
    st.subheader("Mood Distribution (Latest Week) — All Regions")
    st.caption("One polygon per region. Compare how moods differ across the world.")
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
                line=dict(color=REGION_COLORS.get(region, "#888")),
                opacity=0.7,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
            height=500, title=f"Week of {latest_week}",
            margin=dict(l=40, r=40, t=60, b=40),
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
            color_discrete_sequence=[REGION_COLORS.get(selected_region, "#58a6ff")],
        )
        fig.update_layout(height=400, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Energy Over Time")
        mr_df2["avg_energy"] = pd.to_numeric(mr_df2["avg_energy"], errors="coerce")
        fig2 = px.area(
            mr_df2, x="week_start", y="avg_energy",
            labels={"avg_energy": "Avg Energy (0-1)", "week_start": "Week"},
            color_discrete_sequence=[REGION_COLORS.get(selected_region, "#58a6ff")],
        )
        fig2.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load mood timeline: {e}")


with tab_sentiment:
    st.subheader("News Sentiment by Topic")
    st.caption("News data is global (no regional split).")
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
                pivot_emotion, color_continuous_scale="YlGn",
                aspect="auto", labels={"color": "Avg Joy"},
                title="Joy Score by Topic x Week",
            )
            fig_heat.update_layout(height=320, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_heat, use_container_width=True)

            fig_anx = px.line(
                sent_df, x="week_start", y="anxiety_index",
                color="topic", markers=True,
                labels={"anxiety_index": "Anxiety Index", "week_start": "Week"},
                title="Anxiety Index by Topic",
            )
            fig_anx.update_layout(height=360, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_anx, use_container_width=True)

            emotion_counts = sent_df.groupby(["topic", "dominant_emotion"]).size().reset_index(name="weeks")
            fig_bar = px.bar(
                emotion_counts, x="topic", y="weeks",
                color="dominant_emotion", barmode="stack",
                title="Dominant Emotion by Topic",
            )
            fig_bar.update_layout(height=360, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load sentiment data: {e}")


with tab_gen:
    st.subheader("AI-Generated Music Log")
    try:
        gen_data = fetch("/generated-tracks", {"region": selected_region})

        if not gen_data:
            st.info("No generated tracks for this region yet.")
        else:
            latest = gen_data[0]
            mood = latest.get("mood_archetype", "")
            mood_color = MOOD_COLORS.get(mood, "#888")

            st.markdown(f"""
            <div style="border-left: 5px solid {mood_color}; padding: 12px 18px; background: #1e1e1e; border-radius: 6px; margin-bottom: 16px;">
                <h3 style="margin:0; color:{mood_color};">{mood.upper()} — {latest.get('period', '')} — Week of {latest.get('week_start','?')}</h3>
                <p style="color:#ccc; margin:8px 0 4px;"><strong>Prompt:</strong> {latest.get('prompt_text','')}</p>
                <p style="color:#888; font-size:0.85em;">
                    {float(latest.get('duration_seconds',0)):.1f}s &nbsp;|&nbsp;
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
            st.markdown("**All Generations**")
            log_df = pd.DataFrame(gen_data)
            display_cols = ["period", "week_start", "mood_archetype", "duration_seconds", "generated_at"]
            available = [c for c in display_cols if c in log_df.columns]
            st.dataframe(log_df[available], use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not load generation log: {e}")
