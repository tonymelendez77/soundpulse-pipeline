"""
SoundPulse — Module 14
Dashboard 2: "Music Trends"

Story: How music DNA shifts across platforms and time.

Run:
    streamlit run serving/dashboard_trends.py --server.port 8502
"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import datetime

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="SoundPulse — Music Trends",
    page_icon="🎵",
    layout="wide",
)

MOOD_COLORS = {
    "euphoric":   "#ffd166",
    "melancholic": "#118ab2",
    "aggressive": "#ef476f",
    "peaceful":   "#06d6a0",
    "groovy":     "#8338ec",
}

MOOD_COLS = ["euphoric_pct", "melancholic_pct", "aggressive_pct", "peaceful_pct", "groovy_pct"]
MOOD_LABELS = ["Euphoric", "Melancholic", "Aggressive", "Peaceful", "Groovy"]


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
    st.caption("Module 14 — Music Trends")
    st.divider()

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    # Topic filter for news sentiment panel
    try:
        sent_raw = df("/news-sentiment")
        available_topics = sorted(sent_raw["topic"].dropna().unique().tolist()) if not sent_raw.empty else []
    except Exception:
        available_topics = []

    selected_topics = st.multiselect(
        "News sentiment: topics",
        options=available_topics,
        default=available_topics[:5] if available_topics else [],
    )

    st.divider()
    st.markdown(
        "**Data sources**\n"
        "- `stg_audio_mood_weekly`\n"
        "- `stg_news_sentiment_weekly`\n"
        "- `stg_generated_tracks`"
    )


# ── Page header ───────────────────────────────────────────────────────────────

st.title("🎵 Music Trends")
st.markdown(
    "Audio mood distributions across chart sources, sentiment shifts by news topic, "
    "and the AI-generated track that captured this week's sound."
)
st.divider()

tab_radar, tab_timeline, tab_sentiment, tab_gen = st.tabs([
    "🕸️ Mood Radar",
    "📈 Mood Timeline",
    "📰 News Sentiment",
    "🤖 AI Generation Log",
])


# ── Panel A — Cross-Platform Mood Radar ──────────────────────────────────────

with tab_radar:
    st.subheader("Mood Distribution by Chart Source (Latest Week)")
    st.caption(
        "Spider chart of the five mood archetype percentages for each chart source "
        "in the most recent available week."
    )
    try:
        mw_df = df("/mood-weekly")
        for col in MOOD_COLS:
            mw_df[col] = pd.to_numeric(mw_df[col], errors="coerce")

        latest_week = mw_df["week_start"].max()
        latest_df = mw_df[mw_df["week_start"] == latest_week]

        fig = go.Figure()
        for _, row in latest_df.iterrows():
            values = [float(row[c]) for c in MOOD_COLS] + [float(row[MOOD_COLS[0]])]  # close loop
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=MOOD_LABELS + [MOOD_LABELS[0]],
                fill="toself",
                name=str(row["chart_source"]),
                opacity=0.6,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=480,
            title=f"Week of {latest_week}",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Mood % breakdown table")
        display_cols = ["chart_source", "dominant_mood", "track_count"] + MOOD_COLS
        st.dataframe(
            latest_df[display_cols].rename(columns={c: c.replace("_pct", "") for c in MOOD_COLS}),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.error(f"Could not load mood radar: {e}")


# ── Panel B — Mood Timeline by Source ────────────────────────────────────────

with tab_timeline:
    st.subheader("Dominant Mood Trend by Chart Source")
    st.caption(
        "Which mood archetype led each chart source week by week. "
        "Gaps indicate weeks with no data for that source."
    )
    try:
        mw_df2 = df("/mood-weekly")
        mw_df2["week_start"] = pd.to_datetime(mw_df2["week_start"])

        fig = px.line(
            mw_df2,
            x="week_start",
            y="dominant_mood",
            color="chart_source",
            markers=True,
            labels={"week_start": "Week", "dominant_mood": "Dominant Mood", "chart_source": "Source"},
            category_orders={"dominant_mood": list(MOOD_COLORS.keys())},
        )
        fig.update_layout(height=420, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Stacked area of avg_energy per source over time
        st.subheader("Average Audio Energy by Source Over Time")
        mw_df2["avg_energy"] = pd.to_numeric(mw_df2["avg_energy"], errors="coerce")
        fig2 = px.area(
            mw_df2,
            x="week_start",
            y="avg_energy",
            color="chart_source",
            labels={"avg_energy": "Avg Energy (0–1)", "week_start": "Week", "chart_source": "Source"},
        )
        fig2.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load mood timeline: {e}")


# ── Panel C — News Sentiment by Topic ────────────────────────────────────────

with tab_sentiment:
    st.subheader("News Sentiment by Topic Over Time")
    st.caption(
        "Weekly average emotion scores from DistilRoBERTa across different news topics. "
        "Use the sidebar to filter topics."
    )
    try:
        sent_df = df("/news-sentiment")
        sent_df["week_start"] = pd.to_datetime(sent_df["week_start"])
        for col in ["avg_joy", "avg_fear", "avg_anger", "anxiety_index", "positivity_index"]:
            sent_df[col] = pd.to_numeric(sent_df[col], errors="coerce")

        if selected_topics:
            sent_df = sent_df[sent_df["topic"].isin(selected_topics)]

        if sent_df.empty:
            st.info("No data for selected topics.")
        else:
            # Dominant emotion heatmap (week × topic)
            pivot_emotion = sent_df.pivot_table(
                index="topic", columns="week_start", values="avg_joy", aggfunc="mean"
            )
            fig_heat = px.imshow(
                pivot_emotion,
                color_continuous_scale="YlGn",
                aspect="auto",
                labels={"color": "Avg Joy"},
                title="Joy Score by Topic × Week",
            )
            fig_heat.update_layout(height=320, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_heat, use_container_width=True)

            # Anxiety index line chart
            fig_anx = px.line(
                sent_df,
                x="week_start",
                y="anxiety_index",
                color="topic",
                markers=True,
                labels={"anxiety_index": "Anxiety Index", "week_start": "Week", "topic": "Topic"},
                title="Anxiety Index by Topic Over Time",
            )
            fig_anx.update_layout(height=360, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_anx, use_container_width=True)

            # Dominant emotion bar
            emotion_counts = (
                sent_df.groupby(["topic", "dominant_emotion"])
                .size()
                .reset_index(name="weeks")
            )
            fig_bar = px.bar(
                emotion_counts,
                x="topic",
                y="weeks",
                color="dominant_emotion",
                barmode="stack",
                labels={"weeks": "Weeks as Dominant Emotion", "topic": "Topic"},
                title="Most Common Dominant Emotion by Topic",
            )
            fig_bar.update_layout(height=360, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load sentiment data: {e}")


# ── Panel D — MusicGen Generation Log ────────────────────────────────────────

with tab_gen:
    st.subheader("AI-Generated Music Log")
    st.caption(
        "MusicGen (facebook/musicgen-small) generates a 10-second audio clip each run, "
        "informed by the XGBoost-predicted mood and the 10 most sonically similar real tracks."
    )
    try:
        gen_data = fetch("/generated-tracks")

        if not gen_data:
            st.info("No generated tracks yet. Run `python ingestion/music_generation.py` first.")
        else:
            # Latest generation card
            latest = gen_data[0]
            mood = latest.get("mood_archetype", "—")
            mood_color = MOOD_COLORS.get(mood, "#888")

            st.markdown(f"""
            <div style="border-left: 5px solid {mood_color}; padding: 12px 18px; background: #1e1e1e; border-radius: 6px; margin-bottom: 16px;">
                <h3 style="margin:0; color:{mood_color};">🎶 {mood.upper()} — Week of {latest.get('week_start','?')}</h3>
                <p style="color:#ccc; margin:8px 0 4px;"><strong>Prompt:</strong> {latest.get('prompt_text','—')}</p>
                <p style="color:#888; font-size:0.85em;">
                    📁 {latest.get('audio_gcs_path','—')}<br>
                    ⏱ {float(latest.get('duration_seconds',0)):.1f}s &nbsp;|&nbsp;
                    🕒 {latest.get('generated_at','—')}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Similar tracks
            similar_json = latest.get("similar_tracks_json")
            if similar_json:
                try:
                    similar = json.loads(similar_json)
                    st.markdown("**10 Most Similar Tracks Used for Prompt**")
                    sim_df = pd.DataFrame(similar)
                    if "score" in sim_df.columns:
                        sim_df["score"] = sim_df["score"].map(lambda x: f"{float(x):.3f}")
                    st.dataframe(sim_df, use_container_width=True, hide_index=True)
                except Exception:
                    st.text(similar_json)

            # Full log table
            st.divider()
            st.markdown("**All Generations**")
            log_df = pd.DataFrame(gen_data)
            display_cols = ["week_start", "mood_archetype", "duration_seconds", "generated_at", "audio_gcs_path"]
            available = [c for c in display_cols if c in log_df.columns]
            # Truncate generation_id for readability
            if "generation_id" in log_df.columns:
                log_df["id"] = log_df["generation_id"].str[:8] + "..."
                available = ["id"] + available
            st.dataframe(log_df[available], use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Could not load generation log: {e}")
