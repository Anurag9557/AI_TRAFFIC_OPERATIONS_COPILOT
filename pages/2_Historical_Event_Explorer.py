import sqlite3
import pandas as pd
import streamlit as st
import pydeck as pdk
import folium
from streamlit_folium import st_folium

from src.config import get_settings

st.set_page_config(
    page_title="Historical Event Explorer",
    page_icon="📚",
    layout="wide",
)

settings = get_settings()

st.title("📚 Historical Event Explorer")

@st.cache_data
def load_events():

    conn = sqlite3.connect(settings.database_path)

    query = """
        SELECT
            event_id,
            event_type,
            event_cause,
            corridor,
            priority,
            requires_road_closure,
            handling_duration_hours,
            start_datetime,
            police_station,
            latitude,
            longitude
        FROM events
        """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


df = load_events()

# -----------------------
# KPIs
# -----------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Events", len(df))

with c2:
    st.metric(
        "Unique Causes",
        df["event_cause"].nunique()
    )

with c3:
    st.metric(
        "Unique Corridors",
        df["corridor"].nunique()
    )

with c4:
    closure_rate = (
        df["requires_road_closure"]
        .fillna(0)
        .mean()
        * 100
    )

    st.metric(
        "Closure Rate %",
        f"{closure_rate:.1f}"
    )

st.divider()

# -----------------------
# Filters
# -----------------------

col1, col2, col3 = st.columns(3)

with col1:
    cause_filter = st.selectbox(
        "Event Cause",
        ["All"] + sorted(
            df["event_cause"]
            .dropna()
            .unique()
            .tolist()
        ),
        key="cause_filter",
    )

with col2:
    corridor_filter = st.selectbox(
        "Corridor",
        ["All"] + sorted(
            df["corridor"]
            .dropna()
            .unique()
            .tolist()
        ),
        key="corridor_filter",
    )

with col3:
    priority_filter = st.selectbox(
        "Priority",
        ["All"] + sorted(
            df["priority"]
            .dropna()
            .unique()
            .tolist()
        ),
        key="priority_filter",
    )

filtered = df.copy()

if cause_filter != "All":
    filtered = filtered[
        filtered["event_cause"] == cause_filter
    ]

if corridor_filter != "All":
    filtered = filtered[
        filtered["corridor"] == corridor_filter
    ]

if priority_filter != "All":
    filtered = filtered[
        filtered["priority"] == priority_filter
    ]

st.subheader("Filtered Statistics")

s1, s2, s3 = st.columns(3)

with s1:
    st.metric(
        "Filtered Events",
        len(filtered)
    )

with s2:

    avg_duration = (
        filtered["handling_duration_hours"]
        .dropna()
        .mean()
    )

    st.metric(
        "Avg Handling Hours",
        (
            round(avg_duration, 2)
            if pd.notna(avg_duration)
            else "-"
        )
    )

with s3:

    filtered_closure = (
        filtered["requires_road_closure"]
        .fillna(0)
        .mean()
        * 100
    )

    st.metric(
        "Closure %",
        f"{filtered_closure:.1f}"
    )

st.divider()

# -----------------------
# Cause Distribution
# -----------------------

st.subheader("Cause Distribution")

cause_counts = (
    filtered["event_cause"]
    .value_counts()
    .head(15)
)

st.bar_chart(cause_counts)

# -----------------------
# Priority Distribution
# -----------------------

st.subheader("Priority Distribution")

priority_counts = (
    filtered["priority"]
    .value_counts()
)

st.bar_chart(priority_counts)

# -----------------------
# Historical Heatmap
# -----------------------
st.subheader("🏆 Top 10 Traffic Corridors")

top_corridors = (
    filtered["corridor"]
    .value_counts()
    .head(10)
)

st.bar_chart(top_corridors)

st.subheader("🚓 Top 10 Police Stations")

top_ps = (
    filtered["police_station"]
    .value_counts()
    .head(10)
)

st.bar_chart(top_ps)
# -----------------------
# Bengaluru Event Map
# -----------------------

st.divider()

st.subheader("🗺 Bengaluru Historical Event Map")

map_df = filtered[
    [
        "event_id",
        "event_cause",
        "corridor",
        "priority",
        "latitude",
        "longitude",
    ]
].copy()

map_df = map_df.dropna()

map_df["latitude"] = pd.to_numeric(
    map_df["latitude"],
    errors="coerce"
)

map_df["longitude"] = pd.to_numeric(
    map_df["longitude"],
    errors="coerce"
)

map_df = map_df.dropna()

# FIX: building the Folium map (looping over every row to create a
# CircleMarker + popup) is the expensive part of this page. Caching it
# on the filtered dataframe means it only rebuilds when your filters
# actually change, not on every rerun.
@st.cache_data
def build_event_map(points: pd.DataFrame) -> folium.Map:

    center_lat = points["latitude"].mean()
    center_lon = points["longitude"].mean()

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="OpenStreetMap"
    )

    for _, row in points.iterrows():

        popup_text = f"""
        <b>Event ID:</b> {row['event_id']}<br>
        <b>Cause:</b> {row['event_cause']}<br>
        <b>Corridor:</b> {row['corridor']}<br>
        <b>Priority:</b> {row['priority']}
        """

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=1.5,
            color="red",
            fill=True,
            fill_opacity=0.7,
            popup=popup_text,
        ).add_to(fmap)

    return fmap


if len(map_df) > 0:

    m = build_event_map(map_df)

    # FIX: stable key + returned_objects=[] means panning, zooming, or
    # clicking a marker on the map does NOT trigger a Streamlit rerun.
    # Without this, every map interaction re-ran the entire page
    # (filters, charts, and the marker loop above).
    st_folium(
        m,
        width=1200,
        height=600,
        key="historical_event_map",
        returned_objects=[],
    )

    st.caption(
        "Historical traffic events plotted using latitude and longitude from the dataset."
    )

else:

    st.warning(
        "No valid coordinates available for selected filters."
    )

# -----------------------
# Event Table
# -----------------------

st.subheader("Historical Events")

st.dataframe(
    filtered,
    use_container_width=True,
    height=500,
)

csv = filtered.to_csv(index=False)

st.download_button(
    "Download Filtered Events",
    csv,
    file_name="historical_events.csv",
    mime="text/csv",
)