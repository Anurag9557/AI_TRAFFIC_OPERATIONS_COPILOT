import streamlit as st
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium


from src.graph import run_copilot

st.set_page_config(
    page_title="AI Traffic Operations Copilot",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 AI Traffic Operations Copilot")
st.caption(
    "Event-driven traffic planning using historical event intelligence"
)

# -------------------------
# Sidebar
# -------------------------

st.sidebar.header("Event Input")

event_description = st.sidebar.text_area(
    "Event Description",
    height=150,
    placeholder="Example: Large public gathering near MG Road expected from 6 PM to 10 PM",
    key="event_description",
)

# FIX: freeze "now" once per session instead of recomputing it on every
# rerun. datetime.now() changing every rerun was making these widgets'
# defaults shift, which can make them appear to "reset" whenever the
# map (or anything else) triggers a rerun.
if "default_datetime" not in st.session_state:
    st.session_state["default_datetime"] = datetime.now()

start_time = st.sidebar.datetime_input(
    "Start Date & Time",
    value=st.session_state["default_datetime"],
    key="start_time",
)

end_time = st.sidebar.datetime_input(
    "End Date & Time",
    value=st.session_state["default_datetime"],
    key="end_time",
)

event_type = st.sidebar.selectbox(
    "Event Type",
    ["Auto", "planned", "unplanned"],
    key="event_type",
)

event_cause = st.sidebar.text_input(
    "Event Cause",
    placeholder="construction / accident / public_event ...",
    key="event_cause",
)

corridor = st.sidebar.text_input(
    "Corridor",
    key="corridor",
)

latitude = st.sidebar.number_input(
    "Latitude (Optional)",
    value=0.0,
    format="%.6f",
    key="latitude",
)

longitude = st.sidebar.number_input(
    "Longitude (Optional)",
    value=0.0,
    format="%.6f",
    key="longitude",
)

priority = st.sidebar.selectbox(
    "Priority",
    ["Auto", "High", "Low"],
    key="priority",
)

road_closure = st.sidebar.selectbox(
    "Road Closure Required",
    ["Auto", True, False],
    key="road_closure",
)

generate = st.sidebar.button(
    "Generate Traffic Management Plan",
    use_container_width=True,
)

# -------------------------
# Run Graph
# -------------------------
# FIX: only call run_copilot when the button is actually pressed on THIS
# rerun. Previously, a sticky session_state flag meant every later rerun
# (e.g. triggered just by clicking/panning the folium map) re-ran the
# whole pipeline again. Now it runs once and the result is cached.

if generate:

    if not event_description.strip():
        st.error("Please enter an event description.")
        st.stop()

    form_data = {
        "event_type": event_type,
        "event_cause": event_cause,
        "corridor": corridor,
        "priority": priority,
        "requires_road_closure": road_closure,
        "start_datetime": start_time.isoformat(),
        "end_datetime": end_time.isoformat(),
        "latitude": latitude if latitude != 0 else None,
        "longitude": longitude if longitude != 0 else None,
    }

    with st.spinner("Running AI Traffic Operations Copilot..."):

        result = run_copilot(
            user_input=event_description,
            form_data=form_data,
        )

    st.session_state["result"] = result

# -------------------------
# Display (reads cached result only — no recomputation here)
# -------------------------

if "result" in st.session_state:

    result = st.session_state["result"]

    if result.get("validation_errors"):

        st.error("Validation Failed")

        for err in result["validation_errors"]:
            st.write(f"• {err}")

        st.stop()

    risk = result["risk_assessment"]
    resources = result["resource_plan"]

    if risk.band.lower() == "high":
        st.error("🔴 High Operational Risk")
    elif risk.band.lower() == "moderate":
        st.warning("🟡 Moderate Operational Risk")
    else:
        st.success("🟢 Low Operational Risk")

    # -------------------------
    # Metrics
    # -------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Risk Score",
            f"{risk.score}",
        )

    with c2:
        st.metric(
            "Risk Band",
            risk.band,
        )

    with c3:
        st.metric(
            "Confidence",
            f"{risk.confidence*100:.0f}%"
        )

    st.caption(
        "Operational disruption risk derived from historical event intelligence."
    )

    # -------------------------
    # Event Summary
    # -------------------------

    st.subheader("📋 Event Summary")

    event = result["event"]

    st.write("**Event Type:**", event.event_type)
    st.write("**Cause:**", event.event_cause)
    st.write("**Description:**", event.description)
    st.write("**Corridor:**", event.corridor)
    st.write("**Priority:**", event.priority)
    st.write("**Road Closure:**", event.requires_road_closure)

    # -------------------------
    # Risk Components
    # -------------------------

    st.subheader("⚠️ Risk Components")

    component_mapping = {
        "historical_closure_requirement": "Historical Closure Risk",
        "event_priority": "Priority Impact",
        "event_cause_operational_burden": "Cause Impact",
        "historical_handling_duration": "Historical Duration Impact",
        "planned_unplanned_urgency": "Operational Urgency",
        "evidence_uncertainty": "Data Uncertainty",
    }

    comp_df = pd.DataFrame(
        {
            "Component": [
                component_mapping.get(k, k)
                for k in risk.components.keys()
            ],
            "Score": [
                round(v, 2)
                for v in risk.components.values()
            ],
        }
    )

    st.dataframe(
        comp_df,
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------
    # Reasons
    # -------------------------

    st.subheader("🧠 Assessment Reasoning")

    for reason in risk.reasons:
        st.write("•", reason)

    # -------------------------
    # Similar Events
    # -------------------------

    st.subheader("📚 Similar Historical Events")

    events = result["similar_events"]

    st.write(events[0])

    if events:

        display_rows = []

        for e in events:
            display_rows.append(
                {
                    "Rank": e["rank"],
                    "Event ID": e["event_id"],
                    "Cause": e.get("event_cause"),
                    "Distance (km)": round(
                        e["distance_km"],
                        2,
                    ) if e.get("distance_km") else "-",
                    "Similarity": round(
                        e["semantic_similarity"],
                        2,
                    ),
                    "Rerank Score": round(
                        e["rerank_score"],
                        2,
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(display_rows),
            use_container_width=True,
        )

    st.subheader("🗺 Similar Event Locations")

    map_events = []

    for e in events:

        lat = e.get("latitude")
        lon = e.get("longitude")

        if lat is not None and lon is not None:

            map_events.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "event_id": e["event_id"],
                    "cause": e["event_cause"],
                }
            )

    if map_events:

        center_lat = sum(x["lat"] for x in map_events) / len(map_events)
        center_lon = sum(x["lon"] for x in map_events) / len(map_events)

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles="OpenStreetMap",
        )

        # Current Event (Blue)
        if (
            hasattr(event, "latitude")
            and hasattr(event, "longitude")
            and event.latitude is not None
            and event.longitude is not None
        ):

            folium.Marker(
                location=[
                    event.latitude,
                    event.longitude,
                ],
                popup="Current Event",
                icon=folium.Icon(
                    color="blue"
                ),
            ).add_to(m)

        # Historical Events (Red)
        for item in map_events:

            folium.CircleMarker(
                location=[
                    item["lat"],
                    item["lon"],
                ],
                radius=8,
                color="#d62728",
                fill=True,
                fill_color="#d62728",
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"""
                    <b>Event ID:</b> {item['event_id']}<br>
                    <b>Cause:</b> {item['cause']}
                    """,
                    max_width=300,
                ),
            ).add_to(m)

        # FIX: give the map a stable key so Streamlit doesn't treat it as
        # a brand-new widget each rerun, and set returned_objects=[] so
        # panning/zooming/clicking the map does NOT trigger a script
        # rerun at all (since there's nothing for it to return that
        # changes). If you later want click data back, add
        # "last_object_clicked" to this list — it'll be safe now since
        # the pipeline above only runs on an actual button press.
        st_folium(
            m,
            width=1200,
            height=500,
            key="event_map",
            returned_objects=[],
        )

        st.markdown("""
    🔵 Current Event

    🔴 Retrieved Historical Events
    """)

        st.caption(
            "Locations of the top retrieved historical events used for risk assessment."
        )

    # -------------------------
    # Resource Recommendation
    # -------------------------

    st.subheader("👮 Resource Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Recommended Personnel",
            f"{resources.manpower_min} - {resources.manpower_max}",
        )

    with col2:
        st.metric(
            "Recommended Barricades",
            f"{resources.barricades_min} - {resources.barricades_max}",
        )

    with col3:
        st.metric(
            "Support Vehicles",
            len(resources.support_vehicles),
        )

    st.write("### Support Vehicles")

    for vehicle in resources.support_vehicles:
        st.write("•", vehicle)

    st.info(resources.policy_disclaimer)

    # -------------------------
    # AI Report
    # -------------------------

    st.subheader("📝 Traffic Management Plan")

    st.markdown(result["report"])

    # -------------------------
    # Case Info
    # -------------------------

    st.success(
        f"✅ Traffic Management Plan Saved | Reference ID: {result.get('case_id')[:8]}"
    )