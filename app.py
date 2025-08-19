import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.process_data import engineer_features

st.set_page_config(page_title="Uber Ride Analytics 2024", layout="wide")

st.title("🚕 Uber Ride Analytics 2024")
st.write("Upload the Kaggle CSV or use a processed file. The app will clean and engineer features automatically.")

# Sidebar: file upload
uploaded = st.file_uploader("Upload CSV (raw Kaggle file)", type=["csv"])

@st.cache_data
def load_process(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return engineer_features(df)

df = None
if uploaded is not None:
    df = load_process(uploaded)
    st.success(f"Loaded {len(df):,} rows.")
else:
    st.info("Upload a CSV to begin.")

if df is not None and not df.empty:
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        # Date range
        if "datetime" in df.columns:
            min_d = pd.to_datetime(df["datetime"]).min()
            max_d = pd.to_datetime(df["datetime"]).max()
            date_range = st.date_input("Date range", value=(min_d.date(), max_d.date()))
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = map(pd.to_datetime, date_range)
                df = df[(df["datetime"] >= start) & (df["datetime"] <= end + pd.Timedelta(days=1))]

        # Vehicle type
        veh_options = sorted([v for v in df.get("vehicle_type", pd.Series(dtype=object)).dropna().unique()])
        veh_selected = st.multiselect("Vehicle Type", veh_options, default=veh_options[:5] if veh_options else [])
        if veh_selected:
            df = df[df["vehicle_type"].isin(veh_selected)]

        # Booking status
        status_options = sorted([v for v in df.get("booking_status", pd.Series(dtype=object)).dropna().unique()])
        status_selected = st.multiselect("Booking Status", status_options, default=status_options[:5] if status_options else [])
        if status_selected:
            df = df[df["booking_status"].isin(status_selected)]

        # Payment category
        pay_options = sorted([v for v in df.get("payment_category", pd.Series(dtype=object)).dropna().unique()])
        pay_selected = st.multiselect("Payment", pay_options, default=pay_options)
        if pay_selected:
            df = df[df["payment_category"].isin(pay_selected)]

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    total_rides = len(df)
    completion_rate = (df.get("trip_completed", pd.Series(False, index=df.index)).mean() * 100) if total_rides else 0
    cancel_rate = (df.get("is_cancelled", pd.Series(False, index=df.index)).mean() * 100) if total_rides else 0
    revenue = df.get("booking_value", pd.Series(dtype=float)).sum()
    avg_distance = df.get("ride_distance", pd.Series(dtype=float)).mean()

    col1.metric("Total rides", f"{total_rides:,}")
    col2.metric("Completion rate", f"{completion_rate:,.1f}%")
    col3.metric("Cancellation rate", f"{cancel_rate:,.1f}%")
    col4.metric("Total revenue", f"{revenue:,.0f}")
    col5.metric("Avg distance (km)", f"{avg_distance:,.2f}" if not np.isnan(avg_distance) else "N/A")

    st.divider()

    # Charts
    c1, c2 = st.columns(2)
    if "hour" in df.columns:
        hourly = df.groupby("hour", dropna=False).size().reset_index(name="rides")
        c1.subheader("Rides by Hour")
        c1.plotly_chart(px.bar(hourly, x="hour", y="rides"), use_container_width=True)

    if "weekday_name" in df.columns:
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        daily = df.groupby("weekday_name").size().reindex(order).reset_index(name="rides")
        c2.subheader("Rides by Weekday")
        c2.plotly_chart(px.bar(daily, x="weekday_name", y="rides"), use_container_width=True)

    c3, c4 = st.columns(2)
    if "vehicle_type" in df.columns and "booking_value" in df.columns:
        rev_by_vehicle = df.groupby("vehicle_type", dropna=False)["booking_value"].sum().reset_index()
        c3.subheader("Revenue by Vehicle Type")
        c3.plotly_chart(px.bar(rev_by_vehicle, x="vehicle_type", y="booking_value"), use_container_width=True)

    if "distance_bin_km" in df.columns:
        dist = df["distance_bin_km"].value_counts().sort_index().reset_index()
        dist.columns = ["distance_bin_km", "rides"]
        c4.subheader("Distance Distribution")
        c4.plotly_chart(px.bar(dist, x="distance_bin_km", y="rides"), use_container_width=True)

    st.subheader("Cancellation Breakdown")
    if "cancellation_party" in df.columns:
        can_party = df["cancellation_party"].value_counts(dropna=False).reset_index()
        can_party.columns = ["party","rides"]
        st.plotly_chart(px.pie(can_party, names="party", values="rides"), use_container_width=True)

    # Data preview
    st.subheader("Sample Data")
    st.dataframe(df.head(50))
else:
    st.stop()
