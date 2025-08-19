#!/usr/bin/env python3
"""
process_data.py

Cleans and engineers features for the "Uber Ride Analytics 2024" dataset.

Usage:
    python src/process_data.py --input data/uber_ride_analytics_2024.csv --output results/uber_2024_processed.parquet
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dateutil import parser as dtparser

# --------- Helpers ---------

ORIG2SNAKE = {
    "Date": "date",
    "Time": "time",
    "Booking ID": "booking_id",
    "Booking Status": "booking_status",
    "Customer ID": "customer_id",
    "Vehicle Type": "vehicle_type",
    "Pickup Location": "pickup_location",
    "Drop Location": "drop_location",
    "Avg VTAT": "avg_vtat",
    "Avg CTAT": "avg_ctat",
    "Cancelled Rides by Customer": "cancelled_by_customer",
    "Reason for cancelling by Customer": "cancel_reason_customer",
    "Cancelled Rides by Driver": "cancelled_by_driver",
    "Driver Cancellation Reason": "cancel_reason_driver",
    "Incomplete Rides": "incomplete_rides",
    "Incomplete Rides Reason": "incomplete_reason",
    "Booking Value": "booking_value",
    "Ride Distance": "ride_distance",
    "Driver Ratings": "driver_ratings",
    "Customer Rating": "customer_rating",
    "Payment Method": "payment_method",
}

PEAK_HOURS = set(range(7,10)) | set(range(17,20))

def to_snake_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: ORIG2SNAKE.get(c, c).strip().lower().replace(" ", "_") for c in df.columns}
    return df.rename(columns=rename)

def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.extract(r'([-+]?[0-9]*\.?[0-9]+)')[0], errors="coerce")

def coalesce(*args):
    out = None
    for a in args:
        if out is None:
            out = a.copy() if isinstance(a, pd.Series) else a
        else:
            mask = out.isna() if isinstance(out, pd.Series) else pd.isna(out)
            if isinstance(a, pd.Series):
                out[mask] = a[mask]
            else:
                out = np.where(mask, a, out)
    return out

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize columns to snake_case
    df = to_snake_columns(df)

    # Parse dates & times
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        df["date_parsed"] = pd.NaT

    if "time" in df.columns:
        # try parsing HH:MM[:SS]
        df["time_parsed"] = pd.to_datetime(df["time"], errors="coerce").dt.time
    else:
        df["time_parsed"] = pd.NaT

    # Combine datetime
    def combine_dt(row):
        d = row.get("date_parsed")
        t = row.get("time_parsed")
        if pd.isna(d):
            return pd.NaT
        if pd.notna(pd.Series([t])[0]):
            # construct datetime with time
            return pd.to_datetime(f"{d.date()} {t}")
        return pd.to_datetime(d)

    df["datetime"] = df.apply(combine_dt, axis=1)

    # Calendar features
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["weekday"] = df["datetime"].dt.weekday
    df["weekday_name"] = df["datetime"].dt.day_name()
    df["hour"] = df["datetime"].dt.hour
    df["weekofyear"] = df["datetime"].dt.isocalendar().week.astype("Int64")
    df["is_weekend"] = df["weekday"].isin([5,6])

    # Part of day
    def part_of_day(h):
        if pd.isna(h):
            return np.nan
        if 5 <= h < 12:
            return "morning"
        elif 12 <= h < 17:
            return "afternoon"
        elif 17 <= h < 21:
            return "evening"
        else:
            return "night"
    df["part_of_day"] = df["hour"].apply(part_of_day)
    df["is_peak_hour"] = df["hour"].isin(list(PEAK_HOURS))

    # Booking Status derived flags
    status = df.get("booking_status", pd.Series(index=df.index, dtype=object)).astype(str).str.lower().str.strip()
    df["trip_completed"] = status.eq("completed")
    df["is_cancelled"] = status.str.contains("cancelled", na=False)
    df["is_incomplete"] = coalesce(df.get("incomplete_rides"), pd.Series(False, index=df.index)).astype(str).str.lower().isin(["1","true","yes"])

    # Unified cancellation
    cust_flag = df.get("cancelled_by_customer")
    drv_flag  = df.get("cancelled_by_driver")
    cust_flag = cust_flag.astype(str).str.lower().isin(["1","true","yes"]) if cust_flag is not None else pd.Series(False, index=df.index)
    drv_flag  = drv_flag.astype(str).str.lower().isin(["1","true","yes"])  if drv_flag is not None else pd.Series(False, index=df.index)

    df["cancellation_party"] = np.select(
        [cust_flag, drv_flag, df["is_cancelled"]],
        ["customer", "driver", "unknown"],
        default=np.where(df["is_cancelled"], "unknown", "none")
    )

    df["cancellation_reason"] = coalesce(
        df.get("cancel_reason_customer"),
        df.get("cancel_reason_driver"),
        pd.Series(pd.NA, index=df.index)
    )

    # Numeric cleaning
    if "booking_value" in df.columns:
        df["booking_value"] = parse_numeric(df["booking_value"])
    if "ride_distance" in df.columns:
        df["ride_distance"] = parse_numeric(df["ride_distance"])

    # Derived numeric
    df["fare_per_km"] = np.where((df.get("booking_value").notna()) & (df.get("ride_distance") > 0),
                                 df["booking_value"] / df["ride_distance"], np.nan)

    # High value rides (top 10% by fare)
    if "booking_value" in df.columns:
        thresh = df["booking_value"].quantile(0.90)
        df["high_value_ride"] = df["booking_value"] >= thresh
    else:
        df["high_value_ride"] = False

    # Ratings
    if "driver_ratings" in df.columns:
        df["driver_ratings"] = parse_numeric(df["driver_ratings"])
    if "customer_rating" in df.columns:
        df["customer_rating"] = parse_numeric(df["customer_rating"])
    df["rating_gap"] = df.get("driver_ratings") - df.get("customer_rating")

    # Distance bins
    if "ride_distance" in df.columns:
        bins = [0, 2, 5, 10, np.inf]
        labels = ["0-2 km", "2-5 km", "5-10 km", "10+ km"]
        df["distance_bin_km"] = pd.cut(df["ride_distance"], bins=bins, labels=labels, include_lowest=True)

    # Payment normalization
    pm = df.get("payment_method")
    if pm is not None:
        pm_norm = pm.astype(str).str.lower().str.strip()
        mapping = {
            "upi": "UPI",
            "cash": "Cash",
            "credit card": "Card",
            "debit card": "Card",
            "uber wallet": "Wallet"
        }
        df["payment_category"] = pm_norm.map(mapping).fillna(pm.str.title())
    else:
        df["payment_category"] = pd.NA

    # Vehicle grouping (light touch)
    vt = df.get("vehicle_type")
    if vt is not None:
        vt_norm = vt.astype(str).str.lower()
        df["vehicle_category"] = np.select(
            [
                vt_norm.str.contains("sedan|premier"),
                vt_norm.str.contains("xl"),
                vt_norm.str.contains("auto|bike|ebike")
            ],
            ["Sedan", "XL", "Two/Three-Wheeler"],
            default=vt.str.title()
        )
    else:
        df["vehicle_category"] = pd.NA

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV file")
    ap.add_argument("--output", required=True, help="Path to output file (.parquet or .csv)")
    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.input)

    # Process
    df_out = engineer_features(df)

    # Save
    out = args.output
    if out.lower().endswith(".parquet"):
        df_out.to_parquet(out, index=False)
    elif out.lower().endswith(".csv"):
        df_out.to_csv(out, index=False)
    else:
        print("Output must end with .parquet or .csv", file=sys.stderr)
        sys.exit(2)

    print(f"Saved {len(df_out):,} rows to {out}")

if __name__ == "__main__":
    main()
