#!/usr/bin/env python3
"""
prepare_dataset.py
-------------------
Harmonizes any crime dataset (Pune CSV, Bengaluru Excel/CSV) into the standard format:
['Time of Occurrence', 'latitude', 'longitude', 'Distance from Police Station', 'Crime Type']
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def standardize_dataset(input_path: str, output_path: str):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    print(f"Reading {path.name}...")
    if path.suffix in [".xlsx", ".xls"]:
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    # 1. Normalize Column Names
    col_map = {c.strip().lower(): c for c in raw.columns}
    
    def find_col(*candidates):
        for c in candidates:
            if c.lower() in col_map:
                return col_map[c.lower()]
        return None

    c_lat = find_col("latitude", "lat")
    c_lon = find_col("longitude", "lon", "lng")
    c_time = find_col("time of occurrence", "time")
    c_type = find_col("crime type", "type", "crime_type")
    c_police = find_col("distance from police station", "police_km", "distance_police")
    c_station = find_col("police station", "station")

    if not all([c_lat, c_lon, c_time, c_type]):
        raise ValueError(f"Missing essential columns. Found: {list(raw.columns)}")

    df = pd.DataFrame({
        "latitude": pd.to_numeric(raw[c_lat], errors="coerce"),
        "longitude": pd.to_numeric(raw[c_lon], errors="coerce"),
        "Time of Occurrence": raw[c_time].astype(str),
        "Crime Type": raw[c_type].fillna("Unknown").astype(str).str.strip(),
    })

    # Drop invalid coordinates
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    
    # Auto-detect if Bengaluru dataset: clip to actual South Bengaluru metro area
    if df["latitude"].median() < 15.0:  # Bengaluru (~12.9)
        df = df[df["latitude"].between(12.85, 12.985) & df["longitude"].between(77.52, 77.635)].copy()
    else:  # Pune (~18.5)
        df = df[df["latitude"].between(18.35, 18.65) & df["longitude"].between(73.70, 74.05)].copy()

    # Clean time format: handle times like "21:00:00" or datetime strings
    cleaned_hours = pd.to_datetime(df["Time of Occurrence"], format="%H:%M:%S", errors="coerce").dt.time
    # Fallback for alternative date-time string formats
    alt_hours = pd.to_datetime(df["Time of Occurrence"], errors="coerce").dt.time
    df["Time of Occurrence"] = cleaned_hours.fillna(alt_hours).fillna("12:00:00").astype(str)

    # 2. Police Station Distance
    if c_police and c_police in raw.columns:
        df["Distance from Police Station"] = pd.to_numeric(raw[c_police], errors="coerce").fillna(2.0)
    else:
        print("Distance to police station not present; calculating via local centroids offline...")
        if c_station and c_station in raw.columns:
            stations = raw.loc[df.index, c_station].fillna("Unknown")
            # Calculate centroid for each named station in the data
            centroids = df.groupby(stations)[["latitude", "longitude"]].median()
        else:
            # Fallback: divide incident points into spatial grid centroids
            centroids = df[["latitude", "longitude"]].sample(min(15, len(df)), random_state=42)

        # Coordinate projection to meters (Equirectangular approximation)
        lat_m = df["latitude"].median()
        kx = 111.32 * np.cos(np.radians(lat_m)) * 1000.0
        ky = 110.574 * 1000.0

        ref_xy = np.column_stack([centroids["longitude"] * kx, centroids["latitude"] * ky])
        pts_xy = np.column_stack([df["longitude"] * kx, df["latitude"] * ky])

        tree = cKDTree(ref_xy)
        dists_m, _ = tree.query(pts_xy)
        df["Distance from Police Station"] = np.round(dists_m / 1000.0, 3)

    out_cols = ["Time of Occurrence", "latitude", "longitude", "Distance from Police Station", "Crime Type"]
    df[out_cols].to_csv(output_path, index=False)
    print(f" Successfully exported {len(df)} standardized incidents to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize Crime Datasets for SafeRoute")
    parser.add_argument("--input", required=True, help="Input CSV or Excel file")
    parser.add_argument("--output", default="standardized_incidents.csv", help="Output CSV path")
    args = parser.parse_args()
    standardize_dataset(args.input, args.output)