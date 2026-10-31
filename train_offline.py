#!/usr/bin/env python3
"""
train_offline.py
----------------
OFFLINE training for the two-stage, interpretable safe-route model.  Run it ONCE (or whenever
crime datasets change); the Flask app only ever loads the exported artifacts.

Single-city usage:
    python train_offline.py --graph-file pune.graphml --pois-file pune_pois.geojson --csv FINAL_CRIME_DATASET.csv

Joint Multi-City usage (trains on Pune + Bengaluru simultaneously without forgetting):
    python train_offline.py \
        --csv FINAL_CRIME_DATASET.csv bglr_standardized.csv \
        --graph-file training_cache_pune/training_graph.graphml training_cache_bglr/training_graph.graphml \
        --pois-file training_cache_pune/training_pois.geojson training_cache_bglr/training_pois.geojson

PEDESTRIAN-SAFETY PHYSICS (this version)
    * the target keeps ONLY interpersonal / pedestrian-threat incidents: property & vehicle crimes
      (EXCLUDE_CRIMES) are dropped at ingestion; they are reported in busy daytime markets / for insurance and
      say little about a pedestrian's personal vulnerability
    * incidents whose clock time is a data-entry PLACEHOLDER (e.g. 20% of the Bengaluru file is stamped exactly
      12:00:00) carry location information but NO time information: their mass is spread over all 24 hours
    * label mass is weighted by an hourly pedestrian-vulnerability curve V(h) (NIGHT_MULTIPLIER): night hours
      (low visibility, few bystanders) up to x2.5, midday x0.5
    * the regressor is trained with MONOTONIC CONSTRAINTS (ml_engine.MONOTONIC_CONSTRAINTS): risk can only rise
      with police distance, isolation_index and darkness_hazard, and can only fall with junction degree and with
      shop density (commercial_poi_count, active_open_shops_estimate)

STAGE 1  Crime typology (unsupervised)
    * each incident's free-text "Crime Type" (comma-separated, multi-label) -> severity 0.1..5
      via an explicit, editable weight table (CRIME_SEVERITY below / --severity-json)
    * K-Means (k chosen from {3, 4} by silhouette) on [severity, is_night, police distance km]
    * clusters are named from their own statistics (highest mean severity = "Violent Physical Threat")
    -> crime_cluster_explainer.joblib

STAGE 2  Road risk regressor (supervised, NO coordinates)
    * CONTINUOUS GAUSSIAN KERNEL instead of a binary buffer: every incident deposits severity mass on EVERY road
      edge within 2.5 sigma (sigma = 90 m -> 225 m), weighted by its perpendicular distance d to the edge geometry
          w(d) = exp(-d^2 / (2 sigma^2))
      so an incident 39 m away and one 42 m away carry almost the same weight (no step function, far fewer
      artificial zeros).  The label is therefore a smooth kernel-density risk field along the street network.
    * TARGET-LENGTH DECOUPLING: the label is severity mass x V(hour) divided by a LOG-EXPOSURE
          exposure(L) = log1p(max(L, 50 m)) / log1p(100 m)           (= 1.0 for a 100 m street)
      rather than by linear length, and the model is given log_length = log1p(length) instead of raw length.  The
      trees no longer have to split on `length` to undo a 1/length denominator, so their splits go to police
      distance, connectivity, shop frontage and hour.
    * label per (edge, hour) is circularly smoothed over 3 hours and NORMALISED for natural surveillance
          density_adj = density / (1 + 0.35 * commercial_poi_count) ** 0.85
      then scaled to [0, 1] by the 95th percentile of positive values (strictly non-negative), plus an AMBIENT
      HAZARD FLOOR so zero-incident dead-ends and unlit night corridors do not get zero risk:
          y += 0.004 * clip(isolation_index, 0, 3) + 0.005 * darkness_hazard      (disable: --no-ambient-floor)
    * LOSS = POISSON DEVIANCE (log link): the target is non-negative, heavy-tailed and mostly near zero; the
      Poisson deviance does not assume symmetric Gaussian residuals and does not penalise hot-spot outliers
      quadratically.  Predictions are exp(raw score) > 0, so the exported risk thresholds / risk_calibration
      anchors are taken from the Poisson-model prediction distribution at export time.
    * model sees ONLY: road_type_ord, log_length, commercial_poi_count, active_open_shops_estimate,
      distance_to_nearest_police, degree_of_connectivity, query_hour, is_night, plus the engineered
      pedestrian-vulnerability features isolation_index and darkness_hazard (see feature_engineering.py)
    * evaluation uses 500 m spatial blocks (GroupShuffleSplit hold-out + GroupKFold CV) so that
      neighbouring edges sharing the same incident never straddle train/test
    -> road_risk_regressor.joblib  (+ training_report.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import ml_engine as me
from feature_engineering import (ACTIVITY_ARRAY, NIGHT_END_HOUR, NIGHT_START_HOUR, compute_darkness_hazard,
                                 compute_static_and_poi_features, compute_temporal_features, is_night_flag)

log = logging.getLogger("train_offline")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
# Continuous spatial kernel (replaces the old 40 m binary buffer):  w(d) = exp(-d^2 / (2 sigma^2)),  d <= cutoff * sigma
KERNEL_SIGMA_M = 90.0           # bandwidth; 80-100 m is a one-to-two-block neighbourhood
KERNEL_CUTOFF_SIGMAS = 2.5      # edges farther than 2.5 sigma (225 m) from an incident get no mass (w < 0.044)
COVERAGE_M = 750.0              # edges farther than this from every incident are not trained on
SPATIAL_CELL_M = 500.0
LENGTH_FLOOR_M = 50.0           # min length used in the exposure term (guards tiny stubs)
EXPOSURE_REF_M = 100.0          # exposure(100 m) == 1.0, so the label keeps its "per ~100 m" magnitude
LABEL_SCALE_QUANTILE = 0.95     # label = density / (this quantile of the positive densities), clipped to [0, 1]
MAX_DISTANCE_FROM_CENTRE_KM = 35.0   # drops obviously wrong coordinates
SEVERITY_EXTRA_WEIGHT = 0.10    # severity = max + 0.10 * sum(other categories), capped at 5
SEVERITY_CAP = 5.0
COMMERCIAL_DAMPENING = 0.35     # adjusted_density = raw_density / (1 + COMMERCIAL_DAMPENING * commercial_poi_count) ** EXPONENT
COMMERCIAL_EXPONENT = 0.85
# Ambient environmental hazard added to the [0, 1] label so zero-incident dead-ends / unlit night corridors are not
# trained as "zero risk" (under-reporting there is the problem, not safety).  NOTE: this is an assumed prior, not data.
AMBIENT_ISOLATION_COEF = 0.004  # * clip(isolation_index, 0, AMBIENT_ISOLATION_CAP)
AMBIENT_ISOLATION_CAP = 3.0
AMBIENT_DARKNESS_COEF = 0.005   # * darkness_hazard
DEFAULT_UNLISTED_SEVERITY = 1.0
NIGHT_START, NIGHT_END = NIGHT_START_HOUR, NIGHT_END_HOUR   # single source: feature_engineering.is_night_flag
CLUSTER_FEATURE_WEIGHTS = np.array([3.0, 0.5, 0.3])   # severity, is_night, police_km
MIN_CLUSTER_SHARE = 0.05

# Expert prior: how much a reported incident type signals danger to a PEDESTRIAN (0.1 low .. 5 high).
# Design rule: interpersonal violence (3.8-5.0) must dominate; property / vehicle theft (0.1-0.3) is
# reported far more often (busy daytime markets, FIR logging for insurance) but says little about a
# pedestrian's personal vulnerability, so it must not drive the risk map.
CRIME_SEVERITY = {
    # Interpersonal violent offences (3.8 - 5.0)
    "offences affecting life": 5.0, "robbery and dacoity": 5.0, "kidnapping": 5.0,
    "murder": 5.0, "rape": 5.0, "abduction": 4.5, "attempted murder": 4.5,
    "slavery and forced labour": 4.5, "criminal force and assault": 4.5, "kidnap": 4.5,
    "robbery": 4.5, "hurt": 4.0, "extortion": 4.0, "wrongful restraint and confinement": 4.0,
    "assault": 4.0, "chain snatching": 3.8,

    # Moderate nuisance / intimidation offences
    "intimidation & annoyance": 3.0, "public tranquillity": 3.0,
    "criminal trespass": 2.5,
    "attempts to commit offences": 2.0, "drug trafficking and substance abuse": 2.0,
    "damage to public property": 1.5, "marriage-related offences": 1.5,

    # Property & vehicle crime (0.1 - 0.3): high volume, low personal-safety signal
    "theft": 0.3, "ordinary theft": 0.3, "mischief": 0.2,
    "2 wheeler theft": 0.1, "4 wheeler theft": 0.1,

    # Low / regulatory / non-violent offences
    "alcohol prohibition violations": 1.0, "of abetment": 1.0, "general criminal offenses": 1.0,
    "general exceptions": 1.0, "unknown": 1.0, "gambling-related offenses": 0.8,
    "criminal conspiracy": 0.8, "public health & safety": 0.6, "public safety violations": 0.6,
    "traffic violations and motor vehicle offenses": 0.5, "state-specific motor vehicle violations": 0.5,
    "police-related offenses and misconduct": 0.5, "offences against the state": 0.5,
    "elections & authority": 0.4, "public servants": 0.3, "cheating": 0.2,
    "criminal breach of trust": 0.2, "false evidence & justice": 0.2,
    "cybercrimes and digital fraud": 0.1, "offences relating to documents": 0.1,
    "copyright infringement and intellectual property violations": 0.1,
    "weights & measures": 0.1, "coin & govt stamps": 0.1,
}

CATEGORY_ALIASES = {"public health": "public health & safety", "safety": "public health & safety"}

# ---- target construction: which incidents describe a PEDESTRIAN threat ----------------------------------
# Property / vehicle crimes are dropped at ingestion (load_incidents).  They are logged in large numbers in busy
# daytime commercial areas and carry little information about personal vulnerability on the street.
EXCLUDE_CRIMES = {
    "2 wheeler theft", "4 wheeler theft", "ordinary theft",
    "theft", "vehicle theft", "mischief", "damage to public property",
}
# Retained (interpersonal / pedestrian-threat) categories.  The first ten are the specified set; the second
# group are the equivalent labels used by the Pune (IPC-chapter style) file, so Pune is filtered by the same rule.
INTERPERSONAL_CATEGORIES = {
    "murder", "rape", "attempted murder", "kidnap", "robbery", "assault", "chain snatching", "hurt",
    "extortion", "intimidation & annoyance",
    # Pune-schema equivalents
    "offences affecting life", "robbery and dacoity", "criminal force and assault", "kidnapping",
    "abduction", "wrongful restraint and confinement",
}

# Hourly pedestrian vulnerability V(h): exposure-adjusted weight applied to the severity mass of an incident at
# hour h.  Night = low visibility + few bystanders (x2.2-2.5); midday = crowded and well lit (x0.5).
NIGHT_MULTIPLIER = {
    0: 2.5, 1: 2.5, 2: 2.5, 3: 2.5, 4: 2.5, 5: 2.2,
    6: 1.0, 7: 0.8, 8: 0.7, 9: 0.5, 10: 0.5, 11: 0.5,
    12: 0.5, 13: 0.5, 14: 0.5, 15: 0.6, 16: 0.7, 17: 0.9,
    18: 1.1, 19: 1.3, 20: 1.6, 21: 2.0, 22: 2.5, 23: 2.5,
}
HOURLY_VULNERABILITY = np.array([NIGHT_MULTIPLIER.get(h, 1.0) for h in range(24)], dtype=float)

# A clock time shared by more than this fraction of a file's rows (e.g. 12:00:00 on 20% of Bengaluru rows)
# is a data-entry placeholder, not an observation.
PLACEHOLDER_TIME_MIN_SHARE = 0.05
HOURS_NIGHT_CHECK = (22, 23, 0, 1, 2, 3)      # sanity check on the learned hourly profile ...
HOURS_DAY_CHECK = (10, 11, 12, 13, 14, 15)    # ... night mean must exceed midday mean


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs) -> np.ndarray:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) / w.sum()
    return np.interp(qs, cum, v)


def log_exposure(length_m: np.ndarray) -> np.ndarray:
    """
    Sub-linear (logarithmic) exposure of a road segment:  log1p(max(L, 50 m)) / log1p(100 m).
    Replaces the old linear divisor max(L, 50) / 100, whose 1/length shape the trees had to invert with their top
    splits on `length`.  A 100 m street -> 1.0, 50 m -> 0.85, 500 m -> 1.35 (instead of 0.5 / 5.0).
    """
    return np.log1p(np.maximum(np.asarray(length_m, dtype=float), LENGTH_FLOOR_M)) / math.log1p(EXPOSURE_REF_M)


def gaussian_kernel(d_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """w(d) = exp(-d^2 / (2 sigma^2)); w(0) = 1, w(sigma) = 0.61, w(2.5 sigma) = 0.044."""
    return np.exp(-np.square(np.asarray(d_m, dtype=float)) / (2.0 * sigma_m ** 2))


def smooth_hours(mat: np.ndarray) -> np.ndarray:
    """Circular 3-hour moving average (23h, 0h, 1h are neighbours); conserves total mass."""
    return (mat + np.roll(mat, 1, axis=1) + np.roll(mat, -1, axis=1)) / 3.0


def ox_major_version() -> int:
    import osmnx as ox
    return int(str(ox.__version__).split(".")[0])


# =========================================================================== #
# STAGE 1 -- incidents, severity, archetypes
# =========================================================================== #
def parse_categories(raw: str) -> list:
    cats = []
    for part in str(raw).split(","):
        key = part.strip().lower()
        if not key:
            continue
        key = CATEGORY_ALIASES.get(key, key)
        if key not in cats:
            cats.append(key)
    return cats or ["unknown"]


def incident_severity(cats: list, table: dict, unlisted: set) -> float:
    weights = []
    for c in cats:
        if c in table:
            weights.append(table[c])
        else:
            unlisted.add(c)
            weights.append(DEFAULT_UNLISTED_SEVERITY)
    weights.sort(reverse=True)
    return float(min(SEVERITY_CAP, weights[0] + SEVERITY_EXTRA_WEIGHT * sum(weights[1:])))


def load_incidents(csv_path: Path, severity_table: dict, strict_interpersonal: bool = True,
                   mask_placeholder_times: bool = True) -> pd.DataFrame:
    """
    Read + clean a standardized incident CSV -> one row per usable PEDESTRIAN-THREAT incident.

    * property / vehicle crimes (EXCLUDE_CRIMES) are dropped; for multi-label incidents only the excluded
      LABELS are removed, so "theft, assault" survives as an assault and its severity is not inflated by the theft
    * strict_interpersonal=True keeps only INTERPERSONAL_CATEGORIES; False keeps every other category
      (their severity weights still apply)
    * `time_known` is False when the clock time is unparseable or a placeholder shared by more than
      PLACEHOLDER_TIME_MIN_SHARE of the file's rows (mask_placeholder_times=True)
    """
    raw = pd.read_csv(csv_path)
    need = ["Time of Occurrence", "latitude", "longitude", "Distance from Police Station", "Crime Type"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing required columns: {missing}")
    n0 = len(raw)

    df = pd.DataFrame({
        "lat": pd.to_numeric(raw["latitude"], errors="coerce"),
        "lon": pd.to_numeric(raw["longitude"], errors="coerce"),
        "police_km": pd.to_numeric(raw["Distance from Police Station"], errors="coerce"),
        "crime_type": raw["Crime Type"].fillna("Unknown").astype(str),
    })
    hour = pd.to_datetime(raw["Time of Occurrence"], format="%H:%M:%S", errors="coerce").dt.hour
    if hour.isna().any():
        alt = pd.to_datetime(raw["Time of Occurrence"], errors="coerce").dt.hour
        hour = hour.fillna(alt)
    if hour.isna().any() and "DateTime" in raw.columns:
        alt2 = pd.to_datetime(raw["DateTime"], errors="coerce").dt.hour
        hour = hour.fillna(alt2)

    # --- time reliability: unparseable or placeholder clock times carry no time-of-day information ---
    unparsed = hour.isna().to_numpy()
    time_str = raw["Time of Occurrence"].astype(str).str.strip()
    share = time_str.map(time_str.value_counts(normalize=True)).to_numpy(float)
    placeholder = share > PLACEHOLDER_TIME_MIN_SHARE
    if placeholder.any():
        top = time_str[placeholder].value_counts().head(3).to_dict()
        log.warning("[%s] %d rows (%.0f%%) share a placeholder-looking timestamp %s",
                    csv_path.name, int(placeholder.sum()), 100 * placeholder.mean(), top)
    time_known = ~(unparsed | (placeholder if mask_placeholder_times else False))
    df["hour"] = hour.fillna(12)                # value is IGNORED for rows with time_known == False
    df["time_known"] = time_known

    df = df.dropna(subset=["lat", "lon"]).copy()
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]

    # --- crime-category filter: property / vehicle crimes out, interpersonal threats in ---
    cats = df["crime_type"].map(parse_categories).tolist()
    n_loc = len(df)
    n_property_only = sum(all(c in EXCLUDE_CRIMES for c in cl) for cl in cats)
    cats = [[c for c in cl if c not in EXCLUDE_CRIMES] for cl in cats]
    n_non_property = sum(len(cl) > 0 for cl in cats)
    if strict_interpersonal:
        cats = [[c for c in cl if c in INTERPERSONAL_CATEGORIES] for cl in cats]
    keep = np.array([len(cl) > 0 for cl in cats], dtype=bool)
    log.info("[%s] category filter: %d located incidents -> %d property/vehicle-only dropped -> %d "
             "non-interpersonal dropped%s -> %d kept", csv_path.name, n_loc, n_property_only,
             n_non_property - int(keep.sum()), "" if strict_interpersonal else " (disabled: --allow-all-categories)",
             int(keep.sum()))
    if not keep.any():
        raise SystemExit(f"{csv_path}: no interpersonal-threat incidents left after the category filter")
    df = df[keep].copy()
    cats = [cl for cl, k in zip(cats, keep) if k]

    # Spatial outliers: relative to median incident
    med_lat, med_lon = df["lat"].median(), df["lon"].median()
    dkm = np.hypot((df["lat"] - med_lat) * 111.32, (df["lon"] - med_lon) * 111.32 * math.cos(math.radians(med_lat)))
    far = (dkm > MAX_DISTANCE_FROM_CENTRE_KM).to_numpy()
    if far.any():
        log.warning("[%s] dropping %d incident(s) more than %.0f km from median location",
                    csv_path.name, int(far.sum()), MAX_DISTANCE_FROM_CENTRE_KM)
    df = df[~far].copy()
    cats = [cl for cl, f in zip(cats, far) if not f]

    df["hour"] = df["hour"].astype(int)
    df["is_night"] = np.where(df["time_known"], is_night_flag(df["hour"].to_numpy()), 0).astype(int)
    df["police_km"] = df["police_km"].fillna(df["police_km"].median())
    cap = float(df["police_km"].quantile(0.99)) if len(df) > 10 else 5.0
    df["police_km"] = df["police_km"].clip(0, max(cap, 1.0))
    df.attrs["police_km_clip"] = cap

    unlisted: set = set()
    df["severity"] = [incident_severity(c, severity_table, unlisted) for c in cats]
    df["top_category"] = [max(c, key=lambda k: severity_table.get(k, DEFAULT_UNLISTED_SEVERITY)) for c in cats]
    if unlisted:
        log.warning("[%s] categories without exact severity weight: %s", csv_path.name, sorted(unlisted)[:5])
    df = df.reset_index(drop=True)
    log.info("[%s] incidents: %d read, %d usable | severity mean %.2f | night share (known times) %.0f%% | "
             "time unknown for %d (%.0f%%)", csv_path.name, n0, len(df), df["severity"].mean(),
             100 * df.loc[df["time_known"], "is_night"].mean() if df["time_known"].any() else 0.0,
             int((~df["time_known"]).sum()), 100 * (~df["time_known"]).mean())
    return df


def _name_clusters(profile: pd.DataFrame) -> dict:
    order = profile.sort_values("mean_severity").index.tolist()
    k = len(order)
    base = {3: ["Low-Risk Dispute", "Property/Nuisance Risk", "Violent Physical Threat"],
            4: ["Low-Risk Dispute", "Property/Nuisance Risk", "Serious Mixed-Offence Risk",
                "Violent Physical Threat"]}[k]
    overall_night = float(np.average(profile["night_share"], weights=profile["size"]))
    names = {}
    for rank, cid in enumerate(order):
        tag = ""
        diff = profile.loc[cid, "night_share"] - overall_night
        if diff > 0.15:
            tag = " (night-skewed)"
        elif diff < -0.15:
            tag = " (daytime)"
        names[int(cid)] = base[rank] + tag
    return names


def fit_crime_clusters(inc: pd.DataFrame, seed: int) -> tuple:
    feats = ["severity", "is_night", "police_km"]
    X = inc[feats].to_numpy(float)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X) * CLUSTER_FEATURE_WEIGHTS

    candidates = []
    for k in (3, 4):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
        sil = float(silhouette_score(Z, km.labels_))
        share_min = float(np.bincount(km.labels_).min() / len(Z))
        log.info("  K-Means k=%d: silhouette %.3f | smallest cluster %.1f%%", k, sil, 100 * share_min)
        candidates.append((k, sil, share_min, km))
    valid = [c for c in candidates if c[2] >= MIN_CLUSTER_SHARE] or [candidates[0]]
    k, sil, _, km = max(valid, key=lambda c: c[1])

    out = inc.copy()
    out["cluster"] = km.labels_
    profile = out.groupby("cluster").agg(
        size=("severity", "size"), mean_severity=("severity", "mean"),
        night_share=("is_night", "mean"), mean_police_km=("police_km", "mean"))
    names = _name_clusters(profile)
    out["archetype"] = out["cluster"].map(names)

    profiles = {}
    for cid in profile.index:
        sub = out[out["cluster"] == cid]
        profiles[int(cid)] = {
            "name": names[int(cid)], "size": int(len(sub)),
            "mean_severity": round(float(sub["severity"].mean()), 3),
            "night_share": round(float(sub["is_night"].mean()), 3),
            "mean_police_km": round(float(sub["police_km"].mean()), 3),
            "top_categories": sub["top_category"].value_counts().head(5).to_dict(),
        }
    log.info("Stage 1: chose k=%d (silhouette %.3f). Archetypes:", k, sil)
    for cid in sorted(profiles, key=lambda c: profiles[c]["mean_severity"]):
        p = profiles[cid]
        log.info("  [%d] %-38s n=%4d  severity %.2f  night %3.0f%%  police %.1f km  top: %s",
                 cid, p["name"], p["size"], p["mean_severity"], 100 * p["night_share"],
                 p["mean_police_km"], list(p["top_categories"])[:3])

    explainer = {
        "kind": "crime_cluster_explainer", "schema_version": 1,
        "scaler": scaler, "kmeans": km, "feature_names": feats,
        "feature_weights": CLUSTER_FEATURE_WEIGHTS.copy(),
        "cluster_names": {int(c): names[int(c)] for c in sorted(names)},
        "cluster_profiles": profiles, "k": int(k), "silhouette": sil,
        "severity_range": (float(inc["severity"].quantile(0.05)), float(inc["severity"].quantile(0.95))),
        "police_km_clip": float(inc.attrs.get("police_km_clip", 5.0)),
        "severity_weights": dict(CRIME_SEVERITY), "n_incidents": int(len(inc)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return explainer, out


# =========================================================================== #
# Single Area Loading Helper
# =========================================================================== #
POI_TAGS = {"amenity": ["police"], "shop": True}


def load_single_area(graph_file: str, pois_file: str):
    import geopandas as gpd
    import osmnx as ox

    for f in (graph_file, pois_file):
        if not Path(f).is_file():
            raise SystemExit(f"Offline file not found: {f}")
    log.info("Loading graph %s and POIs %s", graph_file, pois_file)
    G = ox.load_graphml(graph_file)
    pois = gpd.read_file(pois_file)
    for col in ("amenity", "shop"):
        if col not in pois.columns:
            pois[col] = None
    if pois.crs is None:
        pois = pois.set_crs("EPSG:4326")
    return G, pois


# =========================================================================== #
# STAGE 2 -- Ground-Truth Labels via Edge Buffer Join
# =========================================================================== #
def accumulate_hourly_mass(n_edges: int, eid: np.ndarray, hour: np.ndarray, mass: np.ndarray,
                           known: np.ndarray) -> np.ndarray:
    """
    Exposure-adjusted temporal vulnerability.  Returns mat[n_edges, 24] with

        mat[e, h] = sum over incidents i on edge e at hour h of  severity_i * share_i * V(h)

    where V(h) = HOURLY_VULNERABILITY[h] (NIGHT_MULTIPLIER: night x2.2-2.5, midday x0.5).  `mass` is already
    severity * share.  Incidents whose clock time is unknown / a placeholder (known == False) are spread
    uniformly over the 24 hours (mass / 24 per hour) and weighted by the same V(h): they inform WHERE, not WHEN.
    """
    hour = np.asarray(hour, dtype=int) % 24
    known = np.asarray(known, dtype=bool)
    mat = np.zeros((n_edges, 24))
    weighted = mass * HOURLY_VULNERABILITY[hour]
    np.add.at(mat, (eid[known], hour[known]), weighted[known])
    if (~known).any():
        np.add.at(mat, eid[~known], (mass[~known] / 24.0)[:, None] * HOURLY_VULNERABILITY[None, :])
    return mat


def ambient_hazard(static: pd.DataFrame) -> np.ndarray:
    """
    Baseline environmental risk per (edge, hour), shape [E, 24]:
        0.004 * clip(isolation_index, 0, 3)  +  0.005 * darkness_hazard(hour)
    darkness_hazard is built from the SAME functions as the model feature (is_night_flag, ACTIVITY_ARRAY,
    compute_darkness_hazard), so label and feature definitions cannot drift apart.
    """
    iso = np.clip(np.nan_to_num(static["isolation_index"].to_numpy(float), nan=0.0), 0.0, AMBIENT_ISOLATION_CAP)
    poi = np.maximum(np.nan_to_num(static["commercial_poi_count"].to_numpy(float), nan=0.0), 0.0)
    night = is_night_flag(np.arange(24))[None, :]                                   # [1, 24]
    active = poi[:, None] * ACTIVITY_ARRAY[None, :]                                 # [E, 24]
    dark = compute_darkness_hazard(np.broadcast_to(night, active.shape), active)    # [E, 24]
    return AMBIENT_ISOLATION_COEF * iso[:, None] + AMBIENT_DARKNESS_COEF * dark


def build_edge_labels(G, static: pd.DataFrame, inc: pd.DataFrame, sigma_m: float, cutoff_sigmas: float,
                      coverage_m: float, ambient_floor: bool = True, normalize_kernel: bool = False) -> dict:
    import geopandas as gpd
    import osmnx as ox
    from scipy.spatial import cKDTree

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges = edges[edges.index.isin(static.index)]
    edges = edges[~edges.index.duplicated(keep="first")]
    edges = edges.reindex(static.index)
    ok = edges.geometry.notna().to_numpy()
    edges, static = edges[ok], static[ok]
    E = len(edges)

    crs_m = edges.estimate_utm_crs()
    edges_m = edges.to_crs(crs_m)
    geom = edges_m.geometry.reset_index(drop=True)
    radius_m = float(cutoff_sigmas) * float(sigma_m)

    pts = gpd.GeoDataFrame(inc[["severity", "hour", "time_known"]].copy(),
                           geometry=gpd.points_from_xy(inc["lon"], inc["lat"]), crs="EPSG:4326").to_crs(crs_m)
    minx, miny, maxx, maxy = edges_m.total_bounds
    pad = radius_m * 2
    pts = pts.cx[minx - pad:maxx + pad, miny - pad:maxy + pad].reset_index(drop=True)
    pts["pid"] = np.arange(len(pts))
    n_in_area = len(pts)
    if n_in_area == 0:
        raise SystemExit("No incidents fall inside this graph extent. Check that incidents and graph match!")

    # Candidate (incident, edge) pairs: a 2.5-sigma disc around each incident intersected with the edge geometries.
    # (Buffering the POINTS is far cheaper than buffering every edge by 225 m; the exact distance is computed next.)
    disc = gpd.GeoDataFrame({"pid": pts["pid"].to_numpy()}, geometry=pts.geometry.buffer(radius_m), crs=crs_m)
    road = gpd.GeoDataFrame({"eid": np.arange(E)}, geometry=geom, crs=crs_m)
    joined = gpd.sjoin(disc, road, how="inner", predicate="intersects")
    eid = joined["eid"].to_numpy(int)
    pid = joined["pid"].to_numpy(int)

    # Exact perpendicular distance incident -> edge polyline, then the Gaussian decay weight.
    d = geom.iloc[eid].reset_index(drop=True).distance(pts.geometry.iloc[pid].reset_index(drop=True),
                                                       align=False).to_numpy(float)
    within = d <= radius_m
    eid, pid, d = eid[within], pid[within], d[within]
    w_kernel = gaussian_kernel(d, sigma_m)
    if normalize_kernel:      # optional: conserve each incident's severity (mass split across the edges it touches)
        w_kernel = w_kernel / pd.Series(w_kernel).groupby(pid).transform("sum").to_numpy()
    n_matched = int(np.unique(pid).size)
    log.info("Gaussian kernel (sigma %.0f m, cutoff %.1f sigma = %.0f m): %d/%d incidents inside graph area reach "
             ">= 1 road edge (%.0f%%); %d incident-edge pairs, mean %.1f edges/incident, median weight %.2f",
             sigma_m, cutoff_sigmas, radius_m, n_matched, n_in_area, 100 * n_matched / max(1, n_in_area),
             len(eid), len(eid) / max(1, n_matched), float(np.median(w_kernel)) if len(w_kernel) else 0.0)

    sev = pts["severity"].to_numpy(float)[pid]
    mass = sev * w_kernel                                   # severity mass deposited on the edge, distance-weighted
    hour = pts["hour"].to_numpy(int)[pid] % 24
    known = pts["time_known"].to_numpy(bool)[pid]

    # Exposure-adjusted temporal vulnerability (see accumulate_hourly_mass): mass[e, h] = sum(severity * share) * V(h)
    mat = accumulate_hourly_mass(E, eid, hour, mass, known)
    log.info("Incident-hour joins: %d with a known clock time, %d spread over 24h (placeholder / unparseable)",
             int(known.sum()), int((~known).sum()))
    mat = smooth_hours(mat)
    by_hour = mat.sum(axis=0)
    log.info("Label mass by hour (night mean / midday mean = %.2f): %s",
             float(by_hour[list(HOURS_NIGHT_CHECK)].mean() / max(by_hour[list(HOURS_DAY_CHECK)].mean(), 1e-12)),
             " ".join(f"{h:02d}:{v:.0f}" for h, v in enumerate(by_hour)))
    # TARGET-LENGTH DECOUPLING: divide by a logarithmic exposure instead of linear length.
    lengths = static["length"].to_numpy(float)
    exposure = log_exposure(lengths)
    log.info("Log-exposure divisor: mean %.2f (min %.2f, max %.2f) vs. linear length/100 m: mean %.2f (max %.2f)",
             float(exposure.mean()), float(exposure.min()), float(exposure.max()),
             float((np.maximum(lengths, LENGTH_FLOOR_M) / 100.0).mean()),
             float((np.maximum(lengths, LENGTH_FLOOR_M) / 100.0).max()))
    raw_density = mat / exposure[:, None]
    # Natural-surveillance normalisation ("eyes on the street"): the same severity mass counts for less where shops
    # line the street; the sub-linear exponent keeps 5-15 shops from being penalised twice:
    #   density_adj = density / (1 + 0.35 * commercial_poi_count) ** 0.85
    poi_count = np.nan_to_num(static["commercial_poi_count"].to_numpy(float), nan=0.0)
    damp = (1.0 + COMMERCIAL_DAMPENING * np.maximum(poi_count, 0.0)) ** COMMERCIAL_EXPONENT
    density = raw_density / damp[:, None]
    log.info("Natural-surveillance normalisation: mean divisor %.2f (max %.2f); total label mass %.1f -> %.1f",
             float(damp.mean()), float(damp.max()), float(raw_density.sum()), float(density.sum()))
    positive = density[density > 0]
    scale = float(np.quantile(positive, LABEL_SCALE_QUANTILE)) if len(positive) > 0 else 1.0
    y = np.clip(density / scale, 0.0, 1.0)            # strictly non-negative, as the Poisson deviance requires
    zero_share = float((y == 0).mean())
    log.info("Kernel label: %.1f%% of (edge, hour) cells are exactly zero (a 40 m binary buffer left far more); "
             "positive-label quantiles p50 %.4f | p90 %.4f | p99 %.4f", 100 * zero_share,
             *(np.quantile(y[y > 0], [0.5, 0.9, 0.99]) if (y > 0).any() else (0.0, 0.0, 0.0)))

    # Which edges actually carry incidents is decided BEFORE the ambient floor is added: the floor makes every
    # label positive, and the zero-edge subsampling / re-weighting in assemble_training_frame must keep keying on
    # real incidents.
    has_incident = (y > 0).any(axis=1)
    floor_share = 0.0
    if ambient_floor:
        floor = ambient_hazard(static)
        y_obs_mass = float(y.sum())
        y = np.clip(y + floor, 0.0, 1.0)
        floor_share = float(floor.sum() / max(float(y.sum()), 1e-12))
        log.info("Ambient hazard floor: mean %.4f vs mean incident label %.4f; floor is %.0f%% of final label mass "
                 "(an assumed prior, not observed data)", float(floor.mean()), y_obs_mass / y.size, 100 * floor_share)

    mid = geom.interpolate(0.5, normalized=True)
    mid_xy = np.column_stack([mid.x.to_numpy(), mid.y.to_numpy()])
    tree = cKDTree(np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()]))
    near, _ = tree.query(mid_xy, k=1)
    covered = near <= coverage_m
    if not np.all(np.isfinite(y)) or float(y.min()) < 0.0:
        raise SystemExit("Poisson loss needs a finite, non-negative target; the label contains NaN/negative values.")
    cx = np.floor(mid_xy[:, 0] / SPATIAL_CELL_M).astype(np.int64)
    cy = np.floor(mid_xy[:, 1] / SPATIAL_CELL_M).astype(np.int64)
    cells = pd.factorize(pd.Series(cx.astype(str)) + "_" + pd.Series(cy.astype(str)))[0]

    info = {"edges_total": int(E), "edges_covered": int(covered.sum()), "edges_with_incidents": int(has_incident.sum()),
            "incidents_in_area": int(n_in_area), "incidents_matched": n_matched,
            "label_scale_p95": scale, "label_zero_share": zero_share, "commercial_dampening": COMMERCIAL_DAMPENING,
            "commercial_exponent": COMMERCIAL_EXPONENT, "ambient_floor_share": floor_share,
            "spatial_blocks": int(len(np.unique(cells[covered])))}
    log.info("Labels: %d edges, %d within %.0f m of an incident, %d carry kernel mass; p95 scale %.4f; %d blocks",
             E, info["edges_covered"], coverage_m, info["edges_with_incidents"], scale, info["spatial_blocks"])
    return {"static": static, "y": y, "cells": cells, "covered": covered, "scale": scale, "info": info,
            "has_incident": has_incident}


# =========================================================================== #
# Training Frame Assembly & Models
# =========================================================================== #
def assemble_training_frame(static, y, cells, covered, zero_keep: float, rng, has_incident=None):
    E = len(static)
    # `has_incident` = edges with real incidents (before any ambient floor); falls back to y > 0 for old callers.
    pos_edge = (y > 0).any(axis=1) if has_incident is None else np.asarray(has_incident, dtype=bool)
    keep = covered & (pos_edge | (rng.random(E) < zero_keep))
    weight_edge = np.where(pos_edge, 1.0, 1.0 / zero_keep)
    idx = np.flatnonzero(keep)
    sub = static.iloc[idx]

    frames, ys, ws, gs, es = [], [], [], [], []
    for h in range(24):
        feats = compute_temporal_features(sub, h)
        if "log_length" not in feats.columns:                 # feature_engineering.add_temporal_columns adds it too
            feats = feats.assign(log_length=np.log1p(feats["length"].to_numpy(float)))
        frames.append(feats[me.FEATURE_COLUMNS].reset_index(drop=True))
        ys.append(y[idx, h])
        ws.append(weight_edge[idx])
        gs.append(cells[idx])
        es.append(idx)
    X = pd.concat(frames, ignore_index=True).astype(float)
    return X, np.concatenate(ys), np.concatenate(ws), np.concatenate(gs), np.concatenate(es)


def make_regressor(kind: str, seed: int, n_rows: int):
    """
    Gradient-boosted trees, POISSON-DEVIANCE loss (log link: prediction = exp(score) > 0), WITH MONOTONIC
    CONSTRAINTS (ml_engine.MONOTONIC_CONSTRAINTS, aligned with ml_engine.FEATURE_COLUMNS):
        commercial_poi_count -1 | active_open_shops_estimate -1 | degree_of_connectivity -1
        distance_to_nearest_police +1 | isolation_index +1 | darkness_hazard +1
    All other features (road_type_ord, log_length, query_hour, is_night) are unconstrained (0).
    The Poisson target is already a rate (mass / log-exposure), so no `offset` / exposure weight is added:
    doing both would divide by exposure twice.  The trees therefore cannot learn that more shops, farther-from-police,
    more-isolated or darker streets or dead ends are SAFER / RISKIER in the wrong direction, however sparse or
    under-reported the data is there.

    Collinearity: isolation_index = police_km / degree and darkness_hazard depend on police distance, degree,
    hour and shops.  To stop one raw feature from absorbing all the splits, trees are shallower (depth 6), the
    learning rate is lower and each split considers only a random subset of the features (max_features /
    colsample_bytree), so isolation_index and darkness_hazard get to be the primary splitter wherever the raw
    features they are built from are not sampled.
    """
    min_leaf = int(max(20, min(200, n_rows // 500)))
    mono = me.monotonic_vector()
    if kind == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise SystemExit("--model lgbm requires `pip install lightgbm`") from exc
        return LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.04, num_leaves=31, max_depth=7,
                             min_child_samples=min_leaf, subsample=0.8, subsample_freq=1, colsample_bytree=0.75,
                             reg_lambda=2.0, monotone_constraints=mono, monotone_constraints_method="intermediate",
                             random_state=seed, n_jobs=-1, verbose=-1), "LGBMRegressor"
    params = dict(loss="poisson", learning_rate=0.04, max_iter=300, max_leaf_nodes=31, max_depth=7,
                  min_samples_leaf=min_leaf, l2_regularization=2.0, early_stopping=False,
                  monotonic_cst=mono, random_state=seed)
    if "max_features" in HistGradientBoostingRegressor().get_params():      # scikit-learn >= 1.4
        params["max_features"] = 0.8
    else:
        log.warning("scikit-learn %s has no HistGradientBoostingRegressor(max_features=...); upgrade to >= 1.4 or "
                    "use --model lgbm for per-split feature subsampling", __import__("sklearn").__version__)
    return HistGradientBoostingRegressor(**params), "HistGradientBoostingRegressor"


def check_monotonicity(model, X: pd.DataFrame, n_rows: int = 300, seed: int = 0) -> dict:
    """
    Empirical check of the constraints: for a sample of real rows, sweep each constrained feature across its
    2nd-98th percentile range (all else fixed) and report the WORST violation, in raw-risk units
    (0.0 = perfectly monotone in the required direction).
    """
    rng = np.random.default_rng(seed)
    sub = X.iloc[rng.choice(len(X), size=min(n_rows, len(X)), replace=False)].reset_index(drop=True)
    worst = {}
    for feat, sign in me.MONOTONIC_CONSTRAINTS.items():
        grid = np.unique(np.quantile(X[feat].to_numpy(float), np.linspace(0.02, 0.98, 25)))
        if len(grid) < 2:
            worst[feat] = 0.0
            continue
        preds = []
        for g in grid:
            d = sub.copy()
            d[feat] = g
            preds.append(np.asarray(model.predict(d), dtype=float))
        steps = np.diff(np.vstack(preds), axis=0) * sign       # must be >= 0 along the grid
        worst[feat] = float(max(0.0, -steps.min()))
    return worst


def evaluate(model, X, y, w, train_mean: float, rng) -> dict:
    from scipy.stats import spearmanr
    pred = np.clip(model.predict(X), 0.0, 1.0)
    mae = float(mean_absolute_error(y, pred, sample_weight=w))
    base_mae = float(mean_absolute_error(y, np.full(len(y), train_mean), sample_weight=w))
    r2 = float(r2_score(y, pred, sample_weight=w))
    # Poisson deviance (the training loss) and D^2 = 1 - deviance / deviance of the constant-mean predictor
    eps = 1e-9
    dev = float(mean_poisson_deviance(y, np.maximum(pred, eps), sample_weight=w))
    base_dev = float(mean_poisson_deviance(y, np.full(len(y), max(train_mean, eps)), sample_weight=w))
    d2 = float(1.0 - dev / base_dev) if base_dev > 0 else float("nan")
    n = len(y)
    sel = rng.choice(n, size=min(n, 200_000), replace=False)
    rho = spearmanr(pred[sel], y[sel])[0] if np.ptp(pred[sel]) > 0 and np.ptp(y[sel]) > 0 else float("nan")
    order = np.argsort(-pred)
    top = order[: max(1, n // 10)]
    total = float((y * w).sum())
    capture = float((y[top] * w[top]).sum() / total) if total > 0 else float("nan")
    return {"mae": mae, "baseline_mae": base_mae, "r2": r2, "spearman": float(rho),
            "top10pct_risk_capture": capture, "poisson_deviance": dev, "baseline_poisson_deviance": base_dev,
            "d2_poisson": d2, "n_rows": int(n)}


def fmt(m: dict) -> str:
    return (f"MAE {m['mae']:.4f} (mean-predictor {m['baseline_mae']:.4f}) | R^2 {m['r2']:+.3f} | "
            f"Poisson dev {m['poisson_deviance']:.4f} (mean-predictor {m['baseline_poisson_deviance']:.4f}, "
            f"D^2 {m['d2_poisson']:+.3f}) | Spearman {m['spearman']:+.3f} | "
            f"top-10% rows capture {100 * m['top10pct_risk_capture']:.0f}% of risk mass")


def train_and_evaluate(X, y, w, groups, args, rng) -> dict:
    n_groups = len(np.unique(groups))
    if n_groups < 10:
        raise SystemExit(f"Only {n_groups} spatial blocks; need >= 10 for spatial cross-validation. Use a larger area.")
    log.info("Training rows: %d (%d spatial blocks of %.0f m)", len(X), n_groups, SPATIAL_CELL_M)

    # ---- spatial hold-out ----
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(X, y, groups))
    model, name = make_regressor(args.model, args.seed, len(tr))
    t0 = time.time()
    model.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
    train_mean = float(np.average(y[tr], weights=w[tr]))
    holdout = evaluate(model, X.iloc[te], y[te], w[te], train_mean, rng)
    log.info("HOLD-OUT (20%% of spatial blocks, fit %.0fs): %s", time.time() - t0, fmt(holdout))

    # ---- permutation importance ----
    sub = rng.choice(te, size=min(len(te), 30_000), replace=False)
    pi = permutation_importance(model, X.iloc[sub], y[sub], scoring="r2", n_repeats=3,
                                random_state=args.seed, sample_weight=w[sub], n_jobs=1)
    importances = pd.Series(pi.importances_mean, index=me.FEATURE_COLUMNS).sort_values(ascending=False)
    log.info("Feature importance (permutation drop in hold-out R^2):")
    for feat, val in importances.items():
        log.info("    %-28s %+.4f", feat, val)

    # ---- spatial-block K-fold ----
    cv = None
    folds = min(args.cv_folds, len(np.unique(groups[tr])))
    if folds >= 2:
        rows = []
        for i, (a, b) in enumerate(GroupKFold(n_splits=folds).split(X.iloc[tr], y[tr], groups[tr]), 1):
            a, b = tr[a], tr[b]
            m, _ = make_regressor(args.model, args.seed, len(a))
            m.fit(X.iloc[a], y[a], sample_weight=w[a])
            r = evaluate(m, X.iloc[b], y[b], w[b], float(np.average(y[a], weights=w[a])), rng)
            rows.append(r)
            log.info("  CV fold %d/%d: %s", i, folds, fmt(r))
        cv = {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows]))}
              for k in ("mae", "baseline_mae", "r2", "spearman", "top10pct_risk_capture",
                        "poisson_deviance", "baseline_poisson_deviance", "d2_poisson")}
        log.info("SPATIAL %d-FOLD CV: MAE %.4f+-%.4f | R^2 %+.3f+-%.3f | Poisson D^2 %+.3f+-%.3f | "
                 "Spearman %+.3f+-%.3f",
                 folds, cv["mae"]["mean"], cv["mae"]["std"], cv["r2"]["mean"], cv["r2"]["std"],
                 cv["d2_poisson"]["mean"], cv["d2_poisson"]["std"],
                 cv["spearman"]["mean"], cv["spearman"]["std"])

    # ---- final model on 100% data ----
    final, _ = make_regressor(args.model, args.seed, len(X))
    final.fit(X, y, sample_weight=w)
    mono_check = check_monotonicity(final, X, seed=args.seed)
    for feat, viol in mono_check.items():
        log.info("Monotonic constraint %+d on %-28s worst violation %.2e %s", me.MONOTONIC_CONSTRAINTS[feat], feat,
                 viol, "OK" if viol < 1e-9 else "<-- VIOLATED")
    # labelled hold-out sample, consumed by evaluate_and_visualize.py --holdout-csv
    hold = X.iloc[te].copy()
    hold["y"], hold["w"] = y[te], w[te]
    hold["length"] = np.expm1(hold["log_length"].to_numpy(float))     # evaluate_and_visualize.py samples raw `length`
    if len(hold) > 60_000:
        hold = hold.sample(60_000, random_state=args.seed)
    return {"model": final, "model_name": name, "holdout": holdout, "cv": cv, "importances": importances,
            "holdout_frame": hold, "monotonic_check": mono_check}


# =========================================================================== #
# Orchestration
# =========================================================================== #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline training: crime archetypes + road risk regressor",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--csv", nargs="+", default=[str(here / "FINAL_CRIME_DATASET.csv")],
                    help="One or more incident CSV files (e.g. Pune, Bengaluru)")
    ap.add_argument("--graph-file", nargs="+", default=None,
                    help="One or more GraphML files matching the incident CSVs")
    ap.add_argument("--pois-file", nargs="+", default=None,
                    help="One or more GeoJSON POI files matching the incident CSVs")
    ap.add_argument("--out-dir", default=str(here), help="where the .joblib files are written")
    ap.add_argument("--cache-dir", default=str(here / "training_cache"))
    ap.add_argument("--place", help='geocodable place name (with --dist)')
    ap.add_argument("--dist", type=int, default=5000, help="radius in metres for --place")
    ap.add_argument("--severity-json", help="JSON file {category: weight} overriding CRIME_SEVERITY")
    ap.add_argument("--model", choices=["hgb", "lgbm"], default="hgb")
    ap.add_argument("--kernel-sigma-m", type=float, default=KERNEL_SIGMA_M,
                    help="Gaussian kernel bandwidth sigma in metres (80-100 recommended)")
    ap.add_argument("--kernel-cutoff-sigmas", type=float, default=KERNEL_CUTOFF_SIGMAS,
                    help="edges farther than this many sigmas from an incident receive no mass")
    ap.add_argument("--normalize-kernel", action="store_true",
                    help="split each incident's severity across the edges it touches (mass-conserving) instead of "
                         "the default un-normalised kernel-density sum")
    ap.add_argument("--coverage-m", type=float, default=COVERAGE_M)
    ap.add_argument("--zero-keep", type=float, default=0.25,
                    help="fraction of incident-free edges kept to bound training size")
    ap.add_argument("--cv-folds", type=int, default=5, help="spatial-block CV folds")
    ap.add_argument("--allow-all-categories", action="store_true",
                    help="keep every non-property crime category (default: ONLY interpersonal / pedestrian-threat "
                         "categories, see INTERPERSONAL_CATEGORIES). Property/vehicle crimes are always dropped.")
    ap.add_argument("--no-ambient-floor", action="store_true",
                    help="do NOT add the ambient hazard floor (0.004*isolation + 0.005*darkness) to the label; use to "
                         "A/B how much of the learned isolation/darkness effect comes from the assumed floor")
    ap.add_argument("--keep-placeholder-times", action="store_true",
                    help="trust placeholder clock times (e.g. 12:00:00) as real hours (NOT recommended)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "log_length" not in me.FEATURE_COLUMNS or "length" in me.FEATURE_COLUMNS:
        raise SystemExit("ml_engine.FEATURE_COLUMNS must contain 'log_length' (and not raw 'length'). "
                         "Replace ml_engine.py and feature_engineering.py with the updated versions first.")
    if args.kernel_sigma_m <= 0 or args.kernel_cutoff_sigmas <= 0:
        raise SystemExit("--kernel-sigma-m and --kernel-cutoff-sigmas must be positive")

    severity = dict(CRIME_SEVERITY)
    if args.severity_json:
        severity.update({k.lower(): float(v) for k, v in json.loads(Path(args.severity_json).read_text()).items()})

    # Validate file inputs
    csv_paths = [Path(p) for p in args.csv]
    for p in csv_paths:
        if not p.is_file():
            raise SystemExit(f"Incident CSV not found: {p}")

    graph_files = args.graph_file or []
    pois_files = args.pois_file or []
    if graph_files and len(graph_files) != len(pois_files):
        raise SystemExit("The number of --graph-file arguments must match the number of --pois-file arguments!")
    if graph_files and len(graph_files) != len(csv_paths):
        raise SystemExit(f"Passed {len(csv_paths)} CSV(s) but {len(graph_files)} graph file(s). "
                         "Provide one graph and POI file per CSV.")

    # ---------------- Stage 1: Typology Clustering ----------------
    log.info("=" * 78)
    log.info("STAGE 1: Crime typology clustering")
    inc_dfs = [load_incidents(p, severity, strict_interpersonal=not args.allow_all_categories,
                              mask_placeholder_times=not args.keep_placeholder_times) for p in csv_paths]
    inc_all = pd.concat(inc_dfs, ignore_index=True)
    n_time_unknown = int((~inc_all["time_known"]).sum())
    log.info("Total pedestrian-threat incidents across all datasets: %d (%d with unknown clock time)",
             len(inc_all), n_time_unknown)

    # Archetypes use is_night, so only incidents with a trustworthy clock time take part in the clustering.
    inc_for_clusters = inc_all[inc_all["time_known"]]
    if len(inc_for_clusters) < 50:
        inc_for_clusters = inc_all
    explainer, _ = fit_crime_clusters(inc_for_clusters.reset_index(drop=True), args.seed)
    explainer["severity_weights"] = severity
    joblib.dump(explainer, out_dir / me.EXPLAINER_FILE, compress=3)
    log.info("Saved %s", out_dir / me.EXPLAINER_FILE)

    # ---------------- Stage 2: Road Risk Regressor ----------------
    log.info("=" * 78)
    log.info("STAGE 2: Multi-City road risk regressor")

    all_X, all_y, all_w, all_groups = [], [], [], []
    group_offset = 0
    total_info = {"edges_total": 0, "edges_covered": 0, "edges_with_incidents": 0, "incidents_matched": 0}
    floor_shares = []
    scale_values = []

    # Iterate through each city dataset & graph
    if graph_files:
        pairs = list(zip(csv_paths, inc_dfs, graph_files, pois_files))
        for idx, (c_path, c_inc, g_file, p_file) in enumerate(pairs, 1):
            log.info("-" * 60)
            log.info("Processing City %d/%d [%s]", idx, len(pairs), c_path.name)
            G, pois = load_single_area(g_file, p_file)
            
            t0 = time.time()
            static = compute_static_and_poi_features(G, pois)
            log.info("Features for %d edges computed in %.0fs", len(static), time.time() - t0)

            lab = build_edge_labels(G, static, c_inc, args.kernel_sigma_m, args.kernel_cutoff_sigmas,
                                    args.coverage_m, ambient_floor=not args.no_ambient_floor,
                                    normalize_kernel=args.normalize_kernel)
            scale_values.append(lab["scale"])
            for k in total_info:
                total_info[k] += lab["info"].get(k, 0)
            floor_shares.append(lab["info"].get("ambient_floor_share", 0.0))

            X_c, y_c, w_c, g_c, _ = assemble_training_frame(
                lab["static"], lab["y"], lab["cells"], lab["covered"], args.zero_keep, rng,
                has_incident=lab["has_incident"]
            )

            all_X.append(X_c)
            all_y.append(y_c)
            all_w.append(w_c)
            # Offset spatial group IDs to keep cross-city neighborhoods distinct
            all_groups.append(g_c + group_offset)
            group_offset += int(np.max(g_c) + 1) if len(g_c) > 0 else 0

        X = pd.concat(all_X, ignore_index=True)
        y = np.concatenate(all_y)
        w = np.concatenate(all_w)
        groups = np.concatenate(all_groups)
        mean_scale = float(np.mean(scale_values)) if scale_values else 1.0

    else:
        raise SystemExit("Pass one --graph-file and one --pois-file per --csv (see the usage examples at the top of "
                         "this file). Download them first with osm_fetch.download_training_area(...).")

    log.info("Combined Matrix: %d training rows across %d unique spatial blocks", len(X), len(np.unique(groups)))
    res = train_and_evaluate(X, y, w, groups, args, rng)

    # ---- Artifact Metadata ----
    base = {c: float(weighted_quantile(X[c].to_numpy(), w, [0.5])[0]) for c in me.FEATURE_COLUMNS}
    base["length"] = float(np.expm1(base["log_length"]))     # raw metres: the explanation texts read baseline["length"]
    raw_cols = {"length": np.expm1(X["log_length"].to_numpy(float))}
    ranges = {}
    for c in me.STATIC_FEATURE_COLUMNS:
        col = X[c].to_numpy(float) if c in X.columns else raw_cols[c]
        ranges[c] = (float(np.quantile(col, 0.01)), float(np.quantile(col, 0.99)))
    # Percentile thresholds are re-derived from the POISSON model's own prediction distribution on every retrain
    # (exp(score) is always > 0, so the calibration anchors in risk_calibration.py stay valid).
    final_pred = np.clip(np.asarray(res["model"].predict(X), dtype=float), 1e-9, 1.0)
    q_elev, q_high = weighted_quantile(final_pred, w, [0.80, 0.92])
    q_med = weighted_quantile(final_pred, w, [0.50])[0]
    thresholds = {"elevated": float(q_elev), "high": float(max(q_high, q_elev + 1e-6)),
                  "median": float(min(q_med, q_elev * 0.999))}   # 'median' anchors the 0-100 calibration
    if not (0.0 < thresholds["median"] < thresholds["elevated"] < thresholds["high"]):
        log.warning("Risk thresholds are degenerate (median %.6f, elevated %.6f, high %.6f): the prediction "
                    "distribution has almost no spread; risk_calibration will fall back to its quadratic ramp",
                    thresholds["median"], thresholds["elevated"], thresholds["high"])
    log.info("Calibrated Risk Thresholds (Poisson predictions): median %.5f | Elevated (p80) >= %.5f | High (p92) >= %.5f",
             thresholds["median"], thresholds["elevated"], thresholds["high"])

    # ---- sanity: the learned hour-of-day profile must put night above midday ----
    hours_arr = X["query_hour"].to_numpy(int)
    hour_profile = {int(h): float(np.average(final_pred[hours_arr == h], weights=w[hours_arr == h]))
                    for h in range(24) if (hours_arr == h).any()}
    night_mean = float(np.mean([hour_profile[h] for h in HOURS_NIGHT_CHECK if h in hour_profile]))
    day_mean = float(np.mean([hour_profile[h] for h in HOURS_DAY_CHECK if h in hour_profile]))
    peak_hour = max(hour_profile, key=hour_profile.get)
    log.info("Learned hour profile: night(22-03) mean %.4f vs midday(10-15) mean %.4f (ratio %.2f); peak hour %02d:00",
             night_mean, day_mean, night_mean / max(day_mean, 1e-12), peak_hour)

    notes = []
    ho = res["holdout"]
    if night_mean <= day_mean:
        notes.append("Learned risk is not higher at night (22-03h) than at midday; check timestamps / placeholder times.")
    bad_mono = {k: v for k, v in res["monotonic_check"].items() if v >= 1e-9}
    if bad_mono:
        notes.append(f"Monotonic constraints violated for: {sorted(bad_mono)}")
    # Only flag warning if relative ranking correlation fails
    if not np.isnan(ho["spearman"]) and ho["spearman"] < 0.08:
        notes.append("Safety model ranking correlation is weak on held-out blocks; treat routes with caution.")
    for n in notes:
        log.warning(n)

    import sklearn
    artifact = {
        "kind": "road_risk_regressor", "schema_version": me.ARTIFACT_SCHEMA_VERSION,
        "model": res["model"], "model_name": res["model_name"],
        "features": list(me.FEATURE_COLUMNS), "static_features": list(me.STATIC_FEATURE_COLUMNS),
        "target_scale": mean_scale,
        "target_definition": "interpersonal/pedestrian-threat incidents only (property & vehicle crimes dropped); "
                             f"Gaussian-kernel severity mass w(d)=exp(-d^2/(2*{args.kernel_sigma_m:g}^2)) on every edge "
                             f"within {args.kernel_cutoff_sigmas:g} sigma of an incident"
                             + (" (mass-conserving per incident)" if args.normalize_kernel else "") +
                             ", weighted by the hourly pedestrian vulnerability V(h) (placeholder-time incidents "
                             "spread over 24 h), 3-hour smoothed, divided by log-exposure "
                             "log1p(max(L,50))/log1p(100) (not linear length), then by "
                             "(1 + 0.35 * commercial_poi_count) ** 0.85 and by the 95th percentile of positive "
                             "values, clipped to [0, 1]; trained with Poisson deviance (log link)"
                             + ("" if args.no_ambient_floor else
                                "; plus ambient hazard floor 0.004*clip(isolation_index,0,3) + 0.005*darkness_hazard, "
                                "clipped to [0, 1]"),
        "monotonic_constraints": dict(me.MONOTONIC_CONSTRAINTS), "monotonic_check": res["monotonic_check"],
        "loss": "poisson" if args.model == "hgb" else "poisson (LightGBM objective)",
        "hour_profile": hour_profile, "hourly_vulnerability": [float(v) for v in HOURLY_VULNERABILITY],
        "baseline_values": base, "feature_ranges": ranges, "risk_thresholds": thresholds,
        "metrics": {"holdout": ho, "spatial_cv": res["cv"]},
        "importances": {k: float(v) for k, v in res["importances"].items()},
        "notes": notes,
        "training": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": args.seed, "csv": [str(p) for p in csv_paths], "n_rows": int(len(X)),
            "n_incidents": int(len(inc_all)), "kernel_sigma_m": args.kernel_sigma_m,
            "kernel_cutoff_sigmas": args.kernel_cutoff_sigmas, "kernel_normalized": bool(args.normalize_kernel),
            "target_exposure": "log1p(max(L,50))/log1p(100)", "loss": "poisson", "coverage_m": args.coverage_m,
            "zero_keep": args.zero_keep, "sklearn_version": sklearn.__version__,
            "strict_interpersonal": not args.allow_all_categories, "ambient_floor": not args.no_ambient_floor,
            "ambient_floor_share": float(np.mean(floor_shares)) if floor_shares else 0.0,
            "commercial_normalisation": {"k": COMMERCIAL_DAMPENING, "exponent": COMMERCIAL_EXPONENT},
            "excluded_crimes": sorted(EXCLUDE_CRIMES), "n_time_unknown": n_time_unknown, **total_info,
        },
    }
    joblib.dump(artifact, out_dir / me.REGRESSOR_FILE, compress=3)
    res["holdout_frame"].to_csv(out_dir / "holdout_sample.csv.gz", index=False)
    (out_dir / "training_report.json").write_text(json.dumps(
        {k: artifact[k] for k in ("model_name", "metrics", "importances", "risk_thresholds", "notes", "training",
                                  "monotonic_constraints", "hour_profile")}
        | {"archetypes": explainer["cluster_profiles"]}, indent=2, default=float))
    log.info("Saved %s and training_report.json in %s", me.REGRESSOR_FILE, out_dir)
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())