"""
ml_engine.py
------------
ONLINE, ZERO-SHOT road-risk inference.  Nothing in this module trains anything.

    train_offline.py  --->  road_risk_regressor.joblib      (Stage 2 model)
                            crime_cluster_explainer.joblib  (Stage 1 archetypes)
                                      |
                                      v
    ml_engine.py  loads both ONCE, then scores ANY osmnx graph (Hauz Khas, Dasna,
    Avantika, Pune, ...) from abstract OSM features only -- no coordinates, no retraining.

Public API
    load_models(model_dir=None, force=False) -> bool
    models_available() -> bool
    model_info() -> dict
    build_edge_feature_frame(G, pois)                    static per-edge features
    predict_road_safety(edges_gdf, query_hour)           -> frame + predicted_risk, risk_level,
                                                            risk_archetype (+ risk_reason if explain=True)
    explain_risk(edge_features, query_hour=None)         -> human-readable rationale (str)
    explain_risk_detailed(edge_features, query_hour)     -> dict (risk, level, archetype, drivers, reason)
    explain_edges(static_df, edge_ids, query_hour)       -> {(u, v, key): detailed dict}  (batched; each dict has
                                                            `drivers` = top RISK factors and `protective` = top
                                                            POSITIVE (risk-lowering) factors)
    attach_risk_to_graph(G, scored_edges)                writes predicted_risk etc. onto G's edges
    score_graph(G, static_df, query_hour)                predict + attach, with constant-risk fallback

How explanations work (and what they are NOT)
    * Feature contributions are COUNTERFACTUAL: for each feature group we re-predict the same
      edge with that group set to a neutral reference (training median; midday for time of day)
      and report prediction - counterfactual.  Works for any model type, needs no SHAP.
      They describe what the MODEL has learned, not causes in the real world.
    * The archetype is a DESCRIPTIVE match: Stage 1 clusters were fit on (severity, night,
      police distance).  An edge has no severity, so we impute one from its predicted risk and
      pick the nearest archetype centroid.  Read it as "this edge/time looks like the context
      in which <archetype> incidents were reported", not as a prediction of crime type.

Models are plain-dict joblib artifacts (sklearn objects + metadata) so they unpickle in any
process, regardless of which script created them.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

from feature_engineering import (ROAD_TYPE_ORDER, compute_darkness_hazard, compute_isolation_index,
                                 compute_static_and_poi_features, compute_temporal_features)
from risk_calibration import (AT_ELEVATED, bind, calibrate_risk_score,  # noqa: F401  (re-exported)
                              calibrate_risk_scores, calibration_table, score_band)

log = logging.getLogger("ml_engine")

# --------------------------------------------------------------------------- #
# Constants shared with train_offline.py (single source of truth)
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "road_type_ord",
    "log_length",           # log1p(segment length in m): replaces raw `length` (derived in feature_engineering)
    "commercial_poi_count",
    "active_open_shops_estimate",
    "distance_to_nearest_police",
    "degree_of_connectivity",
    "query_hour",
    "is_night",
    # engineered pedestrian-vulnerability interactions
    "isolation_index",      # (police distance km) / max(degree, 1)          -- static
    "darkness_hazard",      # is_night / (1 + active_open_shops_estimate)    -- temporal
]
# The five raw OSM-derived columns a caller must supply.
BASE_STATIC_COLUMNS = [
    "road_type_ord",
    "length",
    "commercial_poi_count",
    "distance_to_nearest_police",
    "degree_of_connectivity",
]
# All time-independent model inputs: the 5 raw columns + the derived isolation_index.
STATIC_FEATURE_COLUMNS = BASE_STATIC_COLUMNS + ["isolation_index"]
# Both engineered pedestrian-vulnerability features must ALWAYS be model inputs (train + inference).
assert "isolation_index" in FEATURE_COLUMNS and "darkness_hazard" in FEATURE_COLUMNS, \
    "FEATURE_COLUMNS must include isolation_index and darkness_hazard"
assert "isolation_index" in STATIC_FEATURE_COLUMNS, "STATIC_FEATURE_COLUMNS must include isolation_index"

# Structural "pedestrian safety physics" imposed on the tree model (sklearn monotonic_cst /
# LightGBM monotone_constraints).  +1 = predicted risk may only rise or stay flat as the feature rises,
# -1 = may only fall or stay flat.  Every feature not listed is unconstrained (0).
MONOTONIC_CONSTRAINTS = {
    "commercial_poi_count": -1,         # more shops on the street -> never riskier (an empty alley is not safer)
    "active_open_shops_estimate": -1,   # more OPEN shops -> never riskier
    "distance_to_nearest_police": +1,   # farther from police response -> never safer
    "degree_of_connectivity": -1,       # more connections / escape routes -> never riskier (no "safe" dead ends)
    "isolation_index": +1,              # more isolated -> never safer
    "darkness_hazard": +1,              # darker, emptier night street -> never safer
}
assert set(MONOTONIC_CONSTRAINTS) <= set(FEATURE_COLUMNS), "monotonic spec names an unknown feature"


def monotonic_vector() -> list:
    """Constraint list aligned with FEATURE_COLUMNS (0 for unconstrained features)."""
    return [int(MONOTONIC_CONSTRAINTS.get(c, 0)) for c in FEATURE_COLUMNS]


ARTIFACT_SCHEMA_VERSION = 4   # 4 = Poisson loss, Gaussian-kernel labels, log-exposure target, log_length feature;
                              # 3 = + commercial-frontage monotonic constraints, stronger surveillance normalisation,
                              #     ambient hazard floor; 2 = violent-only target, hourly vulnerability, monotonic

# Edge attributes copied onto the graph so router.py can apply its structural (CPTED) rules and
# build edge-level micro-explanations.
ROUTER_EDGE_FIELDS = ["is_night", "degree_of_connectivity", "commercial_poi_count",
                      "active_open_shops_estimate", "isolation_index", "darkness_hazard"]
REGRESSOR_FILE = "road_risk_regressor.joblib"
EXPLAINER_FILE = "crime_cluster_explainer.joblib"
MODEL_DIR_ENV = "SAFE_ROUTE_MODEL_DIR"

REFERENCE_HOUR = 12          # neutral "midday" used for the time-of-day counterfactual
FALLBACK_RISK = 0.5          # constant risk used only when no trained model exists
MIN_DRIVER_DELTA_FRAC = 0.08  # ignore contributions smaller than this fraction of the 'elevated' threshold
MAX_DRIVERS = 3
MAX_PROTECTIVE = 2
LEVELS = ("low", "elevated", "high")

# Feature groups used for explanations.  Features that carry the same information
# (query_hour / is_night / active_open_shops_estimate) are explained together; permuting
# or resetting one of a correlated pair would otherwise understate its effect.
_STATIC_GROUPS = {
    "shop_activity": ["commercial_poi_count", "active_open_shops_estimate"],
    "police_distance": ["distance_to_nearest_police"],
    "road_type": ["road_type_ord"],
    "connectivity": ["degree_of_connectivity"],
    "segment_length": ["log_length"],
}
EXPLANATION_GROUPS = ["time_of_day"] + list(_STATIC_GROUPS)

_ROAD_NAMES = {
    0: "footpath / pedestrian way", 1: "living street", 2: "local residential road",
    3: "tertiary road", 4: "secondary road", 5: "primary road", 6: "trunk road",
}
assert set(_ROAD_NAMES) == set(ROAD_TYPE_ORDER.values()), "road type names out of sync"


class ModelNotAvailableError(RuntimeError):
    """Raised when scoring is requested but the offline-trained artifacts are missing."""


# --------------------------------------------------------------------------- #
# Model store: loaded once, shared by every request thread
# --------------------------------------------------------------------------- #
class _Store:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.attempted = False
        self.regressor: Optional[dict] = None
        self.explainer: Optional[dict] = None
        self.model_dir: Optional[Path] = None
        self.errors: list = []


_STORE = _Store()


def _default_model_dir() -> Path:
    return Path(os.environ.get(MODEL_DIR_ENV, Path(__file__).resolve().parent))


def _load_artifact(path: Path, expected_kind: str) -> Optional[dict]:
    if not path.exists():
        _STORE.errors.append(f"{path.name} not found in {path.parent}")
        return None
    try:
        art = joblib.load(path)
    except Exception as exc:  # corrupt file, missing lightgbm, sklearn version mismatch...
        _STORE.errors.append(f"could not load {path.name}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(art, dict) or art.get("kind") != expected_kind:
        _STORE.errors.append(f"{path.name} is not a '{expected_kind}' artifact")
        return None
    return art


def load_models(model_dir: Optional[os.PathLike] = None, force: bool = False) -> bool:
    """
    Load both .joblib artifacts into memory (idempotent).  Never raises: returns True when the
    regressor is usable.  Call once at server start-up; scoring functions also call it lazily.
    """
    with _STORE.lock:
        if _STORE.attempted and not force and model_dir is None:
            return _STORE.regressor is not None
        directory = Path(model_dir) if model_dir else _default_model_dir()
        _STORE.attempted = True
        _STORE.errors = []
        _STORE.model_dir = directory
        _STORE.regressor = _load_artifact(directory / REGRESSOR_FILE, "road_risk_regressor")
        _STORE.explainer = _load_artifact(directory / EXPLAINER_FILE, "crime_cluster_explainer")

        reg = _STORE.regressor
        if reg is not None and list(reg.get("features", [])) != FEATURE_COLUMNS:
            _STORE.errors.append("regressor was trained on a different feature list than ml_engine.FEATURE_COLUMNS; "
                                 "re-run train_offline.py")
            _STORE.regressor = None
        if _STORE.regressor is not None and int(_STORE.regressor.get("schema_version", 1)) < ARTIFACT_SCHEMA_VERSION:
            _STORE.regressor.setdefault("notes", [])
            _STORE.regressor["notes"] = list(_STORE.regressor["notes"]) + [
                "This model predates the latest pedestrian-safety retrain (monotonic shop-density / isolation / "
                "darkness constraints, surveillance normalisation, ambient hazard floor). Re-run "
                "`python train_offline.py` for corrected shop-density, isolation and darkness behaviour."]
        if _STORE.regressor is None:
            log.warning("Road-risk regressor NOT loaded (%s). Run `python train_offline.py` first; "
                        "routing falls back to distance-only.", "; ".join(_STORE.errors))
        else:
            log.info("Loaded %s (trained %s)%s", _STORE.regressor.get("model_name"),
                     _STORE.regressor.get("training", {}).get("trained_at", "?"),
                     "" if _STORE.explainer else " -- explainer missing, archetypes disabled")
        return _STORE.regressor is not None


def _ensure_loaded() -> None:
    if not _STORE.attempted:
        load_models()


def models_available() -> bool:
    _ensure_loaded()
    return _STORE.regressor is not None


def _require_regressor() -> dict:
    _ensure_loaded()
    if _STORE.regressor is None:
        raise ModelNotAvailableError(
            "road_risk_regressor.joblib is not available (" + "; ".join(_STORE.errors) + "). "
            "Run `python train_offline.py` to create it.")
    return _STORE.regressor


def model_info() -> dict:
    """Plain-dict summary of the loaded models (safe to jsonify except `importances` is already a dict)."""
    _ensure_loaded()
    reg, ex = _STORE.regressor, _STORE.explainer
    if reg is None:
        # Short and path-free on purpose: this text is shown in the UI. Full detail is in the server log.
        return {"available": False, "notes": ["Safety model not trained yet, so routes are distance-only. "
                                              "Run `python train_offline.py`, then restart the server."],
                "importances": {}, "metrics": {}}
    notes = list(reg.get("notes", []))
    if ex is None:
        notes.append("Crime-archetype explainer missing; risk reasons omit the archetype.")
    return {
        "available": True,
        "model_name": reg.get("model_name"),
        "trained_at": reg.get("training", {}).get("trained_at"),
        "metrics": reg.get("metrics", {}),
        "importances": dict(reg.get("importances", {})),
        "risk_thresholds": dict(reg.get("risk_thresholds", {})),
        "target_scale": reg.get("target_scale"),
        "monotonic_constraints": dict(reg.get("monotonic_constraints", {})),
        "archetypes": list(ex["cluster_names"].values()) if ex else [],
        "notes": notes,
        "model_dir": str(_STORE.model_dir),
    }


# --------------------------------------------------------------------------- #
# Feature preparation
# --------------------------------------------------------------------------- #
def build_edge_feature_frame(G, pois) -> pd.DataFrame:
    """Static, location-free per-edge features for ANY osmnx graph (indexed by (u, v, key))."""
    return compute_static_and_poi_features(G, pois)


def _norm_hour(query_hour) -> int:
    try:
        return int(round(float(query_hour))) % 24
    except (TypeError, ValueError) as exc:
        raise ValueError(f"query_hour must be a number 0-23, got {query_hour!r}") from exc


def _prepare_static(edges, art: dict) -> pd.DataFrame:
    """
    Validate the 5 raw static columns, repair NaN/inf with training medians, warn on drift, then DERIVE
    isolation_index from the repaired values (so a repaired distance/degree can never disagree with it).
    Any isolation_index already present in `edges` is ignored and recomputed.
    """
    cols = {}
    for c in BASE_STATIC_COLUMNS:
        if c in edges.columns:
            cols[c] = pd.to_numeric(edges[c], errors="coerce")
        elif c == "length" and getattr(edges, "geometry", None) is not None:
            geom = edges.geometry
            cols[c] = pd.Series(geom.to_crs(geom.estimate_utm_crs()).length.to_numpy(), index=edges.index)
        else:
            raise ValueError(f"edges frame is missing required feature column '{c}'. "
                             f"Build it with ml_engine.build_edge_feature_frame(G, pois). "
                             f"Required: {BASE_STATIC_COLUMNS}")
    static = pd.DataFrame(cols, index=edges.index).replace([np.inf, -np.inf], np.nan)

    baseline = art["baseline_values"]
    bad = int(static.isna().sum().sum())
    if bad:
        log.warning("%d missing/non-finite feature values replaced by training medians", bad)
        static = static.fillna({c: baseline[c] for c in BASE_STATIC_COLUMNS})
    static["isolation_index"] = compute_isolation_index(static["distance_to_nearest_police"],
                                                        static["degree_of_connectivity"])

    ranges = art.get("feature_ranges", {})
    for c in STATIC_FEATURE_COLUMNS:
        if c in ranges and len(static):
            lo, hi = ranges[c]
            span = max(hi - lo, 1e-9)
            frac = float(((static[c] < lo - span) | (static[c] > hi + span)).mean())
            if frac > 0.05:
                log.warning("feature '%s': %.0f%% of edges lie far outside the training range [%.1f, %.1f]; "
                            "predictions for this area are extrapolations", c, 100 * frac, lo, hi)
    return static


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _levels(pred: np.ndarray, thresholds: Mapping[str, float]) -> np.ndarray:
    out = np.full(pred.shape, "low", dtype=object)
    out[pred >= thresholds["elevated"]] = "elevated"
    out[pred >= thresholds["high"]] = "high"
    return out


def _match_archetypes(pred: np.ndarray, is_night: np.ndarray, police_m: np.ndarray) -> np.ndarray:
    """Nearest Stage-1 archetype for each (imputed severity, night flag, police distance) context."""
    ex = _STORE.explainer
    if ex is None:
        return np.full(pred.shape, "", dtype=object)
    lo, hi = ex["severity_range"]
    severity = lo + (hi - lo) * np.clip(pred, 0.0, 1.0)
    police_km = np.clip(np.asarray(police_m, float) / 1000.0, 0.0, ex["police_km_clip"])
    z = np.column_stack([severity, np.asarray(is_night, float), police_km])
    z = ex["scaler"].transform(z) * ex["feature_weights"]
    labels = ex["kmeans"].predict(z)
    names = np.array([ex["cluster_names"][i] for i in range(len(ex["cluster_names"]))], dtype=object)
    return names[labels]


def _score(static: pd.DataFrame, hour: int, art: dict):
    X = compute_temporal_features(static, hour)[FEATURE_COLUMNS]
    pred = np.clip(np.asarray(art["model"].predict(X), dtype=float), 0.0, 1.0)
    levels = _levels(pred, art["risk_thresholds"])
    archetypes = _match_archetypes(pred, X["is_night"].to_numpy(), X["distance_to_nearest_police"].to_numpy())
    return X, pred, levels, archetypes


def predict_road_safety(edges_gdf, query_hour, explain: bool = False,
                        explain_min_level: str = "elevated"):
    """
    Score every edge of ANY area at `query_hour` (0-23) with the pre-trained regressor.

    `edges_gdf` : DataFrame / GeoDataFrame indexed by (u, v, key) holding the static OSM features
                  (road_type_ord, length, commercial_poi_count, distance_to_nearest_police,
                  degree_of_connectivity) -- i.e. the output of build_edge_feature_frame(G, pois).
                  If it is a GeoDataFrame without a `length` column, length is measured from geometry.

    Returns a copy of the input with extra columns:
        predicted_risk  float in [0, 1]       (feeds the router cost directly)
        risk_level      "low" | "elevated" | "high"   (relative to the training distribution)
        risk_archetype  nearest Stage-1 crime archetype ("" if the explainer is not loaded)
        risk_reason     str, only when explain=True and only for edges at/above `explain_min_level`
                        (explaining all edges is wasted work; use explain_edges() for route edges)
    """
    art = _require_regressor()
    hour = _norm_hour(query_hour)
    static = _prepare_static(edges_gdf, art)
    X, pred, levels, archetypes = _score(static, hour, art)

    out = edges_gdf.copy()
    for col in ROUTER_EDGE_FIELDS:     # structural context for the router (includes isolation_index, darkness_hazard)
        out[col] = X[col].to_numpy()
    out["predicted_risk"] = pred
    out["risk_level"] = levels
    out["risk_archetype"] = archetypes
    reasons = np.full(len(out), "", dtype=object)
    if explain and len(out):
        keep = np.isin(levels, LEVELS[LEVELS.index(explain_min_level):])
        idx = np.flatnonzero(keep)
        if idx.size:
            detail = _explain_rows(static.iloc[idx], X.iloc[idx], pred[idx], levels[idx], archetypes[idx], hour, art)
            reasons[idx] = [d["reason"] for d in detail]
    out["risk_reason"] = reasons
    return out


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #
def _group_contributions(static: pd.DataFrame, X: pd.DataFrame, pred: np.ndarray, art: dict) -> pd.DataFrame:
    """contribution_g = prediction - prediction with group g reset to its neutral reference."""
    model, base = art["model"], art["baseline_values"]
    contrib = {}

    cf = compute_temporal_features(static, REFERENCE_HOUR)[FEATURE_COLUMNS]   # same street at midday
    contrib["time_of_day"] = pred - np.clip(np.asarray(model.predict(cf), float), 0.0, 1.0)

    for group, feats in _STATIC_GROUPS.items():
        cf = X.copy()
        for f in feats:
            cf[f] = base[f]
        # derived interactions must follow the reset raw values, otherwise the counterfactual is incoherent
        cf["isolation_index"] = compute_isolation_index(cf["distance_to_nearest_police"], cf["degree_of_connectivity"])
        cf["darkness_hazard"] = compute_darkness_hazard(cf["is_night"], cf["active_open_shops_estimate"])
        contrib[group] = pred - np.clip(np.asarray(model.predict(cf), float), 0.0, 1.0)
    return pd.DataFrame(contrib, index=X.index)[EXPLANATION_GROUPS]


def _driver_text(group: str, v: Mapping[str, float], base: Mapping[str, float], hour: int) -> str:
    if group == "time_of_day":
        if int(v["is_night"]):
            return f"night-time hour ({hour:02d}:00) with few shops open"
        return f"time of day ({hour:02d}:00), a period with more reported incidents"
    if group == "shop_activity":
        count, active = v["commercial_poi_count"], v["active_open_shops_estimate"]
        if active >= base["active_open_shops_estimate"] and count > 0:
            return f"busy commercial frontage ({count:.0f} shops within 100 m, ~{active:.0f} open)"
        return f"little open commercial activity nearby (~{active:.0f} of {count:.0f} shops open)"
    if group == "police_distance":
        d = v["distance_to_nearest_police"]
        if d >= base["distance_to_nearest_police"]:
            return f"far from the nearest police station ({d / 1000:.1f} km)"
        return f"near a police station ({d:.0f} m), where reported incidents concentrate"
    if group == "road_type":
        return f"{_ROAD_NAMES.get(int(round(v['road_type_ord'])), 'road')} segment"
    if group == "connectivity":
        deg = v["degree_of_connectivity"]
        if deg >= base["degree_of_connectivity"]:
            return f"busy junction area ({deg:.0f} connecting streets)"
        return f"poorly connected, dead-end-like segment ({deg:.0f} link{'s' if deg != 1 else ''})"
    if group == "segment_length":
        length = math.expm1(v["log_length"])
        if length >= base["length"]:
            return f"long unbroken stretch ({length:.0f} m)"
        return f"short segment ({length:.0f} m) between junctions"
    return group


def _protective_text(group: str, v: Mapping[str, float], base: Mapping[str, float], hour: int) -> str:
    """Phrase for a factor that LOWERS the predicted risk of this edge (a top positive factor)."""
    if group == "time_of_day":
        if int(v["is_night"]):
            return f"time of day ({hour:02d}:00) carries less modelled risk than the midday reference"
        return f"daytime hour ({hour:02d}:00): more people about and better visibility"
    if group == "shop_activity":
        count, active = v["commercial_poi_count"], v["active_open_shops_estimate"]
        return f"open shops nearby (~{active:.0f} of {count:.0f}) give light and eyes on the street"
    if group == "police_distance":
        return f"close to a police station ({v['distance_to_nearest_police']:.0f} m)"
    if group == "road_type":
        return f"{_ROAD_NAMES.get(int(round(v['road_type_ord'])), 'road')} segment, associated with lower risk"
    if group == "connectivity":
        deg = v["degree_of_connectivity"]
        return f"well-connected area ({deg:.0f} connecting streets) with several escape routes"
    if group == "segment_length":
        return f"short segment ({math.expm1(v['log_length']):.0f} m) between junctions"
    return group


def _explain_rows(static: pd.DataFrame, X: pd.DataFrame, pred: np.ndarray, levels: np.ndarray,
                  archetypes: np.ndarray, hour: int, art: dict) -> list:
    contrib = _group_contributions(static, X, pred, art)
    base = art["baseline_values"]
    thr = art["risk_thresholds"]
    min_delta = MIN_DRIVER_DELTA_FRAC * float(thr["elevated"])
    scores = bind(art).scores(pred)
    values = X.to_dict("records")
    out = []
    for i, row in enumerate(values):
        c = contrib.iloc[i]
        positive = c[c > 0].sort_values(ascending=False)
        cutoff = max(min_delta, 0.15 * float(positive.sum()))
        chosen = positive[positive >= cutoff].head(MAX_DRIVERS)
        drivers = []
        for g, d in chosen.items():
            pts = float(scores[i] - calibrate_risk_score(max(float(pred[i]) - float(d), 0.0), thr))
            drivers.append({"factor": g, "text": _driver_text(g, row, base, hour),
                            "contribution": round(float(d), 5), "points": round(pts, 1)})

        # top POSITIVE (risk-lowering) factors: groups whose removal would RAISE the prediction
        negative = c[c < 0].sort_values()                         # most protective first
        n_cut = max(min_delta, 0.15 * float(-negative.sum()))
        protective = []
        for g, d in negative[-negative >= n_cut].head(MAX_PROTECTIVE).items():
            pts = float(scores[i] - calibrate_risk_score(float(pred[i]) - float(d), thr))   # d < 0 -> pts < 0
            protective.append({"factor": g, "text": _protective_text(g, row, base, hour),
                               "contribution": round(float(d), 5), "points": round(pts, 1)})

        head = f"{str(levels[i]).capitalize()} risk (score {scores[i]:.0f}/100)"
        if archetypes[i]:
            head += f"; context resembles '{archetypes[i]}' incidents"
        if drivers:
            tail = "; main factors: " + "; ".join(f"{d['text']} ({d['points']:+.0f} pts)" for d in drivers)
        else:
            tail = "; no single factor stands out"
        out.append({"predicted_risk": round(float(pred[i]), 5), "risk_score": round(float(scores[i]), 1),
                    "risk_level": str(levels[i]), "archetype": str(archetypes[i]), "drivers": drivers,
                    "protective": protective, "reason": head + tail + "."})
    return out


def explain_risk_detailed(edge_features: Mapping[str, Any], query_hour: Optional[int] = None) -> dict:
    """
    Explain ONE edge.  `edge_features` needs the 5 raw static features (see BASE_STATIC_COLUMNS; isolation_index is derived);
    the hour comes from `query_hour`, else edge_features['query_hour'], else midday.
    """
    art = _require_regressor()
    hour = _norm_hour(query_hour if query_hour is not None else edge_features.get("query_hour", REFERENCE_HOUR))
    frame = pd.DataFrame([{c: edge_features.get(c, np.nan) for c in BASE_STATIC_COLUMNS}])
    static = _prepare_static(frame, art)
    X, pred, levels, archetypes = _score(static, hour, art)
    return _explain_rows(static, X, pred, levels, archetypes, hour, art)[0]


def explain_risk(edge_features: Mapping[str, Any], query_hour: Optional[int] = None) -> str:
    """Human-readable rationale for one edge, e.g.
    "High risk (0.41); context resembles 'Violent Physical Threat' incidents; main factors: ..."."""
    return explain_risk_detailed(edge_features, query_hour)["reason"]


def explain_edges(static_df: pd.DataFrame, edge_ids: Iterable, query_hour: int) -> dict:
    """
    Batched explanations for specific edges (e.g. the high-risk edges of a returned route).
    `static_df` is the (u, v, key)-indexed feature frame; returns {(u, v, key): detailed dict}.
    """
    art = _require_regressor()
    ids = list(dict.fromkeys(tuple(e) for e in edge_ids))
    if not ids:
        return {}
    hour = _norm_hour(query_hour)
    sub = static_df.reindex(pd.MultiIndex.from_tuples(ids, names=static_df.index.names))
    static = _prepare_static(sub, art)
    X, pred, levels, archetypes = _score(static, hour, art)
    detail = _explain_rows(static, X, pred, levels, archetypes, hour, art)
    return dict(zip(ids, detail))


# --------------------------------------------------------------------------- #
# Graph integration
# --------------------------------------------------------------------------- #
def attach_risk_to_graph(G, scored_edges: pd.DataFrame):
    """
    Write predicted_risk (raw model output, 0-1), risk_score (CALIBRATED 0-100, see risk_calibration.py),
    insecurity_score (= risk_score, kept for the frontend), risk_level, risk_archetype and risk_reason
    onto every edge of G, plus the structural fields router.py needs (is_night, degree_of_connectivity,
    commercial_poi_count, active_open_shops_estimate).  Values are ALWAYS overwritten, so a cached graph never keeps a previous hour's labels.
    Without a loaded model (constant fallback risk) the score degrades to 100 x raw risk.
    """
    idx = scored_edges.index
    if not (isinstance(idx, pd.MultiIndex) and idx.nlevels == 3):
        raise ValueError("scored_edges must be indexed by (u, v, key)")
    raw = scored_edges["predicted_risk"].to_numpy(float)
    reg = _STORE.regressor
    cal = bind(reg).scores(raw) if reg is not None else 100.0 * np.clip(raw, 0.0, 1.0)
    reasons = scored_edges["risk_reason"] if "risk_reason" in scored_edges else [""] * len(scored_edges)
    # Structural context for router.py's CPTED rules (absent when the constant fallback scored the graph).
    ctx = {c: scored_edges[c].to_numpy(float) for c in ROUTER_EDGE_FIELDS if c in scored_edges}
    for i, ((u, v, k), risk, score, level, arche, reason) in enumerate(zip(
            idx, raw, cal, scored_edges["risk_level"], scored_edges["risk_archetype"], reasons)):
        if G.has_edge(u, v, k):
            d = G.edges[u, v, k]
            for c, arr in ctx.items():
                d[c] = float(arr[i])
            d["predicted_risk"] = float(risk)
            d["risk_score"] = float(score)
            d["insecurity_score"] = float(score)
            d["risk_level"] = str(level)
            d["risk_archetype"] = str(arche)
            d["risk_reason"] = str(reason)
    return G


def score_graph(G, static_df: pd.DataFrame, query_hour: int):
    """
    Zero-shot scoring + attach in one call.  Returns (G, notes).
    If the offline models are missing, every edge gets a constant risk (routing degrades to
    plain shortest-path) and `notes` says so -- the server never crashes for lack of a model.
    """
    if models_available():
        scored = predict_road_safety(static_df, query_hour, explain=False)
        return attach_risk_to_graph(G, scored), list(model_info()["notes"])
    scored = pd.DataFrame({"predicted_risk": FALLBACK_RISK, "risk_level": "low",
                           "risk_archetype": "", "risk_reason": ""}, index=static_df.index)
    return attach_risk_to_graph(G, scored), model_info()["notes"]


def make_explain_fn(static_df: pd.DataFrame, query_hour: int) -> Optional[Callable[[list], dict]]:
    """Callable for router.high_risk_segments(explain_fn=...); None when no model is loaded."""
    if not models_available():
        return None
    return lambda edge_ids: explain_edges(static_df, edge_ids, query_hour)


# --------------------------------------------------------------------------- #
# Calibration info, model directory, and baseline-comparison "reasons"
# --------------------------------------------------------------------------- #
def get_model_dir() -> Optional[Path]:
    """Directory the artifacts were loaded from (also holds training_report.json / evaluation_plots/)."""
    _ensure_loaded()
    return _STORE.model_dir


def calibration_info() -> list:
    """Anchor table of the raw->0-100 calibration for the loaded model ([] when no model)."""
    _ensure_loaded()
    reg = _STORE.regressor
    return bind(reg).table() if reg is not None else []


def _edge_reason_list(row: Mapping[str, float], base: Mapping[str, float], hour: int) -> list:
    """Plain-language facts about ONE edge compared with the training baseline (median street)."""
    g = lambda k, default=0.0: float(row.get(k, base.get(k, default)))      # noqa: E731
    b = lambda k, default=0.0: float(base.get(k, default))                  # noqa: E731
    out = []
    if int(g("is_night")):
        out.append(f"Late-night window ({hour:02d}:00; night = 20:00-06:00)")
    shops, open_now = g("commercial_poi_count"), g("active_open_shops_estimate")
    if shops > 0 and open_now < 0.25 * max(shops, 1.0) and not int(g("is_night")):
        out.append("Most nearby shops are probably closed at this hour")
    police, police_med = g("distance_to_nearest_police"), b("distance_to_nearest_police")
    if police >= 2500:
        out.append("Isolated from police response (>2.5 km)")
    elif police >= 1000 and police >= 1.5 * police_med:
        out.append(f"Farther from police than a typical street ({police / 1000:.1f} km vs {police_med / 1000:.1f} km median)")
    elif police <= 300:
        out.append(f"Close to a police station ({police:.0f} m)")
    if shops >= max(10.0, 2.0 * b("commercial_poi_count")):
        out.append(f"Active commercial area ({shops:.0f} shops within 100 m)")
    elif shops == 0:
        out.append("No shops within 100 m (little passive surveillance)")
    deg = g("degree_of_connectivity")
    if deg <= 1:
        out.append("Dead-end or poorly connected segment")
    elif deg >= max(4.0, b("degree_of_connectivity") + 2.0):
        out.append(f"Busy junction area ({deg:.0f} connecting streets)")
    iso = g("isolation_index")
    if iso >= max(1.0, 2.0 * b("isolation_index", 0.5)):
        out.append("Isolated lane: poorly connected and far from police response")
    if g("darkness_hazard") >= 0.9:
        out.append("Dark stretch at night: no open shops nearby for light or passive surveillance")
    length = math.expm1(g("log_length"))
    if length >= max(150.0, 2.0 * b("length")):
        out.append(f"Long unbroken stretch ({length:.0f} m vs {b('length'):.0f} m median)")
    rt = int(round(g("road_type_ord")))
    if rt <= 1:
        out.append(f"{_ROAD_NAMES.get(rt, 'Footpath').capitalize()} (pedestrian-only or traffic-calmed)")
    elif rt >= 5:
        out.append(f"{_ROAD_NAMES.get(rt, 'Major road').capitalize()} (fast, high-traffic road)")
    return out


def _edge_feature_frame(static_df: pd.DataFrame, edge_ids: list, hour: int, art: dict) -> pd.DataFrame:
    sub = static_df.reindex(pd.MultiIndex.from_tuples(edge_ids, names=static_df.index.names))
    return compute_temporal_features(_prepare_static(sub, art), hour)[FEATURE_COLUMNS]


def reasons_for_edges(static_df: pd.DataFrame, edge_ids: Iterable, query_hour: int) -> dict:
    """{(u, v, key): [reason, ...]} comparing each edge's features with artifact['baseline_values']."""
    art = _require_regressor()
    ids = list(dict.fromkeys(tuple(e) for e in edge_ids))
    if not ids:
        return {}
    hour = _norm_hour(query_hour)
    X = _edge_feature_frame(static_df, ids, hour, art)
    base = art["baseline_values"]
    return {eid: _edge_reason_list(row, base, hour) for eid, row in zip(ids, X.to_dict("records"))}


def route_reasons(static_df: pd.DataFrame, edge_ids: Iterable, query_hour: int, profile: Optional[dict] = None,
                  peak_reasons: Optional[list] = None) -> list:
    """
    Route-level `reasons`: length-weighted comparison of the route against the training baseline,
    plus the risk profile (share of elevated length, peak segment).  `edge_ids` are the route's (u, v, key).
    """
    art = _require_regressor()
    ids = [tuple(e) for e in edge_ids]
    hour = _norm_hour(query_hour)
    if not ids:
        return []
    X = _edge_feature_frame(static_df, ids, hour, art)
    base = art["baseline_values"]
    lengths = np.expm1(X["log_length"].to_numpy(float))
    total = float(lengths.sum()) or 1.0
    share = lambda mask: 100.0 * float(lengths[np.asarray(mask)].sum()) / total            # noqa: E731
    out = []
    if hour >= 20 or hour < 6:
        out.append(f"Late-night window ({hour:02d}:00): shops are mostly shut and streets are quiet")
    p = share(X["distance_to_nearest_police"].to_numpy() >= 2500)
    if p >= 20:
        out.append(f"{p:.0f}% of the route is isolated from police response (>2.5 km)")
    p = share(X["commercial_poi_count"].to_numpy() >= max(10.0, 2.0 * float(base["commercial_poi_count"])))
    if p >= 25:
        out.append(f"{p:.0f}% of the route runs through an active commercial area")
    p = share(X["commercial_poi_count"].to_numpy() == 0)
    if p >= 50:
        out.append(f"{p:.0f}% of the route has no shops within 100 m")
    p = share(X["degree_of_connectivity"].to_numpy() <= 1)
    if p >= 10:
        out.append(f"{p:.0f}% of the route is on dead-end or poorly connected segments")
    if profile:
        if profile.get("pct_elevated_length", 0) >= 10:
            out.append(f"{profile['pct_elevated_length']:.0f}% of the route length is rated elevated or high risk")
        if profile.get("peak_segment_risk", 0) >= AT_ELEVATED:
            tail = f": {'; '.join(peak_reasons[:2])}" if peak_reasons else ""
            out.append(f"Peak segment scores {profile['peak_segment_risk']:.0f}/100{tail}")
    return out or ["No factor stands out against the training baseline."]


# --------------------------------------------------------------------------- #
# Smoke test on any cached area:
#   python ml_engine.py --graph cache_graph_hauz_khas_delhi_india.graphml \
#                       --pois  cache_pois_hauz_khas_delhi_india.geojson --hour 23
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    import geopandas as gpd
    import osmnx as ox

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    ap = argparse.ArgumentParser(description="Zero-shot scoring of a cached OSM area")
    ap.add_argument("--graph", required=True, help="GraphML file")
    ap.add_argument("--pois", required=True, help="GeoJSON POI file")
    ap.add_argument("--hour", type=int, default=23)
    args = ap.parse_args()

    if not load_models():
        raise SystemExit("No trained model found. Run: python train_offline.py")
    G_ = ox.load_graphml(args.graph)
    static_ = build_edge_feature_frame(G_, gpd.read_file(args.pois))
    scored_ = predict_road_safety(static_, args.hour)
    print(f"{len(scored_)} edges | mean risk {scored_['predicted_risk'].mean():.3f} | "
          f"levels: {scored_['risk_level'].value_counts().to_dict()}")
    for hr in (3, 12, 22):
        print(f"  hour {hr:02d}: mean predicted risk = {predict_road_safety(static_, hr)['predicted_risk'].mean():.3f}")
    top = scored_.nlargest(5, "predicted_risk").index.tolist()
    for eid, det in explain_edges(static_, top, args.hour).items():
        print(f"  edge {eid}: {det['reason']}")
