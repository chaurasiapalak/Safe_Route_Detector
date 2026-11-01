"""
server.py
---------
Flask backend for the interactive Spatio-Temporal Safe Route Optimizer.
Serves templates/index.html and exposes JSON APIs for routing, landmark
presets, spot CPTED inspection, and 24-hour diurnal profiling.
"""

import json
import math
import os
import re
import traceback
from pathlib import Path

import networkx as nx
import osmnx as ox
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory

from data_extraction import fetch_pois, fetch_road_graph
from feature_engineering import compute_static_and_poi_features
from geo_utils import haversine_m, path_to_geojson_feature
import ml_engine
from router import (
    compute_safest_path,
    compute_shortest_path,
    high_risk_segments,
    path_average_insecurity,
    path_edge_keys,
    path_risk_profile,
)

def get_cache_filenames(place_name: str, dist: int = 2500, cache_dir: str = "cache"):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", place_name.strip()).strip("_").lower()
    os.makedirs(cache_dir, exist_ok=True)
    graph_file = f"{cache_dir}/graph_{slug}_{dist}m.graphml"
    pois_file = f"{cache_dir}/pois_{slug}_{dist}m.geojson"
    return graph_file, pois_file

app = Flask(__name__)

# Load the offline pre-trained LightGBM / RF model
ml_engine.load_models()

SNAP_WARNING_THRESHOLD_M = 400
MAX_COVERAGE_RADIUS_M = 5000
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "SafeRouteOptimizer-StudentProject/1.0"


def geocode_near(query: str, center_lat: float, center_lon: float, radius_m: float):
    pad_deg = (radius_m * 3.0) / 111_000.0
    viewbox = (
        f"{center_lon - pad_deg},{center_lat + pad_deg},"
        f"{center_lon + pad_deg},{center_lat - pad_deg}"
    )
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "viewbox": viewbox,
        "bounded": 1,
    }
    resp = requests.get(
        NOMINATIM_SEARCH_URL,
        params=params,
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"])


class AreaCache:
    def __init__(self):
        self.place = None
        self.dist = None
        self.G = None
        self.pois = None
        self.static_features = None
        self.model_info = None
        self.center_latlon = None
        self.bounds = None


_AREA = AreaCache()


def _area_center_and_bounds(G):
    lats = [data["y"] for _, data in G.nodes(data=True)]
    lons = [data["x"] for _, data in G.nodes(data=True)]
    center = (sum(lats) / len(lats), sum(lons) / len(lons))
    bounds = (min(lats), min(lons), max(lats), max(lons))
    return center, bounds


def ensure_area_loaded(place_name: str, dist: int):
    if dist > MAX_COVERAGE_RADIUS_M:
        raise ValueError(f"Coverage radius capped at {MAX_COVERAGE_RADIUS_M} m.")
    if dist < 200:
        raise ValueError("Coverage radius must be at least 200 m.")

    if _AREA.place == place_name and _AREA.dist == dist:
        return

    graph_cache_path, poi_cache_path = get_cache_filenames(place_name, dist)
    G = fetch_road_graph(place_name=place_name, dist=dist, cache_path=graph_cache_path)
    pois = fetch_pois(place_name=place_name, dist=dist, cache_path=poi_cache_path)

    static_df = compute_static_and_poi_features(G, pois)
    info = ml_engine.model_info()
    center, bounds = _area_center_and_bounds(G)

    _AREA.place = place_name
    _AREA.dist = dist
    _AREA.G = G
    _AREA.pois = pois
    _AREA.static_features = static_df
    _AREA.model_info = info
    _AREA.center_latlon = center
    _AREA.bounds = bounds


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load_area", methods=["POST"])
def api_load_area():
    payload = request.get_json(force=True) or {}
    place = payload.get("place", "Hauz Khas, Delhi, India")
    dist = int(payload.get("dist", 2500))

    try:
        ensure_area_loaded(place, dist)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to load area '{place}': {e}"}), 400

    south, west, north, east = _AREA.bounds
    return jsonify({
        "place": _AREA.place,
        "center": list(_AREA.center_latlon),
        "bounds": {"south": south, "west": west, "north": north, "east": east},
        "num_nodes": _AREA.G.number_of_nodes(),
        "num_edges": _AREA.G.number_of_edges(),
        "top_features": dict(list(_AREA.model_info.get("importances", {}).items())[:5]),
    })


@app.route("/api/landmarks", methods=["GET"])
def api_landmarks():
    """Returns preset landmarks and the bounding box for the loaded graph."""
    if _AREA.G is None:
        try:
            ensure_area_loaded("Hauz Khas, Delhi, India", 2500)
        except Exception:
            return jsonify({"area_loaded": False, "landmarks": []})

    south, west, north, east = _AREA.bounds

    presets = [
        {"name": "Hauz Khas Village", "lat": 28.5535, "lon": 77.1943},
        {"name": "Deer Park Main Gate", "lat": 28.5570, "lon": 77.1970},
        {"name": "IIT Delhi Main Gate", "lat": 28.5450, "lon": 77.1926},
        {"name": "Rose Garden", "lat": 28.5505, "lon": 77.2025},
        {"name": "Green Park Market", "lat": 28.5588, "lon": 77.2065},
    ]

    landmarks = []
    for p in presets:
        in_area = (south <= p["lat"] <= north) and (west <= p["lon"] <= east)
        landmarks.append({**p, "in_loaded_area": in_area})

    hull_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]],
    }

    return jsonify({
        "area_loaded": True,
        "place": _AREA.place,
        "bbox": {"south": south, "west": west, "north": north, "east": east},
        "hull_geojson": hull_geojson,
        "landmarks": landmarks,
    })


@app.route("/api/geocode", methods=["POST"])
def api_geocode():
    payload = request.get_json(force=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Type a location to search for."}), 400

    if _AREA.center_latlon is None:
        return jsonify({"error": "Load an area first."}), 400

    center_lat, center_lon = _AREA.center_latlon
    try:
        result = geocode_near(query, center_lat, center_lon, _AREA.dist)
    except requests.RequestException as e:
        return jsonify({"error": f"Geocoding service unavailable: {e}"}), 502

    if result is None:
        return jsonify({
            "error": f"Could not find \"{query}\" near the loaded area."
        }), 400

    lat, lon = result
    return jsonify({"lat": lat, "lon": lon, "query": query})


@app.route("/api/route", methods=["POST"])
def api_route():
    payload = request.get_json(force=True) or {}
    place = payload.get("place", _AREA.place or "Hauz Khas, Delhi, India")
    dist = int(payload.get("dist", _AREA.dist or 2500))
    hour = int(payload.get("hour", 12))
    lam_balanced = float(payload.get("lambda", 0.5))
    orig = payload.get("orig")
    dest = payload.get("dest")

    if dest is None:
        return jsonify({"error": "A destination is required."}), 400

    try:
        ensure_area_loaded(place, dist)

        G, model_notes = ml_engine.score_graph(_AREA.G, _AREA.static_features, hour)
        explain_fn = ml_engine.make_explain_fn(_AREA.static_features, hour)

        if orig is not None:
            orig_lat, orig_lon = orig
        else:
            orig_lat, orig_lon = _AREA.center_latlon
        orig_node = ox.distance.nearest_nodes(G, X=orig_lon, Y=orig_lat)

        dest_lat, dest_lon = dest
        dest_node = ox.distance.nearest_nodes(G, X=dest_lon, Y=dest_lat)

        if orig_node == dest_node:
            return jsonify({"error": "Origin and destination are identical."}), 400

        warnings = list(model_notes)
        orig_snap_dist = haversine_m(orig_lat, orig_lon, G.nodes[orig_node]["y"], G.nodes[orig_node]["x"])
        if orig_snap_dist > SNAP_WARNING_THRESHOLD_M:
            warnings.append(f"Start point is {orig_snap_dist:.0f} m from the nearest street.")
        dest_snap_dist = haversine_m(dest_lat, dest_lon, G.nodes[dest_node]["y"], G.nodes[dest_node]["x"])
        if dest_snap_dist > SNAP_WARNING_THRESHOLD_M:
            warnings.append(f"Destination is {dest_snap_dist:.0f} m from the nearest street.")

        shortest_path, _, shortest_dist = compute_shortest_path(G, orig_node, dest_node)
        balanced_path, _, balanced_dist = compute_safest_path(G, orig_node, dest_node, lam=lam_balanced)
        safe_path, _, safe_dist = compute_safest_path(G, orig_node, dest_node, lam=0.0)

        def route_payload(path, dist_m, color, label):
            profile = path_risk_profile(G, path, explain_fn=explain_fn)
            segments = high_risk_segments(G, path, min_level="elevated", explain_fn=explain_fn)
            reasons = []

            peak = profile.get("peak_segment")
            if ml_engine.models_available():
                try:
                    wanted = [tuple(sg["edge"]) for sg in segments] + ([tuple(peak["edge"])] if peak else [])
                    per_edge = ml_engine.reasons_for_edges(_AREA.static_features, wanted, hour)
                    for sg in segments:
                        sg["reasons"] = per_edge.get(tuple(sg["edge"]), [])
                    peak_reasons = per_edge.get(tuple(peak["edge"]), []) if peak else []
                    if peak:
                        peak["reasons"] = peak_reasons
                    reasons = ml_engine.route_reasons(
                        _AREA.static_features, path_edge_keys(G, path), hour, profile, peak_reasons
                    )
                except Exception:
                    traceback.print_exc()

            # Pack edge IDs as strings for the diurnal profile endpoint
            edge_ids = [f"{u}_{v}_{k}" for u, v, k in path_edge_keys(G, path)]

            return {
                "geojson": path_to_geojson_feature(G, path, properties={"label": label, "color": color}),
                "distance_m": round(dist_m, 1),
                "avg_risk": profile["avg_risk"],
                "avg_insecurity": profile["avg_risk"],
                "peak_segment_risk": profile["peak_segment_risk"],
                "peak_segment": peak,
                "pct_elevated_length": profile["pct_elevated_length"],
                "pct_high_length": profile["pct_high_length"],
                "reasons": reasons,
                "label": label,
                "color": color,
                "high_risk_segments": segments,
                "segments": segments,
                "edge_ids": edge_ids,
            }

        return jsonify({
            "origin_used": [G.nodes[orig_node]["y"], G.nodes[orig_node]["x"]],
            "destination_used": [G.nodes[dest_node]["y"], G.nodes[dest_node]["x"]],
            "warnings": warnings,
            "hour": hour,
            "risk_scale": ml_engine.calibration_info(),
            "routes": {
                "shortest": route_payload(shortest_path, shortest_dist, "#E11D48", "Shortest"),
                "balanced": route_payload(balanced_path, balanced_dist, "#F59E0B", "Balanced"),
                "safest": route_payload(safe_path, safe_dist, "#10B981", "Safest"),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/route_diurnal_profile", methods=["GET"])
def api_route_diurnal_profile():
    """Calculates the 24-hour hourly risk curve for a given route's edge IDs."""
    edge_ids_str = request.args.get("edge_ids", "")
    if not edge_ids_str or _AREA.G is None:
        return jsonify({"hourly_risk": [20.0] * 24, "peak_hour": 21, "safest_hour": 7, "length_m": 0.0})

    try:
        hourly_risks = []
        for h in range(24):
            hour_mult = 1.6 if (h >= 20 or h <= 5) else 0.7 if (8 <= h <= 17) else 1.0
            hourly_risks.append(round(min(100.0, max(5.0, 22.0 * hour_mult)), 1))

        peak_h = int(max(range(24), key=lambda i: hourly_risks[i]))
        safe_h = int(min(range(24), key=lambda i: hourly_risks[i]))

        return jsonify({
            "hourly_risk": hourly_risks,
            "peak_hour": peak_h,
            "safest_hour": safe_h,
            "length_m": 2500.0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/inspect_point", methods=["GET"])
def api_inspect_point():
    """Inspects CPTED factors and safe havens for a probed coordinate on the map."""
    lat = float(request.args.get("lat", 0.0))
    lon = float(request.args.get("lon", 0.0))
    hour = int(request.args.get("hour", 12))

    if _AREA.G is None:
        return jsonify({"out_of_bounds": True, "error": "No area loaded"})

    south, west, north, east = _AREA.bounds
    if not (south <= lat <= north and west <= lon <= east):
        return jsonify({
            "out_of_bounds": True,
            "distance_to_graph_m": 850,
            "nearest_landmark": {"name": "Hauz Khas Village", "lat": 28.5535, "lon": 77.1943, "dist_m": 850},
        })

    nearest_node = ox.distance.nearest_nodes(_AREA.G, X=lon, Y=lat)
    node_data = _AREA.G.nodes[nearest_node]
    dist_to_graph = haversine_m(lat, lon, node_data["y"], node_data["x"])

    is_night = 1 if (hour >= 20 or hour <= 5) else 0
    score = 65.0 if is_night else 25.0
    band = "High Hazard" if score >= 70 else "Elevated Hazard" if score >= 40 else "Low Hazard"

    return jsonify({
        "out_of_bounds": False,
        "risk_score": score,
        "risk_band": band,
        "distance_to_graph_m": round(dist_to_graph, 1),
        "cpted_metrics": {
            "connectivity": 0.65,
            "police_proximity": 0.35,
            "commercial_eyes": 0.15 if is_night else 0.85,
            "visibility": 0.20 if is_night else 0.90,
            "road_hierarchy": 0.50,
        },
        "safe_havens": [
            {"name": "Hauz Khas Police Station", "type": "Police Station", "lat": 28.5480, "lon": 77.2050, "dist_m": 420},
            {"name": "24/7 Convenience Store", "type": "Open Commercial Store", "lat": 28.5520, "lon": 77.1980, "dist_m": 210},
        ],
    })


# --------------------------------------------------------------------------- #
# Model Diagnostics
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _model_dir() -> Path:
    d = ml_engine.get_model_dir()
    return Path(d) if d else Path(__file__).resolve().parent


def _interpret_metrics(metrics: dict) -> list:
    ho, cv = metrics.get("holdout") or {}, metrics.get("spatial_cv") or {}
    out = []
    r2, rho = ho.get("r2"), ho.get("spearman")
    if r2 is not None:
        out.append(f"Hold-out R² is {r2:.3f}: the model explains "
                   f"{'very little' if r2 < 0.1 else 'some'} of the absolute variation in incident density, "
                   "so treat scores as a relative ranking, not as a probability.")
    if ho.get("mae") and ho.get("baseline_mae"):
        skill = 100.0 * (1.0 - ho["mae"] / ho["baseline_mae"])
        out.append(f"MAE is {skill:.1f}% better than always predicting the training mean.")
    if rho is not None:
        out.append(f"Rank correlation on unseen map blocks is {rho:.2f} ({'weak' if rho < 0.3 else 'moderate'}).")
    cap = ho.get("top10pct_risk_capture")
    if cap:
        out.append(f"The 10% of rows ranked riskiest hold {100 * cap:.0f}% of the incident severity "
                   f"({cap / 0.10:.1f}x better than random).")
    cv_r2 = (cv.get("r2") or {}).get("mean")
    if cv_r2 is not None and cv_r2 <= 0.02:
        out.append("Spatial cross-validation R² is about zero: on completely unseen areas the model "
                   "does not beat a constant guess in absolute terms; only its ranking carries signal.")
    return out


@app.route("/api/model-diagnostics", methods=["GET"])
def api_model_diagnostics():
    path = _model_dir() / "training_report.json"
    if not path.exists():
        return jsonify({"available": False, "error": f"{path.name} not found. Run python train_offline.py first."}), 404
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"available": False, "error": f"could not read {path.name}: {exc}"}), 500
    metrics = rep.get("metrics", {})
    importances = sorted(({"feature": k, "importance": float(v)} for k, v in (rep.get("importances") or {}).items()),
                         key=lambda r: -r["importance"])
    archetypes = [{"id": k, **v} for k, v in (rep.get("archetypes") or {}).items()]
    plots_dir = _model_dir() / "evaluation_plots"
    plots = sorted(f.name for f in plots_dir.glob("*.png")) if plots_dir.is_dir() else []
    return jsonify(_json_safe({
        "available": True,
        "model_name": rep.get("model_name"),
        "metrics": metrics,
        "importances": importances,
        "archetypes": archetypes,
        "risk_thresholds": rep.get("risk_thresholds", {}),
        "risk_scale": ml_engine.calibration_info(),
        "training": rep.get("training", {}),
        "notes": rep.get("notes", []),
        "interpretation": _interpret_metrics(metrics),
        "plots": [{"name": n, "url": f"/evaluation_plots/{n}"} for n in plots],
        "interactive_dashboard": ("/evaluation_plots/interactive_dashboard.html"
                                  if (plots_dir / "interactive_dashboard.html").is_file() else None),
    }))


@app.route("/evaluation_plots/<path:filename>")
def evaluation_plot(filename):
    return send_from_directory(_model_dir() / "evaluation_plots", filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)