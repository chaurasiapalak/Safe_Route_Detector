"""
server.py
---------
Flask backend for the interactive Spatio-Temporal Safe Route Optimizer
frontend. Serves the Leaflet-based single-page UI (templates/index.html)
and exposes a small JSON API the frontend calls via fetch():

  POST /api/load_area
      body: { "place": str, "dist": int }
      -> downloads/loads (and caches in-process) the road graph + POIs
         within `dist` metres of the geocoded `place`, scores its
         edges with the PRE-TRAINED risk regressor (ml_engine), and returns the area's centre
         point (used as the DEFAULT origin) and bounding box (used to
         fit the map view).

  POST /api/geocode
      body: { "query": str }
      -> resolves free-text typed by the user (in the Start/Destination
         search boxes) to a (lat, lon) point, RESTRICTED to a bounding
         box around the currently loaded area (see geocode_near()) so a
         same-named place in a different city is never returned. This is
         how locations are chosen now -- typed and geocoded, not clicked
         or dragged on the map.

  POST /api/route
      body: { "place": str, "dist": int, "hour": int, "lambda": float,
              "orig": [lat, lon] | null, "dest": [lat, lon] }
      -> computes the shortest / safest / balanced paths between `orig`
         (or the area's centre point if `orig` is omitted -- the
         "default start location" requirement) and `dest` (REQUIRED),
         and returns each route as a GeoJSON LineString plus summary
         stats. Also returns `warnings` when a typed point snaps to a
         graph node far from where it was actually typed (i.e. it likely
         falls outside the loaded area's coverage radius).

Run:
    python server.py
    then open http://127.0.0.1:5000 in a browser.

Design decisions (for viva defense):
- IN-PROCESS CACHING PER AREA: geocoding + graph download + RandomForest
  training are all expensive (seconds to tens of seconds). Re-running
  them on every single route request -- which would happen on every
  slider drag in an interactive UI -- would make the frontend feel
  broken. Instead a single module-level `_AREA` cache holds the built
  graph, POIs, and trained model, and `/api/route` only ever redoes the
  genuinely TIME-DEPENDENT, CHEAP step (temporal feature derivation +
  model prediction + weight normalization). This mirrors the two-tier
  static/temporal split already used inside feature_engineering.py --
  we're just applying the same idea at the web-request layer.
- LOADING A NEW AREA IS A SEPARATE, EXPLICIT ACTION: `/api/load_area`
  is only called when the user clicks "Load Area", never implicitly from
  `/api/route`, so an expensive re-download/re-train never happens by
  surprise from a slider drag or a map click.
- DESTINATION REQUIRED, ORIGIN DEFAULTED: implements the requirement
  directly -- there's no sensible universal default endpoint, so `dest`
  must always be supplied by the user (clicking the map), while `orig`
  defaults to the loaded area's geocoded centre point and can be
  overridden by dragging the origin marker.
- SINGLE-PROCESS, SINGLE-AREA CACHE: this is a student-project demo
  server (Flask's single-threaded dev server), so one shared `_AREA`
  object is safe and simple. A production multi-user deployment would
  need a per-session cache (e.g. keyed in Redis by session id) instead
  of one global -- worth stating explicitly if asked about scaling this
  beyond a single-user demo.
"""

import json
import math
import os
import traceback
from pathlib import Path

import requests
from flask import Flask, jsonify, request, render_template, send_from_directory
import osmnx as ox

from data_extraction import fetch_road_graph, fetch_pois
from feature_engineering import compute_static_and_poi_features
import ml_engine
from router import (
    compute_shortest_path,
    compute_safest_path,
    path_average_insecurity,  # noqa: F401  (kept for backward compatibility)
    path_edge_keys,
    path_risk_profile,
    high_risk_segments,
)
from geo_utils import path_to_geojson_feature, haversine_m
import re

def get_cache_filenames(place_name: str, dist: int = 2500, cache_dir: str = "cache"):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", place_name.strip()).strip("_").lower()
    graph_file = f"{cache_dir}/graph_{slug}_{dist}m.graphml"
    pois_file = f"{cache_dir}/pois_{slug}_{dist}m.geojson"
    return graph_file, pois_file

app = Flask(__name__)

# The risk model is trained OFFLINE (python train_offline.py) and loaded once here. Loading a new
# area never trains anything: edge features are extracted on the fly and scored zero-shot.
ml_engine.load_models()

# If a geocoded start/destination point snaps to a graph node farther
# than this many metres away, we surface a warning to the user rather
# than silently returning a route that starts/ends somewhere they didn't
# expect. This should rarely fire now that /api/geocode is bounded to
# search near the loaded area (see geocode_near below); it mainly
# catches genuinely borderline cases just outside the coverage radius.
SNAP_WARNING_THRESHOLD_M = 400

# HARD CAP on the coverage radius. A pedestrian "walk" network's edge
# count grows roughly with the AREA of the circle, i.e. with dist^2 --
# a 2500 m radius (the default) already downloads several thousand
# edges; a 40,000 m radius (as tried in testing) covers ~5,000 km^2 and
# would mean an Overpass query and RandomForest training pass that are
# impractical for a live demo (likely minutes-to-hours, if the Overpass
# query doesn't just time out first). Capping this server-side (not just
# via the HTML <input max>, which a user can bypass by typing a larger
# number directly) keeps every request within what's actually feasible.
MAX_COVERAGE_RADIUS_M = 5000

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "SafeRouteOptimizer-StudentProject/1.0 (contact: set-your-email-here)"


def geocode_near(query: str, center_lat: float, center_lon: float, radius_m: float):
    """
    Geocode `query` to a (lat, lon) point, but RESTRICTED to a bounding
    box around the loaded area's centre -- rather than osmnx's plain
    `ox.geocode()`, which does a global, unscoped Nominatim lookup and
    can happily return a same-named place in a totally different city
    (e.g. searching "Wave City" while a small Mehrauli-radius graph is
    loaded matched Wave City, Ghaziabad -- ~30+ km away -- because
    nothing told Nominatim to prefer results near the loaded area).

    We call Nominatim's /search endpoint directly with `viewbox` +
    `bounded=1`, which makes Nominatim return ONLY results that actually
    fall inside that box, instead of ranking a distant same-named match
    highest globally. The box is padded well beyond `radius_m` so a
    place just past the loaded street network's edge can still resolve
    -- the ROUTER (not the geocoder) is what should ultimately decide
    whether a point is usably close, via the snap-distance warning in
    /api/route.

    Returns (lat, lon) or None if nothing matched inside the box.
    """
    # ~111,000 m per degree of latitude; pad generously (3x the coverage
    # radius) so genuinely-nearby places aren't excluded by an overly
    # tight box, while a different city is still reliably excluded.
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
        "bounded": 1,  # hard-restrict results to inside the viewbox, not just prefer them
    }
    resp = requests.get(
        NOMINATIM_SEARCH_URL, params=params,
        headers={"User-Agent": NOMINATIM_USER_AGENT}, timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"])


class AreaCache:
    """Holds everything expensive-to-compute for the CURRENTLY loaded area."""

    def __init__(self):
        self.place = None
        self.dist = None
        self.G = None
        self.pois = None
        self.static_features = None   # time-independent edge features (see feature_engineering.py)
        self.model_info = None        # ml_engine.model_info() snapshot (metrics, importances, notes)
        self.center_latlon = None     # (lat, lon) -- the DEFAULT origin
        self.bounds = None            # (south, west, north, east)


_AREA = AreaCache()


def _area_center_and_bounds(G):
    """Compute a simple centroid + bounding box from node coordinates,
    used to (a) give the frontend a sensible default map view and
    (b) provide the DEFAULT origin point when the user hasn't dragged
    the start marker."""
    lats = [data["y"] for _, data in G.nodes(data=True)]
    lons = [data["x"] for _, data in G.nodes(data=True)]
    center = (sum(lats) / len(lats), sum(lons) / len(lons))
    bounds = (min(lats), min(lons), max(lats), max(lons))
    return center, bounds


def ensure_area_loaded(place_name: str, dist: int):
    """
    Idempotent load: if `place_name`/`dist` already match what's cached
    in `_AREA`, this is a no-op (near-instant). Otherwise it runs the
    full data + training pipeline once and repopulates the cache.

    Enforces MAX_COVERAGE_RADIUS_M here (the single choke point both
    /api/load_area and /api/route funnel through) rather than only in
    the HTML <input max>, which a user can bypass by typing a larger
    number directly.
    """
    if dist > MAX_COVERAGE_RADIUS_M:
        raise ValueError(
            f"Coverage radius capped at {MAX_COVERAGE_RADIUS_M} m -- a pedestrian network's "
            f"size grows with the AREA of the circle (roughly dist^2), so {dist} m would mean "
            f"an impractically large Overpass download and RandomForest training pass."
        )
    if dist < 200:
        raise ValueError("Coverage radius must be at least 200 m.")

    if _AREA.place == place_name and _AREA.dist == dist:
        return

    graph_cache_path, poi_cache_path = get_cache_filenames(place_name)
    G = fetch_road_graph(place_name=place_name, dist=dist, cache_path=graph_cache_path)
    pois = fetch_pois(place_name=place_name, dist=dist, cache_path=poi_cache_path)

    # Time-independent features computed ONCE per area (KDTree builds,
    # POI spatial joins, etc.) -- see feature_engineering.py.
    static_df = compute_static_and_poi_features(G, pois)

    # No training here: the pre-trained regressor scores these edges per request (see api_route).
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


@app.route("/api/geocode", methods=["POST"])
def api_geocode():
    """
    Resolve a free-text location string (typed by the user in the Start
    or Destination search box) to a (lat, lon) point -- RESTRICTED to
    search near the currently loaded area (see geocode_near() docstring
    for why: an unscoped global geocode can match a same-named place in
    a different city entirely, which is what produced the "11,771,760 m
    away" nonsense warning). This replaces click-on-map / drag-marker
    location picking entirely -- the frontend calls this once per "Set
    Start" / "Set Destination" action and places a plain (non-draggable)
    marker at the result.
    """
    payload = request.get_json(force=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Type a location to search for."}), 400

    if _AREA.center_latlon is None:
        return jsonify({"error": "Load an area first, so there's somewhere to search near."}), 400

    center_lat, center_lon = _AREA.center_latlon
    try:
        result = geocode_near(query, center_lat, center_lon, _AREA.dist)
    except requests.RequestException as e:
        return jsonify({"error": f"Geocoding service unavailable right now: {e}"}), 502

    if result is None:
        return jsonify({
            "error": f"Could not find \u201c{query}\u201d near the loaded area "
                     f"(\u201c{_AREA.place}\u201d, {_AREA.dist} m radius). Try a more specific "
                     f"local name (a nearby landmark, market, or road), or load the area that "
                     f"actually contains this place instead."
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
    orig = payload.get("orig")   # [lat, lon] or None -> defaults to area centre
    dest = payload.get("dest")   # [lat, lon] -- REQUIRED

    if dest is None:
        return jsonify({"error": "A destination is required. Click the map to set one."}), 400

    try:
        ensure_area_loaded(place, dist)

        # --- Recompute ONLY the genuinely time-dependent parts (cheap) ---
        G, model_notes = ml_engine.score_graph(_AREA.G, _AREA.static_features, hour)
        explain_fn = ml_engine.make_explain_fn(_AREA.static_features, hour)

        # Origin: explicit user point if the marker was dragged, else the
        # area's centre point (the "default start location" requirement).
        if orig is not None:
            orig_lat, orig_lon = orig
        else:
            orig_lat, orig_lon = _AREA.center_latlon
        orig_node = ox.distance.nearest_nodes(G, X=orig_lon, Y=orig_lat)

        dest_lat, dest_lon = dest
        dest_node = ox.distance.nearest_nodes(G, X=dest_lon, Y=dest_lat)

        if orig_node == dest_node:
            return jsonify({"error": "Origin and destination snapped to the same street node -- "
                                      "pick points farther apart."}), 400

        # Warn (but don't block) when a typed point is meaningfully far
        # from the nearest street actually present in the loaded graph --
        # this happens when the address falls outside the current
        # coverage radius, and nearest_nodes() would otherwise silently
        # snap it to whatever node is closest, however far that is.
        warnings = list(model_notes)  # e.g. model missing / weak-skill notices (usually empty)
        orig_snap_dist = haversine_m(orig_lat, orig_lon, G.nodes[orig_node]["y"], G.nodes[orig_node]["x"])
        if orig_snap_dist > SNAP_WARNING_THRESHOLD_M:
            warnings.append(
                f"Start point is {orig_snap_dist:.0f} m from the nearest loaded street. "
                f"It may be outside the current coverage radius -- try increasing it."
            )
        dest_snap_dist = haversine_m(dest_lat, dest_lon, G.nodes[dest_node]["y"], G.nodes[dest_node]["x"])
        if dest_snap_dist > SNAP_WARNING_THRESHOLD_M:
            warnings.append(
                f"Destination is {dest_snap_dist:.0f} m from the nearest loaded street. "
                f"It may be outside the current coverage radius -- try increasing it."
            )

        shortest_path, _, shortest_dist = compute_shortest_path(G, orig_node, dest_node)
        balanced_path, _, balanced_dist = compute_safest_path(G, orig_node, dest_node, lam=lam_balanced)
        safe_path, _, safe_dist = compute_safest_path(G, orig_node, dest_node, lam=0.0)

        def route_payload(path, dist_m, color, label):
            profile = path_risk_profile(G, path)                       # calibrated 0-100 metrics
            segments = high_risk_segments(G, path, min_level="elevated", explain_fn=explain_fn)
            reasons = []
            if ml_engine.models_available():
                try:
                    peak = profile["peak_segment"]
                    wanted = [tuple(sg["edge"]) for sg in segments] + ([tuple(peak["edge"])] if peak else [])
                    per_edge = ml_engine.reasons_for_edges(_AREA.static_features, wanted, hour)
                    for sg in segments:
                        sg["reasons"] = per_edge.get(tuple(sg["edge"]), [])
                    peak_reasons = per_edge.get(tuple(peak["edge"]), []) if peak else []
                    if peak:
                        peak["reasons"] = peak_reasons
                    reasons = ml_engine.route_reasons(_AREA.static_features, path_edge_keys(G, path), hour,
                                                      profile, peak_reasons)
                except Exception:                      # reasons are a nicety; never break routing
                    traceback.print_exc()
            return {
                "geojson": path_to_geojson_feature(G, path, properties={"label": label, "color": color}),
                "distance_m": round(dist_m, 1),
                "avg_risk": profile["avg_risk"],                    # distance-weighted, calibrated 0-100
                "avg_insecurity": profile["avg_risk"],              # legacy field name used by old frontends
                "peak_segment_risk": profile["peak_segment_risk"],  # worst single segment on the route
                "peak_segment": profile["peak_segment"],
                "pct_elevated_length": profile["pct_elevated_length"],
                "pct_high_length": profile["pct_high_length"],
                "reasons": reasons,
                "label": label,
                "color": color,
                "high_risk_segments": segments,
            }

        return jsonify({
            "origin_used": [G.nodes[orig_node]["y"], G.nodes[orig_node]["x"]],
            "destination_used": [G.nodes[dest_node]["y"], G.nodes[dest_node]["x"]],
            "warnings": warnings,
            "hour": hour,
            "risk_scale": ml_engine.calibration_info(),
            "routes": {
                "shortest": route_payload(shortest_path, shortest_dist, "#1f6feb", "Shortest (\u03bb=1.0)"),
                "balanced": route_payload(balanced_path, balanced_dist, "#f0883e", f"Balanced (\u03bb={lam_balanced})"),
                "safest": route_payload(safe_path, safe_dist, "#2ea043", "Safest (\u03bb=0.0)"),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# --------------------------------------------------------------------------- #
# Model diagnostics (reads the files written by train_offline.py / evaluate_and_visualize.py)
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    """NaN/inf are not valid JSON -> None; recurse through dicts/lists."""
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
    """Honest, plain-language reading of the hold-out / spatial-CV numbers."""
    ho, cv = metrics.get("holdout") or {}, metrics.get("spatial_cv") or {}
    out = []
    r2, rho = ho.get("r2"), ho.get("spearman")
    if r2 is not None:
        out.append(f"Hold-out R\u00b2 is {r2:.3f}: the model explains "
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
        out.append("Spatial cross-validation R\u00b2 is about zero: on completely unseen areas the model "
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
    # debug=True (auto-reload + interactive traceback) is fine for a local
    # student-project demo; turn it off for anything beyond localhost.
    app.run(debug=True, port=5000)
