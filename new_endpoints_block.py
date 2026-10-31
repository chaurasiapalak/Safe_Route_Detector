# =========================================================================== #
# NEW: /api/landmarks, /api/inspect_point, /api/route_diurnal_profile
# (paste this whole block ABOVE `if __name__ == "__main__":`)
# =========================================================================== #
# Everything below is read-only with respect to the shared `_AREA` cache and to the
# graph `_AREA.G`: it never calls ml_engine.score_graph / attach_risk_to_graph, so it can
# never overwrite the per-hour edge scores that /api/route just wrote.

OUT_OF_BOUNDS_M = 500.0     # point farther than this from any loaded street -> out_of_bounds
HAVEN_RADIUS_M = 500.0      # emergency safe-haven search radius
MAX_HAVENS = 2
SAMPLE_STEP_M = 15.0        # spacing of the points the nearest-edge KD-tree is built on
MAX_DIURNAL_EDGES = 5000

# Approximate coordinates -- sanity-checked against the cached graphs but NOT surveyed.
# Edit freely; the only contract is (name, lat, lon) + the city key.
PRESET_LANDMARKS = {
    "Hauz Khas (Delhi)": [
        {"name": "Hauz Khas Village",         "lat": 28.5535, "lon": 77.1941},
        {"name": "Hauz Khas Metro Station",   "lat": 28.5433, "lon": 77.2066},
        {"name": "Deer Park, Hauz Khas",      "lat": 28.5504, "lon": 77.1996},
        {"name": "IIT Delhi Main Gate",       "lat": 28.5450, "lon": 77.1926},
        {"name": "AIIMS Delhi",               "lat": 28.5672, "lon": 77.2100},
        {"name": "Green Park Metro Station",  "lat": 28.5593, "lon": 77.2069},
    ],
    "Shivaji Nagar (Pune)": [
        {"name": "Shivajinagar Railway Station", "lat": 18.5305, "lon": 73.8478},
        {"name": "Fergusson College",            "lat": 18.5236, "lon": 73.8410},
        {"name": "FC Road (Goodluck Chowk)",     "lat": 18.5195, "lon": 73.8413},
        {"name": "Deccan Gymkhana",              "lat": 18.5167, "lon": 73.8412},
        {"name": "Shaniwar Wada",                "lat": 18.5195, "lon": 73.8553},
        {"name": "Pune Railway Station",         "lat": 18.5286, "lon": 73.8742},
    ],
    "Jayanagar (Bengaluru)": [
        {"name": "Jayanagar 4th Block Shopping Complex", "lat": 12.9250, "lon": 77.5938},
        {"name": "Jayanagar Metro Station",              "lat": 12.9293, "lon": 77.5802},
        {"name": "South End Circle",                     "lat": 12.9378, "lon": 77.5800},
        {"name": "Jayadeva Hospital",                    "lat": 12.9176, "lon": 77.5997},
        {"name": "Ragigudda Temple",                     "lat": 12.9186, "lon": 77.5920},
        {"name": "Lalbagh West Gate",                    "lat": 12.9507, "lon": 77.5848},
    ],
}

# OSM `highway` tag -> 0.2 (footpath) .. 1.0 (arterial). Unknown tags default to residential.
HIGHWAY_HIERARCHY = {
    "motorway": 1.0, "motorway_link": 1.0, "trunk": 1.0, "trunk_link": 1.0,
    "primary": 1.0, "primary_link": 1.0,
    "secondary": 0.85, "secondary_link": 0.85,
    "tertiary": 0.70, "tertiary_link": 0.70,
    "unclassified": 0.55, "residential": 0.50,
    "living_street": 0.40, "service": 0.35,
    "pedestrian": 0.25, "track": 0.25, "cycleway": 0.25,
    "path": 0.20, "footway": 0.20, "steps": 0.20, "corridor": 0.20,
}
DEFAULT_HIERARCHY = 0.50

# Amenities counted as "commercial" when a place is not tagged shop=*.
COMMERCIAL_AMENITIES = {"pharmacy", "fuel", "cafe", "restaurant", "fast_food", "marketplace", "bank", "atm"}
_ALWAYS_OPEN = re.compile(r"^(24/7|(mo-su\s*)?00:00\s*-\s*24:00\+?)$")


def _flat_landmarks():
    return [dict(lm, city=city) for city, lms in PRESET_LANDMARKS.items() for lm in lms]


def _first(value, default=None):
    """OSM tags may be a scalar or a list (after simplification) -> one representative value."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _clean_str(value):
    v = _first(value)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):                        # list that was serialised as text
        parts = [t.strip(" '\"") for t in s[1:-1].split(",")]
        s = parts[0] if parts else ""
    return None if s.lower() in ("", "nan", "none") else s


def _haversine_vec(lat, lon, lats, lons):
    """Great-circle distance in metres from one point to arrays of points (same maths as geo_utils.haversine_m)."""
    p1, p2 = np.radians(lat), np.radians(lats)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lons - lon) / 2) ** 2)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# Per-area spatial index (built lazily once per loaded area, then reused)
# --------------------------------------------------------------------------- #
class _AreaIndex:
    """
    The app keeps no persistent edge KD-tree (nearest_nodes() builds one internally and throws it
    away), so this builds one: a scipy cKDTree over points sampled every ~SAMPLE_STEP_M along each
    edge's TRUE geometry, in local metres.  Querying it returns the nearest edge in O(log N), with a
    snapping error of at most ~SAMPLE_STEP_M / 2.

    Also owns the edge_id scheme used by all three endpoints:
        edge_id == positional row of the edge in _AREA.static_features.
    """

    def __init__(self, G, static_df):
        self.G, self.static = G, static_df
        self.edge_keys = list(static_df.index)                       # [(u, v, key), ...]
        self.pos = {k: i for i, k in enumerate(self.edge_keys)}

        lats = np.array([d["y"] for _, d in G.nodes(data=True)], float)
        lons = np.array([d["x"] for _, d in G.nodes(data=True)], float)
        self.lat0, self.lon0 = float(lats.mean()), float(lons.mean())
        self.kx = 111320.0 * math.cos(math.radians(self.lat0))
        self.ky = 110540.0

        pts_lat, pts_lon, owner = [], [], []
        for i, (u, v, k) in enumerate(self.edge_keys):
            if not G.has_edge(u, v, k):
                continue
            line = self._polyline(u, v, k)
            for (la1, lo1), (la2, lo2) in zip(line[:-1], line[1:]):
                seg_m = math.hypot((la2 - la1) * self.ky, (lo2 - lo1) * self.kx)
                n = max(1, int(math.ceil(seg_m / SAMPLE_STEP_M)))
                for t in range(n):
                    f = t / n
                    pts_lat.append(la1 + (la2 - la1) * f)
                    pts_lon.append(lo1 + (lo2 - lo1) * f)
                    owner.append(i)
            pts_lat.append(line[-1][0]); pts_lon.append(line[-1][1]); owner.append(i)

        self.pt_lat, self.pt_lon = np.array(pts_lat), np.array(pts_lon)
        self.owner = np.array(owner, dtype=int)
        self.tree = cKDTree(np.column_stack([(self.pt_lon - self.lon0) * self.kx,
                                             (self.pt_lat - self.lat0) * self.ky]))
        self._hull = None

    def _polyline(self, u, v, k):
        geom = self.G.edges[u, v, k].get("geometry")
        if geom is not None:
            return [(y, x) for x, y in geom.coords]                  # shapely gives (x=lon, y=lat)
        return [(self.G.nodes[u]["y"], self.G.nodes[u]["x"]), (self.G.nodes[v]["y"], self.G.nodes[v]["x"])]

    def nearest_edge(self, lat, lon):
        """-> (edge_id, snapped_lat, snapped_lon, distance_m)."""
        _, j = self.tree.query([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky], k=1)
        slat, slon = float(self.pt_lat[j]), float(self.pt_lon[j])
        return int(self.owner[j]), slat, slon, haversine_m(lat, lon, slat, slon)

    def hull(self):
        """Convex hull of all graph nodes -> (ring_latlon [[lat, lon], ...] closed, bbox dict)."""
        if self._hull is None:
            from scipy.spatial import ConvexHull
            lats = np.array([d["y"] for _, d in self.G.nodes(data=True)], float)
            lons = np.array([d["x"] for _, d in self.G.nodes(data=True)], float)
            bbox = {"south": float(lats.min()), "west": float(lons.min()),
                    "north": float(lats.max()), "east": float(lons.max())}
            try:
                idx = ConvexHull(np.column_stack([(lons - self.lon0) * self.kx,
                                                  (lats - self.lat0) * self.ky])).vertices
                ring = [[float(lats[i]), float(lons[i])] for i in idx]
            except Exception:                                        # <3 nodes / collinear -> use the bbox
                ring = [[bbox["south"], bbox["west"]], [bbox["south"], bbox["east"]],
                        [bbox["north"], bbox["east"]], [bbox["north"], bbox["west"]]]
            ring.append(ring[0])
            self._hull = (ring, bbox)
        return self._hull


_INDEX = {"obj": None}


def _area_index() -> _AreaIndex:
    """Index for the CURRENTLY loaded area; rebuilt automatically when /api/load_area swaps the area."""
    obj = _INDEX["obj"]
    if obj is None or obj.G is not _AREA.G or obj.static is not _AREA.static_features:
        _INDEX["obj"] = obj = _AreaIndex(_AREA.G, _AREA.static_features)
    return obj


def edge_ids_for_path(G, path) -> list:
    """edge_ids (the ids /api/route_diurnal_profile expects) of every hop of a computed route."""
    pos = _area_index().pos
    return [pos[k] for k in path_edge_keys(G, path) if k in pos]


# --------------------------------------------------------------------------- #
# Emergency safe havens: police stations + always-open (24/7) commercial POIs
# --------------------------------------------------------------------------- #
_HAVENS = {"src": None, "table": None}


def _haven_table():
    pois = _AREA.pois
    if _HAVENS["src"] is pois and _HAVENS["table"] is not None:
        return _HAVENS["table"]
    empty = {"lat": np.array([]), "lon": np.array([]), "name": [], "type": []}
    table = empty
    if pois is not None and len(pois):
        if getattr(pois, "crs", None) is not None and not pois.crs.is_geographic:
            pois = pois.to_crs(4326)
        pts = pois.geometry.representative_point()                  # works for points, lines and polygons
        lons, lats = pts.x.to_numpy(float), pts.y.to_numpy(float)

        def col(name):
            return list(pois[name]) if name in pois.columns else [None] * len(pois)

        amenity, shop, hours = col("amenity"), col("shop"), col("opening_hours")
        names = [(_clean_str(a) or _clean_str(b) or _clean_str(c))
                 for a, b, c in zip(col("name"), col("name:en"), col("brand"))]
        keep, types, out_names = [], [], []
        for i in range(len(pois)):
            am, sh, oh = _clean_str(amenity[i]), _clean_str(shop[i]), _clean_str(hours[i])
            if am == "police":
                kind, fallback = "police", "Police station"
            elif (sh or am in COMMERCIAL_AMENITIES) and oh and _ALWAYS_OPEN.match(oh.lower()):
                kind, fallback = "24/7 commercial", "24/7 shop"
            else:
                continue
            if not (math.isfinite(lats[i]) and math.isfinite(lons[i])):
                continue
            keep.append(i); types.append(kind); out_names.append(names[i] or fallback)
        if keep:
            table = {"lat": lats[keep], "lon": lons[keep], "name": out_names, "type": types}
    _HAVENS["src"], _HAVENS["table"] = _AREA.pois, table
    return table


def _nearest_havens(lat, lon):
    t = _haven_table()
    if not len(t["lat"]):
        return []
    d = _haversine_vec(lat, lon, t["lat"], t["lon"])
    order = [i for i in np.argsort(d) if d[i] <= HAVEN_RADIUS_M][:MAX_HAVENS]
    return [{"name": t["name"][i], "type": t["type"][i], "distance_m": round(float(d[i]), 1),
             "lat": float(t["lat"][i]), "lon": float(t["lon"][i])} for i in order]


def _nearest_landmark(lat, lon):
    lms = _flat_landmarks()
    d = _haversine_vec(lat, lon, np.array([l["lat"] for l in lms]), np.array([l["lon"] for l in lms]))
    i = int(np.argmin(d))
    return {"name": lms[i]["name"], "city": lms[i]["city"], "lat": lms[i]["lat"], "lon": lms[i]["lon"],
            "distance_m": round(float(d[i]), 1)}


# --------------------------------------------------------------------------- #
# Request helpers
# --------------------------------------------------------------------------- #
def _parse_hour(raw, default=12) -> int:
    if raw is None or str(raw).strip() == "":
        return default
    h = int(float(raw))
    if not 0 <= h <= 23:
        raise ValueError("hour must be between 0 and 23")
    return h


def _calibrated_scores(raw_risk):
    """Raw model output -> the same calibrated 0-100 score /api/route shows (no graph mutation)."""
    return calibrate_risk_scores(np.asarray(raw_risk, float), ml_engine.model_info()["risk_thresholds"])


def _clip01(x) -> float:
    return float(min(1.0, max(0.0, x)))


# --------------------------------------------------------------------------- #
# 1) GET /api/landmarks
# --------------------------------------------------------------------------- #
@app.route("/api/landmarks", methods=["GET"])
def api_landmarks():
    """
    Convex hull + bbox of the loaded graph, and the preset landmarks (6 per supported city).
    Works before any area is loaded (hull/bbox are then null).  Each landmark carries
    `in_loaded_area` so the UI can show only the relevant city's pins.
    """
    hull_ring = bbox = None
    loaded = _AREA.G is not None and _AREA.static_features is not None
    if loaded:
        hull_ring, bbox = _area_index().hull()

    landmarks = []
    for lm in _flat_landmarks():
        inside = None
        if bbox:
            inside = bool(bbox["south"] <= lm["lat"] <= bbox["north"] and bbox["west"] <= lm["lon"] <= bbox["east"])
        landmarks.append({**lm, "in_loaded_area": inside})

    return jsonify({
        "area_loaded": loaded,
        "place": _AREA.place,
        "bbox": bbox,
        "hull_latlon": hull_ring,                                    # [[lat, lon], ...] for L.polygon
        "hull_geojson": ({"type": "Polygon", "coordinates": [[[lo, la] for la, lo in hull_ring]]}
                         if hull_ring else None),
        "landmarks": landmarks,
        "landmarks_by_city": PRESET_LANDMARKS,
    })


# --------------------------------------------------------------------------- #
# 2) GET /api/inspect_point?lat=&lon=&hour=
# --------------------------------------------------------------------------- #
@app.route("/api/inspect_point", methods=["GET"])
def api_inspect_point():
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("lat/lon out of range")
        hour = _parse_hour(request.args.get("hour"))
    except KeyError:
        return jsonify({"error": "lat and lon query parameters are required (hour is optional, 0-23)."}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid parameter: {e}"}), 400

    if _AREA.G is None or _AREA.static_features is None:
        return jsonify({"error": "Load an area first (POST /api/load_area)."}), 400

    try:
        idx = _area_index()
        edge_id, slat, slon, snap_m = idx.nearest_edge(lat, lon)

        # ---- far from every loaded street -> fallback to the nearest preset landmark ----
        if snap_m > OUT_OF_BOUNDS_M:
            return jsonify({
                "out_of_bounds": True,
                "query": {"lat": lat, "lon": lon, "hour": hour},
                "distance_to_graph_m": round(snap_m, 1),
                "threshold_m": OUT_OF_BOUNDS_M,
                "nearest_landmark": _nearest_landmark(lat, lon),
                "message": f"Point is {snap_m:.0f} m from the nearest loaded street "
                           f"(limit {OUT_OF_BOUNDS_M:.0f} m).",
            })

        if not ml_engine.models_available():
            return jsonify({"error": "Safety model not trained yet. Run `python train_offline.py`, "
                                     "then restart the server."}), 503

        u, v, k = idx.edge_keys[edge_id]
        row = _AREA.static_features.iloc[[edge_id]]
        scored = ml_engine.predict_road_safety(row, hour).iloc[0]    # runs compute_temporal_features inside
        raw = float(scored["predicted_risk"])
        score = float(_calibrated_scores([raw])[0])

        # ---- 5 CPTED radar metrics, each in [0, 1] (higher = safer) ----
        edata = _AREA.G.edges[u, v, k]
        highway = _clean_str(edata.get("highway")) or "residential"
        degree = float(scored["degree_of_connectivity"])
        police_m = float(scored["distance_to_nearest_police"])
        open_shops = float(scored["active_open_shops_estimate"])
        darkness = float(scored["darkness_hazard"])
        cpted = {
            # dead end (deg 1) = 0.0 ... 5+-way junction = 1.0
            "connectivity": round(_clip01((degree - 1.0) / 4.0), 3),
            # 1 at the station, 0.5 at 200 m, ~0.17 at 1 km
            "police_proximity": round(_clip01(1.0 / (1.0 + police_m / 200.0)), 3),
            # saturating: ~0.63 with 3 open shops nearby, ~0.86 with 6
            "commercial_eyes": round(_clip01(1.0 - math.exp(-open_shops / 3.0)), 3),
            "visibility": round(_clip01(1.0 - darkness), 3),
            "road_hierarchy": HIGHWAY_HIERARCHY.get(highway, DEFAULT_HIERARCHY),
        }

        return jsonify(_json_safe({
            "out_of_bounds": False,
            "query": {"lat": lat, "lon": lon, "hour": hour},
            "snapped": {"lat": slat, "lon": slon, "distance_m": round(snap_m, 1)},
            "edge": {"edge_id": edge_id, "u": u, "v": v, "key": k, "highway": highway,
                     "name": _clean_str(edata.get("name")),
                     "length_m": round(float(row["length"].iloc[0]), 1)},
            "risk_score": round(score, 1),                           # calibrated 0-100, same scale as /api/route
            "risk_band": score_band(score),
            "raw_risk": raw,
            "risk_level": str(scored["risk_level"]),
            "risk_archetype": str(scored["risk_archetype"]),
            "cpted": cpted,
            "cpted_inputs": {"degree": degree, "police_distance_m": round(police_m, 1),
                             "active_open_shops": round(open_shops, 2), "darkness_hazard": round(darkness, 3)},
            "safe_havens": _nearest_havens(lat, lon),
            "safe_haven_radius_m": HAVEN_RADIUS_M,
        }))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# --------------------------------------------------------------------------- #
# 3) GET /api/route_diurnal_profile?edge_ids=1,2,3
# --------------------------------------------------------------------------- #
@app.route("/api/route_diurnal_profile", methods=["GET"])
def api_route_diurnal_profile():
    raw_ids = (request.args.get("edge_ids") or "").strip()
    if not raw_ids:
        return jsonify({"error": "edge_ids is required, e.g. ?edge_ids=12,845,3021"}), 400
    try:
        ids = list(dict.fromkeys(int(t) for t in raw_ids.split(",") if t.strip() != ""))
    except ValueError:
        return jsonify({"error": "edge_ids must be comma-separated integers."}), 400
    if not ids:
        return jsonify({"error": "edge_ids is empty."}), 400
    if len(ids) > MAX_DIURNAL_EDGES:
        return jsonify({"error": f"At most {MAX_DIURNAL_EDGES} edges per request."}), 400

    if _AREA.G is None or _AREA.static_features is None:
        return jsonify({"error": "Load an area first (POST /api/load_area)."}), 400
    if not ml_engine.models_available():
        return jsonify({"error": "Safety model not trained yet. Run `python train_offline.py`."}), 503

    n_edges = len(_AREA.static_features)
    bad = [i for i in ids if not 0 <= i < n_edges]
    if bad:
        return jsonify({"error": f"Unknown edge_ids {bad[:5]} (valid range 0-{n_edges - 1}) -- edge ids are "
                                 f"only valid for the currently loaded area."}), 400

    try:
        subset = _AREA.static_features.iloc[ids]
        w = subset["length"].to_numpy(float)
        w = np.where(w > 0, w, 1.0)
        hourly = []
        for h in range(24):
            # predict_road_safety -> compute_temporal_features(subset, h) -> regressor
            raw = ml_engine.predict_road_safety(subset, h)["predicted_risk"].to_numpy(float)
            hourly.append(float(np.average(_calibrated_scores(raw), weights=w)))   # length-weighted, like path_risk_profile
        hourly = [round(s, 1) for s in hourly]
        return jsonify({
            "hours": list(range(24)),
            "hourly_risk": hourly,                                   # 24 values, calibrated 0-100
            "peak_hour": int(np.argmax(hourly)),
            "safest_hour": int(np.argmin(hourly)),
            "num_edges": len(ids),
            "length_m": round(float(w.sum()), 1),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400
