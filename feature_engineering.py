"""
feature_engineering.py
-----------------------
Converts raw graph + POI geometry into a per-edge feature table that the
ML model (model.py) can consume.

Design decisions (for viva defense):
- We compute features PER EDGE (not per node) because insecurity is a
  property of *traversing a segment* (a dark alley vs. a lit main road),
  which matches how routing algorithms assign cost to edges.
- We use a projected CRS (metres, via osmnx's UTM auto-projection) for all
  distance/buffer calculations. Lat/lon degrees are NOT a uniform distance
  unit, so computing "distance to nearest police station in metres" or a
  "100m buffer" directly in EPSG:4326 would silently give wrong results
  depending on latitude. This is a common and important geospatial
  correctness point to mention in the viva.
- We use a scipy cKDTree for nearest-POI queries instead of a naive O(N*M)
  loop. For a graph with thousands of edges and hundreds of POIs, KDTree
  nearest-neighbour queries are O(log M) per query after an O(M log M)
  build, which is essential for keeping the pipeline interactive.
- Two engineered interaction features target PEDESTRIAN PERSONAL VULNERABILITY rather than
  reporting volume (see isolation_index / darkness_hazard below):
      isolation_index = (distance_to_nearest_police / 1000) / max(degree_of_connectivity, 1)
      darkness_hazard = is_night / (1 + active_open_shops_estimate)
  isolation_index is time-independent (computed with the static features); darkness_hazard needs the
  hour, so it is computed with the temporal features.
- Temporal features (is_night, active_open_shops_estimate, darkness_hazard) are computed
  from a single `query_hour` parameter, NOT baked permanently into the
  graph -- because the whole point of the project is that risk is
  time-dependent. feature_engineering.compute_edge_features() is designed
  to be called AGAIN at query time with a different hour, and model.py's
  predict_and_attach() does exactly that.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree

# NOTE: osmnx is imported lazily inside _project_graph_and_pois(). The pure-pandas helpers below
# (add_temporal_columns, compute_isolation_index, compute_darkness_hazard, ...) therefore work in
# lightweight environments (diagnostics, unit tests) that have no OSM stack installed, and the SAME
# code path computes the engineered features offline (training), in tests and at routing time.

# Road-type risk priors used only as an ordinal feature for the model
# (the model itself learns the actual weighting from training data;
# this dict just turns OSM's free-text 'highway' tag into a bounded
# category the model can use consistently).
ROAD_TYPE_ORDER = {
    "footway": 0, "pedestrian": 0, "path": 0, "living_street": 1,
    "residential": 2, "unclassified": 2, "tertiary": 3,
    "secondary": 4, "primary": 5, "trunk": 6,
}


# Shop-activity multiplier per hour of day (shared by compute_temporal_features and the diagnostics script).
ACTIVITY_CURVE = {
    0: 0.02, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.02, 5: 0.05,
    6: 0.15, 7: 0.30, 8: 0.50, 9: 0.70, 10: 0.85, 11: 0.90,
    12: 0.95, 13: 0.90, 14: 0.80, 15: 0.80, 16: 0.85, 17: 0.90,
    18: 0.95, 19: 1.00, 20: 0.85, 21: 0.60, 22: 0.35, 23: 0.15,
}


ACTIVITY_ARRAY = np.array([ACTIVITY_CURVE[h] for h in range(24)], dtype=float)

# Night window [20:00, 06:00). Single source of truth: train_offline.py, ml_engine.py,
# evaluate_and_visualize.py and the router all derive is_night from is_night_flag().
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 6


def is_night_flag(hours):
    """1 for hours in [20:00, 06:00), else 0. Scalar in -> int out; array/Series in -> int ndarray out."""
    h = np.asarray(hours)
    out = ((h >= NIGHT_START_HOUR) | (h < NIGHT_END_HOUR)).astype(int)
    return int(out) if out.ndim == 0 else out


def compute_isolation_index(distance_to_nearest_police, degree_of_connectivity):
    """
    isolation_index = (distance_to_nearest_police / 1000) / max(degree_of_connectivity, 1)

    High for dead ends and poorly connected lanes (low degree) that are also far from emergency
    response; low for well-connected streets near a police station.  Works on scalars, numpy arrays
    and pandas Series (distance in metres).
    """
    dist_km = np.asarray(distance_to_nearest_police, dtype=float) / 1000.0
    deg = np.maximum(np.asarray(degree_of_connectivity, dtype=float), 1.0)
    out = dist_km / deg
    return out.item() if np.ndim(out) == 0 else out


def compute_darkness_hazard(is_night, active_open_shops_estimate):
    """
    darkness_hazard = is_night / (1 + active_open_shops_estimate)

    1.0 on a night-time street with no open shops (dark, nobody watching), shrinking as open shops
    provide light and natural surveillance; exactly 0 in daytime.
    """
    night = np.asarray(is_night, dtype=float)
    active = np.maximum(np.asarray(active_open_shops_estimate, dtype=float), 0.0)
    out = night / (1.0 + active)
    return out.item() if np.ndim(out) == 0 else out


def _first_tag(value, default=None):
    """OSM edges can carry a tag as a scalar OR a list (multi-tagged ways
    after simplification). Normalize to a single representative value."""
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _project_graph_and_pois(G, pois):
    """
    Project both the graph and POIs into the same local metric CRS
    (UTM zone auto-detected by osmnx based on the graph's centroid).
    This is required before any Euclidean distance/buffer math.
    """
    import osmnx as ox   # lazy: keeps the rest of this module importable without the OSM stack
    G_proj = ox.project_graph(G)
    crs = G_proj.graph["crs"]
    pois_proj = pois.to_crs(crs) if pois is not None and len(pois) else pois
    return G_proj, pois_proj


def _edge_midpoints(G_proj):
    """
    Return arrays of (u, v, key, x_mid, y_mid) for every edge, using the
    midpoint between the projected endpoint nodes as the edge's
    representative location for distance/buffer queries. This is a
    standard simplification -- exact geometry-based buffering (using the
    edge's actual polyline) would be more precise but is unnecessary
    overhead for a segment-level risk proxy.
    """
    xs_u = np.array([G_proj.nodes[u]["x"] for u, v, k in G_proj.edges(keys=True)])
    ys_u = np.array([G_proj.nodes[u]["y"] for u, v, k in G_proj.edges(keys=True)])
    xs_v = np.array([G_proj.nodes[v]["x"] for u, v, k in G_proj.edges(keys=True)])
    ys_v = np.array([G_proj.nodes[v]["y"] for u, v, k in G_proj.edges(keys=True)])
    mid_x = (xs_u + xs_v) / 2.0
    mid_y = (ys_u + ys_v) / 2.0
    edge_ids = list(G_proj.edges(keys=True))
    return edge_ids, mid_x, mid_y


def compute_static_and_poi_features(G, pois) -> pd.DataFrame:
    """
    Compute the TIME-INDEPENDENT feature columns:
        length, road_type_ord, degree_of_connectivity,
        distance_to_nearest_police, commercial_poi_count, isolation_index

    These only need to be recomputed if the graph or POI set changes --
    NOT on every routing query -- which is why they're split out from the
    temporal features (see compute_temporal_features below). This
    separation avoids redundant KDTree builds on every user interaction
    with the lambda slider / hour selector in the demo app.

    Returns a DataFrame indexed by (u, v, key) matching G's edge keys.
    """
    G_proj, pois_proj = _project_graph_and_pois(G, pois)
    edge_ids, mid_x, mid_y = _edge_midpoints(G_proj)
    edge_points = np.column_stack([mid_x, mid_y])

    # --- Static features ---
    lengths = [G_proj.edges[e].get("length", 1.0) for e in edge_ids]
    road_types = [ROAD_TYPE_ORDER.get(_first_tag(G_proj.edges[e].get("highway"), "residential"), 2)
                  for e in edge_ids]
    # Degree of connectivity: how many streets meet at the edge's start
    # node. Higher-degree intersections are typically busier / more
    # overlooked (informal surveillance), a well-known urban-safety proxy
    # from Jane Jacobs' "eyes on the street" theory.
    undirected_deg = dict(nx.degree(nx.Graph(G_proj)))
    degree_conn = [undirected_deg.get(u, 1) for u, v, k in edge_ids]

    # --- POI-derived features ---
    if pois_proj is not None and len(pois_proj):
        pois_proj = pois_proj.copy()
        pois_proj["geometry"] = pois_proj.geometry.centroid  # handles polygons
        police_mask = pois_proj.get("amenity").isin(["police"]) if "amenity" in pois_proj else pd.Series(False, index=pois_proj.index)
        police_pts = pois_proj[police_mask]
        shop_mask = pois_proj.get("shop").notna() if "shop" in pois_proj else pd.Series(False, index=pois_proj.index)
        shop_pts = pois_proj[shop_mask]
    else:
        police_pts, shop_pts = pois_proj.iloc[0:0] if pois_proj is not None else None, None

    # Nearest-police distance via KDTree (see module docstring for
    # complexity justification).
    if police_pts is not None and len(police_pts):
        police_xy = np.column_stack([police_pts.geometry.x, police_pts.geometry.y])
        police_tree = cKDTree(police_xy)
        dist_police, _ = police_tree.query(edge_points, k=1)
    else:
        # No police POIs found in area -> treat as "far" (a conservative,
        # safety-pessimistic default rather than silently imputing 0).
        dist_police = np.full(len(edge_ids), 2000.0)

    # Commercial POI count within 100m buffer via KDTree radius query.
    # 100m is chosen as a walkable "immediate surroundings" radius --
    # roughly a 1-2 minute walk, matching the resolution at which a
    # pedestrian actually perceives street-level activity.
    if shop_pts is not None and len(shop_pts):
        shop_xy = np.column_stack([shop_pts.geometry.x, shop_pts.geometry.y])
        shop_tree = cKDTree(shop_xy)
        commercial_count = shop_tree.query_ball_point(edge_points, r=100.0, return_length=True)
    else:
        commercial_count = np.zeros(len(edge_ids))

    df = pd.DataFrame({
        "u": [e[0] for e in edge_ids],
        "v": [e[1] for e in edge_ids],
        "key": [e[2] for e in edge_ids],
        "length": lengths,
        "road_type_ord": road_types,
        "degree_of_connectivity": degree_conn,
        "distance_to_nearest_police": dist_police,
        "commercial_poi_count": commercial_count,
    })
    # Engineered interaction: dead-end / low-connectivity streets far from police response.
    df["isolation_index"] = compute_isolation_index(df["distance_to_nearest_police"], df["degree_of_connectivity"])
    return df.set_index(["u", "v", "key"])


def add_temporal_columns(df: pd.DataFrame, hours) -> pd.DataFrame:
    """
    VECTORISED single source of truth for every time-dependent / interaction column.  `hours` is a
    scalar (one query hour for all rows) or an array with one hour per row.  Adds:

        query_hour                  hour of day, 0-23
        log_length                  log1p(length in metres)
        is_night                    1 if hour in [20:00, 06:00)
        active_open_shops_estimate  commercial_poi_count * ACTIVITY_CURVE[hour]
        darkness_hazard             is_night * 1 / (1 + active_open_shops_estimate)
        isolation_index             (distance_to_nearest_police / 1000) * 1 / max(degree_of_connectivity, 1)

    isolation_index is ALWAYS recomputed from the raw columns, so a frame whose distance/degree were
    edited (counterfactuals, repaired NaNs, partial-dependence sweeps) can never carry a stale value.
    Offline training, hold-out evaluation, diagnostics and real-time routing all call this function.
    """
    out = df.copy()
    h = np.broadcast_to(np.asarray(hours, dtype=int), (len(out),)) % 24
    out["query_hour"] = h
    # log-compressed segment length (5 m .. 500 m -> 1.8 .. 6.2): the model feature that replaces raw `length`, so
    # segment length cannot dominate greedy tree splits.  Always recomputed from the raw metres column.
    out["log_length"] = np.log1p(out["length"].to_numpy(float))
    out["is_night"] = is_night_flag(h)
    out["active_open_shops_estimate"] = out["commercial_poi_count"].to_numpy(float) * ACTIVITY_ARRAY[h]
    out["darkness_hazard"] = compute_darkness_hazard(out["is_night"].to_numpy(), out["active_open_shops_estimate"].to_numpy())
    out["isolation_index"] = compute_isolation_index(out["distance_to_nearest_police"].to_numpy(),
                                                     out["degree_of_connectivity"].to_numpy())
    return out


def compute_temporal_features(static_df: pd.DataFrame, query_hour: int) -> pd.DataFrame:
    """
    Compute the TIME-DEPENDENT columns for a given query_hour (0-23) and join them onto the static
    feature table (thin wrapper around add_temporal_columns).

    - is_night: binary flag, 1 if hour is in [20:00, 06:00). This window (rather than a fixed sunset
      time) is a simplification appropriate for a prototype; a production system could swap in an
      astronomical sunset/sunrise API per city/date.
    - active_open_shops_estimate: a heuristic estimate of how many of the nearby commercial_poi_count
      are actually OPEN and active at this hour (ACTIVITY_CURVE: ~0 at 01:00-05:00, ~1 around 19:00).
      A street lined with shops is only safer ("eyes on the street") when those shops are open -- the
      same street becomes a shuttered, empty corridor at 3am despite an identical commercial_poi_count.
    - darkness_hazard: is_night / (1 + active_open_shops_estimate); penalises night-time streets with no
      open shops (no light, no eyes on the street).  Exactly 0 in daytime.
    - isolation_index: static, but re-derived here so every consumer sees a value consistent with the
      raw distance/degree columns.
    """
    return add_temporal_columns(static_df, int(query_hour))


def compute_edge_features(G, pois, query_hour: int) -> pd.DataFrame:
    """Convenience wrapper: static + POI features, joined with temporal
    features for a specific query hour. This is the single entry point
    the rest of the pipeline (model.py) should call."""
    static_df = compute_static_and_poi_features(G, pois)
    return compute_temporal_features(static_df, query_hour)
