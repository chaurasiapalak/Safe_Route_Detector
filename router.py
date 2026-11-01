"""
router.py
---------
Risk-aware pedestrian routing on top of the zero-shot model in ml_engine.py.

Edge weight:
    weight(edge) = length * (1.0 + risk_lambda * calibrated_risk + penalty)

    calibrated_risk = risk_score / 100  (the 0-100 index of risk_calibration.py, bound to the trained
                                         artifact's thresholds: median ~20, elevated 60, high 85)

    penalty = +0.35  if is_night == 1 and degree_of_connectivity <= 2
                     (dark, low-escape lanes and dead ends: CPTED "entrapment" rule)
            = -0.10  if is_night == 1 and commercial_poi_count >= 4 and active_open_shops_estimate >= 1.0
                     (visible, active commercial street: natural-surveillance discount)
            =  0     otherwise

    predicted_risk in [0, 1]   (written on each edge by ml_engine.attach_risk_to_graph; predicted by
                               the OFFLINE-trained regressor, no per-area retraining)
    risk_lambda    >= 0        (0 = ignore risk, larger = detour more to avoid risk)

Explainability in the output payload:
    * high_risk_segments(...)  -> returns segments with micro_explanation and true polyline coordinates
    * path_risk_profile(...)   -> returns peak_segment_risk and peak_segment with street name, length,
                                  coords, and micro_explanation
    * route_summary(...)       -> aggregates the profile and explained high-risk segments for /api/route
"""

import heapq
import math
import os
from typing import Callable, Optional

import networkx as nx

# Maximum risk multiplier: with risk_lambda = 8 an edge at predicted_risk = 1.0
# costs 9x its length, so the "safest" route may detour up to ~9x around it.
RISK_LAMBDA_MAX = float(os.environ.get("SAFE_ROUTE_RISK_LAMBDA_MAX", "8.0"))


def lam_to_risk_lambda(lam: float) -> float:
    """Map the frontend slider (1 = shortest, 0 = safest) to the risk multiplier."""
    lam = min(1.0, max(0.0, float(lam)))
    return RISK_LAMBDA_MAX * (1.0 - lam)


def normalize_graph_weights(G):
    """Kept for backward compatibility. The multiplicative cost needs no graph-wide
    normalisation, so this returns G unchanged."""
    return G


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Structural urban-design (CPTED) rules
# --------------------------------------------------------------------------- #
CPTED_ENABLED = os.environ.get("SAFE_ROUTE_CPTED", "1") != "0"
NIGHT_LOW_ESCAPE_PENALTY = 0.35      # night + degree_of_connectivity <= LOW_ESCAPE_DEGREE
LOW_ESCAPE_DEGREE = 2
SURVEILLANCE_DISCOUNT = -0.10        # night-time busy commercial street with open shops
SURVEILLANCE_MIN_POIS = 4            # commercial_poi_count must be >= this
SURVEILLANCE_MIN_OPEN = 1.0          # active_open_shops_estimate must be >= this (and is_night == 1)
MICRO_MAX_FACTORS = 3

# Smallest possible cost multiplier -> keeps the A* heuristic admissible.
HEURISTIC_SCALE = 1.0 + (SURVEILLANCE_DISCOUNT if CPTED_ENABLED else 0.0)


def _num(data: dict, key: str):
    """float(data[key]) or None when absent / NaN."""
    v = data.get(key)
    if v is None:
        return None
    try:
        v = float(v)
        return None if v != v else v
    except (ValueError, TypeError):
        return None


def structural_flags(data: dict) -> dict:
    """Which CPTED rules fire for one edge: {"dead_end_night": bool, "surveilled_night": bool}."""
    night = _num(data, "is_night")
    degree = _num(data, "degree_of_connectivity")
    pois = _num(data, "commercial_poi_count")
    open_shops = _num(data, "active_open_shops_estimate")
    is_night = night is not None and int(round(night)) == 1
    return {
        "dead_end_night": bool(is_night and degree is not None and degree <= LOW_ESCAPE_DEGREE),
        "surveilled_night": bool(is_night and pois is not None and open_shops is not None
                                 and pois >= SURVEILLANCE_MIN_POIS and open_shops >= SURVEILLANCE_MIN_OPEN),
    }


def structural_penalty(data: dict) -> float:
    """
    CPTED structural term added inside (1 + lambda * risk + penalty).
    +0.35  night and degree_of_connectivity <= 2  (dead-end / low-escape entrapment)
    -0.10  night and commercial_poi_count >= 4 and active_open_shops_estimate >= 1.0 (surveilled corridor)
    """
    if not CPTED_ENABLED:
        return 0.0
    flags = structural_flags(data)
    penalty = 0.0
    if flags["dead_end_night"]:
        penalty += NIGHT_LOW_ESCAPE_PENALTY
    if flags["surveilled_night"]:
        penalty += SURVEILLANCE_DISCOUNT
    return penalty


COST_USES_CALIBRATED = os.environ.get("SAFE_ROUTE_COST_USES_CALIBRATED", "1") != "0"


def edge_risk_unit(data: dict) -> float:
    """Edge risk in [0, 1] used by the cost function: calibrated score / 100 when available."""
    if COST_USES_CALIBRATED and data.get("risk_score") is not None:
        return min(1.0, max(0.0, float(data["risk_score"]) / 100.0))
    return min(1.0, max(0.0, float(data.get("predicted_risk", 0.0))))


def edge_weight(G, u, v, data: dict, risk_lambda: float) -> float:
    """weight = length * (1 + risk_lambda * calibrated_risk + penalty), with safe defaults."""
    length = data.get("length")
    if length is None:
        length = _haversine_m(G.nodes[u]["y"], G.nodes[u]["x"], G.nodes[v]["y"], G.nodes[v]["x"])
    penalty = structural_penalty(data) if risk_lambda > 0 else 0.0
    multiplier = max(HEURISTIC_SCALE, 1.0 + risk_lambda * edge_risk_unit(data) + penalty)
    return float(length) * multiplier


def _best_parallel_edge(G, u, v, risk_lambda: float):
    """Cheapest of the parallel edges between u and v (MultiDiGraph) -> (weight, key)."""
    best_w, best_k = math.inf, None
    edge_dict = G.get_edge_data(u, v)
    if not edge_dict:
        return best_w, best_k
    for key, data in edge_dict.items():
        w = edge_weight(G, u, v, data, risk_lambda)
        if w < best_w:
            best_w, best_k = w, key
    return best_w, best_k


def risk_weighted_route(G, source, target, risk_lambda: float, use_heuristic: bool = True):
    """
    A* (or plain Dijkstra when use_heuristic=False) with the risk-weighted cost.
    Returns (path_nodes, total_cost, total_distance_m).
    """
    if source not in G or target not in G:
        raise ValueError("Source or target node not in graph.")
    risk_lambda = max(0.0, float(risk_lambda))
    ty, tx = G.nodes[target]["y"], G.nodes[target]["x"]

    h_scale = HEURISTIC_SCALE if risk_lambda > 0 else 1.0

    def h(n):
        if not use_heuristic:
            return 0.0
        return h_scale * _haversine_m(G.nodes[n]["y"], G.nodes[n]["x"], ty, tx)

    g_cost = {source: 0.0}
    prev, closed = {}, set()
    heap = [(h(source), 0.0, source)]

    while heap:
        _, g_u, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        if u == target:
            break
        for v in G.successors(u):
            if v in closed:
                continue
            w, _ = _best_parallel_edge(G, u, v, risk_lambda)
            if math.isinf(w):
                continue
            g_v = g_u + w
            if g_v < g_cost.get(v, math.inf):
                g_cost[v] = g_v
                prev[v] = u
                heapq.heappush(heap, (g_v + h(v), g_v, v))

    if target not in g_cost:
        raise nx.NetworkXNoPath(f"No path found between {source} and {target}.")

    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()

    distance = 0.0
    for a, b in zip(path[:-1], path[1:]):
        _, key = _best_parallel_edge(G, a, b, risk_lambda)
        d = G.edges[a, b, key]
        distance += float(d.get("length")) if d.get("length") is not None else _haversine_m(
            G.nodes[a]["y"], G.nodes[a]["x"], G.nodes[b]["y"], G.nodes[b]["x"]
        )
    return path, g_cost[target], distance


def compute_shortest_path(G, source, target):
    """Pure distance route (risk_lambda = 0, equivalent to slider lam = 1.0)."""
    return risk_weighted_route(G, source, target, risk_lambda=0.0)


def compute_safest_path(G, source, target, lam: float = 0.0):
    """Safety-weighted route with slider lam mapped to risk multiplier."""
    return risk_weighted_route(G, source, target, risk_lambda=lam_to_risk_lambda(lam))


def path_edge_keys(G, path, risk_lambda: float = 0.0) -> list:
    """[(u, v, key), ...] for each hop of path."""
    out = []
    for a, b in zip(path[:-1], path[1:]):
        _, key = _best_parallel_edge(G, a, b, risk_lambda)
        out.append((a, b, key))
    return out


def _edge_score(d: dict) -> float:
    """Calibrated 0-100 score of one edge."""
    if d.get("risk_score") is not None:
        return float(d["risk_score"])
    return 100.0 * float(d.get("predicted_risk", 0.0))


def _edge_street_name(d: dict) -> str:
    """Extract legible street or pathway name from OSM edge attributes."""
    name = d.get("name")
    if isinstance(name, list) and name:
        return str(name[0])
    if isinstance(name, str) and name.strip():
        return name.strip()
    hw = d.get("highway")
    if isinstance(hw, list) and hw:
        hw = hw[0]
    if hw in ("footway", "path", "pedestrian"):
        return "Pedestrian Walkway / Path"
    if hw in ("service", "alley"):
        return "Service Lane / Alley"
    return "Local Road"


def _edge_latlon(G, u, v, data: dict):
    """[[lat, lon], ...] along the actual geometry coordinates of the edge."""
    geom = data.get("geometry")
    if geom is not None and hasattr(geom, "xy"):
        xs, ys = geom.xy
        return [[round(float(y), 6), round(float(x), 6)] for x, y in zip(xs, ys)]
    return [
        [round(float(G.nodes[u]["y"]), 6), round(float(G.nodes[u]["x"]), 6)],
        [round(float(G.nodes[v]["y"]), 6), round(float(G.nodes[v]["x"]), 6)],
    ]


def structural_micro_explanation(d: dict) -> dict:
    """Model-free micro-explanation built from edge attributes stored on the graph."""
    risk, positive = [], []
    flags = structural_flags(d)
    night = _num(d, "is_night") is not None and int(round(_num(d, "is_night"))) == 1
    deg = _num(d, "degree_of_connectivity")
    pois = _num(d, "commercial_poi_count")
    open_shops = _num(d, "active_open_shops_estimate")
    iso = _num(d, "isolation_index")
    dark = _num(d, "darkness_hazard")
    dist_pol = _num(d, "dist_to_police") or _num(d, "dist_to_police_m")
    level = d.get("risk_level")

    if dist_pol is not None and dist_pol > 1400:
        risk.append({"factor": "police_distance", "text": f"far from nearest police station (~{dist_pol/1000:.1f} km away)"})

    if flags["dead_end_night"] and deg is not None:
        risk.append({
            "factor": "dead_end_at_night",
            "text": f"low-connectivity lane at night ({deg:.0f} link{'s' if deg != 1 else ''}): few escape routes",
        })

    if dark is not None and dark >= 0.85:
        risk.append({"factor": "darkness_hazard", "text": "dark stretch at night: no open commercial frontage for light or witnesses"})
    elif pois == 0 or (open_shops is not None and open_shops == 0):
        risk.append({"factor": "no_eyes_on_street", "text": "zero active storefronts nearby: low natural surveillance"})

    if iso is not None and iso >= 1.0:
        risk.append({"factor": "isolation_index", "text": f"isolated: far from emergency response with limited exit corridors (index {iso:.1f})"})

    if level in ("elevated", "high") and not risk:
        risk.append({"factor": "model_rating", "text": f"model identifies elevated street hazard ({_edge_score(d):.0f}/100)"})

    # Positive protective factors
    if flags["surveilled_night"] and pois is not None and open_shops is not None:
        positive.append({
            "factor": "natural_surveillance",
            "text": f"active commercial frontage at night ({pois:.0f} shops, ~{open_shops:.0f} open): eyes on the street",
        })
    elif night and open_shops is not None and open_shops >= SURVEILLANCE_MIN_OPEN:
        positive.append({"factor": "open_shops", "text": f"~{open_shops:.0f} open shops nearby provide light and witnesses"})

    if deg is not None and deg >= 4:
        positive.append({"factor": "connectivity", "text": f"well-connected junction area ({deg:.0f} intersecting streets): open escape paths"})

    if dist_pol is not None and dist_pol < 500:
        positive.append({"factor": "police_proximity", "text": f"close to police station (~{dist_pol:.0f} m)"})

    if level == "low" and not positive:
        positive.append({"factor": "model_rating", "text": f"model rates this segment low risk ({_edge_score(d):.0f}/100)"})

    return {"top_risk_factors": risk[:MICRO_MAX_FACTORS], "top_positive_factors": positive[:MICRO_MAX_FACTORS]}


def edge_micro_explanation(d: dict, detail: Optional[dict] = None) -> dict:
    """Edge-level micro-explanation with applied routing adjustments."""
    adjustments = []
    flags = structural_flags(d) if CPTED_ENABLED else {"dead_end_night": False, "surveilled_night": False}
    if flags["dead_end_night"]:
        adjustments.append({"rule": "dead_end_night", "cost": NIGHT_LOW_ESCAPE_PENALTY})
    if flags["surveilled_night"]:
        adjustments.append({"rule": "active_commercial_night", "cost": SURVEILLANCE_DISCOUNT})

    if detail and (detail.get("drivers") or detail.get("protective")):
        out = {
            "top_risk_factors": [{"factor": x["factor"], "text": x["text"], "points": x.get("points")}
                                 for x in detail.get("drivers", [])[:MICRO_MAX_FACTORS]],
            "top_positive_factors": [{"factor": x["factor"], "text": x["text"], "points": x.get("points")}
                                     for x in detail.get("protective", [])[:MICRO_MAX_FACTORS]],
            "source": "model",
        }
    else:
        out = dict(structural_micro_explanation(d), source="structural")
    out["routing_adjustments"] = adjustments
    return out


def path_risk_profile(G, path, explain_fn: Optional[Callable[[list], dict]] = None) -> dict:
    """
    Route-level risk metrics with detailed peak hazard bottleneck metadata.
    """
    keys = path_edge_keys(G, path)
    total = weighted = elev_len = high_len = 0.0
    peak = None
    for k in keys:
        d = G.edges[k]
        length = float(d.get("length") or 0.0)
        score = _edge_score(d)
        total += length
        weighted += length * score
        level = d.get("risk_level")
        if level in ("elevated", "high"):
            elev_len += length
        if level == "high":
            high_len += length
        if peak is None or score > peak[0]:
            peak = (score, k, d, length)

    if peak is None:
        return {
            "avg_risk": 0.0,
            "peak_segment_risk": 0.0,
            "peak_segment": None,
            "pct_elevated_length": 0.0,
            "pct_high_length": 0.0,
        }

    score, k, d, length = peak
    detail = None
    if explain_fn is not None:
        try:
            detail = (explain_fn([tuple(k)]) or {}).get(tuple(k))
        except Exception:
            detail = None

    coords = _edge_latlon(G, k[0], k[1], d)
    street_name = _edge_street_name(d)
    micro = edge_micro_explanation(d, detail)

    peak_segment_dict = {
        "edge": list(k),
        "street_name": street_name,
        "length_m": round(length, 1),
        "risk_score": round(score, 1),
        "risk_level": d.get("risk_level") or ("high" if score >= 80 else "elevated" if score >= 50 else "low"),
        "coords": coords,
        "geojson": {
            "type": "LineString",
            "coordinates": [[pt[1], pt[0]] for pt in coords],
        },
        "micro_explanation": micro,
    }

    return {
        "avg_risk": round(weighted / total, 1) if total else 0.0,
        "peak_segment_risk": round(score, 1),
        "peak_segment": peak_segment_dict,
        "pct_elevated_length": round(100.0 * elev_len / total, 1) if total else 0.0,
        "pct_high_length": round(100.0 * high_len / total, 1) if total else 0.0,
    }


def path_average_insecurity(G, path):
    """Length-weighted mean calibrated risk (0-100)."""
    return path_risk_profile(G, path)["avg_risk"]


def high_risk_segments(G, path, min_level: str = "elevated", max_segments: int = 25,
                       risk_lambda: float = 0.0,
                       explain_fn: Optional[Callable[[list], dict]] = None):
    """
    Returns high-risk and elevated segments in route traversal order.
    """
    allowed = {"high"} if min_level == "high" else {"high", "elevated"}
    found = []
    for order, (a, b) in enumerate(zip(path[:-1], path[1:])):
        _, key = _best_parallel_edge(G, a, b, risk_lambda)
        d = G.edges[a, b, key]
        score = _edge_score(d)
        level = d.get("risk_level") or ("high" if score >= 80 else "elevated" if score >= 50 else "low")
        if level in allowed:
            found.append((order, a, b, key, d))

    found.sort(key=lambda t: -_edge_score(t[4]))
    found = sorted(found[:max_segments], key=lambda t: t[0])

    details = {}
    if explain_fn is not None and found:
        try:
            details = explain_fn([(a, b, k) for _, a, b, k, _ in found]) or {}
        except Exception:
            details = {}

    out = []
    for order, a, b, key, d in found:
        det = details.get((a, b, key), {})
        coords = _edge_latlon(G, a, b, d)
        micro = edge_micro_explanation(d, det)
        top_risk = micro.get("top_risk_factors", [])
        driver_text = top_risk[0]["text"] if top_risk else ""

        out.append({
            "edge": [a, b, key],
            "order": order,
            "street_name": _edge_street_name(d),
            "length_m": round(float(d.get("length") or 0.0), 1),
            "predicted_risk": round(float(d.get("predicted_risk", 0.0)), 4),
            "risk_score": round(_edge_score(d), 1),
            "risk_level": d.get("risk_level"),
            "archetype": det.get("archetype") or d.get("risk_archetype", ""),
            "reason": det.get("reason") or d.get("risk_reason") or driver_text or (
                f"{str(d.get('risk_level', 'elevated')).capitalize()} hazard score ({_edge_score(d):.0f}/100)."
            ),
            "drivers": det.get("drivers", []),
            "protective": det.get("protective", []),
            "micro_explanation": micro,
            "coords": coords,
            "geojson": {
                "type": "LineString",
                "coordinates": [[pt[1], pt[0]] for pt in coords],
            },
        })
    return out


def route_summary(G, path, risk_lambda: float = 0.0, explain_fn: Optional[Callable[[list], dict]] = None,
                  min_level: str = "elevated", max_segments: int = 25) -> dict:
    """
    One-call explainable payload for a route. Aggregates the risk profile,
    the peak hazard bottleneck stretch, and all explained high-risk segments.
    """
    out = path_risk_profile(G, path, explain_fn=explain_fn)
    out["high_risk_segments"] = high_risk_segments(
        G, path, min_level=min_level, max_segments=max_segments,
        risk_lambda=risk_lambda, explain_fn=explain_fn
    )
    return out