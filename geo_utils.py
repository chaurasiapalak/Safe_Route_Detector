"""
geo_utils.py
------------
Shared geometry helpers for turning a computed route (a list of graph
node ids) into renderable coordinates / GeoJSON.

Kept as its own module so the CLI demo (app.py, which draws routes with
folium) and the web backend (server.py, which sends GeoJSON to Leaflet)
never end up with two independently-maintained copies of the same "walk
the true edge geometry" logic -- that correctness only needs to be
gotten right once.
"""


def edge_geometry_latlon(G, u, v):
    """
    Return the list of (lat, lon) points tracing the TRUE physical shape
    of the street segment between nodes u and v, instead of a straight
    chord between the two intersections.

    osmnx keeps the original curved polyline in an edge's 'geometry'
    attribute whenever that edge was simplified out of a longer OSM way
    during graph construction. Segments with no stored geometry are, by
    construction, already straight between their two endpoint nodes, so
    the straight chord IS the correct shape in that case.

    A (u, v) pair in a MultiDiGraph can have several parallel edges; for
    rendering we take the first available one -- the visual difference
    between parallel edges (e.g. two carriageways of a divided road) is
    negligible at map-drawing scale.
    """
    edge_data_dict = G.get_edge_data(u, v)
    if not edge_data_dict:
        return [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]

    first_edge = next(iter(edge_data_dict.values()))
    if "geometry" in first_edge:
        x, y = first_edge["geometry"].xy
        return list(zip(y, x))

    return [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]


def path_to_latlon_polyline(G, path):
    """
    Flatten a full route (list of node ids) into ONE continuous list of
    (lat, lon) points, using true edge geometry for every segment and
    de-duplicating the shared point between consecutive segments.
    """
    coords = []
    for u, v in zip(path[:-1], path[1:]):
        seg = edge_geometry_latlon(G, u, v)
        if coords and coords[-1] == seg[0]:
            coords.extend(seg[1:])
        else:
            coords.extend(seg)
    return coords


def haversine_m(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in metres between two (lat, lon) points.

    Used by server.py to measure how far a user's TYPED location (from
    geocoding) sits from the graph node the router actually snapped it
    to -- nearest_nodes() always returns SOME node even if the typed
    address is far outside the loaded area's coverage radius, so this
    distance is what lets the backend warn the user ("that address is
    900m from the nearest loaded street -- try increasing the radius")
    instead of silently returning a misleading route.
    """
    import math
    R = 6371000.0  # mean Earth radius, metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def path_to_geojson_feature(G, path, properties=None):
    """
    Convert a route into a GeoJSON Feature (LineString).

    NOTE ON COORDINATE ORDER: GeoJSON mandates [longitude, latitude]
    ordering, which is the OPPOSITE of the (lat, lon) tuples used above
    (matching folium/Leaflet's own (lat, lon) convention for
    L.polyline/L.marker). Mixing the two orders up is a classic subtle
    bug in mapping code, so the swap is isolated to this one function.
    """
    latlon_coords = path_to_latlon_polyline(G, path)
    lonlat_coords = [[lon, lat] for lat, lon in latlon_coords]
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": lonlat_coords},
        "properties": properties or {},
    }
