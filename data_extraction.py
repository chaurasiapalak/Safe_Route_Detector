"""
data_extraction.py
-------------------
Responsible for pulling raw geospatial data from OpenStreetMap:
  1. The pedestrian street network (as a NetworkX MultiDiGraph via osmnx).
  2. Points of Interest (POIs) relevant to safety scoring (police stations,
     shops, streetlights) as a GeoDataFrame.

Hardening changes in this revision (for viva defense):
- OVERPASS ENDPOINT / RELIABILITY: point osmnx at the Kumi Systems mirror
  instead of the default (frequently-504ing) German server, disable the
  defensive inter-request rate-limit stall, and use a long timeout.
- ALWAYS GEOCODE-TO-POINT + FIXED RADIUS (this is the key fix in this
  revision): earlier versions tried `ox.graph_from_place()` first, which
  asks Nominatim for the named place's OWN administrative/neighbourhood
  polygon boundary. That boundary is whatever a mapper happened to draw
  for that name -- for many places it is tiny (a single named market
  square, a few dozen nodes) and for others it is huge (an entire
  district). This produced wildly inconsistent, uncontrollable coverage
  and is why routes appeared to "not go beyond a certain area" -- the
  loaded graph itself was small, not a routing bug.
  We now ALWAYS geocode the place name down to a single (lat, lon) point
  via `ox.geocode()`, then build the graph as a circular buffer of
  radius `dist` metres around that point via `ox.graph_from_point()`.
  This makes coverage size a single, predictable, USER-CONTROLLED
  parameter (`dist`) instead of an accident of OSM's tagging for that
  specific place -- the closest practical equivalent to "search radius"
  behaviour in a consumer maps product, within what a single Overpass
  query can reasonably return.
- DEDICATED POI CACHING: POIs are cached independently of the graph
  (separate GeoJSON file) so a POI-only re-run doesn't force a redundant
  graph re-download, and vice versa.
"""

import os
import osmnx as ox
import networkx as nx
import geopandas as gpd

# --- osmnx global settings ---
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 180
ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api"
ox.settings.overpass_rate_limit = False


def fetch_road_graph(place_name: str = None,
                      bbox: tuple = None,
                      dist: int = 2500,
                      network_type: str = "walk",
                      cache_path: str = None) -> nx.MultiDiGraph:
    """
    Download (or load from cache) the pedestrian street network.

    Parameters
    ----------
    place_name : str, optional
        A geocodable place string, e.g. "Hauz Khas, Delhi, India". This is
        ONLY used to resolve a (lat, lon) centre point via `ox.geocode()`
        -- the graph itself is then built as a `dist`-metre radius circle
        around that point (see module docstring for why).
    bbox : tuple, optional
        (north, south, east, west) in decimal degrees, for precise manual
        control over the study area instead of geocoding a place name.
    dist : int
        Radius in metres of the circular study area around the geocoded
        centre point. This is now the SINGLE parameter that controls how
        large the routable area is -- increase it for a larger,
        "Google-Maps-like" coverage area, at the cost of a slower
        Overpass download and RandomForest training pass (both scale
        roughly with the number of edges, i.e. with dist^2).
    network_type : str
        osmnx network filter. 'walk' is correct for a pedestrian safety
        router: it includes footways/pedestrian streets and excludes
        motorways.
    cache_path : str, optional
        If provided and the file exists, load from it instead of hitting
        the Overpass API. If provided and the file does NOT exist, the
        freshly downloaded graph is saved there for next time.

    Returns
    -------
    networkx.MultiDiGraph
        Graph with edge-length attributes guaranteed to be present,
        required by both feature engineering and the routing cost
        function.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[data_extraction] Loading cached graph from {cache_path}")
        return ox.load_graphml(cache_path)

    if bbox:
        north, south, east, west = bbox
        print(f"[data_extraction] Downloading walk network for bbox: {bbox}")
        G = ox.graph_from_bbox(bbox=(west, south, east, north),
                                network_type=network_type, simplify=True)

    elif place_name:
        try:
            lat, lon = ox.geocode(place_name)
        except Exception as e:
            raise ValueError(f"Could not geocode place_name '{place_name}': {e}") from e

        print(f"[data_extraction] Geocoded '{place_name}' -> ({lat:.5f}, {lon:.5f}); "
              f"downloading walk network within {dist} m.")
        G = ox.graph_from_point((lat, lon), dist=dist, network_type=network_type, simplify=True)
    else:
        raise ValueError("Provide either place_name or bbox.")

    # Guarantee every edge has a 'length' attribute in metres -- required
    # by feature_engineering.py and router.py.
    G = ox.distance.add_edge_lengths(G)

    if cache_path:
        ox.save_graphml(G, cache_path)
        print(f"[data_extraction] Cached graph to {cache_path}")

    return G


def fetch_pois(place_name: str = None,
                bbox: tuple = None,
                dist: int = 2500,
                tags: dict = None,
                cache_path: str = None) -> gpd.GeoDataFrame:
    """
    Download (or load from cache) Points of Interest relevant to the
    safety model, from the SAME `dist`-metre circular area used for the
    road graph (see fetch_road_graph docstring) so POIs and streets
    always cover identical ground.

    Parameters
    ----------
    tags : dict
        OSM tag filter dict, e.g.
        {"amenity": ["police", "hospital"], "shop": True, "lit": True}.
        `True` as a value means "any value accepted for this key".
    cache_path : str, optional
        If provided and the file exists, POIs are loaded directly from
        disk via `gpd.read_file()` -- no network call at all.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[data_extraction] Loading cached POIs from {cache_path}")
        return gpd.read_file(cache_path)

    default_tags = {
        "amenity": ["police", "hospital", "pharmacy", "fire_station"],
        "shop": True,
        "lit": True,
    }
    tags = tags or default_tags

    if bbox:
        north, south, east, west = bbox
        print(f"[data_extraction] Fetching POIs for bbox: {bbox}")
        pois = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)

    elif place_name:
        try:
            lat, lon = ox.geocode(place_name)
        except Exception as e:
            raise ValueError(f"Could not geocode place_name '{place_name}': {e}") from e

        print(f"[data_extraction] Fetching POIs within {dist} m of geocoded '{place_name}'.")
        pois = ox.features_from_point((lat, lon), tags=tags, dist=dist)
    else:
        raise ValueError("Provide either place_name or bbox.")

    if cache_path:
        pois.to_file(cache_path, driver="GeoJSON")
        print(f"[data_extraction] Cached POIs to {cache_path}")

    return pois


if __name__ == "__main__":
    # Quick smoke test / demo usage
    PLACE = "Connaught Place, New Delhi, India"
    graph = fetch_road_graph(place_name=PLACE, dist=2500, cache_path="cache_graph_demo.graphml")
    pois = fetch_pois(place_name=PLACE, dist=2500, cache_path="cache_pois_demo.geojson")
    print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"POIs fetched: {len(pois)}")
