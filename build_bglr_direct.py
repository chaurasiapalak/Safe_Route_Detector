#!/usr/bin/env python3
"""
build_bglr_direct.py
--------------------
Downloads raw OpenStreetMap data directly from the official OSM API (api.openstreetmap.org)
using 4 small sub-boxes to strictly respect the 50,000 node API limit.
"""

from pathlib import Path
import time
import xml.etree.ElementTree as ET
import geopandas as gpd
import networkx as nx
import osmnx as ox
import requests

CACHE_DIR = Path("training_cache_bglr")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = CACHE_DIR / "raw_osm"
RAW_DIR.mkdir(exist_ok=True)

graph_path = CACHE_DIR / "training_graph.graphml"
poi_path = CACHE_DIR / "training_pois.geojson"

# Main extent covering Jayanagar, Banashankari, Basavanagudi
MIN_LON, MIN_LAT = 77.565, 12.915
MAX_LON, MAX_LAT = 77.595, 12.945

# Split into a 2x2 grid (4 chunks) to stay well under the 50,000 node limit
MID_LON = (MIN_LON + MAX_LON) / 2.0
MID_LAT = (MIN_LAT + MAX_LAT) / 2.0

sub_boxes = [
    (MIN_LON, MIN_LAT, MID_LON, MID_LAT),  # SW
    (MID_LON, MIN_LAT, MAX_LON, MID_LAT),  # SE
    (MIN_LON, MID_LAT, MID_LON, MAX_LAT),  # NW
    (MID_LON, MID_LAT, MAX_LON, MAX_LAT),  # NE
]

headers = {"User-Agent": "SafeRouteResearch/1.0 (academic.project@bennett.edu.in)"}
graphs = []
poi_records = []

print(f"Downloading map data in {len(sub_boxes)} chunks directly from api.openstreetmap.org...\n")

for i, (min_x, min_y, max_x, max_y) in enumerate(sub_boxes, 1):
    xml_file = RAW_DIR / f"chunk_{i}.osm"
    print(f"[{i}/{len(sub_boxes)}] Fetching chunk ({min_x:.3f}, {min_y:.3f}) to ({max_x:.3f}, {max_y:.3f})...")

    # 1. Download chunk XML if not cached
    if not xml_file.exists():
        url = f"https://api.openstreetmap.org/api/0.6/map?bbox={min_x},{min_y},{max_x},{max_y}"
        resp = requests.get(url, headers=headers, timeout=120)
        if resp.status_code != 200:
            raise SystemExit(f"OSM API request failed for chunk {i} ({resp.status_code}): {resp.text[:120]}")
        xml_file.write_bytes(resp.content)
        print(f"    Saved {len(resp.content) // 1024} KB.")
        time.sleep(1.0)
    else:
        print("    Loaded from disk cache.")

    # 2. Parse chunk into OSMnx Graph
    try:
        g = ox.graph_from_xml(str(xml_file), simplify=False, retain_all=True)
        graphs.append(g)
    except Exception as e:
        print(f"    Warning: Could not extract graph from chunk {i}: {e}")

    # 3. Extract POIs (police & shops) directly from XML tags
    tree = ET.parse(xml_file)
    for node in tree.getroot().findall("node"):
        tags = {tag.attrib.get("k"): tag.attrib.get("v") for tag in node.findall("tag")}
        amenity = tags.get("amenity")
        shop = tags.get("shop")
        if amenity == "police" or shop is not None:
            lat = float(node.attrib["lat"])
            lon = float(node.attrib["lon"])
            poi_records.append({
                "amenity": "police" if amenity == "police" else None,
                "shop": shop,
                "geometry": gpd.points_from_xy([lon], [lat])[0],
            })

if not graphs:
    raise SystemExit("No road graphs could be extracted.")

# 4. Compose and simplify the stitched graph
print("\nStitching chunks into unified road graph...")
G_combined = nx.compose_all(graphs)
simplify = ox.simplification.simplify_graph if hasattr(ox, "simplification") else ox.simplify_graph
G_final = simplify(G_combined)
G_final = ox.distance.add_edge_lengths(G_final)
ox.save_graphml(G_final, graph_path)
print(f" Saved graph ({len(G_final.nodes)} nodes, {len(G_final.edges)} edges) to: {graph_path}")

# 5. Save combined POIs
if poi_records:
    pois_gdf = gpd.GeoDataFrame(poi_records, geometry="geometry", crs="EPSG:4326").drop_duplicates(subset=["geometry"])
else:
    pois_gdf = gpd.GeoDataFrame(
        {"amenity": ["police"], "shop": [None], "geometry": gpd.points_from_xy([77.58], [12.93])},
        crs="EPSG:4326",
    )

pois_gdf.to_file(poi_path, driver="GeoJSON")
print(f" Saved {len(pois_gdf)} POIs to: {poi_path}")
print("\nDone! South Bengaluru offline files are ready in training_cache_bglr/")