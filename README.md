# Spatio-Temporal Safe Route Optimizer

Pedestrian routing that trades off shortest distance against a
time-dependent, ML-predicted "Insecurity Score" for each street segment.

## Setup
```bash
pip install -r requirements.txt
```
(Requires internet access to query OpenStreetMap's Overpass API on first run;
results are cached to `cache_graph.graphml` afterward.)

## Option A — CLI demo (static HTML map)
```bash
python app.py --place "Hauz Khas, Delhi, India" --hour 23 --dist 1200
```
This will:
1. Download the walk network + POIs for `--place`.
2. Engineer static + temporal edge features for `--hour`.
3. Bootstrap synthetic training labels and train a `RandomForestRegressor`.
4. Predict a 0-100 `insecurity_score` for every edge at that hour.
5. Compute three routes (shortest / balanced / safest) between the
   south-west-most and north-east-most graph nodes by default, or between
   `--orig_lat/--orig_lon` and `--dest_lat/--dest_lon` if supplied.
6. Export `safe_route_map.html` — open it in a browser to see all three
   routes overlaid (blue = shortest, orange = balanced, green = safest).

## Option B — Interactive web frontend (recommended for demos/viva)
```bash
python server.py
```
Then open **http://127.0.0.1:5000** in a browser:
1. Type a place name and a **coverage radius** (metres), then click
   **Load Area**. The graph is built as a circle of that radius around
   the geocoded place — not OSM's own (often tiny/arbitrary)
   administrative boundary — so radius is the one knob that controls how
   much ground is covered. Increase it for a larger, more
   "Google Maps"-like area, at the cost of a slower first load.
2. The **blue marker** is placed at the area's centre automatically as
   the *default* start point. To use somewhere else, type an address in
   **Start location** and click **Set** (or press Enter); leaving it
   blank keeps the default.
3. Type an address in **Destination** and click **Set** — this is
   required before routing.
4. Adjust the **hour** and **λ (lambda)** sliders and click **Find
   Route** (or just release a slider) to see the shortest, balanced, and
   safest paths update live on the map, with distance and average-risk
   stats per route. If a typed address falls outside the loaded area, a
   warning is shown (it still routes, using the nearest street it found).

Locations are always chosen by **typing and geocoding** — there's no
click-to-place or marker-dragging, so every point on the map corresponds
to an address you searched for.

## Module map
| File | Responsibility |
|---|---|
| `data_extraction.py` | osmnx graph + POI download/caching, with Point/Polygon geocoding fallback |
| `feature_engineering.py` | static, POI-derived, and temporal edge features |
| `model.py` | synthetic label generation, RandomForest training, score attachment |
| `router.py` | custom Dijkstra with the `lam`-blended cost function |
| `geo_utils.py` | shared route-geometry helpers (used by both `app.py` and `server.py`) |
| `app.py` | CLI orchestration + static folium HTML map |
| `server.py` | Flask backend for the interactive web frontend |
| `templates/index.html`, `static/js/app.js`, `static/css/style.css` | Leaflet-based interactive frontend |

## Key parameter: `lam` (lambda)
```
cost(edge) = lam * normalized_distance + (1 - lam) * normalized_insecurity
```
- `lam = 1.0` → pure shortest path
- `lam = 0.0` → pure safest path
- Anything in between → blended trade-off


