"""
osm_fetch.py
------------
Resilient OpenStreetMap / Overpass access for train_offline.py.  Works with OSMnx 1.x AND 2.x.

What it fixes
    * OSMnx renamed its settings between 1.x and 2.x (overpass_endpoint -> overpass_url, timeout ->
      requests_timeout, default_user_agent -> http_user_agent).  `configure_osmnx()` sets the right
      ones for whichever version is installed (setting the wrong name is silently ignored, which
      means you keep hammering overpass-api.de without noticing).
    * One dead server no longer kills the run: `MirrorPool` fails over to the next mirror and puts a
      failing mirror in an exponentially growing "cool-down".
    * Adaptive tiles: a tile that fails on every mirror is split into 4 quarter-tiles (down to
      --max-split-depth levels) and the pieces are cached independently, so a re-run resumes.
    * The heavy POI query (every shop = node + way + relation geometry) is replaced by a tiny
      "out center tags" query over larger chunks.  train_offline only needs one point per shop/police
      station, so nothing is lost.
    * Overpass can answer HTTP 200 with a "runtime error: timed out" remark and a TRUNCATED result.
      That is treated as a failure, not as data.
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("osm_fetch")

# Tried strictly in this order; a mirror that fails is skipped for a growing cool-down period.
# overpass-api.de allows only ~2 concurrent query slots per IP, so this code is strictly sequential.
# NOTE: some community mirrors have been seen serving stale snapshots -- they are last-resort fallbacks.
DEFAULT_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
"https://overpass.openstreetmap.fr/api/interpreter",
]
# Removed after real runs: maps.mail.ru (403 Forbidden), overpass.kumi.systems and
# overpass.private.coffee (500 Internal Server Error).  Add working ones with --overpass-url.
# User-Agent: None = leave OSMnx's own default.  A fake/placeholder UA (e.g. you@example.com) is what
# overpass-api.de answers with "406 Not Acceptable", so only pass --user-agent with a REAL contact.
DEFAULT_USER_AGENT = None

POI_TAGS = {"amenity": ["police"], "shop": True}   # documented for reference; the query below mirrors it


class DownloadFailed(RuntimeError):
    """Every attempt on every mirror failed for one query."""


class DownloadIncomplete(RuntimeError):
    """Some tiles could not be fetched; `.failed` lists their bounding boxes."""

    def __init__(self, message, failed):
        super().__init__(message)
        self.failed = failed


class OverpassReplyError(RuntimeError):
    """HTTP error, non-JSON body, or a 200 reply whose 'remark' says the result is truncated."""


# --------------------------------------------------------------------------- #
# OSMnx 1.x / 2.x configuration
# --------------------------------------------------------------------------- #
def ox_major_version() -> int:
    import osmnx as ox
    return int(str(ox.__version__).split(".")[0])


_V1_ENDPOINT_IS_FULL_URL: Optional[bool] = None


def _v1_endpoint_value(url: str) -> str:
    """OSMnx 1.x: `overpass_endpoint` is normally a BASE ('https://host/api') and OSMnx appends
    '/interpreter'.  Very old 1.x releases expected the full '.../api/interpreter'.  Detect which
    style the installed version uses from its default value (read once, before we modify it)."""
    global _V1_ENDPOINT_IS_FULL_URL
    import osmnx as ox
    if _V1_ENDPOINT_IS_FULL_URL is None:
        default = str(getattr(ox.settings, "overpass_endpoint", "") or "")
        _V1_ENDPOINT_IS_FULL_URL = default.rstrip("/").endswith("/interpreter")
    url = url.rstrip("/")
    if _V1_ENDPOINT_IS_FULL_URL:
        return url
    return url[: -len("/interpreter")] if url.endswith("/interpreter") else url


def configure_osmnx(url: str, timeout_s: float = 90.0, user_agent=DEFAULT_USER_AGENT,
                    cache_folder: Optional[Path] = None) -> None:
    """Point OSMnx at `url` with sane timeouts and a descriptive User-Agent (1.x and 2.x)."""
    import osmnx as ox
    s = ox.settings
    if ox_major_version() >= 2:
        s.overpass_url = url                       # full ".../api/interpreter"
        s.requests_timeout = int(timeout_s)        # also used as Overpass "[timeout:N]"
        if user_agent:
            s.http_user_agent = user_agent
    else:
        s.overpass_endpoint = _v1_endpoint_value(url)
        s.timeout = int(timeout_s)
        if user_agent:
            s.default_user_agent = user_agent
    # OSMnx's own pre-query "/status" slot check is one more request that can hang on a mirror;
    # we do our own pacing + fail-over instead.  (Do NOT put 'timeout' into settings.requests_kwargs:
    # OSMnx already passes it and requests would raise "multiple values for keyword argument".)
    s.overpass_rate_limit = False
    s.use_cache = True
    s.log_console = False
    if cache_folder is not None:
        s.cache_folder = str(cache_folder)


# --------------------------------------------------------------------------- #
# Mirror pool + retry with fail-over
# --------------------------------------------------------------------------- #
class MirrorPool:
    """Sticks with the current healthy mirror, fails over on error, and cools failing mirrors down
    for base * 2^(consecutive_failures-1) seconds (capped, jittered; doubled for HTTP 429)."""

    HARD_BLOCK_S = 3600.0

    def __init__(self, urls, base_cooldown: float = 30.0, max_cooldown: float = 240.0):
        self.urls = list(dict.fromkeys(u.strip() for u in urls if u and u.strip()))
        if not self.urls:
            raise ValueError("no Overpass URLs given")
        self.base, self.cap = base_cooldown, max_cooldown
        self._until = {u: 0.0 for u in self.urls}
        self._streak = {u: 0 for u in self.urls}
        self._current = self.urls[0]

    def _healthy(self, now):
        return [u for u in self.urls if self._until[u] <= now]

    def pick(self):
        """-> (url, seconds_to_wait_before_using_it)."""
        now = time.monotonic()
        ok = self._healthy(now)
        if ok:
            if self._current not in ok:
                self._current = ok[0]
            return self._current, 0.0
        url = min(self.urls, key=lambda u: self._until[u])
        self._current = url
        return url, max(0.0, self._until[url] - now)

    def has_other_healthy(self, url) -> bool:
        return any(u != url for u in self._healthy(time.monotonic()))

    def success(self, url):
        self._streak[url] = 0
        self._until[url] = 0.0
        self._current = url

    def failure(self, url, exc: Exception):
        self._streak[url] += 1
        msg = str(exc).lower()
        if re.search(r"\b(403|406|401)\b", msg):
            # deterministic rejection (blocked UA / forbidden): retrying in a minute will not change it
            cd = self.HARD_BLOCK_S
            log.warning("    %s REJECTED the request (%s). 406/403 usually means a blocked or fake "
                        "User-Agent: run without --user-agent, or use a real contact e-mail.", url, _short(exc))
        else:
            mult = 2.0 if ("429" in msg or "too many" in msg) else 1.0
            cd = min(self.cap, self.base * mult * 2 ** (self._streak[url] - 1)) * random.uniform(0.8, 1.2)
        self._until[url] = time.monotonic() + cd
        log.info("    mirror %s cooling down for %.0fs (failure #%d)", url, cd, self._streak[url])


def is_empty_result(exc: Exception) -> bool:
    """OSMnx raises InsufficientResponseError when OSM simply has nothing there (a legitimate empty tile)."""
    return type(exc).__name__ == "InsufficientResponseError"


def _short(exc: Exception) -> str:
    text = str(exc).strip()
    return (text.splitlines()[0][:140] if text else type(exc).__name__)


def with_mirrors(fn: Callable[[str], object], what: str, pool: MirrorPool, *, attempts: int = 4,
                 timeout_s: float = 90.0, user_agent=DEFAULT_USER_AGENT, use_osmnx: bool = True,
                 cache_folder: Optional[Path] = None):
    """Run fn(url) up to `attempts` times, rotating mirrors.  Empty-result errors are re-raised untouched;
    everything else is retried and finally raised as DownloadFailed."""
    last: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        url, delay = pool.pick()
        if delay > 600:
            raise DownloadFailed(f"{what}: every mirror is blocked/unavailable (next retry in {delay:.0f}s). "
                                 "Check --check-mirrors, drop a fake --user-agent, or use the offline route.")
        if delay > 0:
            log.info("    all mirrors are cooling down; waiting %.0fs", delay)
            time.sleep(delay)
        elif attempt > 1:
            time.sleep(random.uniform(1.5, 4.0))     # tiny pause when failing over to a healthy mirror
        if use_osmnx:
            configure_osmnx(url, timeout_s, user_agent, cache_folder)
        try:
            result = fn(url)
            pool.success(url)
            return result
        except Exception as exc:                      # noqa: BLE001 (KeyboardInterrupt is not caught)
            if is_empty_result(exc):
                pool.success(url)
                raise
            last = exc
            pool.failure(url, exc)
            log.warning("%s: attempt %d/%d on %s failed (%s: %s)", what, attempt, attempts, url,
                        type(exc).__name__, _short(exc))
    raise DownloadFailed(f"{what}: {attempts} attempts failed; last error: {_short(last)}") from last


# --------------------------------------------------------------------------- #
# Pre-flight probe
# --------------------------------------------------------------------------- #
def probe_mirrors(urls, user_agent=DEFAULT_USER_AGENT, timeout_s: float = 20.0) -> dict:
    """Send one tiny query to every mirror.  Returns {url: (ok, message)}."""
    import requests
    out = {}
    base_ua = user_agent if user_agent else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SafeRouteBot/1.0"
    headers = {
            "User-Agent": base_ua,
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
        }
    for u in urls:
        try:
            r = requests.post(u, data={"data": "[out:json][timeout:10];node(1);out ids;"}, headers=headers,
                                  timeout=(10, timeout_s))
            ok = r.status_code == 200 and "elements" in r.text[:2000]
            out[u] = (ok, "OK" if ok else f"HTTP {r.status_code} {r.reason}")
        except Exception as exc:                      # noqa: BLE001
            out[u] = (False, f"{type(exc).__name__}: {_short(exc)}")
    return out


def log_probe(results: dict) -> list:
    for u, (ok, msg) in results.items():
        (log.info if ok else log.warning)("  mirror check: %-62s %s", u, msg)
    return [u for u, (ok, _) in results.items() if ok]


# --------------------------------------------------------------------------- #
# Tiling helpers
# --------------------------------------------------------------------------- #
def make_tiles(south, west, north, east, tile_km):
    """Same maths (and therefore the same cache keys) as the original train_offline._tiles."""
    lat_step = tile_km / 111.32
    lon_step = tile_km / (111.32 * math.cos(math.radians((south + north) / 2)))
    lats = np.arange(south, north, lat_step).tolist() + [north]
    lons = np.arange(west, east, lon_step).tolist() + [east]
    return [(lats[i], lons[j], lats[i + 1], lons[j + 1])
            for i in range(len(lats) - 1) for j in range(len(lons) - 1)]


def split4(box):
    s, w, n, e = box
    ms, mw = (s + n) / 2, (w + e) / 2
    return [(s, w, ms, mw), (s, mw, ms, e), (ms, w, n, mw), (ms, mw, n, e)]


def bbox_key(box) -> str:
    s, w, n, e = box
    return f"{s:.4f}_{w:.4f}_{n:.4f}_{e:.4f}"


@dataclass
class _Ctx:
    cache_dir: Path
    prefix: str
    ext: str
    fetch: Callable[[tuple], object]      # box -> object (may raise DownloadFailed / empty-result error)
    load: Callable[[Path], object]
    save: Callable[[object, Path], None]
    max_depth: int
    pause: float


def _adaptive(box, depth: int, ctx: _Ctx) -> list:
    """Fetch `box`; if it fails on every mirror, split into 4 and recurse.  Each leaf is cached as
    <prefix>_<key><ext>; '.empty' = legitimately empty, '.split' = this box is handled by its children."""
    path = ctx.cache_dir / f"{ctx.prefix}_{bbox_key(box)}{ctx.ext}"
    empty, split = path.with_suffix(".empty"), path.with_suffix(".split")
    if empty.exists():
        return []
    if path.exists():
        return [ctx.load(path)]
    if not split.exists():
        try:
            obj = ctx.fetch(box)
            ctx.save(obj, path)
            return [obj]
        except Exception as exc:                      # noqa: BLE001
            if is_empty_result(exc):
                empty.touch()
                return []
            if not isinstance(exc, DownloadFailed):
                raise
            if depth >= ctx.max_depth:
                raise
            log.warning("  %s %s failed on all mirrors -> splitting into 4 quarter-tiles (level %d/%d)",
                        ctx.prefix, bbox_key(box), depth + 1, ctx.max_depth)
            split.touch()
    parts = []
    for sub in split4(box):                           # the first failing child aborts this parent
        parts.extend(_adaptive(sub, depth + 1, ctx))
        time.sleep(ctx.pause)
    return parts


# --------------------------------------------------------------------------- #
# Road graph (OSMnx)
# --------------------------------------------------------------------------- #
def graph_from_bbox_compat(w, s, e, n, **kw):
    import osmnx as ox
    if ox_major_version() >= 2:
        return ox.graph_from_bbox((w, s, e, n), **kw)      # 2.x: (left, bottom, right, top)
    return ox.graph_from_bbox(n, s, e, w, **kw)            # 1.x: north, south, east, west


# --------------------------------------------------------------------------- #
# POIs: one light raw-Overpass query ("out center tags") instead of OSMnx features_from_bbox
# --------------------------------------------------------------------------- #
POI_QUERY = """[out:json][timeout:{t}];
(
  nwr["amenity"="police"]({s},{w},{n},{e});
  nwr["shop"]({s},{w},{n},{e});
);
out center tags qt;"""


def _overpass_json(session, url: str, query: str, timeout_s: float, user_agent: str) -> dict:
    headers = {"User-Agent": user_agent} if user_agent else {}
    r = session.post(url, data={"data": query}, headers=headers, timeout=(15, timeout_s + 30))  # (connect, read)
    if r.status_code != 200:
        raise OverpassReplyError(f"HTTP {r.status_code} from {url}: {r.text[:100]!r}")
    try:
        data = r.json()
    except ValueError as exc:
        raise OverpassReplyError(f"non-JSON reply from {url}: {r.text[:100]!r}") from exc
    remark = str(data.get("remark", "")).lower()
    if "runtime error" in remark or "timed out" in remark or "out of memory" in remark:
        raise OverpassReplyError(f"truncated result from {url}: {remark[:120]}")
    return data


def parse_poi_elements(elements: list) -> pd.DataFrame:
    rows = []
    for el in elements:
        tags = el.get("tags") or {}
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center") or {}
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        rows.append({"osm_type": el.get("type"), "osm_id": el.get("id"), "lat": lat, "lon": lon,
                     "amenity": tags.get("amenity"), "shop": tags.get("shop")})
    return pd.DataFrame(rows, columns=["osm_type", "osm_id", "lat", "lon", "amenity", "shop"])


def fetch_poi_chunk(box, pool: MirrorPool, session, *, attempts, timeout_s, user_agent) -> pd.DataFrame:
    s, w, n, e = box
    q = POI_QUERY.format(t=int(timeout_s), s=s, w=w, n=n, e=e)
    data = with_mirrors(lambda url: _overpass_json(session, url, q, timeout_s, user_agent),
                        f"POIs {bbox_key(box)}", pool, attempts=attempts, timeout_s=timeout_s,
                        user_agent=user_agent, use_osmnx=False)
    return parse_poi_elements(data.get("elements", []))


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def download_training_area(south, west, north, east, cache_dir, *, mirrors=None, tile_km=3.0, poi_tile_km=6.0,
                           timeout_s=90.0, attempts=4, pause=3.0, max_split_depth=2, max_failed_tiles=3,
                           user_agent=DEFAULT_USER_AGENT):
    """Download (resumably) the walk graph + shop/police POIs for the bounding box.
    Returns (networkx MultiDiGraph, GeoDataFrame[amenity, shop, geometry]).
    Raises DownloadIncomplete if any tile could not be fetched (nothing is half-built)."""
    import geopandas as gpd
    import networkx as nx
    import osmnx as ox
    import requests

    cand = list(mirrors or DEFAULT_MIRRORS)
    alive = log_probe(probe_mirrors(cand, user_agent))
    if not alive:
        raise SystemExit("No Overpass mirror answered the pre-flight check (see 'mirror check' lines above).\n"
                         "  406/403 -> blocked User-Agent: remove --user-agent or use a real e-mail.\n"
                         "  5xx/timeouts -> servers down: retry later or use the offline route "
                         "(osm_extract_from_pbf.py, then --graph-file/--pois-file).")
    pool = MirrorPool(alive)
    tiles_dir = Path(cache_dir) / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    log.info("Overpass mirrors (in order): %s", ", ".join(pool.urls))
    failed: list = []

    # ---------------- POIs ----------------
    session = requests.Session()
    poi_ctx = _Ctx(
        cache_dir=tiles_dir, prefix="poi", ext=".csv",
        fetch=lambda box: fetch_poi_chunk(box, pool, session, attempts=attempts, timeout_s=timeout_s,
                                          user_agent=user_agent),
        load=lambda p: pd.read_csv(p), save=lambda df, p: df.to_csv(p, index=False),
        max_depth=max_split_depth, pause=pause)
    poi_parts = []
    poi_boxes = make_tiles(south, west, north, east, poi_tile_km)
    log.info("POIs: %d chunk(s) of ~%.0f km (lightweight 'out center' query)", len(poi_boxes), poi_tile_km)
    for i, box in enumerate(poi_boxes):
        try:
            poi_parts.extend(_adaptive(box, 0, poi_ctx))
            log.info("  POI chunk %d/%d done", i + 1, len(poi_boxes))
        except DownloadFailed as exc:
            log.error("  POI chunk %d/%d FAILED (%s)", i + 1, len(poi_boxes), _short(exc))
            failed.append(("poi", box))
            if len(failed) > max_failed_tiles:
                break
        time.sleep(pause)

    # ---------------- road graph ----------------
    graph_ctx = _Ctx(
        cache_dir=tiles_dir, prefix="graph", ext=".graphml",
        fetch=lambda box: with_mirrors(
            lambda url: graph_from_bbox_compat(box[1], box[0], box[3], box[2], network_type="walk",
                                               simplify=False, retain_all=True, truncate_by_edge=True),
            f"graph {bbox_key(box)}", pool, attempts=attempts, timeout_s=timeout_s, user_agent=user_agent,
            cache_folder=Path(cache_dir) / "osmnx_http_cache"),
        load=lambda p: ox.load_graphml(p), save=lambda g, p: ox.save_graphml(g, p),
        max_depth=max_split_depth, pause=pause)
    graphs = []
    g_boxes = make_tiles(south, west, north, east, tile_km)
    log.info("road graph: %d tile(s) of ~%.1f km; each failing tile is auto-split up to %d time(s)",
             len(g_boxes), tile_km, max_split_depth)
    if len(failed) <= max_failed_tiles:
        for i, box in enumerate(g_boxes):
            try:
                graphs.extend(_adaptive(box, 0, graph_ctx))
                log.info("  graph tile %d/%d done", i + 1, len(g_boxes))
            except DownloadFailed as exc:
                log.error("  graph tile %d/%d FAILED (%s)", i + 1, len(g_boxes), _short(exc))
                failed.append(("graph", box))
                if len(failed) > max_failed_tiles:
                    log.error("too many failed tiles (> %d); stopping early -- finished tiles stay cached",
                              max_failed_tiles)
                    break
            time.sleep(pause)

    if failed:
        raise DownloadIncomplete(f"{len(failed)} tile(s) could not be downloaded", failed)
    if not graphs:
        raise SystemExit("OSM returned no roads for the training extent; check the incident coordinates.")

    G = nx.compose_all(graphs)
    simplify = ox.simplification.simplify_graph if hasattr(ox, "simplification") else ox.simplify_graph
    G = simplify(G)                                    # simplify ONCE, after composing -> no seams

    df = pd.concat(poi_parts, ignore_index=True) if poi_parts else parse_poi_elements([])
    df = df.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)
    attrs = df[["amenity", "shop"]].astype(object)
    attrs = attrs.where(attrs.notna(), None)
    pois = gpd.GeoDataFrame(attrs, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    log.info("downloaded: %d graph nodes, %d edges | POIs: %d police, %d shops", G.number_of_nodes(),
             G.number_of_edges(), int((pois["amenity"] == "police").sum()), int(pois["shop"].notna().sum()))
    return G, pois
