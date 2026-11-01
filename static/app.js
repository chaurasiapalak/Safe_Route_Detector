/*
 * app.js
 * ------
 * Client-side interactivity for the Safe Route Optimizer frontend.
 * Enhanced with clean tabbed route switching and streamlined diagnostic views.
 */

// ---------------------------------------------------------------------------------------------
// Basemap.  CARTO now requires a (free) API key on its raster tiles: without one every tile is
// stamped "API KEY REQUIRED".  Get a key at https://carto.com/basemaps/apikey (free, no account,
// emailed instantly), paste it below, and restrict it to your domain in dashboard.basemaps.carto.com.
// While CARTO_API_KEY is empty the map uses OpenStreetMap's standard tiles, which need no key.
// ---------------------------------------------------------------------------------------------
const CARTO_API_KEY = "";   // <-- paste your CARTO basemaps key here

const BASEMAP = CARTO_API_KEY
  ? {
      url: `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=${encodeURIComponent(CARTO_API_KEY)}`,
      subdomains: "abcd",
      maxZoom: 20,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors ' +
        '&copy; <a href="https://carto.com/attributions">CARTO</a>',
    }
  : {
      url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      subdomains: "abc",
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    };

const ROUTE_COLORS = {
  shortest: "#f43f5e", // hot coral
  balanced: "#eab308", // amber
  safest: "#14b8a6",   // teal
};

const DEFAULT_VIEW = { center: [28.5535, 77.1998], zoom: 14 };

// --- Map setup ---
const map = L.map("map").setView(DEFAULT_VIEW.center, DEFAULT_VIEW.zoom);
L.tileLayer(BASEMAP.url, { attribution: BASEMAP.attribution, maxZoom: BASEMAP.maxZoom, subdomains: BASEMAP.subdomains, detectRetina: true }).addTo(map);

let originMarker = null;
let destMarker = null;
let routeLayers = {};
let latestRoutesData = null;
let activeRouteKey = "safest"; // default active view

const startIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41], iconAnchor: [12, 41], shadowSize: [41, 41],
  className: "start-marker-cyan",
});
const destIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41], iconAnchor: [12, 41], shadowSize: [41, 41],
  className: "dest-marker-coral",
});

// --- Glowing route styling -------------------------------------------------------------
// Every route is TWO stacked polylines: a wide, translucent halo (opacity 0.3 = the "glow") under a
// thin, fully opaque core line that stays crisp on the light basemap.
const ROUTE_STYLE = {
  active:   { core: 5,   halo: 14, coreOpacity: 1.0,  haloOpacity: 0.3 },
  inactive: { core: 3.5, halo: 10, coreOpacity: 0.6,  haloOpacity: 0.15 },
};

function makeGlowPair(latlngs, color, active, cap) {
  const st = active ? ROUTE_STYLE.active : ROUTE_STYLE.inactive;
  const common = { color, lineCap: cap || "round", lineJoin: "round" };
  const halo = L.polyline(latlngs, { ...common, weight: st.halo, opacity: st.haloOpacity, interactive: false, className: "route-halo" });
  const core = L.polyline(latlngs, { ...common, weight: st.core, opacity: st.coreOpacity, className: "route-core" });
  return { halo, core };
}

function makeGlowRoute(latlngs, color, label, active) {
  const { halo, core } = makeGlowPair(latlngs, color, active);
  core.bindTooltip(label, { sticky: true });
  return { halo, core, group: L.featureGroup([halo, core]), segments: null };
}

function styleRouteEntry(entry, isActive) {
  const st = isActive ? ROUTE_STYLE.active : ROUTE_STYLE.inactive;
  entry.halo.setStyle({ weight: st.halo, opacity: st.haloOpacity });
  entry.core.setStyle({ weight: st.core, opacity: st.coreOpacity });
}

// --- DOM references ---
const placeInput = document.getElementById("place-input");
const distInput = document.getElementById("dist-input");
const loadAreaBtn = document.getElementById("load-area-btn");
const areaStatus = document.getElementById("area-status");

const startInput = document.getElementById("start-input");
const setStartBtn = document.getElementById("set-start-btn");
const destInput = document.getElementById("dest-input");
const setDestBtn = document.getElementById("set-dest-btn");
const locationStatus = document.getElementById("location-status");

const hourSlider = document.getElementById("hour-slider");
const hourValue = document.getElementById("hour-value");
const lambdaSlider = document.getElementById("lambda-slider");
const lambdaValue = document.getElementById("lambda-value");
const findRouteBtn = document.getElementById("find-route-btn");
const routeStatus = document.getElementById("route-status");

const resultsPanel = document.getElementById("results-panel");

// --- State ---
let areaLoaded = false;
let areaCenterLatLng = null;
let originLatLng = null;
let destLatLng = null;

function setStatus(el, message, kind) {
  if (!el) return;
  el.textContent = message;
  el.className = "status" + (kind ? " " + kind : "");
}

function updateFindButtonState() {
  if (findRouteBtn) {
    findRouteBtn.disabled = !(areaLoaded && destLatLng);
  }
}

// Default hour slider to current local time
if (hourSlider && hourValue) {
  const nowHour = new Date().getHours();
  hourSlider.value = nowHour;
  hourValue.textContent = nowHour;

  hourSlider.addEventListener("input", () => { hourValue.textContent = hourSlider.value; });
  hourSlider.addEventListener("change", () => { if (!findRouteBtn.disabled) findRoute(); });
}

if (lambdaSlider && lambdaValue) {
  lambdaSlider.addEventListener("input", () => { lambdaValue.textContent = parseFloat(lambdaSlider.value).toFixed(2); });
  lambdaSlider.addEventListener("change", () => { if (!findRouteBtn.disabled) findRoute(); });
}

// --- Load Area ---
if (loadAreaBtn) {
  loadAreaBtn.addEventListener("click", loadArea);
}

async function loadArea() {
  const place = placeInput.value.trim();
  const dist = parseInt(distInput.value, 10) || 2500;

  if (!place) {
    setStatus(areaStatus, "Enter a place name first.", "error");
    return;
  }

  setStatus(areaStatus, "Loading area network...", "loading");
  loadAreaBtn.disabled = true;
  areaLoaded = false;
  updateFindButtonState();

  try {
    const res = await fetch("/api/load_area", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ place, dist }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to load area.");

    const [lat, lon] = data.center;
    const b = data.bounds;
    map.fitBounds([[b.south, b.west], [b.north, b.east]]);

    startInput.value = "";
    destInput.value = "";
    if (destMarker) { map.removeLayer(destMarker); destMarker = null; }
    destLatLng = null;
    clearRouteLayers();
    if (resultsPanel) resultsPanel.style.display = "none";

    areaCenterLatLng = [lat, lon];
    originLatLng = areaCenterLatLng;
    if (originMarker) map.removeLayer(originMarker);
    originMarker = L.marker(originLatLng, { draggable: false, icon: startIcon })
      .addTo(map)
      .bindPopup("Start (area centre by default)");

    areaLoaded = true;
    startInput.disabled = false;
    setStartBtn.disabled = false;
    destInput.disabled = false;
    setDestBtn.disabled = false;
    [startInput, setStartBtn, destInput, setDestBtn].forEach((el) => el && el.removeAttribute("title"));

    setStatus(
      areaStatus,
      `Loaded "${data.place}" \u2014 ${data.num_nodes} nodes, ${data.num_edges} edges.`,
      "ok"
    );
    if (window.SRO) window.SRO.onAreaLoaded(data);   // boundary, minimap, landmark pills, inspector
  } catch (err) {
    setStatus(areaStatus, err.message, "error");
  } finally {
    loadAreaBtn.disabled = false;
    updateFindButtonState();
  }
}

// --- Geocode Helpers ---
async function geocode(query) {
  const res = await fetch("/api/geocode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Could not find that location.");
  return [data.lat, data.lon];
}

if (setStartBtn) {
  setStartBtn.addEventListener("click", setStart);
}
if (startInput) {
  startInput.addEventListener("keydown", (e) => { if (e.key === "Enter") setStart(); });
}

async function setStart() {
  const query = startInput.value.trim();
  if (!query) {
    originLatLng = areaCenterLatLng;
    if (originMarker) {
      originMarker.setLatLng(originLatLng).setPopupContent("Start (area centre by default)");
    }
    setStatus(locationStatus, "Start reset to area centre.", "ok");
    return;
  }
  setStatus(locationStatus, `Searching for "${query}"...`, "loading");
  setStartBtn.disabled = true;
  try {
    const latlng = await geocode(query);
    originLatLng = latlng;
    if (originMarker) {
      originMarker.setLatLng(latlng).setPopupContent(`Start: ${query}`).openPopup();
    }
    map.panTo(latlng);
    setStatus(locationStatus, `Start set to "${query}".`, "ok");
  } catch (err) {
    setStatus(locationStatus, err.message, "error");
  } finally {
    setStartBtn.disabled = false;
  }
}

if (setDestBtn) {
  setDestBtn.addEventListener("click", setDestination);
}
if (destInput) {
  destInput.addEventListener("keydown", (e) => { if (e.key === "Enter") setDestination(); });
}

async function setDestination() {
  const query = destInput.value.trim();
  if (!query) {
    setStatus(locationStatus, "Type a destination first.", "error");
    return;
  }
  setStatus(locationStatus, `Searching for "${query}"...`, "loading");
  setDestBtn.disabled = true;
  try {
    const latlng = await geocode(query);
    destLatLng = latlng;
    if (destMarker) {
      destMarker.setLatLng(latlng).setPopupContent(`Destination: ${query}`).openPopup();
    } else {
      destMarker = L.marker(latlng, { draggable: false, icon: destIcon })
        .addTo(map)
        .bindPopup(`Destination: ${query}`)
        .openPopup();
    }
    map.panTo(latlng);
    setStatus(locationStatus, `Destination set to "${query}".`, "ok");
    updateFindButtonState();
  } catch (err) {
    setStatus(locationStatus, err.message, "error");
  } finally {
    setDestBtn.disabled = false;
  }
}

// --- Set start / destination from a known coordinate (landmark pills, inspector "Go here") ---
function flyToPoint(latlng) {
  map.flyTo(latlng, Math.max(map.getZoom(), 16), { duration: 0.6 });
}

function setStartLatLng(latlng, label, opts) {
  if (!areaLoaded || !originMarker) return false;
  originLatLng = latlng;
  startInput.value = label || "";
  originMarker.setLatLng(latlng).setPopupContent(`Start: ${label || "selected point"}`).openPopup();
  if (!opts || opts.fly !== false) flyToPoint(latlng);       // picking on the map keeps the current view
  setStatus(locationStatus, `Start set to "${label || "selected point"}".`, "ok");
  return true;
}

function setDestLatLng(latlng, label, opts) {
  if (!areaLoaded) return false;
  destLatLng = latlng;
  destInput.value = label || "";
  if (destMarker) {
    destMarker.setLatLng(latlng).setPopupContent(`Destination: ${label || "selected point"}`).openPopup();
  } else {
    destMarker = L.marker(latlng, { draggable: false, icon: destIcon })
      .addTo(map)
      .bindPopup(`Destination: ${label || "selected point"}`)
      .openPopup();
  }
  if (!opts || opts.fly !== false) flyToPoint(latlng);
  setStatus(locationStatus, `Destination set to "${label || "selected point"}".`, "ok");
  updateFindButtonState();
  return true;
}
window.setStartLatLng = setStartLatLng;
window.setDestLatLng = setDestLatLng;

function clearRouteLayers() {
  Object.values(routeLayers).forEach((entry) => {
    map.removeLayer(entry.group);
    if (entry.segments) map.removeLayer(entry.segments);
  });
  routeLayers = {};
  if (window.SRO) window.SRO.onRoutesCleared();
}

// --- Find Route ---
if (findRouteBtn) {
  findRouteBtn.addEventListener("click", findRoute);
}

// Bottom-HUD-aware padding for route fits: keeps polylines clear of the collapsed #bottom-hud.
// paddingBottomRight is [right, bottom] in Leaflet.
function fitOpts(extra) {
  return Object.assign({ paddingTopLeft: [30, 30], paddingBottomRight: [20, 220] }, extra || {});
}

async function findRoute() {
  if (!areaLoaded || !destLatLng) return;

  const place = placeInput.value.trim();
  const dist = parseInt(distInput.value, 10) || 2500;
  const hour = parseInt(hourSlider.value, 10);
  const lambda = parseFloat(lambdaSlider.value);

  setStatus(routeStatus, "Computing routes...", "loading");
  findRouteBtn.disabled = true;

  try {
    const res = await fetch("/api/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ place, dist, hour, lambda, orig: originLatLng, dest: destLatLng }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Routing failed.");

    clearRouteLayers();
    const boundsPoints = [];
    latestRoutesData = data.routes;

    // Render glowing (halo + core) polylines on the map; the active route is drawn last, on top.
    [...["shortest", "balanced", "safest"].filter((k) => k !== activeRouteKey), activeRouteKey].forEach((key) => {
      const route = data.routes[key];
      if (!route) return;
      // GeoJSON is [lon, lat]; Leaflet polylines want [lat, lon].
      const latlngs = route.geojson.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
      const entry = makeGlowRoute(latlngs, ROUTE_COLORS[key],
                                  `${route.label} (${route.distance_m.toFixed(0)} m)`, key === activeRouteKey);
      entry.group.addTo(map);
      routeLayers[key] = entry;
      boundsPoints.push(...latlngs);
    });

    if (boundsPoints.length) map.fitBounds(L.latLngBounds(boundsPoints), fitOpts());

    if (resultsPanel) resultsPanel.style.display = "block";
    if (window.SRO) window.SRO.onRoutesRendered(data);   // starts the 24-hour profile fetch
    renderStreamlinedRoutes();

    const warnings = data.warnings || [];
    if (warnings.length) {
      setStatus(routeStatus, warnings.join(" "), "error");
    } else {
      setStatus(routeStatus, `Routes computed for ${hour}:00, \u03bb=${lambda.toFixed(2)}.`, "ok");
    }
  } catch (err) {
    setStatus(routeStatus, err.message, "error");
  } finally {
    updateFindButtonState();
  }
}

// ============================================================================================
// Route-card helpers: peak-hazard card, factor chips, structured segments, "locate on map"
// ============================================================================================
function escHtml(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const prettyWord = (w) => String(w || "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
const CROSSHAIR_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">' +
  '<circle cx="12" cy="12" r="7"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/></svg>';

// Short, human tag for one model driver ({factor, text}). null = not worth a tag.
const FACTOR_TAGS = {
  time_of_day:     (t) => (/night/i.test(t) ? "night hours" : "high-incident hour"),
  shop_activity:   (t) => (/little|few/i.test(t) ? "few open shops" : "busy frontage"),
  police_distance: (t) => (/far/i.test(t) ? "far from police" : "near police"),
  road_type:       (t) => t.replace(/\s*segment$/i, "").toLowerCase(),
  connectivity:    (t) => (/dead-end|poorly/i.test(t) ? "dead-end" : "busy junction"),
  segment_length:  (t) => (/long/i.test(t) ? "long stretch" : "short segment"),
  dead_end_at_night: () => "dead-end at night",
  darkness_hazard:   () => "dark at night",
  isolation_index:   () => "isolated",
  model_rating:      () => null,
};
function shortFactor(f) {
  const fn = FACTOR_TAGS[f.factor];
  if (fn) return fn(f.text || "");
  return String(f.text || "").split(" (")[0].slice(0, 28) || null;
}
function tagFromText(t) {                       // fallback when only free-text reasons exist
  const s = String(t).toLowerCase();
  if (/police/.test(s)) return "far from police";
  if (/dead-end|poorly connected/.test(s)) return "dead-end";
  if (/no shops|little open|few shops/.test(s)) return "few open shops";
  if (/night|dark/.test(s)) return "night hours";
  if (/footpath|footway|path/.test(s)) return "footpath";
  return null;
}
function peakTags(peak) {
  const mx = peak.micro_explanation || {};
  let tags = (mx.top_risk_factors || []).map(shortFactor);
  if (!tags.filter(Boolean).length) tags = (peak.reasons || []).map(tagFromText);
  return [...new Set(tags.filter(Boolean))].slice(0, 4);
}

function peakCardHtml(key, peak) {
  const score = Math.round(peak.risk_score);
  const street = peak.street || (peak.highway ? prettyWord(peak.highway) : "");
  const tags = peakTags(peak);
  return `
    <div class="peak-card" id="peak-card">
      <div class="peak-head">
        <span class="peak-title">Peak hazard \u00b7 bottleneck</span>
        <span class="peak-score">${score}<small>/100</small></span>
      </div>
      <div class="peak-pills">
        <span class="glow-pill len">${Math.round(peak.length_m || 0)} m</span>
        ${street ? `<span class="glow-pill street" title="${escHtml(street)}">${escHtml(street)}</span>` : ""}
      </div>
      ${tags.length ? `<div class="peak-tags">${tags.map((t) => `<span class="risk-tag">${escHtml(t)}</span>`).join("")}</div>` : ""}
      <button type="button" class="locate-btn" onclick="locatePeak('${key}')">${CROSSHAIR_SVG}Locate this stretch on map</button>
    </div>`;
}

// Factor chips: one neutral glass surface; only the icon badge is coloured by topic.
function chipKind(text) {
  const t = text.toLowerCase();
  if (/police/.test(t)) return { cls: "blue", icon: "\u{1F6E1}\uFE0F" };                              // shield
  if (/late-night|night window|dead-end|dark|isolated/.test(t)) return { cls: "rose", icon: "\u26A0\uFE0F" };
  if (/shop|commercial|frontage|storefront|eyes on|witness/.test(t)) return { cls: "amber", icon: "\u{1F441}\uFE0F" }; // eye
  return { cls: "rose", icon: "\u26A0\uFE0F" };                                                      // general risk / isolation
}
function factorChipsHtml(bullets, peakShown) {
  const seen = new Set();
  const items = bullets.filter((b) => {
    const key = String(b).trim().toLowerCase();
    if (!key || seen.has(key)) return false;
    seen.add(key);
    if (peakShown && /^peak segment/.test(key)) return false;     // the Peak Hazard card already says this
    return true;
  });
  if (!items.length) return "";
  return `<div class="factor-list">` + items.map((b) => {
    const k = chipKind(String(b));
    return `<div class="factor-chip"><span class="chip-ico ${k.cls}">${k.icon}</span><span class="chip-text">${escHtml(b)}</span></div>`;
  }).join("") + `</div>`;
}

// Segment: structured data first (drivers[] / archetype / risk_score); the legacy
// "Elevated risk (score 93/100); context resembles 'X' incidents; main factors: ...; ..." string is only a fallback.
function parseSegment(seg) {
  const raw = String(seg.reason || seg.explanation || "");
  const out = {
    score: Math.round(Number(seg.risk_score ?? seg.score ?? seg.risk ?? 0)),
    length: Math.round(Number(seg.length_m) || 0),
    archetype: seg.archetype || "",
    factors: [],
  };
  (seg.drivers || []).forEach((d) => out.factors.push({ text: d.text, pts: d.points, pos: false }));
  (seg.protective || []).slice(0, 2).forEach((d) => out.factors.push({ text: d.text, pts: d.points, pos: true }));

  if (!out.score) { const m = raw.match(/score\s+(\d+)\s*\/\s*100/i); if (m) out.score = parseInt(m[1], 10); }
  if (!out.archetype) { const m = raw.match(/context resembles '([^']+)'/i); if (m) out.archetype = m[1]; }
  if (!out.factors.length) {
    const i = raw.search(/main factors:/i);
    if (i >= 0) {
      raw.slice(i + "main factors:".length).replace(/\.\s*$/, "").split(";").forEach((part) => {
        const p = part.trim();
        if (!p) return;
        const m = p.match(/^(.*?)\s*\(([+-]?\d+(?:\.\d+)?)\s*pts\)$/i);
        out.factors.push(m ? { text: m[1].trim(), pts: parseFloat(m[2]), pos: false } : { text: p, pts: null, pos: false });
      });
    }
  }
  return out;
}
const fmtPts = (p) => `${p >= 0 ? "+" : "\u2212"}${Math.abs(Math.round(p))} pts`;

function segmentsAccordionHtml(key, segs, cut) {
  const cards = segs.map((seg, i) => {
    const p = parseSegment(seg);
    const band = p.score >= cut.high ? "high" : "elevated";
    const tags = p.factors.map((f) =>
      `<span class="seg-tag ${f.pos ? "pos" : ""}" title="${escHtml(f.text)}">` +
      (f.pts !== null && f.pts !== undefined && !Number.isNaN(f.pts) ? `<i class="pts">${fmtPts(f.pts)}</i>` : "") +
      `<span>${escHtml(f.text)}</span></span>`).join("");
    return `
      <div class="seg-card" onclick="locateSegment('${key}', ${i})" title="Click to locate on map">
        <div class="seg-head"><span class="seg-score ${band}">Risk ${p.score}/100</span><span class="seg-len">${p.length} m</span></div>
        ${p.archetype ? `<div class="seg-context"><b>Context:</b> ${escHtml(p.archetype)}</div>` : ""}
        ${tags ? `<div class="seg-factors">${tags}</div>` : ""}
      </div>`;
  }).join("");
  return `
    <div class="accordion-header" onclick="toggleAccordion(this)">
      <span>Highlighted Route Segments (${segs.length})</span>
      <span class="chevron">\u25bc</span>
    </div>
    <div id="seg-accordion-body" class="accordion-body" style="display: none;">${cards}</div>`;
}

// "Locate on map": fly to a stretch and flash a cyan highlight for a few seconds.
let locateLayer = null, locateTimer = null;
function locateCoords(coords) {
  if (!coords || !coords.length) return;
  map.flyToBounds(L.latLngBounds(coords), { maxZoom: 18, padding: [90, 90], duration: 0.7 });
  if (locateLayer) map.removeLayer(locateLayer);
  locateLayer = L.featureGroup([
    L.polyline(coords, { color: "#22d3ee", weight: 18, opacity: 0.35, lineCap: "round", interactive: false }),
    L.polyline(coords, { color: "#ffffff", weight: 5, opacity: 0.95, lineCap: "round", interactive: false }),
  ]).addTo(map);
  clearTimeout(locateTimer);
  locateTimer = setTimeout(() => { if (locateLayer) { map.removeLayer(locateLayer); locateLayer = null; } }, 6000);
}
window.locatePeak = (key) => { const p = latestRoutesData && latestRoutesData[key] && latestRoutesData[key].peak_segment; if (p) locateCoords(p.coords); };
window.locateSegment = (key, i) => { const r = latestRoutesData && latestRoutesData[key]; const s = r && (r.high_risk_segments || [])[i]; if (s) locateCoords(s.coords); };

// --- Streamlined Tabbed Panel 4 Renderer ---
function renderStreamlinedRoutes() {
  if (!latestRoutesData) return;
  if (window.SRO) window.SRO.unmountDiurnal();    // the card is rebuilt below: destroy the old Chart.js instance first

  // Hide legacy stacked rows if present in template
  ["row-shortest", "row-balanced", "row-safest"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });

  // Resilient target container resolution:
  // 1. Try dedicated #route-explain container
  // 2. Fall back to #results-panel container
  let target = document.getElementById("route-explain");
  let preserveHeader = false;

  if (!target) {
    target = document.getElementById("results-panel");
    preserveHeader = true;
  }
  if (!target) {
    console.error("No results container element found in HTML.");
    return;
  }

  const keys = ["shortest", "balanced", "safest"];
  const titles = { shortest: "Shortest", balanced: "Balanced", safest: "Safest" };

  let html = `<div class="route-selector">`;
  keys.forEach(k => {
    const r = latestRoutesData[k];
    if (!r) return;
    const isActive = (k === activeRouteKey);
    const avgScore = Math.round(r.avg_risk ?? r.avg_insecurity ?? 0);
    const peakScore = Math.round(r.peak_segment_risk ?? r.peak_risk ?? avgScore);

    html += `
      <div class="route-tab-btn ${isActive ? 'active-' + k : ''}" onclick="selectRoute('${k}')">
        <div class="route-tab-title" style="color: ${ROUTE_COLORS[k]}">
          <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:${ROUTE_COLORS[k]};"></span>
          ${titles[k]}
        </div>
        <div class="route-tab-metric">${r.distance_m.toFixed(0)} m</div>
        <div class="route-tab-sub">Risk ${avgScore} \u00b7 Peak ${peakScore}</div>
      </div>
    `;
  });
  html += `</div>`;

  // Render Details for Active Tab
  const cur = latestRoutesData[activeRouteKey];
  if (cur) {
    const avgScore = Math.round(cur.avg_risk ?? cur.avg_insecurity ?? 0);
    const peakScore = Math.round(cur.peak_segment_risk ?? cur.peak_risk ?? avgScore);
    const riskColor = avgScore >= 60 ? "#f43f5e" : (avgScore >= 25 ? "#f59e0b" : "#14b8a6");
    const peakColor = peakScore >= 60 ? "#f43f5e" : (peakScore >= 30 ? "#f59e0b" : "#14b8a6");

    html += `
      <div class="route-stats-grid" id="route-stats-grid">
        <div class="route-stat-item">
          <span class="stat-label">Distance</span>
          <span class="stat-val">${cur.distance_m.toFixed(0)} m</span>
        </div>
        <div class="route-stat-item">
          <span class="stat-label">Avg Risk</span>
          <span class="stat-val" style="color:${riskColor}">${avgScore}/100</span>
        </div>
        <div class="route-stat-item">
          <span class="stat-label">Peak Hazard</span>
          <span class="stat-val" style="color:${peakColor}">${peakScore}/100</span>
        </div>
      </div>
    `;

    // 24-hour sparkline + time scrubber (markup only; charts are mounted after the innerHTML swap below)
    if (window.SRO) html += window.SRO.diurnalMarkup();

    // Peak-hazard HUD card -> calm glass factor chips -> structured highlighted segments
    const cut = (window.SRO && window.SRO.RISK_CUT) || { elevated: 60, high: 85 };
    const peak = cur.peak_segment;
    const showPeak = !!(peak && Number(peak.risk_score) >= cut.elevated);
    if (showPeak) html += peakCardHtml(activeRouteKey, peak);

    html += factorChipsHtml(cur.summary_bullets || cur.reasons || [], showPeak);

    const highRisk = cur.high_risk_segments || cur.peak_segments || [];
    if (highRisk.length > 0) html += segmentsAccordionHtml(activeRouteKey, highRisk, cut);
  }

  // Inject content cleanly
  if (preserveHeader) {
    target.innerHTML = `<h2>4. Routes</h2>` + html;
  } else {
    target.innerHTML = html;
  }
  if (window.SRO) window.SRO.mountDiurnal();
}

// Global window switchers for HTML onclick callbacks
window.selectRoute = function(key) {
  activeRouteKey = key;
  // Selected route: thick glowing line on top; the others dim down.
  Object.entries(routeLayers).forEach(([k, entry]) => styleRouteEntry(entry, k === key));
  if (window.SRO) window.SRO.onRouteSelected();
  else if (routeLayers[key]) routeLayers[key].group.bringToFront();
  renderStreamlinedRoutes();
};

window.toggleAccordion = function(el) {
  const body = el.nextElementSibling;
  const chevron = el.querySelector(".chevron");
  if (body.style.display === "none") {
    body.style.display = "flex";
    chevron.textContent = "\u25b2";
  } else {
    body.style.display = "none";
    chevron.textContent = "\u25bc";
  }
};