/*
 * app.js
 * ------
 * Client-side interactivity for the Safe Route Optimizer frontend.
 * Enhanced with clean tabbed route switching and streamlined diagnostic views.
 */

const OSM_TILES_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

const ROUTE_COLORS = {
  shortest: "#f43f5e", // hot coral
  balanced: "#eab308", // amber
  safest: "#14b8a6",   // teal
};

const DEFAULT_VIEW = { center: [28.5535, 77.1998], zoom: 14 };

// --- Map setup ---
const map = L.map("map").setView(DEFAULT_VIEW.center, DEFAULT_VIEW.zoom);
L.tileLayer(OSM_TILES_URL, { attribution: OSM_ATTRIBUTION, maxZoom: 19, subdomains: "abc" }).addTo(map);

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

function clearRouteLayers() {
  Object.values(routeLayers).forEach(l => map.removeLayer(l));
  routeLayers = {};
}

// --- Find Route ---
if (findRouteBtn) {
  findRouteBtn.addEventListener("click", findRoute);
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

    // Render polylines on map
    ["shortest", "balanced", "safest"].forEach((key) => {
      const route = data.routes[key];
      if (!route) return;
      const isDefaultActive = (key === activeRouteKey);
      const layer = L.geoJSON(route.geojson, {
        style: {
          color: ROUTE_COLORS[key],
          weight: isDefaultActive ? 6 : 3.5,
          opacity: isDefaultActive ? 0.95 : 0.45,
        },
      }).bindTooltip(`${route.label} (${route.distance_m.toFixed(0)} m)`);
      layer.addTo(map);
      routeLayers[key] = layer;
      layer.eachLayer((l) => boundsPoints.push(...l.getLatLngs()));
    });

    if (boundsPoints.length) map.fitBounds(L.latLngBounds(boundsPoints), { padding: [30, 30] });

    if (resultsPanel) resultsPanel.style.display = "block";
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

// --- Streamlined Tabbed Panel 4 Renderer ---
function renderStreamlinedRoutes() {
  if (!latestRoutesData) return;

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
      <div class="route-stats-grid">
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

    // Factored Badges / Environmental Summary Chips
    const bullets = cur.summary_bullets || cur.reasons || [];
    if (bullets.length > 0) {
      html += `<div class="factor-list">`;
      bullets.forEach(b => {
        let chipClass = "neutral";
        const bLower = b.toLowerCase();
        if (bLower.includes("night") || bLower.includes("no shops") || bLower.includes("isolated") || bLower.includes("dark")) {
          chipClass = "danger";
        } else if (bLower.includes("police") || bLower.includes("open") || bLower.includes("lower risk") || bLower.includes("surveillance")) {
          chipClass = "safe";
        }
        html += `<div class="factor-chip ${chipClass}"><span>\u2022</span><span>${b}</span></div>`;
      });
      html += `</div>`;
    }

    // Collapsible High-Risk Segments Accordion
    const highRisk = cur.high_risk_segments || cur.peak_segments || [];
    if (highRisk.length > 0) {
      html += `
        <div class="accordion-header" onclick="toggleAccordion(this)">
          <span>Highlighted Route Segments (${highRisk.length})</span>
          <span class="chevron">\u25bc</span>
        </div>
        <div class="accordion-body" style="display: none;">
      `;
      highRisk.forEach(seg => {
        const segRisk = Math.round(seg.score ?? seg.risk ?? 0);
        const sColor = segRisk >= 60 ? "#f43f5e" : "#f59e0b";
        html += `
          <div class="segment-card">
            <div class="segment-card-top">
              <span style="color: ${sColor}">Risk ${segRisk}/100</span>
              <span style="color: var(--text-dim)">${Math.round(seg.length_m || 0)} m</span>
            </div>
            <div style="color: #cbd5e1; font-size: 11px; line-height: 1.35;">
              ${seg.reason || seg.explanation || ""}
            </div>
          </div>
        `;
      });
      html += `</div>`;
    }
  }

  // Inject content cleanly
  if (preserveHeader) {
    target.innerHTML = `<h2>4. Routes</h2>` + html;
  } else {
    target.innerHTML = html;
  }
}

// Global window switchers for HTML onclick callbacks
window.selectRoute = function(key) {
  activeRouteKey = key;
  // Update map polyline styles (make selected route bold, dim others)
  Object.entries(routeLayers).forEach(([k, layer]) => {
    const isSel = (k === key);
    layer.setStyle({
      weight: isSel ? 6 : 3.5,
      opacity: isSel ? 0.95 : 0.45,
    });
    if (isSel) layer.bringToFront();
  });
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