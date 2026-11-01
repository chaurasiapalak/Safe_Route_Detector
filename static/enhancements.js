/*
 * enhancements.js
 * ---------------
 * Presentation layer for the Safe Route Optimizer. Loaded AFTER app.js (it uses app.js globals: map,
 * routeLayers, latestRoutesData, activeRouteKey, BASEMAP, ROUTE_STYLE, makeGlowPair, setStartLatLng,
 * setDestLatLng, findRoute) and exposes ONE namespace, window.SRO, that app.js calls into:
 *
 *   SRO.onAreaLoaded(data)      boundary polygon + minimap + landmark pills + click-to-inspect
 *   SRO.onRoutesRendered(data)  start fetching the 24-hour profile of all three routes
 *   SRO.onRouteSelected()       recolour / re-focus after a route tab is clicked
 *   SRO.onRoutesCleared()
 *   SRO.diurnalMarkup()         HTML for the sparkline + time scrubber inside the results card
 *   SRO.mountDiurnal() / SRO.unmountDiurnal()   create / destroy the Chart.js instance (the card is
 *                                                rebuilt via innerHTML on every tab click)
 *
 * Backend contract (server.py):
 *   GET /api/landmarks
 *   GET /api/inspect_point?lat=&lon=&hour=
 *   GET /api/route_diurnal_profile?edge_ids=1,2,3&detail=1   (detail=1 adds per-edge geometry + 24 scores)
 *   POST /api/route must return `edge_ids` for each route.
 */
(function () {
  "use strict";

  const SRO = (window.SRO = {});
  const $ = (id) => document.getElementById(id);
  const pad2 = (n) => String(n).padStart(2, "0");
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ======================================================================================
  // Risk helpers: score (0-100, higher = riskier) -> colour / grade
  // ======================================================================================
  const RISK_CUT = { elevated: 60, high: 85 };       // overwritten from the server's `risk_scale` table
  function setCutoffs(scale) {
    (scale || []).forEach((r) => {
      const key = r.key || (/^elevated/i.test(r.label || "") ? "elevated" : /^high/i.test(r.label || "") ? "high" : "");
      const sc = Number(r.score);
      if ((key === "elevated" || key === "high") && Number.isFinite(sc)) RISK_CUT[key] = sc;
    });
  }

  SRO.RISK_CUT = RISK_CUT;                           // app.js reads this to decide which segments are "elevated"

  const GREEN = [34, 197, 94], AMBER = [245, 158, 11], RED = [239, 68, 68];
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * Math.min(1, Math.max(0, t))));
  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

  /** Green (low) -> amber (at the 'elevated' cut-off) -> red (at/after the 'high' cut-off). */
  function riskColor(score) {
    const s = Number(score) || 0;
    if (s <= 20) return rgb(GREEN);
    if (s < RISK_CUT.elevated) return rgb(mix(GREEN, AMBER, (s - 20) / (RISK_CUT.elevated - 20)));
    if (s < RISK_CUT.high) return rgb(mix(AMBER, RED, (s - RISK_CUT.elevated) / (RISK_CUT.high - RISK_CUT.elevated)));
    return rgb(RED);
  }

  // Grade is a letter for the RISK score: A+ is the calmest street. Letters change at fixed risk values;
  // 60 ("elevated") and 85 ("high") line up with the model's own band cut-offs.
  const GRADES = [[12, "A+"], [18, "A"], [25, "A-"], [32, "B+"], [40, "B"], [50, "B-"], [60, "C"],
                  [68, "D+"], [76, "D"], [85, "D-"], [Infinity, "F"]];
  function gradeFor(score) {
    for (const [limit, letter] of GRADES) if (score < limit) return letter;
    return "F";
  }
  function bandLabel(score) {
    return score >= RISK_CUT.high ? "High risk" : score >= RISK_CUT.elevated ? "Elevated risk" : "Low risk";
  }

  function pointInRing(lat, lon, ring) {            // ring: [[lat, lon], ...]; ray casting
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [yi, xi] = ring[i], [yj, xj] = ring[j];
      if ((yi > lat) !== (yj > lat) && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  function toast(msg, ms = 3200) {
    const el = $("map-toast");
    if (!el) return;
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => el.classList.remove("show"), ms);
  }

  // The hour used for point inspections: the scrubbed hour while a route is on screen, else the sidebar hour.
  function currentHour() {
    if (D.scrub !== null && Object.keys(D.data).length) return D.scrub;
    return parseInt(document.getElementById("hour-slider").value, 10);
  }

  // ======================================================================================
  // 1) Boundary polygon + MiniMap
  // ======================================================================================
  let boundaryPoly = null, miniMap = null, miniBoundary = null;
  let hullRing = null;
  let landmarks = [];

  function initMiniMap() {
    if (miniMap || !(L.Control && L.Control.MiniMap)) return;
    const tiles = L.tileLayer(BASEMAP.url, { subdomains: BASEMAP.subdomains, minZoom: 0, maxZoom: 18, attribution: "" });
    miniMap = new L.Control.MiniMap(tiles, {
      position: "bottomright",
      width: 176, height: 176,
      toggleDisplay: true, minimized: false, zoomAnimation: false,
      aimingRectOptions: { color: "#06b6d4", weight: 2, interactive: false },        // the viewfinder
      shadowRectOptions: { color: "#0f172a", weight: 1, opacity: 0.4, fillOpacity: 0.12, interactive: false },
    }).addTo(map);
  }

  function drawBoundary(ring) {
    if (boundaryPoly) map.removeLayer(boundaryPoly);
    boundaryPoly = L.polygon(ring, {
      color: "#2563eb", weight: 2, dashArray: "8 6",
      fillColor: "#60a5fa", fillOpacity: 0.1,
      interactive: false,                              // clicks fall through to the map (click-to-inspect)
    }).addTo(map);
    boundaryPoly.bringToBack();

    if (!miniMap) return;
    try {
      const inner = miniMap._miniMap;                  // the plugin's internal Leaflet map
      if (miniBoundary) inner.removeLayer(miniBoundary);
      miniBoundary = L.polygon(ring, {
        color: "#2563eb", weight: 1.5, dashArray: "4 3", fillColor: "#60a5fa", fillOpacity: 0.22, interactive: false,
      }).addTo(inner);
      const b = boundaryPoly.getBounds();
      inner.invalidateSize();
      // Pin the minimap on the whole city (instead of following the main map) so the viewfinder rectangle
      // moves INSIDE a fixed overview of the loaded boundary.
      miniMap.options.centerFixed = b.getCenter();
      miniMap.options.zoomLevelFixed = Math.max(0, inner.getBoundsZoom(b, false, L.point(10, 10)));
      inner.setView(miniMap.options.centerFixed, miniMap.options.zoomLevelFixed, { animate: false });
      map.fire("moveend");                             // make the plugin redraw the viewfinder
    } catch (err) {
      console.warn("MiniMap boundary setup failed:", err);
    }
  }

  // ======================================================================================
  // 2) Landmark pills
  // ======================================================================================
  let pillTarget = "dest";                             // "dest" | "start"
  const chosen = { start: null, dest: null };          // landmark names currently set, for pill highlighting

  function emojiFor(name) {
    const n = name.toLowerCase();
    if (/metro|subway/.test(n)) return "\u{1F687}";
    if (/railway|station|junction/.test(n)) return "\u{1F689}";
    if (/hospital|aiims|clinic/.test(n)) return "\u{1F3E5}";
    if (/college|iit|university|school/.test(n)) return "\u{1F393}";
    if (/temple|mandir|church|mosque/.test(n)) return "\u{1F6D5}";
    if (/park|garden|lalbagh|deer/.test(n)) return "\u{1F333}";
    if (/market|shopping|complex|mall|road|chowk|village/.test(n)) return "\u{1F6CD}\uFE0F";
    if (/fort|wada|tomb|palace|gymkhana/.test(n)) return "\u{1F3DB}\uFE0F";
    if (/circle|gate/.test(n)) return "\u{1F6A6}";
    return "\u{1F4CD}";
  }

  function renderPills() {
    const box = $("landmark-pills");
    if (!box) return;
    const list = landmarks.filter((l) => l.in_loaded_area);
    if (!list.length) {
      box.innerHTML = `<span class="pill-hint">No preset landmarks for this area.</span>`;
      return;
    }
    box.innerHTML = list.map((l, i) => {
      const cls = chosen.start === l.name ? " is-start" : chosen.dest === l.name ? " is-dest" : "";
      return `<button type="button" class="pill${cls}" data-i="${i}" title="${esc(l.name)}">` +
             `<span class="pill-emoji">${emojiFor(l.name)}</span><span class="pill-text">${esc(l.name)}</span></button>`;
    }).join("");
    box.querySelectorAll(".pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        const l = list[Number(btn.dataset.i)];
        const ll = [l.lat, l.lon];
        if (pillTarget === "start") { if (setStartLatLng(ll, l.name)) chosen.start = l.name; }
        else if (setDestLatLng(ll, l.name)) chosen.dest = l.name;
        renderPills();
      });
    });
  }

  function bindPillToolbar() {
    const seg = $("pill-target");
    if (!seg) return;
    seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      pillTarget = b.dataset.target;
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
    }));
  }

  async function loadLandmarks() {
    try {
      const res = await fetch("/api/landmarks");
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "landmarks failed");
      landmarks = j.landmarks || [];
      hullRing = j.hull_latlon || null;
      if (hullRing) drawBoundary(hullRing);
      renderPills();
    } catch (err) {
      console.warn("Could not load /api/landmarks:", err);
      const box = $("landmark-pills");
      if (box) box.innerHTML = `<span class="pill-hint">Landmarks unavailable.</span>`;
    }
  }

  // ======================================================================================
  // 2b) Set start / destination straight from the map
  //     - the target buttons beside each input ("pick" mode: next map click sets that point)
  //     - right-click (or long-press) menu: Start here / Destination here / Inspect
  //     - Set-as buttons inside the inspector drawer
  // ======================================================================================
  const PICK = { mode: null };

  function setPickMode(mode) {
    PICK.mode = mode;
    map.getContainer().classList.toggle("pick-mode", !!mode);
    [["start", "pick-start-btn"], ["dest", "pick-dest-btn"]].forEach(([k, id]) => {
      const b = $(id);
      if (b) b.classList.toggle("active", mode === k);
    });
    if (mode) toast(`Click the map to set the ${mode === "start" ? "start" : "destination"} \u00b7 Esc to cancel`, 6000);
  }

  /** Put the start/destination at a map position (must be inside the loaded boundary). */
  function placePoint(kind, latlng, label) {
    if (!hullRing || !pointInRing(latlng.lat, latlng.lng, hullRing)) {
      toast("That spot is outside the dashed boundary \u2014 pick a point inside it.");
      return false;
    }
    const name = label || `Map point (${latlng.lat.toFixed(5)}, ${latlng.lng.toFixed(5)})`;
    const ll = [latlng.lat, latlng.lng];
    const ok = kind === "start" ? setStartLatLng(ll, name, { fly: false }) : setDestLatLng(ll, name, { fly: false });
    if (ok) { chosen[kind] = null; renderPills(); }
    return ok;
  }

  function openPointMenu(latlng) {
    if (typeof areaLoaded === "undefined" || !areaLoaded || !hullRing) return;
    if (!pointInRing(latlng.lat, latlng.lng, hullRing)) { toast("Right-click inside the dashed boundary."); return; }
    const box = document.createElement("div");
    box.className = "pt-menu";
    box.innerHTML =
      `<div class="pt-menu-title">${latlng.lat.toFixed(5)}, ${latlng.lng.toFixed(5)}</div>` +
      `<button type="button" data-a="start"><i class="dot start"></i>Set as start</button>` +
      `<button type="button" data-a="dest"><i class="dot dest"></i>Set as destination</button>` +
      `<button type="button" data-a="inspect"><i class="dot inspect"></i>Inspect safety here</button>`;
    box.addEventListener("click", (ev) => {
      const b = ev.target.closest("button");
      if (!b) return;
      map.closePopup();
      if (b.dataset.a === "inspect") inspectAt(latlng);
      else placePoint(b.dataset.a, latlng);
    });
    L.popup({ closeButton: false, className: "pt-popup", offset: [0, -2], autoPan: false })
      .setLatLng(latlng).setContent(box).openOn(map);
  }

  function bindPickControls() {
    [["start", "pick-start-btn"], ["dest", "pick-dest-btn"]].forEach(([kind, id]) => {
      const b = $(id);
      if (b) b.addEventListener("click", () => setPickMode(PICK.mode === kind ? null : kind));
    });
    map.on("contextmenu", (e) => { setPickMode(null); openPointMenu(e.latlng); });
    // drawer shortcuts act on the point that was clicked for inspection
    [["start", "insp-set-start"], ["dest", "insp-set-dest"]].forEach(([kind, id]) => {
      const b = $(id);
      if (!b) return;
      b.addEventListener("click", () => {
        if (!INSP.last) return;
        const street = $("insp-street").textContent.trim();
        placePoint(kind, INSP.last, street && street !== "\u2014" ? `${street} (map point)` : null);
      });
    });
  }

  // ======================================================================================
  // 3) Area Safety Inspector (slide-out drawer)
  // ======================================================================================
  const INSP = { token: 0, last: null, radar: null, layer: null, timer: null };
  const CPTED_AXES = [
    ["connectivity", ["Connectivity"]],
    ["police_proximity", ["Police", "proximity"]],
    ["commercial_eyes", ["Commercial", "eyes"]],
    ["visibility", ["Visibility"]],
    ["road_hierarchy", ["Road", "hierarchy"]],
  ];

  function drawerEl() { return $("inspector"); }
  function showPane(which) {                            // which: loading | error | oob | content
    ["loading", "error", "oob", "content"].forEach((p) => { const el = $("insp-" + p); if (el) el.hidden = p !== which; });
  }
  function openDrawer() {
    drawerEl().classList.add("open");
    drawerEl().setAttribute("aria-hidden", "false");
    $("map-wrap").classList.add("drawer-open");
  }
  function closeDrawer() {
    drawerEl().classList.remove("open");
    drawerEl().setAttribute("aria-hidden", "true");
    $("map-wrap").classList.remove("drawer-open");
    INSP.token++;                                       // drop any in-flight response
    INSP.last = null;
    if (INSP.layer) { map.removeLayer(INSP.layer); INSP.layer = null; }
  }

  function prettyHighway(h) {
    return String(h || "street").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
  }

  async function inspectAt(latlng, opts) {
    const quiet = !!(opts && opts.quiet);
    const token = ++INSP.token;
    INSP.last = latlng;
    const hour = currentHour();
    if (!quiet) { openDrawer(); showPane("loading"); }
    try {
      const url = `/api/inspect_point?lat=${latlng.lat.toFixed(6)}&lon=${latlng.lng.toFixed(6)}&hour=${hour}`;
      const res = await fetch(url);
      const j = await res.json();
      if (token !== INSP.token) return;                 // a newer click superseded this one
      if (!res.ok) throw new Error(j.error || "Inspection failed.");
      renderInspection(j, latlng);
    } catch (err) {
      if (token !== INSP.token) return;
      $("insp-error").textContent = err.message;
      showPane("error");
    }
  }

  function renderInspection(j, clicked) {
    if (INSP.layer) { map.removeLayer(INSP.layer); INSP.layer = null; }

    if (j.out_of_bounds) {
      const lm = j.nearest_landmark || {};
      $("insp-street").textContent = "Outside coverage";
      $("insp-oob").innerHTML =
        `<p class="insp-note">This point is <b>${Math.round(j.distance_to_graph_m)} m</b> from the nearest loaded street ` +
        `(limit ${Math.round(j.threshold_m)} m), so it can't be scored.</p>` +
        (lm.name ? `<div class="haven"><span class="haven-ico">\u{1F4CD}</span><div class="haven-main"><b>${esc(lm.name)}</b>` +
          `<span class="dim">${esc(lm.city || "")} \u00b7 ${Math.round(lm.distance_m)} m away</span></div>` +
          `<button type="button" class="ghost sm" id="oob-center">Go</button></div>` : "");
      const go = $("oob-center");
      if (go) go.addEventListener("click", () => map.flyTo([lm.lat, lm.lon], 16, { duration: 0.6 }));
      showPane("oob");
      return;
    }

    const score = Math.round(j.risk_score);
    const color = riskColor(score);
    const edge = j.edge || {};
    $("insp-street").textContent = edge.name || prettyHighway(edge.highway);

    const badge = $("grade-badge");
    badge.style.setProperty("--grade", color);
    badge.querySelector(".grade-letter").textContent = gradeFor(score);
    badge.querySelector(".grade-score").textContent = `${score}/100`;
    $("grade-band").textContent = bandLabel(score);
    $("grade-band").style.color = color;
    $("grade-hour").textContent = `${prettyHighway(edge.highway)} \u00b7 evaluated at ${pad2(j.query.hour)}:00`;

    showPane("content");
    renderRadar(j.cpted_metrics || j.cpted || {});
    renderCptedDetails(j);
    renderHavens(j.safe_havens || [], j.safe_haven_radius_m || 800);

    // Map feedback: snapped point, the click-to-snap link, and the havens.
    const parts = [];
    const snap = [j.snapped.lat, j.snapped.lon];
    if (j.snapped.distance_m > 12) parts.push(L.polyline([[clicked.lat, clicked.lng], snap], { color: "#334155", weight: 1.5, dashArray: "3 4", opacity: 0.8, interactive: false }));
    parts.push(L.circleMarker(snap, { radius: 8, color: "#fff", weight: 3, fillColor: color, fillOpacity: 1, interactive: false }));
    (j.safe_havens || []).forEach((h) => parts.push(
      L.circleMarker([h.lat, h.lon], { radius: 6, color: "#fff", weight: 2, fillColor: havenStyle(h.type).color, fillOpacity: 1 })
        .bindTooltip(`${esc(h.name)} \u00b7 ${Math.round(h.distance_m)} m`)));
    INSP.layer = L.layerGroup(parts).addTo(map);

    requestAnimationFrame(() => INSP.radar && INSP.radar.resize());
  }

  function renderRadar(cpted) {
    const data = CPTED_AXES.map(([k]) => Number(cpted[k] ?? 0));
    const canvas = $("radar-chart");
    if (!canvas) return;
    if (typeof Chart === "undefined") { canvas.replaceWith(Object.assign(document.createElement("p"), { className: "insp-note", textContent: "Chart.js failed to load." })); return; }
    if (INSP.radar) {
      INSP.radar.data.datasets[0].data = data;
      INSP.radar.update();
      return;
    }
    INSP.radar = new Chart(canvas, {
      type: "radar",
      data: {
        labels: CPTED_AXES.map(([, label]) => label),
        datasets: [{
          label: "CPTED score", data,
          backgroundColor: "rgba(20,184,166,0.28)", borderColor: "#14b8a6", borderWidth: 2,
          pointBackgroundColor: "#14b8a6", pointBorderColor: "#fff", pointBorderWidth: 1.5, pointRadius: 3.5,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 450 },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${Math.round(c.parsed.r * 100)}% (higher = safer)` } } },
        scales: {
          r: {
            min: 0, max: 1,
            ticks: { display: false, stepSize: 0.25 },
            grid: { color: "rgba(255,255,255,0.14)" },
            angleLines: { color: "rgba(255,255,255,0.14)" },
            pointLabels: { color: "#e2e8f0", font: { size: 11, weight: "600" } },
          },
        },
      },
    });
  }

  function renderCptedDetails(j) {
    const c = j.cpted_metrics || j.cpted || {}, inp = j.cpted_inputs || {}, edge = j.edge || {};
    const detail = {
      connectivity: `${inp.degree ?? "?"}-way junction`,
      police_proximity: `${Math.round(inp.police_distance_m ?? 0)} m to nearest police`,
      commercial_eyes: `${inp.pois_within_200m ?? 0} places within 200 m` + (inp.is_night ? `, ~${Number(inp.active_open_shops ?? 0).toFixed(0)} open now` : ""),
      visibility: `darkness ${Number(inp.darkness_hazard ?? 0).toFixed(2)}`,
      road_hierarchy: prettyHighway(edge.highway),
    };
    $("cpted-list").innerHTML = CPTED_AXES.map(([k, label]) => {
      const v = Math.max(0, Math.min(1, Number(c[k] ?? 0)));
      return `<li><span class="cp-name">${esc(label.join(" "))}</span>` +
             `<span class="cp-bar"><i style="width:${Math.round(v * 100)}%;background:${riskColor((1 - v) * 80)}"></i></span>` +
             `<span class="cp-val">${Math.round(v * 100)}</span><span class="cp-detail">${esc(detail[k])}</span></li>`;
    }).join("");
  }

  const HAVEN_STYLE = {
    police: { icon: "\u{1F46E}", color: "#2563eb" }, medical: { icon: "\u{1F3E5}", color: "#e11d48" },
    bank: { icon: "\u{1F3E6}", color: "#0891b2" }, dining: { icon: "\u2615", color: "#d97706" },
    retail: { icon: "\u{1F3EA}", color: "#7c3aed" },
  };
  const havenStyle = (t) => HAVEN_STYLE[t] || { icon: "\u{1F4CD}", color: "#64748b" };

  function renderHavens(havens, radius) {
    const box = $("haven-list");
    $("haven-radius").textContent = `within ${Math.round(radius)} m`;
    if (!havens.length) {
      box.innerHTML = `<p class="insp-note">No police station, shop, cafe, bank or pharmacy within ${Math.round(radius)} m of this point.</p>`;
      return;
    }
    box.innerHTML = havens.map((h, i) =>
      `<div class="haven"><span class="haven-ico">${havenStyle(h.type).icon}</span>` +
      `<div class="haven-main"><b>${esc(h.name)}</b><span class="dim">${esc(h.category_label || h.type)}${h.open_24_7 ? " \u00b7 24/7" : ""} \u00b7 ${Math.round(h.distance_m)} m</span></div>` +
      `<button type="button" class="ghost sm" data-i="${i}" title="Use as destination">Go</button></div>`).join("");
    box.querySelectorAll("button[data-i]").forEach((b) => b.addEventListener("click", () => {
      const h = havens[Number(b.dataset.i)];
      if (setDestLatLng([h.lat, h.lon], h.name)) { chosen.dest = null; renderPills(); }
    }));
  }

  function bindInspector() {
    map.on("click", (e) => {
      if (typeof areaLoaded === "undefined" || !areaLoaded || !hullRing) return;
      if (PICK.mode) {                                   // "pick on map" mode: this click sets start/destination
        const kind = PICK.mode;
        if (placePoint(kind, e.latlng)) setPickMode(null);
        return;
      }
      if (!pointInRing(e.latlng.lat, e.latlng.lng, hullRing)) {
        toast("Click inside the dashed boundary to inspect an area.");
        return;
      }
      inspectAt(e.latlng);
    });
    $("insp-close").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      if (PICK.mode) setPickMode(null);
      else if (drawerEl().classList.contains("open")) closeDrawer();
    });
  }

  // Re-inspect (quietly) when the scrubbed hour changes while the drawer is open.
  function refreshInspectorSoon() {
    if (!INSP.last || !drawerEl().classList.contains("open")) return;
    clearTimeout(INSP.timer);
    INSP.timer = setTimeout(() => INSP.last && inspectAt(INSP.last, { quiet: true }), 280);
  }

  // ======================================================================================
  // 4) 24-hour sparkline + time scrubber
  // ======================================================================================
  const D = { token: 0, data: {}, errors: {}, scrub: null, routeHour: null, chart: null, peak: null, raf: 0 };

  SRO.diurnalMarkup = function () {
    const h = D.scrub ?? 12;
    return `
      <div class="diurnal" id="diurnal-block">
        <div class="diurnal-head">
          <span class="diurnal-title">24-hour risk profile</span>
          <span class="diurnal-clock" id="diurnal-clock"><span id="diurnal-icon"></span> ${pad2(h)}:00</span>
        </div>
        <div class="spark-wrap">
          <canvas id="diurnal-chart" aria-label="Route risk by hour of day"></canvas>
          <div class="diurnal-msg" id="diurnal-msg">Loading 24-hour profile\u2026</div>
        </div>
        <input id="scrub-slider" class="scrub-slider" type="range" min="0" max="23" step="1" value="${h}" aria-label="Preview hour of day" disabled>
        <div class="scrub-ticks"><span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>23:00</span></div>
        <div class="diurnal-live" id="diurnal-live"></div>
        <div class="diurnal-actions">
          <span class="legend-bar" title="Low \u2192 elevated \u2192 high risk"><i>low</i><i>high</i></span>
          <button type="button" class="ghost sm" id="reroute-btn" disabled>Re-optimise path for this hour</button>
        </div>
      </div>`;
  };

  SRO.mountDiurnal = function () {
    const slider = $("scrub-slider");
    if (!slider) return;
    slider.addEventListener("input", () => setScrub(parseInt(slider.value, 10)));
    const re = $("reroute-btn");
    re.addEventListener("click", () => {
      const hs = $("hour-slider");
      hs.value = D.scrub;
      $("hour-value").textContent = D.scrub;
      findRoute();
    });
    refreshDiurnal();
  };

  SRO.unmountDiurnal = function () {
    if (D.chart) { D.chart.destroy(); D.chart = null; }
  };

  function activeDetail() { return D.data[activeRouteKey] || null; }

  /** (Re)build everything that depends on the active route's per-hour data. Safe to call any time. */
  function refreshDiurnal() {
    const slider = $("scrub-slider");
    if (!slider) { applyColoring(); return; }
    const det = activeDetail();
    const msg = $("diurnal-msg");

    if (!det) {
      slider.disabled = true;
      const err = D.errors[activeRouteKey];
      msg.hidden = false;
      msg.textContent = err ? `24-hour profile unavailable: ${err}` : "Loading 24-hour profile\u2026";
      applyColoring();
      return;
    }
    msg.hidden = true;
    slider.disabled = false;
    slider.style.setProperty("--track-bg", trackGradient(det.hourly_risk));
    $("reroute-btn").disabled = false;
    buildChart(det.hourly_risk);
    updateScrubUi();
    applyColoring();
  }

  function trackGradient(hourly) {
    const stops = hourly.map((v, i) => `${riskColor(v)} ${((i + 0.5) / 24 * 100).toFixed(2)}%`);
    return `linear-gradient(90deg, ${stops.join(", ")})`;
  }

  function buildChart(hourly) {
    const canvas = $("diurnal-chart");
    if (!canvas) return;
    if (typeof Chart === "undefined") { $("diurnal-msg").hidden = false; $("diurnal-msg").textContent = "Chart.js failed to load."; return; }
    if (D.chart) { D.chart.destroy(); D.chart = null; }

    const lo = Math.max(0, Math.floor(Math.min(...hourly) - 8)), hi = Math.min(100, Math.ceil(Math.max(...hourly) + 8));
    const cursor = {                                    // dashed vertical line at the scrubbed hour
      id: "scrubCursor",
      afterDatasetsDraw(chart) {
        const pt = chart.getDatasetMeta(0).data[D.scrub];
        if (!pt) return;
        const { ctx, chartArea: { top, bottom } } = chart;
        ctx.save();
        ctx.strokeStyle = "rgba(226,232,240,0.7)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(pt.x, top); ctx.lineTo(pt.x, bottom); ctx.stroke();
        ctx.restore();
      },
    };
    D.chart = new Chart(canvas, {
      type: "line",
      data: {
        labels: hourly.map((_, i) => `${pad2(i)}:00`),
        datasets: [{
          data: hourly, tension: 0.35, borderWidth: 2.5, fill: true,
          backgroundColor: "rgba(148,163,184,0.10)",
          segment: { borderColor: (c) => riskColor((c.p0.parsed.y + c.p1.parsed.y) / 2) },
          pointRadius: (c) => (c.dataIndex === D.scrub ? 5 : 0),
          pointHoverRadius: 5, pointBorderColor: "#fff", pointBorderWidth: 2,
          pointBackgroundColor: (c) => riskColor(c.raw),
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
        interaction: { mode: "index", intersect: false },
        onClick: (e, _els, chart) => {
          const pts = chart.getElementsAtEventForMode(e, "index", { intersect: false }, false);
          if (pts.length) setScrub(pts[0].index);
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { title: (i) => `${pad2(i[0].dataIndex)}:00`, label: (c) => `Avg route risk ${Math.round(c.parsed.y)}/100` } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: "#8b8b93", autoSkip: false, maxRotation: 0, font: { size: 9 }, callback: (v, i) => ([0, 6, 12, 18, 23].includes(i) ? pad2(i) : "") } },
          y: { min: lo, max: hi, ticks: { display: false }, grid: { color: "rgba(255,255,255,0.06)" }, border: { display: false } },
        },
      },
      plugins: [cursor],
    });
  }

  function isNight(h) { return h >= 20 || h < 6; }

  function updateScrubUi() {
    const h = D.scrub;
    const slider = $("scrub-slider");
    if (!slider) return;
    if (parseInt(slider.value, 10) !== h) slider.value = h;
    $("diurnal-clock").innerHTML = `<span id="diurnal-icon">${isNight(h) ? "\u{1F319}" : "\u2600\uFE0F"}</span> ${pad2(h)}:00`;
    if (D.chart) D.chart.update("none");

    const det = activeDetail();
    const live = $("diurnal-live");
    const re = $("reroute-btn");
    if (re) {
      const same = h === D.routeHour;
      re.disabled = !det || same;
      re.textContent = same ? "Paths are optimised for this hour" : `Re-optimise path for ${pad2(h)}:00`;
    }
    if (!det || !live) return;
    const avg = det.hourly_risk[h];
    let peak = 0, totalLen = 0, elevLen = 0;
    det.edges.forEach((e) => {
      const r = e.risk[h];
      totalLen += e.length_m;
      if (r >= RISK_CUT.elevated) elevLen += e.length_m;
      if (r > peak) peak = r;
    });
    const pct = totalLen ? Math.round((100 * elevLen) / totalLen) : 0;
    live.innerHTML =
      `<span>Avg <b style="color:${riskColor(avg)}">${Math.round(avg)}</b></span>` +
      `<span>Peak <b style="color:${riskColor(peak)}">${Math.round(peak)}</b></span>` +
      `<span><b>${pct}%</b> of route elevated</span>`;
  }

  /** Scrub to hour h: slider, sparkline cursor, live numbers, map colours, peak marker, open inspector. */
  function setScrub(h) {
    if (!Number.isFinite(h)) return;
    D.scrub = Math.min(23, Math.max(0, h));
    updateScrubUi();
    if (!D.raf) D.raf = requestAnimationFrame(() => { D.raf = 0; applyColoring(); });   // coalesce slider events
    refreshInspectorSoon();
  }

  // ---- map: per-segment risk colouring of the ACTIVE route + pulsing peak-hazard marker -------------
  function buildSegments(entry, det) {
    if (entry.segments && entry.segDet === det) return;
    if (entry.segments) map.removeLayer(entry.segments);
    const halos = [], cores = [];
    entry.parts = det.edges.map((e) => {
      // butt caps on the halos so overlapping translucent ends don't show as darker blobs at joints
      const { halo, core } = makeGlowPair(e.coords, "#14b8a6", true, "butt");
      core.setStyle({ lineCap: "round" });
      core.bindTooltip(() => `Risk ${Math.round(e.risk[D.scrub])}/100 \u00b7 ${Math.round(e.length_m)} m \u00b7 ${pad2(D.scrub)}:00`, { sticky: true });
      halos.push(halo); cores.push(core);
      return { halo, core, e };
    });
    entry.segments = L.featureGroup([...halos, ...cores]);   // all halos below all cores
    entry.segDet = det;
  }

  function applyColoring() {
    if (typeof routeLayers === "undefined") return;
    Object.entries(routeLayers).forEach(([k, entry]) => {
      const det = D.data[k];
      if (k === activeRouteKey && det && D.scrub !== null) {
        buildSegments(entry, det);
        const h = D.scrub;
        entry.parts.forEach(({ halo, core, e }) => {
          const r = e.risk[h], c = riskColor(r);
          halo.setStyle({ color: c });
          core.setStyle({ color: c, weight: r >= RISK_CUT.elevated ? 6.5 : 5 });
        });
        if (map.hasLayer(entry.group)) map.removeLayer(entry.group);
        if (!map.hasLayer(entry.segments)) entry.segments.addTo(map);
      } else {
        if (entry.segments && map.hasLayer(entry.segments)) map.removeLayer(entry.segments);
        if (!map.hasLayer(entry.group)) entry.group.addTo(map);
      }
    });
    const act = routeLayers[activeRouteKey];
    if (act) (act.segments && map.hasLayer(act.segments) ? act.segments : act.group).bringToFront();
    updatePeakMarker();
  }

  function updatePeakMarker() {
    const det = activeDetail();
    if (!det || D.scrub === null) { removePeak(); return; }
    let best = null;
    det.edges.forEach((e) => { const r = e.risk[D.scrub]; if (!best || r > best.r) best = { r, e }; });
    if (!best || best.r < RISK_CUT.elevated) { removePeak(); return; }     // nothing hazardous at this hour
    const mid = best.e.coords[Math.floor(best.e.coords.length / 2)];
    const color = riskColor(best.r);
    if (!D.peak) {
      D.peak = L.marker(mid, {
        icon: L.divIcon({ className: "peak-pin", html: '<span class="peak-ring"></span><span class="peak-dot"></span>', iconSize: [26, 26], iconAnchor: [13, 13] }),
        keyboard: false, zIndexOffset: 1000,
      }).bindPopup(() => D.peakInfo || "").addTo(map);
    } else {
      D.peak.setLatLng(mid);
    }
    D.peakInfo = `<b>Peak hazard at ${pad2(D.scrub)}:00</b><br>Risk ${Math.round(best.r)}/100 \u00b7 ${Math.round(best.e.length_m)} m segment`;
    const el = D.peak.getElement();
    if (el) el.style.setProperty("--peak", color);
  }
  function removePeak() { if (D.peak) { map.removeLayer(D.peak); D.peak = null; } }

  // ---- data loading --------------------------------------------------------------------------------
  async function fetchDetail(key, route) {
    const ids = route && route.edge_ids;
    if (!ids || !ids.length) throw new Error("server did not return edge_ids (update server.py)");
    const res = await fetch(`/api/route_diurnal_profile?detail=1&edge_ids=${ids.join(",")}`);
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || "profile request failed");
    return j;
  }

  SRO.onRoutesRendered = function (data) {
    setCutoffs(data.risk_scale);
    D.token++;
    const token = D.token;
    D.data = {}; D.errors = {};
    D.scrub = Number.isFinite(data.hour) ? data.hour : parseInt($("hour-slider").value, 10);
    D.routeHour = D.scrub;
    removePeak();
    ["shortest", "balanced", "safest"].forEach((key) => {
      if (!data.routes[key]) return;
      fetchDetail(key, data.routes[key])
        .then((j) => { if (token === D.token) { D.data[key] = j; refreshDiurnal(); } })
        .catch((err) => { if (token === D.token) { D.errors[key] = err.message; refreshDiurnal(); } });
    });
  };

  SRO.onRouteSelected = function () { applyColoring(); };   // the card itself is rebuilt + re-mounted by app.js

  SRO.onRoutesCleared = function () {
    D.token++; D.data = {}; D.errors = {}; D.scrub = null;
    removePeak();
    SRO.unmountDiurnal();
  };

  // ======================================================================================
  // Area lifecycle + boot
  // ======================================================================================
  SRO.onAreaLoaded = function () {
    closeDrawer();
    setPickMode(null);
    ["pick-start-btn", "pick-dest-btn"].forEach((id) => { const b = $(id); if (b) b.disabled = false; });
    chosen.start = chosen.dest = null;
    landmarks = []; hullRing = null;
    const box = $("landmark-pills");
    if (box) box.innerHTML = `<span class="pill-hint">Loading landmarks\u2026</span>`;
    loadLandmarks();
    toast("Tip: click anywhere inside the dashed boundary to inspect its safety.", 5000);
  };

  document.addEventListener("DOMContentLoaded", () => {
    initMiniMap();
    bindPillToolbar();
    bindInspector();
    bindPickControls();
  });
  // Scripts sit at the end of <body>, so DOMContentLoaded may already have fired.
  if (document.readyState !== "loading") { initMiniMap(); bindPillToolbar(); bindInspector(); bindPickControls(); }
})();
