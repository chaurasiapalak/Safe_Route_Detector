/*
 * diagnostics.js
 * --------------
 * (1) "Model diagnostics" modal: metrics, feature importances, risk-scale legend, crime archetypes and the
 *     plots written by evaluate_and_visualize.py  (data: GET /api/model-diagnostics).
 * (2) Route explanation cards under the route list: avg risk, peak segment, % elevated length and the
 *     `reasons` returned by POST /api/route; the peak segment of each route is marked on the map.
 * Loaded after app.js and uses its globals (map, routeLayerGroup, ROUTE_COLORS).
 */
(function () {
  "use strict";

  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : Number(v).toFixed(d));
  const PRETTY = {
    distance_to_nearest_police: "Distance to police", query_hour: "Hour of day", length: "Segment length",
    active_open_shops_estimate: "Open shops (est.)", degree_of_connectivity: "Junction degree",
    road_type_ord: "Road type", commercial_poi_count: "Shops within 100 m", is_night: "Night flag",
    isolation_index: "Isolation index", darkness_hazard: "Darkness hazard",
  };
  // Band cut-offs on the 0-100 display score. They come from the server's `risk_scale` table (the calibration is
  // bound to the trained artifact's thresholds), so they follow every retrain; these defaults only apply until
  // the first response arrives or if an older server omits the table.
  const CUT = { elevated: 60, high: 85 };
  function setCutoffs(scale) {
    (scale || []).forEach((r) => {
      const key = r.key || (/^elevated/i.test(r.label || "") ? "elevated" : /^high/i.test(r.label || "") ? "high" : "");
      const sc = Number(r.score);
      if ((key === "elevated" || key === "high") && Number.isFinite(sc)) CUT[key] = sc;
    });
  }
  const bandClass = (s) => (s >= CUT.high ? "risk-high" : s >= CUT.elevated ? "risk-elev" : "risk-low");

  // ------------------------------------------------------------------ modal
  let modal = null, cache = null;

  function buildModal() {
    modal = document.createElement("div");
    modal.id = "diag-modal";
    modal.innerHTML = `
      <div class="diag-card" role="dialog" aria-modal="true" aria-label="Model diagnostics">
        <div class="diag-head"><h2>Model diagnostics</h2><button id="diag-close" class="secondary" aria-label="Close">&times;</button></div>
        <div class="diag-tabs">
          <button data-tab="metrics" class="active">Performance</button>
          <button data-tab="features">Features</button>
          <button data-tab="scale">Risk scale</button>
          <button data-tab="archetypes">Archetypes</button>
          <button data-tab="plots">Plots</button>
        </div>
        <div id="diag-body" class="diag-body">Loading&hellip;</div>
      </div>`;
    document.body.appendChild(modal);
    modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
    modal.querySelector("#diag-close").addEventListener("click", closeModal);
    modal.querySelectorAll(".diag-tabs button").forEach((b) => b.addEventListener("click", () => {
      modal.querySelectorAll(".diag-tabs button").forEach((x) => x.classList.toggle("active", x === b));
      render(b.dataset.tab);
    }));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
  }

  function closeModal() { if (modal) modal.style.display = "none"; }

  async function openModal() {
    if (!modal) buildModal();
    modal.style.display = "flex";
    const body = modal.querySelector("#diag-body");
    if (!cache) {
      try {
        const res = await fetch("/api/model-diagnostics");
        cache = await res.json();
        if (!res.ok) throw new Error(cache.error || "Diagnostics unavailable.");
        setCutoffs(cache.risk_scale);
      } catch (err) {
        body.innerHTML = `<p class="status error">${esc(err.message)}</p>`;
        cache = null;
        return;
      }
    }
    modal.querySelectorAll(".diag-tabs button").forEach((x) => x.classList.toggle("active", x.dataset.tab === "metrics"));
    render("metrics");
  }

  function metricRow(name, ho, cv, fmt) {
    const c = cv && cv[name] ? `${fmt(cv[name].mean)} &plusmn; ${fmt(cv[name].std)}` : "n/a";
    return `<tr><td>${esc(PRETTY[name] || name)}</td><td>${fmt(ho[name])}</td><td>${c}</td></tr>`;
  }

  function render(tab) {
    const body = modal.querySelector("#diag-body");
    const d = cache;
    if (!d) return;
    if (tab === "metrics") {
      const ho = d.metrics.holdout || {}, cv = d.metrics.spatial_cv || {};
      const f4 = (v) => num(v, 4), f3 = (v) => num(v, 3);
      body.innerHTML = `
        <p class="diag-sub">${esc(d.model_name || "")} &middot; trained ${esc((d.training.trained_at || "").slice(0, 10))} on
          ${esc((d.training.n_incidents || 0).toLocaleString())} incidents / ${esc((d.training.n_rows || 0).toLocaleString())} edge-hour rows</p>
        <table class="diag-table"><thead><tr><th>Metric</th><th>Hold-out (unseen blocks)</th><th>Spatial CV (mean &plusmn; sd)</th></tr></thead><tbody>
          <tr><td>MAE</td><td>${f4(ho.mae)}</td><td>${cv.mae ? f4(cv.mae.mean) + " &plusmn; " + f4(cv.mae.std) : "n/a"}</td></tr>
          <tr><td>MAE of a constant guess</td><td>${f4(ho.baseline_mae)}</td><td>${cv.baseline_mae ? f4(cv.baseline_mae.mean) : "n/a"}</td></tr>
          <tr><td>R&sup2;</td><td>${f3(ho.r2)}</td><td>${cv.r2 ? f3(cv.r2.mean) + " &plusmn; " + f3(cv.r2.std) : "n/a"}</td></tr>
          <tr><td>Spearman rank correlation</td><td>${f3(ho.spearman)}</td><td>${cv.spearman ? f3(cv.spearman.mean) + " &plusmn; " + f3(cv.spearman.std) : "n/a"}</td></tr>
          <tr><td>Top-decile risk capture</td><td>${num(100 * ho.top10pct_risk_capture, 1)}%</td><td>${cv.top10pct_risk_capture ? num(100 * cv.top10pct_risk_capture.mean, 1) + "%" : "n/a"}</td></tr>
        </tbody></table>
        <h3>How to read this</h3><ul class="diag-list">${(d.interpretation || []).map((t) => `<li>${esc(t)}</li>`).join("")}</ul>
        ${(d.notes || []).map((n) => `<p class="status error">${esc(n)}</p>`).join("")}`;
    } else if (tab === "features") {
      const max = Math.max(...(d.importances || []).map((r) => Math.abs(r.importance)), 1e-9);
      body.innerHTML = `<p class="diag-sub">Permutation importance: drop in hold-out R&sup2; when the feature is shuffled.</p>` +
        (d.importances || []).map((r) => `<div class="imp-row"><span class="imp-name">${esc(PRETTY[r.feature] || r.feature)}</span>
          <span class="imp-bar"><i style="width:${Math.max(2, 100 * Math.abs(r.importance) / max)}%"></i></span>
          <span class="imp-val">${r.importance >= 0 ? "+" : ""}${num(r.importance, 4)}</span></div>`).join("") +
        `<p class="diag-sub">Hour, night flag and open-shops estimate overlap, so each understates its shared influence.
        Police distance, isolation index and darkness hazard can only raise risk as they grow; junction degree can only lower it (monotonic constraints).</p>`;
    } else if (tab === "scale") {
      const t = d.risk_thresholds || {};
      setCutoffs(d.risk_scale);
      body.innerHTML = `<p class="diag-sub">Routes show a calibrated 0&ndash;100 index. It is relative to the training cities
        (${num(CUT.elevated, 0)} = as risky as the top 20% of streets, ${num(CUT.high, 0)} = top 8%), not a probability of crime.</p>
        <table class="diag-table"><thead><tr><th>Anchor</th><th>Raw model output</th><th>Display score</th></tr></thead><tbody>
        ${(d.risk_scale || []).map((r) => `<tr><td>${esc(r.label)}</td><td>${num(r.raw, 4)}</td><td><b class="${bandClass(r.score)}">${num(r.score, 0)}</b></td></tr>`).join("")}
        </tbody></table><p class="diag-sub">Thresholds: elevated &ge; ${num(t.elevated, 4)}, high &ge; ${num(t.high, 4)} (raw).</p>`;
    } else if (tab === "archetypes") {
      body.innerHTML = (d.archetypes || []).map((a) => `<div class="arch"><h3>${esc(a.name)} <small>${esc((a.size || 0).toLocaleString())} incidents</small></h3>
        <p class="diag-sub">mean severity ${num(a.mean_severity, 2)} &middot; ${num(100 * a.night_share, 0)}% at night &middot; ${num(a.mean_police_km, 1)} km from police</p>
        <p>${Object.entries(a.top_categories || {}).map(([k, v]) => `<span class="chip">${esc(k)} <b>${esc(v)}</b></span>`).join("")}</p></div>`).join("")
        || "<p>No archetypes in the report.</p>";
    } else if (tab === "plots") {
      const dash = d.interactive_dashboard
        ? `<p><a href="${esc(d.interactive_dashboard)}" target="_blank" rel="noopener">Open interactive dashboard in a new tab &#8599;</a></p>` +
          `<iframe src="${esc(d.interactive_dashboard)}" title="Interactive model dashboard" style="width:100%;height:70vh;border:1px solid #cbd5e1;border-radius:10px"></iframe>`
        : "";
      body.innerHTML = dash + ((d.plots || []).length
        ? d.plots.map((p) => `<figure><img src="${esc(p.url)}" alt="${esc(p.name)}" loading="lazy"><figcaption>${esc(p.name)}</figcaption></figure>`).join("")
        : "<p>No plots yet. Run <code>python evaluate_and_visualize.py</code> and reopen this tab.</p>");
    }
  }

  // ------------------------------------------------------------------ route explanations
  function renderRouteDetails(data) {
    const panel = document.getElementById("results-panel");
    if (!panel) return;
    let box = document.getElementById("route-explain");
    if (!box) { box = document.createElement("div"); box.id = "route-explain"; panel.appendChild(box); }
    const names = { shortest: "Shortest", balanced: "Balanced", safest: "Safest" };
    setCutoffs(data.risk_scale);
    box.innerHTML = ["shortest", "balanced", "safest"].map((k) => {
      const r = data.routes[k] || {};
      const avg = r.avg_risk ?? r.avg_insecurity ?? 0, peak = r.peak_segment_risk ?? 0;
      // edge-level micro-explanations: the worst segments with their top risk factor and top positive factor
      const worst = (r.high_risk_segments || []).slice().sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0)).slice(0, 3);
      const worstHtml = worst.map((sg) => {
        const m = sg.micro_explanation || {}, up = (m.top_risk_factors || [])[0], dn = (m.top_positive_factors || [])[0];
        return `<li><b class="${bandClass(sg.risk_score || 0)}">${num(sg.risk_score, 0)}</b>/100 &middot; ${num(sg.length_m, 0)} m` +
          (up ? ` &ndash; risk: ${esc(up.text)}` : "") + (dn ? `; helps: ${esc(dn.text)}` : "") + `</li>`;
      }).join("");
      return `<details class="explain" ${k === "safest" ? "open" : ""}>
        <summary><span class="swatch-dot" style="background:${(typeof ROUTE_COLORS !== "undefined" && ROUTE_COLORS[k]) || "#888"}"></span>
          ${names[k]}: avg <b class="${bandClass(avg)}">${num(avg, 0)}</b> &middot; peak <b class="${bandClass(peak)}">${num(peak, 0)}</b> / 100</summary>
        <p class="diag-sub">${num(r.pct_elevated_length, 0)}% of the route is elevated or high risk (${num(r.pct_high_length, 0)}% high).</p>
        <ul class="diag-list">${(r.reasons || []).map((t) => `<li>${esc(t)}</li>`).join("") || "<li>No explanation available.</li>"}</ul>
        ${worstHtml ? `<p class="diag-sub">Highest-risk segments on this route</p><ul class="diag-list">${worstHtml}</ul>` : ""}</details>`;
    }).join("") + `<p class="diag-sub">Scores are a relative index (see Model diagnostics &rarr; Risk scale).</p>`;

    if (typeof routeLayerGroup === "undefined" || typeof L === "undefined") return;
    ["shortest", "balanced", "safest"].forEach((k) => {
      const seg = (data.routes[k] || {}).peak_segment;
      if (!seg || !seg.coords || !seg.coords.length || seg.risk_score < CUT.elevated) return;
      const mid = seg.coords[Math.floor(seg.coords.length / 2)];
      const mx = seg.micro_explanation || {};
      const lines = [].concat(
        (mx.top_risk_factors || []).map((f) => "&#9650; " + esc(f.text)),
        (mx.top_positive_factors || []).map((f) => "&#9660; " + esc(f.text)),
        (seg.reasons || []).map(esc));
      const html = `<b>${names[k]} route &ndash; peak segment ${num(seg.risk_score, 0)}/100</b><br>${lines.join("<br>")}`;
      L.circleMarker(mid, { radius: 7, color: "#fff", weight: 2, fillColor: seg.risk_score >= CUT.high ? "#ef4444" : "#f59e0b", fillOpacity: 0.95 })
        .bindPopup(html).addTo(routeLayerGroup);
    });
  }
  window.renderRouteDetails = renderRouteDetails;

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("diag-btn");
    if (btn) btn.addEventListener("click", openModal);
  });
})();
