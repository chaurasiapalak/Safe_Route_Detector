/* =========================================================================
   hud.js -- workspace coordinator for the three floating panels:
     * Search island   (#sidebar, top-left)
     * Bottom HUD      (#bottom-hud, route pills + analytics)
     * Area Inspector  (#inspector, right drawer)

   Rules
     1. Inspector open  -> HUD slides away (never overlaps); settings drawer tucks in.
        Inspector closed / new routes rendered -> HUD is back.
     2. New routes -> HUD appears COLLAPSED (3 route pills only). The analytics body slides up
        only on the grab handle or a route card.
     3. Click on the open map -> HUD collapses and the settings drawer folds.
        Esc collapses the HUD. Locating a stretch on the map collapses it too.

   Presentation only: no routing, analytics, Leaflet layers or existing event bindings are touched.
   It reads app.js globals (map, areaLoaded, latestRoutesData) and uses Chart.js's registry.
   ========================================================================= */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const hud = $("bottom-hud"), handle = $("hud-handle"), panel = $("results-panel");
  const insp = $("inspector"), drawer = $("settings-drawer");
  if (!hud || !handle || !panel) return;

  const state = { routes: false, inspector: false };
  let settleTimer = 0, lastRoutes = null;

  /* ---- layout refresh: Leaflet + the 24-hour chart re-measure ---------------------------------- */
  function refreshLayout() {
    requestAnimationFrame(() => {
      if (typeof map !== "undefined" && map && map.invalidateSize) map.invalidateSize({ pan: false });
      const canvas = $("diurnal-chart");
      const chart = canvas && window.Chart && Chart.getChart ? Chart.getChart(canvas) : null;
      if (chart) chart.resize();
    });
  }

  /* ---- expand / collapse ----------------------------------------------------------------------- */
  const isExpanded = () => hud.classList.contains("expanded");
  function setExpanded(on) {
    if (on === isExpanded()) return;
    clearTimeout(settleTimer);
    hud.classList.toggle("expanded", on);
    hud.classList.toggle("collapsed", !on);
    hud.classList.remove("settled");                       // no scrollbar flash while sliding
    document.body.classList.toggle("hud-expanded", on);
    handle.setAttribute("aria-expanded", String(on));
    handle.setAttribute("aria-label", on ? "Collapse route details" : "Expand route details");
    if (on) settleTimer = setTimeout(() => { hud.classList.add("settled"); refreshLayout(); }, 340);
    else hud.scrollTop = 0;
    refreshLayout();
  }

  /* ---- render the HUD's visibility from the two inputs ------------------------------------------ */
  function render() {
    hud.classList.toggle("is-empty", !state.routes);
    hud.classList.toggle("is-suppressed", state.routes && state.inspector);
    document.body.classList.toggle("hud-visible", state.routes && !state.inspector);
    document.body.classList.toggle("inspector-open", state.inspector);
  }
  function closeSettings() { if (drawer) drawer.open = false; }
  const areaIsLoaded = () => typeof areaLoaded !== "undefined" && !!areaLoaded;

  /* ---- 1) grab handle + route cards ------------------------------------------------------------- */
  handle.addEventListener("click", () => setExpanded(!isExpanded()));
  panel.addEventListener("click", (e) => {                 // route card: app.js' selectRoute() runs first, then the sheet slides up
    if (e.target.closest && e.target.closest(".route-tab-btn")) setExpanded(true);
  });
  hud.addEventListener("click", (e) => {                   // "Locate on map" / segment cards: get out of the way of the map
    if (e.target.closest && e.target.closest(".locate-btn, .seg-card")) setTimeout(() => setExpanded(false), 0);
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && isExpanded()) setExpanded(false); });

  /* ---- 2) routes: appear / disappear / re-render ------------------------------------------------ */
  function onPanelChange() {
    state.routes = panel.style.display !== "none";
    if (!state.routes) setExpanded(false);
    const cur = typeof latestRoutesData !== "undefined" ? latestRoutesData : null;
    if (state.routes && cur && cur !== lastRoutes) {       // a NEW route set (not just a tab switch re-render)
      lastRoutes = cur;
      setExpanded(false);                                   // default: collapsed pills
      closeSettings();                                      // keep the map clear
      if (state.inspector && $("insp-close")) $("insp-close").click();   // route is the focus now: use the inspector's own close handler
    }
    render();
  }
  new MutationObserver(onPanelChange).observe(panel, { attributes: true, attributeFilter: ["style"], childList: true });

  /* ---- 3) inspector <-> HUD mutual exclusivity --------------------------------------------------- */
  function onInspectorChange() {
    const open = insp.classList.contains("open");
    if (open === state.inspector) return;
    state.inspector = open;
    if (open) { setExpanded(false); closeSettings(); }
    render();
  }
  if (insp) new MutationObserver(onInspectorChange).observe(insp, { attributes: true, attributeFilter: ["class"] });

  /* ---- 4) click on the open map: minimise whatever isn't in focus --------------------------------- */
  if (typeof map !== "undefined" && map && map.on) {
    map.on("click", () => {
      setExpanded(false);
      if (areaIsLoaded()) closeSettings();                  // before an area is loaded the drawer holds "Load Area": leave it
    });
  }

  /* ---- 5) publish the sheet's live height so corner controls / MiniMap can ride above it ---------- */
  if ("ResizeObserver" in window) {
    new ResizeObserver(() => document.documentElement.style.setProperty("--hud-h", hud.offsetHeight + "px")).observe(hud);
  }
  window.addEventListener("resize", refreshLayout);

  onPanelChange();
  onInspectorChange();
})();
