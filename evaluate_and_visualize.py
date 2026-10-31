#!/usr/bin/env python3
"""
evaluate_and_visualize.py
-------------------------
Standalone diagnostics for the trained road-risk regressor.

Loads  road_risk_regressor.joblib  and  training_report.json  and writes to  evaluation_plots/ :

    1_feature_vs_risk.png        partial-dependence (marginal-effect) curves + ICE lines for police distance,
                                 hour of day (night corridor shaded; must peak at NIGHT), shops within 100 m,
                                 junction degree and shop density (must be non-increasing), isolation_index and
                                 darkness_hazard (must be non-decreasing).  Each constrained panel states whether
                                 the monotonic constraint holds in the plotted curve.  A "Constraints" dashboard tab
                                 and a console table verify EVERY constrained feature (also active_open_shops_estimate).
    2_risk_distribution.png      histogram + KDE of predicted risk with the elevated (p80) / high (p92)
                                 thresholds and the calibrated 0-100 scale on the top axis
    3_permutation_importance.png drop in hold-out R^2 when each of the 10 features is shuffled
    4_summary_table.png          hold-out + spatial-CV MAE / R^2 / Spearman / top-decile capture
    5_calibration_curve.png      raw risk -> 0-100 display score (risk_calibration.py)
    interactive_dashboard.html   ALL of the above as one hover-able page (works offline, no extra packages)
    summary.json                 every number printed to the console

Where the data for the plots comes from (printed on every figure, so nobody mistakes it):
    1. --holdout-csv   labelled hold-out rows written by train_offline.py (best: real data, and it lets this
                       script RECOMPUTE metrics and permutation importance instead of quoting the report)
    2. --graph-file / --pois-file   a real OSM area, scored at random hours
    3. (default)       a SYNTHETIC population sampled inside the artifact's training feature ranges.
                       Curves are still the model's true response, but the *distribution* plot then only
                       illustrates thresholds; it is not evidence about real streets.

Usage
    python evaluate_and_visualize.py
    python evaluate_and_visualize.py --holdout-csv holdout_sample.csv.gz
    python evaluate_and_visualize.py --graph-file training_cache/training_graph.graphml \\
                                     --pois-file  training_cache/training_pois.geojson
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import os

# Set headless backend via environment variable before importing matplotlib
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import add_temporal_columns, is_night_flag  # noqa: E402  (single source of truth)
from risk_calibration import calibrate_risk_scores, uncalibrate_scores  # noqa: E402

STATIC = ["road_type_ord", "length", "commercial_poi_count", "distance_to_nearest_police", "degree_of_connectivity"]
# Engineered features that are deterministic functions of other columns: a partial-dependence sweep must set them
# AFTER the temporal columns are (re)computed, otherwise the recomputation would silently overwrite the sweep.
DERIVED_FEATURES = {"isolation_index", "darkness_hazard", "active_open_shops_estimate"}
NIGHT_CHECK_HOURS = (22, 23, 0, 1, 2, 3)
DAY_CHECK_HOURS = (10, 11, 12, 13, 14, 15)
PRETTY = {
    "isolation_index": "Isolation index (police km / junction degree)", "darkness_hazard": "Darkness hazard (night, no open shops)",
    "distance_to_nearest_police": "Distance to nearest police station",
    "query_hour": "Hour of day", "commercial_poi_count": "Shops within 100 m",
    "degree_of_connectivity": "Junction degree (connecting streets)", "road_type_ord": "Road type (ordinal)",
    "length": "Segment length", "log_length": "Segment length (log1p m)", "active_open_shops_estimate": "Open shops (estimate)", "is_night": "Night flag",
}
BLUE, ORANGE, RED, GREY, GREEN = "#2563eb", "#f59e0b", "#dc2626", "#6b7280", "#16a34a"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 11, "axes.titlesize": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.35, "axes.titleweight": "bold", "axes.linewidth": 1.2,
                     "axes.labelweight": "semibold"})


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(model_dir: Path):
    reg_path, rep_path = model_dir / "road_risk_regressor.joblib", model_dir / "training_report.json"
    for p in (reg_path, rep_path):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    art = joblib.load(reg_path)
    if not isinstance(art, dict) or art.get("kind") != "road_risk_regressor":
        raise SystemExit(f"{reg_path.name} is not a road_risk_regressor artifact")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    import sklearn
    trained = (rep.get("training") or {}).get("sklearn_version")
    if trained and trained != sklearn.__version__:
        warnings.warn(f"model trained with scikit-learn {trained}, running {sklearn.__version__}")
    return art, rep


def with_temporal(df: pd.DataFrame, hours) -> pd.DataFrame:
    """Add query_hour / is_night / active_open_shops_estimate / isolation_index / darkness_hazard.
    Delegates to feature_engineering.add_temporal_columns, the SAME function used by offline training and by
    real-time routing, so diagnostics can never drift from the model's real inputs."""
    return add_temporal_columns(df, np.asarray(hours, dtype=int))


def predict(art, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(art["model"].predict(frame[art["features"]]), dtype=float), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Reference population
# --------------------------------------------------------------------------- #
def synthetic_population(art, n, rng) -> pd.DataFrame:
    fr, base = art.get("feature_ranges", {}), art["baseline_values"]
    rng_of = lambda k, d: tuple(float(x) for x in fr.get(k, d))                      # noqa: E731
    rt_lo, rt_hi = rng_of("road_type_ord", (0, 6))
    ln_lo, ln_hi = rng_of("length", (5, 400))
    poi_lo, poi_hi = rng_of("commercial_poi_count", (0, 30))
    pd_lo, pd_hi = rng_of("distance_to_nearest_police", (0, 4000))
    dg_lo, dg_hi = rng_of("degree_of_connectivity", (1, 6))
    length = np.clip(rng.lognormal(np.log(max(base.get("length", 60.0), 5.0)), 0.8, n), ln_lo, ln_hi)
    poi = np.where(rng.random(n) < 0.4, 0, rng.integers(1, int(max(poi_hi, 2)) + 1, n))
    return pd.DataFrame({
        "road_type_ord": rng.integers(int(rt_lo), int(rt_hi) + 1, n), "length": length,
        "commercial_poi_count": poi, "distance_to_nearest_police": rng.uniform(pd_lo, pd_hi, n),
        "degree_of_connectivity": rng.integers(int(dg_lo), int(dg_hi) + 1, n),
        "query_hour": rng.integers(0, 24, n)})


def reference_population(args, art, rng):
    """-> (population with static cols + query_hour, source label, labelled hold-out frame or None)"""
    n = args.n_ref
    if args.holdout_csv:
        df = pd.read_csv(args.holdout_csv)
        derived = {"isolation_index", "darkness_hazard"}      # recomputed below, so older hold-out CSVs still work
        need = (set(art["features"]) - derived) | {"y"}
        if not need <= set(df.columns):
            raise SystemExit(f"{args.holdout_csv} lacks columns {sorted(need - set(df.columns))}")
        df = with_temporal(df, df["query_hour"].to_numpy(int))
        if "w" not in df:
            df["w"] = 1.0
        pop = df[STATIC + ["query_hour"]].sample(min(n, len(df)), random_state=args.seed).reset_index(drop=True)
        return pop, f"hold-out sample ({len(df):,} rows)", df
    if args.graph_file:
        if not args.pois_file:
            raise SystemExit("--graph-file needs --pois-file")
        import geopandas as gpd
        import osmnx as ox
        from feature_engineering import compute_static_and_poi_features
        static = compute_static_and_poi_features(ox.load_graphml(args.graph_file), gpd.read_file(args.pois_file))
        pop = static[STATIC].reset_index(drop=True)
        pop = pop.sample(min(n, len(pop)), random_state=args.seed).reset_index(drop=True)
        pop["query_hour"] = rng.integers(0, 24, len(pop))
        return pop, f"OSM area {Path(args.graph_file).name} at random hours", None
    return synthetic_population(art, n, rng), "SYNTHETIC population inside training feature ranges", None


# --------------------------------------------------------------------------- #
# Metrics (same definitions as train_offline.evaluate)
# --------------------------------------------------------------------------- #
def metrics_from_frame(art, df: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score
    y, w, pred = df["y"].to_numpy(float), df["w"].to_numpy(float), predict(art, df)
    base = float(np.average(y, weights=w))
    rho = spearmanr(pred, y)[0] if np.ptp(pred) > 0 and np.ptp(y) > 0 else float("nan")
    top = np.argsort(-pred)[: max(1, len(y) // 10)]
    total = float((y * w).sum())
    return {"mae": float(mean_absolute_error(y, pred, sample_weight=w)),
            "baseline_mae": float(mean_absolute_error(y, np.full(len(y), base), sample_weight=w)),
            "r2": float(r2_score(y, pred, sample_weight=w)), "spearman": float(rho),
            "top10pct_risk_capture": float((y[top] * w[top]).sum() / total) if total > 0 else float("nan"),
            "n_rows": int(len(y))}


def recompute_importance(art, df: pd.DataFrame, seed: int) -> dict:
    from sklearn.inspection import permutation_importance
    sub = df.sample(min(len(df), 30_000), random_state=seed)
    pi = permutation_importance(art["model"], sub[art["features"]], sub["y"].to_numpy(float), scoring="r2",
                                n_repeats=5, random_state=seed, sample_weight=sub["w"].to_numpy(float), n_jobs=1)
    return dict(zip(art["features"], map(float, pi.importances_mean)))


# --------------------------------------------------------------------------- #
# Plot 1: partial dependence
# --------------------------------------------------------------------------- #
def pdp(art, pop: pd.DataFrame, feature: str, grid: np.ndarray, n_rows: int, seed: int):
    """Returns ICE matrix [len(grid), n_rows]; PDP = column mean.

    * Raw features (police distance, hour, shops, degree): the sweep is applied first and the dependent columns
      (is_night, open shops, isolation_index, darkness_hazard) are recomputed consistently, because shuffling
      hour / is_night / open-shops independently would break their physical link.
    * Derived features (isolation_index, darkness_hazard): set DIRECTLY on the model input after the temporal
      columns are computed -- this is exactly the quantity the monotonic constraint governs (feature varies,
      every other model input fixed).  darkness_hazard is only meaningful at night, so it is swept over night rows.
    """
    base = pop
    if feature == "darkness_hazard":
        night = pop[is_night_flag(pop["query_hour"].to_numpy(int)) == 1]
        if len(night) >= 50:
            base = night
    sub = base.sample(min(n_rows, len(base)), random_state=seed).reset_index(drop=True)
    big = pd.concat([sub] * len(grid), ignore_index=True)
    values = np.repeat(np.asarray(grid, dtype=float), len(sub))
    if feature in DERIVED_FEATURES:
        big = with_temporal(big, big["query_hour"].to_numpy())
        big[feature] = values
    else:
        big[feature] = values
        big = with_temporal(big, big["query_hour"].to_numpy())
    return predict(art, big).reshape(len(grid), len(sub))


def _monotone_violation(mean: np.ndarray, sign: int) -> float:
    """Worst step against the required direction (0.0 = monotone); sign +1 non-decreasing, -1 non-increasing."""
    if not sign or len(mean) < 2:
        return 0.0
    return float(max(0.0, -(np.diff(mean) * sign).min()))


def feature_panels(art) -> list:
    """(feature, grid, x-scale, unit, tick decimals, tooltip decimals) for the six partial-dependence panels."""
    fr = art.get("feature_ranges", {})
    iso_hi = float(fr.get("isolation_index", (0.0, 2.0))[1])
    iso_hi = max(iso_hi, 0.5)
    return [("distance_to_nearest_police", np.linspace(0, 5000, 26), 1 / 1000, "km", 1, 2),
            ("query_hour", np.arange(24), 1, "h", 0, 0),
            ("commercial_poi_count", np.arange(0, 51, 2), 1, "count", 0, 0),
            ("degree_of_connectivity", np.arange(1, 7), 1, "count", 0, 0),
            ("isolation_index", np.linspace(0, iso_hi, 21), 1, "index", 2, 2),
            ("darkness_hazard", np.linspace(0, 1, 21), 1, "index", 2, 2)]


def verification_grid(feat: str, art, pop: pd.DataFrame):
    """Sweep grid for a constrained feature that has no plotted panel (e.g. active_open_shops_estimate)."""
    if feat == "active_open_shops_estimate":
        hi = float(pop["commercial_poi_count"].quantile(0.99)) if "commercial_poi_count" in pop else 30.0
        return np.linspace(0.0, max(hi, 5.0), 15)
    if feat in pop:
        lo, hi = (float(pop[feat].quantile(q)) for q in (0.01, 0.99))
        return np.linspace(lo, max(hi, lo + 1e-6), 15)
    return None


def verify_constraints(art, pop, curves: dict, n_rows: int, seed: int) -> list:
    """
    Check EVERY monotonic constraint stored in the artifact against the model's own partial-dependence curve.
    Features with a panel reuse that curve; the others (active_open_shops_estimate) get their own sweep.
    Returns [{feature, name, sign, violation, ok, swing, source}], violation in raw-risk units (0.0 = holds).
    """
    rows = []
    for feat, sign in (art.get("monotonic_constraints", {}) or {}).items():
        if feat in curves:
            mean, src = np.asarray(curves[feat]["mean"], dtype=float), "plotted panel"
        else:
            grid = verification_grid(feat, art, pop)
            if grid is None:
                continue
            mean, src = pdp(art, pop, feat, grid, n_rows, seed).mean(axis=1), "extra sweep"
        viol = _monotone_violation(mean, int(sign))
        rows.append({"feature": feat, "name": PRETTY.get(feat, feat), "sign": int(sign), "violation": viol,
                     "ok": bool(viol < 1e-9), "swing": float(mean.max() - mean.min()), "source": src})
    return rows


def plot_feature_effects(art, pop, source, out: Path, n_rows: int, seed: int):
    thr, fr = art["risk_thresholds"], art.get("feature_ranges", {})
    cons = art.get("monotonic_constraints", {}) or {}
    panels = feature_panels(art)
    fig, axes = plt.subplots(3, 2, figsize=(13, 13.5))
    stats, curves = {}, {}
    for ax, (feat, grid, scale, unit, dec, tipdec) in zip(axes.ravel(), panels):
        ice = pdp(art, pop, feat, grid, n_rows, seed)
        x = grid * scale
        lo, hi = np.percentile(ice, [10, 90], axis=1)
        mean = ice.mean(axis=1)
        sign = int(cons.get(feat, 0))
        viol = _monotone_violation(mean, sign)
        curves[feat] = {"x": [float(v) for v in x], "mean": [float(v) for v in mean],
                        "lo": [float(v) for v in lo], "hi": [float(v) for v in hi], "unit": unit,
                        "ice": [[float(v) for v in line] for line in ice[:, :15].T],
                        "title": PRETTY[feat], "dec": int(dec), "tipdec": int(tipdec),
                        "constraint": sign, "monotone_ok": bool(viol < 1e-9)}
        # a few clearly visible individual streets (ICE) instead of 60 invisible ones
        for line in ice[:, :12].T:
            ax.plot(x, line, color="#64748b", alpha=0.35, lw=1.0, zorder=2)
        ax.fill_between(x, lo, hi, color="#93c5fd", alpha=0.55, label="10-90% of streets", zorder=1)
        ax.plot(x, mean, color="#1d4ed8", lw=4.0, marker="o", ms=6, mfc="white", mew=2.0,
                label="average effect (PDP)", zorder=5)
        ytop = max(float(hi.max()) * 1.12, thr["high"] * 1.9)      # outlier streets may run off-chart; the band stays readable
        ax.set_ylim(-ytop * 0.09, ytop)
        ax.axhspan(thr["high"], ytop, color=RED, alpha=0.07, zorder=0)
        ax.axhline(thr["elevated"], color="#d97706", ls="--", lw=2.2, label="elevated threshold (p80)", zorder=4)
        ax.axhline(thr["high"], color="#b91c1c", ls="--", lw=2.2, label="high threshold (p92)", zorder=4)
        ymax = ax.get_ylim()[1]
        i_pk, i_lo = int(np.argmax(mean)), int(np.argmin(mean))
        ax.annotate(f"peak {mean[i_pk]:.3f}", (x[i_pk], mean[i_pk]), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=10, fontweight="bold", color="#b91c1c",
                    arrowprops=dict(arrowstyle="-", color="#b91c1c"), zorder=6)
        ax.annotate(f"lowest {mean[i_lo]:.3f}", (x[i_lo], mean[i_lo]), textcoords="offset points", xytext=(0, -22),
                    ha="center", fontsize=10, fontweight="bold", color="#15803d",
                    arrowprops=dict(arrowstyle="-", color="#15803d"), zorder=6)
        if feat == "query_hour":
            for a, b in ((-0.5, 6), (20, 23.5)):
                ax.axvspan(a, b, color="#1e293b", alpha=0.18)
            ax.text(2.8, ymax, "night", ha="center", va="top", fontsize=10, fontweight="bold", color="#1e293b")
            ax.text(21.8, ymax, "night", ha="center", va="top", fontsize=10, fontweight="bold", color="#1e293b")
            ax.set_xticks(range(0, 24, 3))
            night_mean = float(np.mean([mean[h] for h in NIGHT_CHECK_HOURS]))
            day_mean = float(np.mean([mean[h] for h in DAY_CHECK_HOURS]))
            ok = night_mean > day_mean
            curves[feat]["night_vs_midday"] = night_mean / max(day_mean, 1e-12)
            ax.text(0.99, 0.02, f"night / midday = {curves[feat]['night_vs_midday']:.2f}x  ({'OK' if ok else 'night looks SAFER - check data'})",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold",
                    color=GREEN if ok else RED)
        elif feat in fr:
            f_lo, f_hi = (float(v) * scale for v in fr[feat])
            if x.max() > f_hi:
                ax.axvspan(f_hi, x.max(), color="#f97316", alpha=0.12)
                ax.text(f_hi, ax.get_ylim()[0], " beyond training range", fontsize=9, color="#c2410c",
                        fontweight="bold", va="bottom")
        if sign:
            word = "non-decreasing" if sign > 0 else "non-increasing"
            ax.text(0.99, 0.02, f"constraint: {word} - {'holds' if viol < 1e-9 else 'VIOLATED'}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold",
                    color=GREEN if viol < 1e-9 else RED)
        ax.set_title(PRETTY[feat])
        ax.set_xlabel(f"{PRETTY[feat]} ({unit})")
        ax.set_ylabel("predicted raw risk")
        stats[feat] = {"pdp_min": float(mean.min()), "pdp_max": float(mean.max()),
                       "swing_vs_elevated_threshold": float((mean.max() - mean.min()) / thr["elevated"]),
                       "constraint": sign, "monotone_violation": viol}
        if "night_vs_midday" in curves[feat]:
            stats[feat]["night_vs_midday"] = curves[feat]["night_vs_midday"]
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.suptitle(f"How each feature moves predicted risk (all else held at real values)\nreference: {source}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(out / "1_feature_vs_risk.png")
    plt.close(fig)
    return stats, curves


# --------------------------------------------------------------------------- #
# Plot 2: distribution + thresholds
# --------------------------------------------------------------------------- #
def plot_distribution(art, pop, source, out: Path, seed: int):
    thr = art["risk_thresholds"]
    pred = predict(art, with_temporal(pop, pop["query_hour"].to_numpy()))
    share = {"low": float((pred < thr["elevated"]).mean()),
             "elevated": float(((pred >= thr["elevated"]) & (pred < thr["high"])).mean()),
             "high": float((pred >= thr["high"]).mean())}
    fig, ax = plt.subplots(figsize=(11, 5.6))
    xmax = max(float(np.percentile(pred, 99.5)), thr["high"] * 1.4)
    ax.axvspan(0, thr["elevated"], color=GREEN, alpha=0.08)
    ax.axvspan(thr["elevated"], thr["high"], color=ORANGE, alpha=0.12)
    ax.axvspan(thr["high"], xmax, color=RED, alpha=0.12)
    ax.hist(pred, bins=90, range=(0, xmax), density=True, color="#3b82f6", alpha=0.85, edgecolor="white", linewidth=0.4, label="predicted risk")
    if np.ptp(pred) > 0:
        from scipy.stats import gaussian_kde
        xs = np.linspace(0, xmax, 400)
        ax.plot(xs, gaussian_kde(np.random.default_rng(seed).choice(pred, min(len(pred), 20_000), replace=False))(xs),
                color="#1e3a8a", lw=3.5, label="KDE")
    for name, val, c in (("elevated (p80)", thr["elevated"], ORANGE), ("high (p92)", thr["high"], RED)):
        ax.axvline(val, color=c, ls="--", lw=2)
        ax.text(val, ax.get_ylim()[1] * 0.97, f" {name}\n {val:.4f}", color=c, va="top", fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("predicted raw risk")
    ax.set_ylabel("density")
    top = ax.secondary_xaxis("top", functions=(lambda v: calibrate_risk_scores(v, thr),
                                               lambda s: uncalibrate_scores(s, thr)))
    top.set_xlabel("calibrated display score (0-100)")
    ax.legend(loc="center right", frameon=False)
    ax.set_title(f"Predicted risk distribution  |  low {100 * share['low']:.0f}%  elevated {100 * share['elevated']:.0f}%  "
                 f"high {100 * share['high']:.0f}%\nreference: {source}", pad=34)
    if source.startswith("SYNTHETIC"):
        ax.text(0.01, 0.02, "synthetic population: shares will not match 80/12/8 and say nothing about real streets",
                transform=ax.transAxes, fontsize=8.5, color=RED)
    fig.tight_layout()
    fig.savefig(out / "2_risk_distribution.png")
    plt.close(fig)
    return share, pred


# --------------------------------------------------------------------------- #
# Plot 3: permutation importance
# --------------------------------------------------------------------------- #
def plot_importance(importances: dict, origin: str, out: Path):
    items = sorted(importances.items(), key=lambda kv: kv[1])
    names, vals = [PRETTY.get(k, k) for k, _ in items], [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.barh(names, vals, color=[BLUE if v > 0 else GREY for v in vals])
    span = max(max(vals), 1e-9)
    for b, v in zip(bars, vals):
        ax.text(max(v, 0) + span * 0.015, b.get_y() + b.get_height() / 2, f"{v:+.4f}", va="center", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(min(0, min(vals) * 1.3), span * 1.22)
    ax.set_xlabel("drop in hold-out R\u00b2 when the feature is shuffled")
    ax.set_title(f"Permutation feature importance ({origin})")
    ax.text(0.99, 0.02, "hour, is_night and open-shops carry overlapping information,\n"
                        "so shuffling one of them understates its true influence",
            transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
    fig.tight_layout()
    fig.savefig(out / "3_permutation_importance.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Plot 4: summary table (+ console)
# --------------------------------------------------------------------------- #
def summary_rows(ho: dict, cv: dict | None):
    def cvs(k, f="{:.4f}"):
        if not cv or k not in cv:
            return "n/a"
        return (f + " \u00b1 " + f).format(cv[k]["mean"], cv[k]["std"])
    cap, cap_cv = ho["top10pct_risk_capture"], (cv or {}).get("top10pct_risk_capture", {}).get("mean")
    skill = 100.0 * (1.0 - ho["mae"] / ho["baseline_mae"])
    return [("MAE", f"{ho['mae']:.4f}", cvs("mae")),
            ("MAE of a constant guess", f"{ho['baseline_mae']:.4f}", cvs("baseline_mae")),
            ("MAE improvement vs constant", f"{skill:.1f}%", "n/a"),
            ("R\u00b2", f"{ho['r2']:+.3f}", cvs("r2", "{:+.3f}")),
            ("Spearman rank correlation", f"{ho['spearman']:+.3f}", cvs("spearman", "{:+.3f}")),
            ("Top-decile risk capture", f"{100 * cap:.1f}%  ({cap / 0.10:.1f}x random)",
             f"{100 * cap_cv:.1f}%  ({cap_cv / 0.10:.1f}x random)" if cap_cv is not None else "n/a"),
            ("Rows evaluated", f"{ho.get('n_rows', 0):,}", "")]


def print_and_plot_summary(ho: dict, cv: dict | None, origin: str, out: Path):
    rows = summary_rows(ho, cv)
    w0, w1 = max(len(r[0]) for r in rows) + 2, max(len(r[1]) for r in rows) + 2
    line = "=" * (w0 + w1 + 34)
    print("\n" + line + f"\n MODEL EVALUATION SUMMARY   (metrics: {origin})\n" + line)
    pm="\u00b1"
    print(f" {'Metric':<{w0}}{'Hold-out':<{w1}}{f'Spatial CV (mean {pm} sd)'}")
    print(" " + "-" * (len(line) - 2))
    for r in rows:
        print(f" {r[0]:<{w0}}{r[1]:<{w1}}{r[2]}")
    print(line)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    tab = ax.table(cellText=[list(r) for r in rows], colLabels=["Metric", "Hold-out (20% unseen map blocks)",
                                                                "Spatial CV (mean \u00b1 sd)"],
                   loc="center", cellLoc="left", colWidths=[0.30, 0.38, 0.32])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1, 1.55)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if r == 0:
            cell.set_facecolor("#1e3a8a")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f4f6")
    ax.set_title(f"Model evaluation summary ({origin})", fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(out / "4_summary_table.png")
    plt.close(fig)


def plot_calibration(art, out: Path):
    thr = art["risk_thresholds"]
    xs = np.linspace(0, thr["high"] * 2.5, 500)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(xs, calibrate_risk_scores(xs, thr), color="#1d4ed8", lw=4)
    anchors = [("elevated (p80)", thr["elevated"], ORANGE), ("high (p92)", thr["high"], RED)]
    if thr.get("median"):
        anchors.insert(0, ("median street", thr["median"], GREEN))
    for name, v, c in anchors:
        s = float(calibrate_risk_scores(np.array([v]), thr)[0])
        ax.plot([v], [s], "o", color=c, ms=8)
        ax.annotate(f"{name}\nraw {v:.4f} \u2192 {s:.0f}/100", (v, s), textcoords="offset points", xytext=(12, -28),
                    color=c, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlabel("raw predicted risk")
    ax.set_ylabel("display score (0-100)")
    ax.set_title("Risk calibration: raw model output \u2192 0-100 score")
    fig.tight_layout()
    fig.savefig(out / "5_calibration_curve.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Interactive dashboard (single self-contained HTML file, no internet / no extra packages needed)
# --------------------------------------------------------------------------- #
DASH_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Safe Route - model diagnostics</title>
<style>
:root{--bg:#f8fafc;--card:#fff;--ink:#0f172a;--mut:#475569;--line:#e2e8f0;--blue:#1d4ed8;--band:#bfdbfe;
--low:#16a34a;--elev:#d97706;--high:#dc2626;--night:#1e293b}
@media(prefers-color-scheme:dark){:root{--bg:#0b1220;--card:#111a2e;--ink:#f1f5f9;--mut:#a8b3c7;--line:#26324a;--blue:#60a5fa;--band:#1e3a8a;
--low:#4ade80;--elev:#fbbf24;--high:#f87171;--night:#94a3b8}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:22px 24px 8px;max-width:1200px;margin:auto}h1{margin:0 0 4px;font-size:24px}
.sub{color:var(--mut);font-size:13px}
.wrap{max-width:1200px;margin:auto;padding:8px 24px 40px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
.tabs button{border:2px solid var(--line);background:var(--card);color:var(--ink);padding:8px 14px;border-radius:999px;font-weight:600;cursor:pointer;font-size:14px}
.tabs button.on{background:var(--blue);border-color:var(--blue);color:#fff}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.card h3{margin:0 0 2px;font-size:17px}.insight{color:var(--mut);font-size:13.5px;margin:0 0 8px;min-height:40px}
.insight b{color:var(--ink)}
svg{width:100%;height:auto;display:block;overflow:visible}svg text{fill:var(--ink);font-size:12px}
.axis line,.axis path{stroke:var(--mut)}.grid line{stroke:var(--line)}.mut{fill:var(--mut)!important}
.tip{position:fixed;pointer-events:none;background:#0f172a;color:#fff;padding:9px 12px;border-radius:10px;font-size:13px;
box-shadow:0 8px 24px rgba(0,0,0,.35);opacity:0;transition:opacity .08s;z-index:9;max-width:260px}
.tip b{font-size:14px}.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-weight:700;font-size:12px;margin-left:6px;color:#fff}
.ctrl{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--mut);margin:4px 0 10px}
.ctrl label{cursor:pointer;display:flex;gap:6px;align-items:center}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin-top:6px}
.legend i{display:inline-block;width:22px;height:0;border-top:4px solid;vertical-align:middle;margin-right:6px;border-radius:2px}
table{border-collapse:collapse;width:100%}td,th{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}
th{background:var(--blue);color:#fff}tr:nth-child(even) td{background:rgba(148,163,184,.12)}
.hint{font-size:12.5px;color:var(--mut)}
</style></head><body>
<header><h1>Safe Route &mdash; what the model learned</h1>
<div class="sub" id="src"></div></header>
<div class="wrap">
<div class="tabs" id="tabs"></div>
<div id="view"></div></div>
<div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const $ = (s, r=document) => r.querySelector(s);
const tip = $("#tip");
const T = D.thresholds;
const band = v => v >= T.high ? ["HIGH","var(--high)"] : v >= T.elevated ? ["ELEVATED","var(--elev)"] : ["LOW","var(--low)"];
const fmt = (v,d=4) => Number(v).toFixed(d);
$("#src").textContent = "Reference data: " + D.source + "  \u2022  hover any chart for exact values";

function el(tag, attrs={}, parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e}
function showTip(html, ev){tip.innerHTML=html;tip.style.opacity=1;const w=tip.offsetWidth;let x=ev.clientX+16;if(x+w>innerWidth-8)x=ev.clientX-w-16;tip.style.left=x+"px";tip.style.top=(ev.clientY+14)+"px"}
function hideTip(){tip.style.opacity=0}
const scale=(d0,d1,r0,r1)=>v=>r0+(v-d0)*(r1-r0)/((d1-d0)||1);
function ticks(a,b,n=6){const step=(b-a)/n,out=[];for(let i=0;i<=n;i++)out.push(a+i*step);return out}

/* ---------- partial-dependence panel ---------- */
function pdpPanel(host, c, key){
  const W=560,H=340,m={l:58,r:16,t:14,b:46};
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`},host);
  const ymax=Math.max(T.high*1.9, Math.max(...c.hi)*1.12), xs=c.x, uid="c"+key;
  const X=scale(xs[0],xs[xs.length-1],m.l,W-m.r), Y=scale(0,ymax,H-m.b,m.t);
  const cp=el("clipPath",{id:uid},el("defs",{},svg));el("rect",{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.b-m.t},cp);
  const g=el("g",{},svg);
  // risk zones
  el("rect",{x:m.l,y:Y(T.elevated),width:W-m.l-m.r,height:Y(0)-Y(T.elevated),fill:"var(--low)",opacity:.08},g);
  el("rect",{x:m.l,y:Y(T.high),width:W-m.l-m.r,height:Y(T.elevated)-Y(T.high),fill:"var(--elev)",opacity:.13},g);
  el("rect",{x:m.l,y:m.t,width:W-m.l-m.r,height:Y(T.high)-m.t,fill:"var(--high)",opacity:.11},g);
  if(key==="query_hour"){[[-0.5,6],[20,23.5]].forEach(([a,b])=>{
    el("rect",{x:X(Math.max(a,xs[0])),y:m.t,width:X(Math.min(b,xs[xs.length-1]))-X(Math.max(a,xs[0])),height:H-m.b-m.t,fill:"var(--night)",opacity:.18},g);
    const t=el("text",{x:X((a+b)/2),y:m.t+14,"text-anchor":"middle","font-weight":700},g);t.textContent="night"});}
  ticks(0,ymax,5).forEach(v=>{el("line",{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:"var(--line)"},g);
    const t=el("text",{x:m.l-8,y:Y(v)+4,"text-anchor":"end",class:"mut"},g);t.textContent=fmt(v,3)});
  const dec = c.dec ?? 0, tipdec = c.tipdec ?? dec;
  const xt = key==="query_hour"?xs.filter(v=>v%3===0): key==="degree_of_connectivity"?xs: ticks(xs[0],xs[xs.length-1],5);
  xt.forEach(v=>{const t=el("text",{x:X(v),y:H-m.b+18,"text-anchor":"middle",class:"mut"},g);t.textContent=fmt(v,dec)});
  const xl=el("text",{x:(m.l+W-m.r)/2,y:H-6,"text-anchor":"middle","font-weight":700},g);xl.textContent=c.title+" ("+c.unit+")";
  const yl=el("text",{transform:`translate(14 ${(m.t+H-m.b)/2}) rotate(-90)`,"text-anchor":"middle","font-weight":700},g);yl.textContent="predicted raw risk";
  // thresholds
  const labs=[];
  [[T.elevated,"var(--elev)","elevated (p80)",15],[T.high,"var(--high)","high (p92)",-6]].forEach(([v,col,lab,dy])=>{
    el("line",{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:col,"stroke-width":2.5,"stroke-dasharray":"7 5"},g);
    const t=el("text",{x:m.l+6,y:Y(v)+dy,"font-weight":800,style:"fill:"+col+";stroke:var(--card);stroke-width:4px;paint-order:stroke"},g);t.textContent=lab;labs.push(t)});
  // band, ICE, mean
  const bandG=el("g",{"clip-path":`url(#${uid})`},g), iceG=el("g",{"clip-path":`url(#${uid})`,style:"display:none"},g);
  const up=xs.map((x,i)=>`${X(x)},${Y(c.hi[i])}`), dn=xs.map((x,i)=>`${X(x)},${Y(c.lo[i])}`).reverse();
  el("polygon",{points:up.concat(dn).join(" "),fill:"var(--band)",opacity:.75},bandG);
  (c.ice||[]).forEach(l=>el("polyline",{points:l.map((v,i)=>`${X(xs[i])},${Y(v)}`).join(" "),fill:"none",stroke:"var(--mut)","stroke-opacity":.4,"stroke-width":1.2},iceG));
  const line=el("polyline",{points:c.mean.map((v,i)=>`${X(xs[i])},${Y(v)}`).join(" "),fill:"none",stroke:"var(--blue)","stroke-width":4.5,"stroke-linejoin":"round","stroke-linecap":"round"},g);
  c.mean.forEach((v,i)=>el("circle",{cx:X(xs[i]),cy:Y(v),r:4.5,fill:"var(--card)",stroke:"var(--blue)","stroke-width":2.5},g));
  labs.forEach(t=>g.appendChild(t));   // keep threshold labels above band and lines
  // peak / lowest markers
  const ip=c.mean.indexOf(Math.max(...c.mean)), il=c.mean.indexOf(Math.min(...c.mean));
  [[ip,"peak","var(--high)",-14],[il,"lowest","var(--low)",22]].forEach(([i,lab,col,dy])=>{
    el("circle",{cx:X(xs[i]),cy:Y(c.mean[i]),r:8,fill:"none",stroke:col,"stroke-width":3},g);
    const t=el("text",{x:X(xs[i]),y:Y(c.mean[i])+dy,"text-anchor":"middle","font-weight":800,style:"fill:"+col},g);t.textContent=lab+" "+fmt(c.mean[i],3)});
  // hover
  const cross=el("line",{y1:m.t,y2:H-m.b,stroke:"var(--ink)","stroke-width":1.5,opacity:0},g);
  const dot=el("circle",{r:8,fill:"var(--blue)",stroke:"#fff","stroke-width":3,opacity:0},g);
  const hit=el("rect",{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.b-m.t,fill:"transparent",style:"cursor:crosshair"},svg);
  hit.addEventListener("mousemove",ev=>{
    const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width;
    let bi=0,bd=1e9;xs.forEach((x,i)=>{const d=Math.abs(X(x)-px);if(d<bd){bd=d;bi=i}});
    cross.setAttribute("x1",X(xs[bi]));cross.setAttribute("x2",X(xs[bi]));cross.setAttribute("opacity",.6);
    dot.setAttribute("cx",X(xs[bi]));dot.setAttribute("cy",Y(c.mean[bi]));dot.setAttribute("opacity",1);
    const b=band(c.mean[bi]),xv=fmt(xs[bi],tipdec),un=(c.unit==="count"||c.unit==="index")?"":" "+c.unit;
    showTip(`<b>${c.title}: ${xv}${un}</b><br>average risk <b>${fmt(c.mean[bi])}</b><span class="pill" style="background:${b[1]}">${b[0]}</span><br>`+
      `<span style="opacity:.8">typical streets: ${fmt(c.lo[bi])} \u2013 ${fmt(c.hi[bi])}</span>`,ev)});
  hit.addEventListener("mouseleave",()=>{hideTip();cross.setAttribute("opacity",0);dot.setAttribute("opacity",0)});
  return {bandG,iceG};
}

/* ---------- views ---------- */
const views={
 effects(root){
  root.innerHTML=`<div class="ctrl"><label><input type="checkbox" id="cb" checked> show 10\u201390% band of streets</label>
   <label><input type="checkbox" id="ci"> show example individual streets</label></div>
   <div class="grid" id="g"></div>
   <div class="legend"><span><i style="border-color:var(--blue)"></i>average effect</span><span><i style="border-color:var(--elev);border-top-style:dashed"></i>elevated threshold</span>
   <span><i style="border-color:var(--high);border-top-style:dashed"></i>high threshold</span><span><i style="border-color:var(--band)"></i>10\u201390% of streets</span></div>
   <p class="hint">Reading guide: the thick blue line is what the model predicts on average as you change ONE feature while everything else stays realistic. Above the red dashed line a street is rated HIGH.
   Police distance, isolation index and darkness hazard are constrained to never LOWER risk as they rise; junction degree to never RAISE it.</p>`;
  const panels=[];
  Object.keys(D.curves).forEach(k=>{const c=D.curves[k],card=document.createElement("div");card.className="card";
    card.innerHTML=`<h3>${c.title}</h3><p class="insight">${D.insights[k]||""}</p>`;$("#g",root).appendChild(card);panels.push(pdpPanel(card,c,k))});
  $("#cb",root).onchange=e=>panels.forEach(p=>p.bandG.style.display=e.target.checked?"":"none");
  $("#ci",root).onchange=e=>panels.forEach(p=>p.iceG.style.display=e.target.checked?"":"none");
 },
 dist(root){
  const h=D.hist,W=900,H=400,m={l:60,r:20,t:34,b:52};
  root.innerHTML=`<div class="card"><h3>How risky are the streets overall?</h3>
   <p class="insight">${D.shares_text}</p></div>`;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`},$(".card",root));
  const X=scale(0,h.xmax,m.l,W-m.r),ym=Math.max(...h.dens)*1.12,Y=scale(0,ym,H-m.b,m.t);
  [[0,T.elevated,"var(--low)",.1],[T.elevated,T.high,"var(--elev)",.16],[T.high,h.xmax,"var(--high)",.13]].forEach(([a,b,c,o])=>
    el("rect",{x:X(a),y:m.t,width:X(b)-X(a),height:H-m.b-m.t,fill:c,opacity:o},svg));
  ticks(0,ym,5).forEach(v=>{el("line",{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:"var(--line)"},svg);const t=el("text",{x:m.l-8,y:Y(v)+4,"text-anchor":"end",class:"mut"},svg);t.textContent=Math.round(v)});
  ticks(0,h.xmax,8).forEach(v=>{const t=el("text",{x:X(v),y:H-m.b+18,"text-anchor":"middle",class:"mut"},svg);t.textContent=fmt(v,3);
    const s=el("text",{x:X(v),y:m.t-10,"text-anchor":"middle",class:"mut"},svg);s.textContent=Math.round(D.calib_of(v))});
  const bw=(X(h.xmax)-X(0))/h.dens.length;
  h.dens.forEach((d,i)=>{const x0=h.xmax*i/h.dens.length,r=el("rect",{x:X(x0)+.5,y:Y(d),width:bw-1,height:Y(0)-Y(d),fill:"var(--blue)",opacity:.85,style:"cursor:pointer"},svg);
    r.addEventListener("mousemove",ev=>{const x1=x0+h.xmax/h.dens.length,b=band(x0);r.setAttribute("opacity",1);
      showTip(`<b>risk ${fmt(x0)} \u2013 ${fmt(x1)}</b><span class="pill" style="background:${b[1]}">${b[0]}</span><br>${(h.share[i]*100).toFixed(1)}% of streets<br><span style="opacity:.8">display score \u2248 ${Math.round(D.calib_of(x0))}/100</span>`,ev)});
    r.addEventListener("mouseleave",()=>{hideTip();r.setAttribute("opacity",.85)})});
  el("polyline",{points:h.kde.map((d,i)=>`${X(h.xmax*i/(h.kde.length-1))},${Y(d)}`).join(" "),fill:"none",stroke:"#1e3a8a","stroke-width":4,style:"stroke:var(--ink)"},svg);
  [[T.elevated,"var(--elev)","elevated",16],[T.high,"var(--high)","high",36]].forEach(([v,c,l,dy])=>{
    el("line",{x1:X(v),x2:X(v),y1:m.t,y2:H-m.b,stroke:c,"stroke-width":3,"stroke-dasharray":"8 5"},svg);
    const t=el("text",{x:X(v)+6,y:m.t+dy,"font-weight":800,style:"fill:"+c+";stroke:var(--card);stroke-width:4px;paint-order:stroke"},svg);t.textContent=l+" "+fmt(v)});
  const xl=el("text",{x:W/2,y:H-8,"text-anchor":"middle","font-weight":700},svg);xl.textContent="predicted raw risk  (top axis: 0\u2013100 display score)";
 },
 imp(root){
  const items=D.importance.slice().sort((a,b)=>b.v-a.v),mx=Math.max(...items.map(i=>i.v),1e-9);
  root.innerHTML=`<div class="card"><h3>Which features matter most?</h3><p class="insight">${D.imp_origin}. Longer bar = shuffling that feature hurts the model more.
   Hour, night flag and open-shops overlap, so each one alone looks smaller than their combined effect.</p><div id="b"></div></div>`;
  const W=900,rowH=44,H=items.length*rowH+10,svg=el("svg",{viewBox:`0 0 ${W} ${H}`},$("#b",root)),X=scale(0,mx,300,W-90);
  items.forEach((it,i)=>{const y=i*rowH+6,col=it.v>0?"var(--blue)":"var(--mut)";
    const t=el("text",{x:290,y:y+rowH/2,"text-anchor":"end","font-weight":600,"font-size":14},svg);t.textContent=it.name;
    const r=el("rect",{x:300,y:y+5,width:Math.max(3,X(Math.max(it.v,0))-300),height:rowH-14,rx:6,fill:col,style:"cursor:pointer"},svg);
    const v=el("text",{x:X(Math.max(it.v,0))+8,y:y+rowH/2,"font-weight":700},svg);v.textContent=(it.v>=0?"+":"")+it.v.toFixed(4);
    r.addEventListener("mousemove",ev=>showTip(`<b>${it.name}</b><br>R\u00b2 drop when shuffled: ${it.v.toFixed(4)}<br>rank #${i+1} of ${items.length}`,ev));r.addEventListener("mouseleave",hideTip)});
 },
 cal(root){
  const c=D.calib,W=800,H=420,m={l:60,r:24,t:20,b:52};
  root.innerHTML=`<div class="card"><h3>From raw risk to the 0\u2013100 score shown on the map</h3><p class="insight">Hover the curve. Elevated and high thresholds are marked.</p></div>`;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`},$(".card",root)),X=scale(0,c.x[c.x.length-1],m.l,W-m.r),Y=scale(0,100,H-m.b,m.t);
  ticks(0,100,5).forEach(v=>{el("line",{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:"var(--line)"},svg);const t=el("text",{x:m.l-8,y:Y(v)+4,"text-anchor":"end",class:"mut"},svg);t.textContent=Math.round(v)});
  ticks(0,c.x[c.x.length-1],6).forEach(v=>{const t=el("text",{x:X(v),y:H-m.b+18,"text-anchor":"middle",class:"mut"},svg);t.textContent=fmt(v,3)});
  el("polyline",{points:c.x.map((x,i)=>`${X(x)},${Y(c.y[i])}`).join(" "),fill:"none",stroke:"var(--blue)","stroke-width":5,"stroke-linecap":"round"},svg);
  [[T.elevated,"var(--elev)","elevated"],[T.high,"var(--high)","high"]].forEach(([v,col,l])=>{const s=D.calib_of(v);
    el("circle",{cx:X(v),cy:Y(s),r:9,fill:col,stroke:"#fff","stroke-width":3},svg);
    const t=el("text",{x:X(v)+14,y:Y(s)+22,"font-weight":800,style:"fill:"+col},svg);t.textContent=`${l}: ${fmt(v)} \u2192 ${Math.round(s)}/100`});
  const cross=el("circle",{r:7,fill:"var(--ink)",opacity:0},svg),hit=el("rect",{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.b-m.t,fill:"transparent",style:"cursor:crosshair"},svg);
  hit.addEventListener("mousemove",ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width;
    let bi=0,bd=1e9;c.x.forEach((x,i)=>{const d=Math.abs(X(x)-px);if(d<bd){bd=d;bi=i}});
    cross.setAttribute("cx",X(c.x[bi]));cross.setAttribute("cy",Y(c.y[bi]));cross.setAttribute("opacity",1);
    showTip(`raw risk <b>${fmt(c.x[bi])}</b><br>display score <b>${c.y[bi].toFixed(0)}/100</b>`,ev)});
  hit.addEventListener("mouseleave",()=>{hideTip();cross.setAttribute("opacity",0)});
 },
 con(root){
  const rows=D.constraints||[], dir=r=>r.sign>0?"risk never falls as it rises":"risk never rises as it rises";
  root.innerHTML=`<div class="card"><h3>Monotonic constraint check</h3><p class="insight">The model is trained so that safety physics cannot be violated.
   Each row is verified on the model's own partial-dependence curve (worst step against the required direction; 0 = holds).</p>
   <table><tr><th>Feature</th><th>Required</th><th>Worst violation</th><th>Effect swing</th><th>Status</th></tr>
   ${rows.map(r=>`<tr><td>${r.name}</td><td>${dir(r)}</td><td>${Number(r.violation).toExponential(1)}</td><td>${fmt(r.swing,4)}</td>
     <td><b style="color:${r.ok?"var(--low)":"var(--high)"}">${r.ok?"&#10004; holds":"&#10008; VIOLATED"}</b></td></tr>`).join("")
     ||'<tr><td colspan="5">No constraints recorded in this model artifact (trained before constraints existed) - retrain.</td></tr>'}</table>
   <p class="hint">A passing constraint guarantees direction, not size: a small swing means the data (or the floor) carries little signal for that feature.</p></div>`;
 },
 table(root){
  root.innerHTML=`<div class="card"><h3>Model scorecard</h3><table><tr><th>Metric</th><th>Hold-out</th><th>Spatial CV (mean \u00b1 sd)</th></tr>
   ${D.table.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("")}</table><p class="hint">Metrics: ${D.metric_origin}</p></div>`;
 }};
const names={effects:"Feature effects",con:"Constraints",dist:"Risk distribution",imp:"Feature importance",cal:"Score scale",table:"Scorecard"};
Object.keys(names).forEach((k,i)=>{const b=document.createElement("button");b.textContent=names[k];b.onclick=()=>{
  document.querySelectorAll(".tabs button").forEach(x=>x.classList.toggle("on",x===b));hideTip();const v=$("#view");v.innerHTML="";views[k](v)};
  $("#tabs").appendChild(b);if(i===0)b.click()});
</script></body></html>"""


def _insights(curves: dict, thr: dict) -> dict:
    """Plain-English one-liners computed from the PDP curves (so they stay true after every retrain)."""
    out = {}
    for k, c in curves.items():
        x, m = np.asarray(c["x"]), np.asarray(c["mean"])
        ip, il = int(np.argmax(m)), int(np.argmin(m))
        swing = (m.max() - m.min()) / thr["elevated"]
        strength = "a strong" if swing > 1.0 else "a moderate" if swing > 0.4 else "a weak"
        dec = int(c.get("tipdec", 0))
        xs = lambda v: f"{v:.{dec}f}" if dec else f"{v:g}"                          # noqa: E731
        u = "" if c["unit"] in ("count", "index") else " " + c["unit"]
        out[k] = (f"{strength.capitalize()} effect. Risk is <b>highest</b> at <b>{xs(x[ip])}{u}</b> ({m[ip]:.3f}) "
                  f"and <b>lowest</b> at <b>{xs(x[il])}{u}</b> ({m[il]:.3f}).")
        if k == "query_hour" and len(m) == 24:
            night = float(np.mean([m[h] for h in NIGHT_CHECK_HOURS]))
            mid = float(np.mean([m[h] for h in DAY_CHECK_HOURS]))
            ok = night > mid
            out[k] += (f" {'&#10004;' if ok else '&#10008;'} Night (22-03h) averages <b>{night / max(mid, 1e-12):.2f}x</b> the "
                       f"midday (10-15h) risk. Expectation: night hours must NOT look safer than daytime.")
        elif k == "commercial_poi_count":
            out[k] += (f" Expectation: more shops = more eyes on the street, so risk must never climb "
                       f"(0 shops: {m[0]:.3f}, peak {m[ip]:.3f} at {xs(x[ip])}).")
        elif k == "degree_of_connectivity":
            out[k] += " Expectation: dead ends (degree 1-2) must not look safer than busy junctions."
        elif k == "isolation_index":
            out[k] += " Expectation: more isolation (far from police, few connections) never means safer."
        elif k == "darkness_hazard":
            out[k] += " Expectation: a darker, emptier night street never means safer. Shown for night streets only."
        sign = int(c.get("constraint", 0))
        if sign:
            word = "never falls as this rises" if sign > 0 else "never rises as this rises"
            out[k] += (f" {'&#10004; Monotonic constraint holds' if c.get('monotone_ok', True) else '&#10008; Monotonic constraint VIOLATED'}"
                       f" (risk {word}).")
    return out


def write_interactive_dashboard(art, curves, pred, importances, imp_origin, ho, cv, origin, source, share, out: Path,
                                constraints=None):
    thr = art["risk_thresholds"]
    xmax = max(float(np.percentile(pred, 98)), thr["high"] * 2.0)     # trim the extreme tail so the bars stay readable
    pred = pred[pred <= xmax]
    nb = 60
    counts, edges = np.histogram(pred, bins=nb, range=(0, xmax))
    dens = counts / (len(pred) * (xmax / nb))
    kde = np.zeros(120)
    if np.ptp(pred) > 0:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(np.random.default_rng(0).choice(pred, min(len(pred), 20_000), replace=False))(
            np.linspace(0, xmax, 120))
    cx = np.linspace(0, thr["high"] * 2.5, 120)
    cy = calibrate_risk_scores(cx, thr)
    grid = np.linspace(0, xmax, 400)
    gs = np.asarray(calibrate_risk_scores(grid, thr), dtype=float)
    # JS looks calibration up by interpolating a coarse table
    data = {
        "source": source, "thresholds": thr, "curves": curves, "insights": _insights(curves, thr),
        "hist": {"xmax": xmax, "dens": dens.tolist(), "share": (counts / counts.sum()).tolist(), "kde": kde.tolist()},
        "shares_text": (f"Share of streets (extreme tail beyond the chart is not drawn): <b>{100 * share['low']:.0f}%</b> low &nbsp;|&nbsp; <b>{100 * share['elevated']:.0f}%</b> elevated "
                        f"&nbsp;|&nbsp; <b>{100 * share['high']:.0f}%</b> high. Hover the bars for exact values."),
        "importance": [{"name": PRETTY.get(k, k), "v": float(v)} for k, v in importances.items()],
        "imp_origin": imp_origin, "calib": {"x": cx.tolist(), "y": np.asarray(cy, dtype=float).tolist()},
        "calib_grid": {"x": grid.tolist(), "y": gs.tolist()},
        "table": [list(r) for r in summary_rows(ho, cv)], "metric_origin": origin,
        "constraints": constraints or [],
    }
    html = DASH_HTML.replace("__DATA__", json.dumps(data, default=float))
    # add the interpolation helper right after the data constant
    helper = ("D.calib_of = v => { const g = D.calib_grid, n = g.x.length; if (v <= g.x[0]) return g.y[0]; "
              "if (v >= g.x[n-1]) return g.y[n-1]; let i = Math.floor(v / g.x[n-1] * (n-1)); "
              "while (g.x[i+1] < v) i++; const t = (v - g.x[i]) / (g.x[i+1] - g.x[i]); "
              "return g.y[i] + t * (g.y[i+1] - g.y[i]); };\nconst NS =")
    html = html.replace("const NS =", helper, 1)
    (out / "interactive_dashboard.html").write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("SAFE_ROUTE_MODEL_DIR", str(Path(__file__).resolve().parent)))
    ap.add_argument("--out-dir", default="evaluation_plots")
    ap.add_argument("--holdout-csv", help="labelled hold-out sample from train_offline.py (features + y [+ w])")
    ap.add_argument("--graph-file", help="GraphML of a real area to use as reference population")
    ap.add_argument("--pois-file", help="GeoJSON POIs for --graph-file")
    ap.add_argument("--n-ref", type=int, default=20_000, help="reference rows for the distribution plot")
    ap.add_argument("--n-pdp", type=int, default=1_500, help="rows averaged per partial-dependence point")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    art, rep = load(Path(args.model_dir))
    pop, source, labelled = reference_population(args, art, rng)
    print(f"reference population: {source}")

    effects, curves = plot_feature_effects(art, pop, source, out, args.n_pdp, args.seed)
    share, pred_pop = plot_distribution(art, pop, source, out, args.seed)

    if labelled is not None:
        importances, imp_origin = recompute_importance(art, labelled, args.seed), "recomputed on hold-out sample"
        ho, origin = metrics_from_frame(art, labelled), "recomputed on hold-out sample"
    else:
        importances = rep.get("importances") or art.get("importances") or {}
        imp_origin = "values saved by train_offline.py"
        ho, origin = rep["metrics"]["holdout"], "from training_report.json"
    if not importances:
        raise SystemExit("no feature importances available")
    plot_importance(importances, imp_origin, out)
    cv = rep["metrics"].get("spatial_cv")
    print_and_plot_summary(ho, cv, origin, out)
    plot_calibration(art, out)
    constraint_rows = verify_constraints(art, pop, curves, args.n_pdp, args.seed)
    write_interactive_dashboard(art, curves, pred_pop, importances, imp_origin, ho, cv, origin, source, share, out,
                                constraints=constraint_rows)

    print("\nfeature effect size (PDP max-min, in units of the 'elevated' threshold):")
    for f, st in effects.items():
        tag = ""
        if st.get("constraint"):
            tag = f"   constraint {st['constraint']:+d}: " + ("holds" if st["monotone_violation"] < 1e-9 else f"VIOLATED by {st['monotone_violation']:.4f}")
        print(f"  {PRETTY[f]:<46s} {st['swing_vs_elevated_threshold']:.2f}{tag}")
    print("\nMONOTONIC CONSTRAINT VERIFICATION (model's own partial-dependence curves):")
    if not constraint_rows:
        print("  none recorded in this artifact - it was trained before constraints existed; retrain.")
    for r in constraint_rows:
        print(f"  {r['name']:<48s} required {'non-decreasing' if r['sign'] > 0 else 'non-increasing':<15s} "
              f"{'HOLDS' if r['ok'] else 'VIOLATED by %.4f' % r['violation']:<18s} swing {r['swing']:.4f}  ({r['source']})")
    nm = effects.get("query_hour", {}).get("night_vs_midday")
    if nm is not None:
        print(f"\nhour-of-day check: night (22-03h) risk = {nm:.2f}x midday (10-15h)  ->  "
              + ("OK, night is riskier" if nm > 1 else "PROBLEM: night looks safer than midday"))
    print(f"population shares  low {100 * share['low']:.0f}% | elevated {100 * share['elevated']:.0f}% | "
          f"high {100 * share['high']:.0f}%")
    if ho["r2"] < 0.1:
        print("\nNOTE: hold-out R\u00b2 is below 0.10. The scores rank streets; they do not estimate incident counts.")

    (out / "summary.json").write_text(json.dumps({
        "reference": source, "holdout": ho, "spatial_cv": cv, "importances": importances,
        "feature_effects": effects, "constraint_check": constraint_rows, "population_shares": share,
        "thresholds": art["risk_thresholds"]},
        indent=2, default=float))
    print(f"\nplots written to {out.resolve()}/")
    print(f"interactive dashboard: {(out / 'interactive_dashboard.html').resolve()}  (open in any browser)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
