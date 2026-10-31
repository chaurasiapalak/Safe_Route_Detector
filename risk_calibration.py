"""
risk_calibration.py
-------------------
Maps the regressor's RAW predicted risk (a tiny number: thresholds are typically ~0.01-0.03) onto an
intuitive 0-100 scale for the UI and the router.

The map is monotone, continuous and piecewise, and it is ANCHORED ON THE TRAINED ARTIFACT'S OWN
THRESHOLDS (artifact["risk_thresholds"]), so every retrain re-anchors it automatically:

    raw = 0                  ->   5      (floor)
    raw = median             ->  20      (typical city street; target band 15-25  = LOW risk)
    raw = elevated  (p80)    ->  60      (lower edge of "elevated"; target band 55-65 = MODERATE risk)
    raw = high      (p92)    ->  85      (lower edge of "high";     target band 80-95 = HIGH risk)
    raw >> high              ->  99      (asymptote, never exceeded)

Segments:
    0 .. median        linear   5 -> 20
    median .. elevated linear  20 -> 60
    elevated .. high   linear  60 -> 85
    >= high            85 + 14 * (1 - exp(-(raw - high) / (high - elevated)))

If the artifact has no usable "median" anchor, the part below `elevated` falls back to
    5 + 55 * (raw / elevated) ** 2.

Usage (dynamic binding to a trained artifact -- no hard-coded thresholds anywhere):

    cal = bind(artifact)               # or bind(artifact["risk_thresholds"])
    cal.scores(raw_array)  ->  np.ndarray in [5, 99]
    cal.score(0.012)       ->  float
    cal.band(score)        ->  "low" | "elevated" | "high"
    cal.table()            ->  anchor rows for UI legends / diagnostics

The functional API (calibrate_risk_scores / calibrate_risk_score / uncalibrate_scores /
calibration_table) is unchanged and takes the thresholds mapping explicitly.

IMPORTANT: this is a *relative index anchored on the training distribution*, not a probability.
A score of 60 means "as risky as the top-20% streets of the training cities", nothing more.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

FLOOR = 5.0
AT_MEDIAN = 20.0
AT_ELEVATED = 60.0
AT_HIGH = 85.0
CEILING = 99.0
LOW_GAMMA = 2.0

# Display-score cut-offs of the three bands (the frontend reads them from `calibration_table`).
BAND_CUTOFFS = {"elevated": AT_ELEVATED, "high": AT_HIGH}


def _anchors(thresholds: Mapping[str, float]):
    elev = float(thresholds.get("elevated", 0.0))
    high = float(thresholds.get("high", 0.0))
    if not np.isfinite(elev) or elev <= 0:
        raise ValueError("thresholds['elevated'] must be a positive number")
    if not np.isfinite(high) or high <= elev:
        high = elev * 1.5
    med = thresholds.get("median")
    med = float(med) if med is not None and np.isfinite(med) and 0 < float(med) < elev else None
    return elev, high, med


def calibrate_risk_scores(raw_risk, thresholds: Mapping[str, float]) -> np.ndarray:
    """Vectorised calibration: raw risk (any shape) -> score in [FLOOR, CEILING]."""
    elev, high, med = _anchors(thresholds)
    r = np.clip(np.asarray(raw_risk, dtype=float), 0.0, None)
    out = np.empty_like(r)

    low = r < elev
    if med is None:
        out[low] = FLOOR + (AT_ELEVATED - FLOOR) * (r[low] / elev) ** LOW_GAMMA
    else:
        out[low] = np.interp(r[low], [0.0, med, elev], [FLOOR, AT_MEDIAN, AT_ELEVATED])

    mid = (r >= elev) & (r < high)
    out[mid] = AT_ELEVATED + (AT_HIGH - AT_ELEVATED) * (r[mid] - elev) / (high - elev)

    top = r >= high
    out[top] = AT_HIGH + (CEILING - AT_HIGH) * (1.0 - np.exp(-(r[top] - high) / max(high - elev, 0.5 * high)))
    return out


def calibrate_risk_score(raw_risk: float, thresholds: Mapping[str, float]) -> float:
    """Scalar calibration: raw predicted risk -> 0-100 display score."""
    return float(calibrate_risk_scores(np.array([raw_risk], dtype=float), thresholds)[0])


def uncalibrate_scores(scores, thresholds: Mapping[str, float]) -> np.ndarray:
    """Approximate inverse (display score -> raw risk), via interpolation on a dense grid."""
    elev, high, _ = _anchors(thresholds)
    grid = np.linspace(0.0, high + 8.0 * (high - elev), 4000)
    return np.interp(np.asarray(scores, dtype=float), calibrate_risk_scores(grid, thresholds), grid)


def score_band(score: float) -> str:
    """Human label for a calibrated score (bands match the anchors above)."""
    if score >= AT_HIGH:
        return "high"
    if score >= AT_ELEVATED:
        return "elevated"
    return "low"


def calibration_table(thresholds: Mapping[str, float]) -> list:
    """Anchor points for UI legends / diagnostics: [{key, label, raw, score}, ...]."""
    elev, high, med = _anchors(thresholds)
    rows = [("zero", "no recorded risk", 0.0)]
    if med is not None:
        rows.append(("median", "median street", med))
    rows += [("elevated", "elevated threshold (p80)", elev), ("high", "high threshold (p92)", high),
             ("2x_high", "2x high threshold", 2 * high)]
    return [{"key": k, "label": n, "raw": round(v, 5), "score": round(calibrate_risk_score(v, thresholds), 1)}
            for k, n, v in rows]


# --------------------------------------------------------------------------- #
# Dynamic binding to a trained artifact
# --------------------------------------------------------------------------- #
class RiskCalibrator:
    """Calibration bound to ONE set of thresholds (normally a trained artifact's `risk_thresholds`)."""

    def __init__(self, thresholds: Mapping[str, float]):
        _anchors(thresholds)                                   # validate early, fail loudly
        self.thresholds = {k: float(v) for k, v in thresholds.items() if v is not None}

    def scores(self, raw_risk) -> np.ndarray:
        return calibrate_risk_scores(raw_risk, self.thresholds)

    def score(self, raw_risk: float) -> float:
        return calibrate_risk_score(raw_risk, self.thresholds)

    def raw_from_score(self, scores) -> np.ndarray:
        return uncalibrate_scores(scores, self.thresholds)

    def band(self, score: float) -> str:
        return score_band(score)

    def table(self) -> list:
        return calibration_table(self.thresholds)


_BOUND: dict = {"sig": None, "cal": None}


def bind(source: Any) -> RiskCalibrator:
    """
    Bind the calibration to a trained artifact (dict containing "risk_thresholds") or directly to a
    thresholds mapping ({"elevated": ..., "high": ..., "median": ...}).  The most recent binding is
    cached, so calling bind(artifact) per request is cheap.
    """
    thr = source["risk_thresholds"] if isinstance(source, Mapping) and "risk_thresholds" in source else source
    if not isinstance(thr, Mapping):
        raise TypeError("bind() expects an artifact dict or a thresholds mapping")
    sig = tuple(sorted((k, float(v)) for k, v in thr.items() if v is not None))
    if _BOUND["sig"] != sig:
        _BOUND["cal"], _BOUND["sig"] = RiskCalibrator(thr), sig
    return _BOUND["cal"]
