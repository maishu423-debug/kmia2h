"""Split-conformal calibration and physics/MOS blending (SPEC.md Phase 8, steps 5-6).

Each quantile gets its own additive offset, per group, taken from calibration residuals. Separate
offsets on the two sides shift an interval as well as widen it, so a biased forecast is corrected
rather than only made wider.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def group_key(season, is_day) -> np.ndarray:
    """Season by day/night, e.g. wet|True."""
    return np.array([f"{s}|{bool(d)}" for s, d in zip(season, is_day)], dtype=object)


def _offset(residuals: np.ndarray, tau: float) -> float:
    """Finite-sample order statistic of y - q for one quantile level."""
    r = np.sort(residuals)
    n = len(r)
    if tau > 0.5:
        return float(r[min(int(np.ceil((n + 1) * tau)), n) - 1])
    if tau < 0.5:
        return float(r[max(int(np.floor((n + 1) * tau)), 1) - 1])
    return float(np.median(r))


def fit(q: np.ndarray, y: np.ndarray, taus, groups, cfg: Mapping) -> dict:
    """Offsets per group. Groups with too few rows use the offsets of all rows pooled."""
    c = cfg["conformal"]
    q, y, groups = np.asarray(q, float), np.asarray(y, float), np.asarray(groups, dtype=object)
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    q, y, groups = q[ok], y[ok], groups[ok]

    def offsets(mask):
        return [_offset(y[mask] - q[mask, j], t) for j, t in enumerate(taus)]

    out = {c["all_group"]: offsets(np.ones(len(y), bool))}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() >= c["min_group_n"]:
            out[str(g)] = offsets(mask)
    return {"taus": [float(t) for t in taus], "offsets": out}


def apply(q: np.ndarray, groups, params: Mapping, cfg: Mapping) -> np.ndarray:
    """Calibrated quantiles, sorted so that they do not cross."""
    every = params["offsets"][cfg["conformal"]["all_group"]]
    delta = np.array([params["offsets"].get(str(g), every) for g in groups], float)
    return np.sort(np.asarray(q, float) + delta, axis=1)


def coverage(q: np.ndarray, y: np.ndarray, taus, lower_tau: float) -> float:
    """Share of y inside the central interval from lower_tau to 1 - lower_tau."""
    t = np.asarray(taus, float)
    lo, hi = np.argmin(np.abs(t - lower_tau)), np.argmin(np.abs(t - (1.0 - lower_tau)))
    return float(np.mean((y >= q[:, lo]) & (y <= q[:, hi])))


def log_loss(p: np.ndarray, realised: np.ndarray, cfg: Mapping) -> float:
    """Mean -log P(realised bin). p: (rows, bins). realised: bin index per row."""
    chosen = np.asarray(p, float)[np.arange(len(p)), np.asarray(realised, int)]
    return float(-np.mean(np.log(np.maximum(chosen, cfg["conformal"]["p_floor"]))))


def fit_blend(p_mos: np.ndarray, p_physics: np.ndarray, realised: np.ndarray, groups, cfg: Mapping) -> dict:
    """Weight on MOS per group from a 0..1 grid, minimising log-loss on the calibration rows."""
    c = cfg["conformal"]
    grid = np.round(np.arange(0.0, 1.0 + c["blend_step"] / 2.0, c["blend_step"]), 10)
    groups = np.asarray(groups, dtype=object)

    def best(mask):
        losses = [log_loss(w * p_mos[mask] + (1.0 - w) * p_physics[mask], realised[mask], cfg) for w in grid]
        return float(grid[int(np.argmin(losses))])

    out = {c["all_group"]: best(np.ones(len(p_mos), bool))}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() >= c["min_group_n"]:
            out[str(g)] = best(mask)
    return out


def blend_weight(weights: Mapping, group, cfg: Mapping) -> float:
    return float(weights.get(str(group), weights[cfg["conformal"]["all_group"]]))


def blend_flags(weights: Mapping) -> dict:
    """Groups where the blend chose one side entirely, which the spec asks to log."""
    return {g: ("mos_only" if w >= 1.0 else "physics_only") for g, w in weights.items() if w >= 1.0 or w <= 0.0}


def reliability(p: np.ndarray, realised: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Observed frequency against mean predicted probability, per probability decile.

    Every bin of every row counts as one forecast of that event.
    """
    p = np.asarray(p, float)
    hit = np.zeros_like(p, dtype=bool)
    hit[np.arange(len(p)), np.asarray(realised, int)] = True
    which = np.minimum((p.ravel() * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = which == b
        if mask.any():
            out.append({"decile": b, "n": int(mask.sum()), "predicted": float(p.ravel()[mask].mean()),
                        "observed": float(hit.ravel()[mask].mean())})
    return out
