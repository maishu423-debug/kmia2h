"""Rain and outflow cooling R(t) (SPEC.md Phase 4, step 4).

A cell of size D that arrives at minute t0 should push temperature down by D and let it recover
exponentially. The ODE already relaxes T toward T_adv at rate b, so the rate needed for the
perturbation -D g(s) is R = D (g'(s) + b g(s)). R is negative during recovery by design.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

KM_PER_MS_MIN = 0.06   # 1 m/s in km/min


def shape(minutes, arrival, cfg: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """g(s) and g'(s). g ramps 0 -> 1 from arrival over ramp_min, then decays over recovery_min."""
    r = cfg["physics"]["rain"]
    ramp, tau = r["ramp_min"], r["recovery_min"]
    s = np.asarray(minutes, float) - np.asarray(arrival, float)
    up = (s >= 0) & (s < ramp)
    down = s >= ramp
    g = np.where(up, s / ramp, 0.0) + np.where(down, np.exp(-(s - ramp) / tau), 0.0)
    dg = np.where(up, 1.0 / ramp, 0.0) + np.where(down, -np.exp(-(s - ramp) / tau) / tau, 0.0)
    return g, dg


def rate(cells: pd.DataFrame | None, n: int, b_ref: float, cfg: Mapping) -> np.ndarray:
    """R at minutes 1..n (degF/min). cells: arrival_min (after origin), drop_f. Pulses add."""
    out = np.zeros(n)
    if cells is None or len(cells) == 0:
        return out
    minutes = np.arange(1, n + 1, dtype=float)
    for arrival, drop in zip(cells["arrival_min"].to_numpy(float), cells["drop_f"].to_numpy(float)):
        g, dg = shape(minutes, arrival, cfg)
        out += drop * (dg + b_ref * g)
    return out


def rate_paths(cells: pd.DataFrame | None, n: int, n_paths: int, b_ref: float, rng: np.random.Generator,
               cfg: Mapping) -> np.ndarray:
    """(n_paths, n) rates with each path's arrival time and drop size perturbed."""
    out = np.zeros((n_paths, n))
    if cells is None or len(cells) == 0:
        return out
    r = cfg["physics"]["rain"]
    minutes = np.arange(1, n + 1, dtype=float)[None, :]
    for arrival, drop in zip(cells["arrival_min"].to_numpy(float), cells["drop_f"].to_numpy(float)):
        t0 = arrival + rng.normal(0.0, r["arrival_sigma_min"], n_paths)[:, None]
        d = np.maximum(drop + rng.normal(0.0, r["drop_sigma_f"], n_paths), 0.0)[:, None]
        g, dg = shape(minutes, t0, cfg)
        out += d * (dg + b_ref * g)
    return out


def drop_size(max_dbz, frac35, cfg: Mapping):
    """Regression estimate of the temperature drop (degF), never negative."""
    r = cfg["physics"]["rain"]
    b0, b1, b2 = r["drop_coef"]
    return np.maximum(b0 + b1 * (np.asarray(max_dbz, float) - r["dbz_min"]) + b2 * np.asarray(frac35, float), 0.0)


def find_cells(row: Mapping | None, n: int, cfg: Mapping) -> pd.DataFrame:
    """Cells projected to pass within hit_km of the station inside minutes 1..n.

    row: maxdbz_s{SS}_r{R}, frac35_s{SS}_r{R}, motion_u, motion_v (m/s). Sector and ring geometry
    is that of cloud_advect. Returns columns arrival_min, drop_f, sector, ring.
    """
    empty = pd.DataFrame({"arrival_min": [], "drop_f": [], "sector": [], "ring": []})
    if row is None:
        return empty
    r, c = cfg["physics"]["rain"], cfg["physics"]["cloud"]
    u, v = float(row.get("motion_u", np.nan)), float(row.get("motion_v", np.nan))
    if not np.isfinite([u, v]).all():
        return empty
    step = 360.0 / c["n_sectors"]
    minutes = np.arange(1, n + 1, dtype=float)
    found = []
    for s in range(c["n_sectors"]):
        for ring, dist in enumerate(c["ring_centres_km"]):
            dbz = float(row.get(f"maxdbz_s{s:02d}_r{ring}", np.nan))
            if not (np.isfinite(dbz) and dbz >= r["dbz_min"] and dist <= r["search_km"]):
                continue
            bearing = np.radians((s + 0.5) * step)
            x = dist * np.sin(bearing) + u * KM_PER_MS_MIN * minutes
            y = dist * np.cos(bearing) + v * KM_PER_MS_MIN * minutes
            hit = np.flatnonzero(np.hypot(x, y) <= r["hit_km"])
            if hit.size:
                frac = float(row.get(f"frac35_s{s:02d}_r{ring}", 0.0))
                found.append({"arrival_min": float(minutes[hit[0]]), "sector": s, "ring": ring,
                              "drop_f": float(drop_size(dbz, np.nan_to_num(frac), cfg))})
    return pd.DataFrame(found) if found else empty
