"""Effective cloud factor C(t) for the next two hours (SPEC.md Phase 4, step 2).

Geometry: `cf_field` has shape (n_sectors, n_rings). Sector k covers bearings [k, k+1) * 30 deg
measured clockwise from north, looking outward from the station. Rings are the spec's 0-10, 10-20,
20-40 and 40-80 km. All cloud fractions are 0..1, winds and motion vectors are m/s.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

SECONDS_PER_MIN = 60.0
M_PER_KM = 1000.0


def field_from_row(row: Mapping, cfg: Mapping) -> np.ndarray:
    """(n_sectors, n_rings) cloud fractions from columns cf_s{SS}_r{R}."""
    c = cfg["physics"]["cloud"]
    return np.array([[float(row.get(f"cf_s{s:02d}_r{r}", np.nan)) for r in range(c["n_rings"])]
                     for s in range(c["n_sectors"])], dtype=float)


def motion_vector(row: Mapping | None, u10: float, v10: float, cfg: Mapping) -> tuple[float, float, str]:
    """GOES cloud motion if its quality passes, otherwise scaled HRRR 10 m wind."""
    c = cfg["physics"]["cloud"]
    if row is not None:
        u, v, q = (float(row.get(k, np.nan)) for k in ("motion_u", "motion_v", "motion_q"))
        if np.isfinite([u, v, q]).all() and q > c["motion_q_min"]:
            return u, v, "goes"
    return c["wind_factor"] * float(u10), c["wind_factor"] * float(v10), "hrrr_wind"


def project(cf_disc: float, cf_field: np.ndarray, u, v, n: int, cfg: Mapping,
            tcdc: np.ndarray | None = None) -> np.ndarray:
    """C at minutes 1..n by moving the current field along the motion vector.

    u, v: scalars or arrays of shape (p,), the motion of the cloud in m/s. Returns (p, n), or (1, n)
    for scalars. tcdc: HRRR total cloud (fraction) at those n minutes, used beyond the outer ring.
    """
    c = cfg["physics"]["cloud"]
    if tcdc is not None and np.nanmax(tcdc) > 1.0 + 1e-9:
        raise ValueError("tcdc must be a fraction in 0..1, not percent")
    u, v = np.atleast_1d(np.asarray(u, float)), np.atleast_1d(np.asarray(v, float))
    speed_km_min = np.hypot(u, v) * SECONDS_PER_MIN / M_PER_KM
    lead = np.arange(1, n + 1, dtype=float)
    dist = speed_km_min[:, None] * lead

    # The air now at distance d in the direction the cloud is coming from arrives after d / speed.
    bearing = np.degrees(np.arctan2(-u, -v)) % 360.0
    sector = np.floor(bearing / (360.0 / c["n_sectors"])).astype(int) % c["n_sectors"]

    disc = float(cf_disc) if np.isfinite(cf_disc) else float(np.nanmean(cf_field))
    filled = np.where(np.isnan(cf_field), disc, cf_field)
    profile = np.concatenate([np.full((len(u), 1), disc), filled[sector]], axis=1)
    centres = np.array([c["disc_km"], *c["ring_centres_km"]], dtype=float)

    i = np.clip(np.searchsorted(centres, dist, side="right") - 1, 0, len(centres) - 2)
    frac = np.clip((dist - centres[i]) / (centres[i + 1] - centres[i]), 0.0, 1.0)
    rows = np.arange(len(u))[:, None]
    out = profile[rows, i] * (1.0 - frac) + profile[rows, i + 1] * frac

    if tcdc is not None:
        tc = np.where(np.isnan(tcdc), out, tcdc)
        beyond = dist > c["outer_km"]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_outer = np.where(speed_km_min > 0, c["outer_km"] / speed_km_min, np.inf)
        weight = np.exp(-np.clip(lead - t_outer[:, None], 0.0, None) / c["relax_min"])
        out = np.where(beyond, tc + (out - tc) * weight, out)
    return np.clip(out, 0.0, 1.0)


def convective_growth(valid: pd.DatetimeIndex, tcdc: np.ndarray, tcdc_origin: float, cfg: Mapping) -> np.ndarray:
    """HRRR cloud change since origin, in wet-season late morning and afternoon (LST); else 0."""
    lo, hi = cfg["physics"]["cloud"]["growth_hours_lst"]
    local = valid + pd.Timedelta(hours=cfg["tz_lst_offset_hours"])
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    active = np.isin(local.month.to_numpy(), cfg["features"]["wet_months"]) & (hour >= lo) & (hour < hi)
    return np.where(active, np.nan_to_num(tcdc - tcdc_origin), 0.0)


def kappa_value(cod: float, top_m: float, is_day: bool, cfg: Mapping) -> float:
    """Attenuation for the current cloud class. At night only cloud-top height is used."""
    c = cfg["physics"]["cloud"]
    thin = (np.isfinite(top_m) and top_m > c["thin_top_m"]) or (
        bool(is_day) and np.isfinite(cod) and cod < c["thin_cod_max"])
    return float(c["kappa_thin"] if thin else c["kappa_thick"])


def metar_fallback(sky_code: str | None, n: int, cfg: Mapping) -> dict:
    """Station sky cover held constant, thick cloud. Used when GOES is unavailable."""
    c = cfg["physics"]["cloud"]
    cover = c["metar_cover"].get((sky_code or "").strip().upper(), np.nan)
    return {"c": np.full(n, float(cover)), "kappa": float(c["kappa_thick"]), "source": "metar"}


def build(valid: pd.DatetimeIndex, goes_row: Mapping | None, u10: float, v10: float, is_day: bool,
          sky_code: str | None, cfg: Mapping, tcdc: np.ndarray | None = None,
          tcdc_origin: float | None = None) -> dict:
    """Mean cloud forcing for minutes `valid`: {c, kappa_scale, source, u, v}."""
    n = len(valid)
    prior_kappa = cfg["physics"]["prior"]["kappa"]
    usable = goes_row is not None and (np.isfinite(float(goes_row.get("cf_disc", np.nan)))
                                       or np.isfinite(field_from_row(goes_row, cfg)).any())
    if not usable:
        fb = metar_fallback(sky_code, n, cfg)
        return {"c": fb["c"], "kappa_scale": np.full(n, fb["kappa"] / prior_kappa),
                "source": fb["source"], "u": float("nan"), "v": float("nan")}

    u, v, source = motion_vector(goes_row, u10, v10, cfg)
    c = project(float(goes_row.get("cf_disc", np.nan)), field_from_row(goes_row, cfg), u, v, n, cfg, tcdc)[0]
    if tcdc is not None and tcdc_origin is not None and np.isfinite(tcdc_origin):
        c = np.clip(c + convective_growth(valid, tcdc, tcdc_origin, cfg), 0.0, 1.0)
    kappa = kappa_value(float(goes_row.get("cod_disc", np.nan)), float(goes_row.get("cth_disc", np.nan)), is_day, cfg)
    return {"c": c, "kappa_scale": np.full(n, kappa / prior_kappa), "source": source, "u": u, "v": v}
