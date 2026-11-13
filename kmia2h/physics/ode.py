"""Surface energy-balance ODE for the mean temperature path (SPEC.md Phase 4).

dT/dt = a S_clr (1 - kappa C) - b (T - T_adv) - c sigma (T^4 - T_sky^4) (1 - lam C) - R

T, T_adv and T_sky are degF, the T^4 terms are evaluated in kelvin, time is minutes and the
tendency is degF/min. Every function is vectorised: coefficients and forcing may be scalars or
arrays that broadcast against T.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

FORCING = ("s_clr", "c", "t_adv", "t_sky", "r")
OPTIONAL_FORCING = {"b_scale": 1.0, "kappa_scale": 1.0}
TERMS = ("solar", "advection", "longwave", "rain")


def kelvin(t_f):
    return (np.asarray(t_f, dtype=float) - 32.0) * 5.0 / 9.0 + 273.15


def sky_temperature_f(t_f, dewpoint_f, cfg: Mapping):
    """Effective clear-sky radiating temperature (degF) from air temperature and dewpoint."""
    e0, e1, e2 = cfg["physics"]["sky_emissivity"]
    td = (np.asarray(dewpoint_f, dtype=float) - 32.0) * 5.0 / 9.0 / 100.0
    t_sky_k = kelvin(t_f) * (e0 + e1 * td + e2 * td ** 2) ** 0.25
    return (t_sky_k - 273.15) * 9.0 / 5.0 + 32.0


def terms(t, coef: Mapping, f: Mapping, cfg: Mapping) -> dict:
    """Each right-hand-side term separately (degF/min), so errors can be attributed to a term.

    coef: a, b, c, kappa, lam. f: s_clr, c, t_adv, t_sky, r and optionally b_scale, kappa_scale.
    """
    kappa = np.clip(coef["kappa"] * f.get("kappa_scale", 1.0), 0.0, 1.0)
    b = coef["b"] * f.get("b_scale", 1.0)
    longwave = kelvin(t) ** 4 - kelvin(f["t_sky"]) ** 4
    return {
        "solar": coef["a"] * f["s_clr"] * (1.0 - kappa * f["c"]),
        "advection": -b * (t - f["t_adv"]),
        "longwave": -coef["c"] * cfg["physics"]["sigma_sb"] * longwave * (1.0 - coef["lam"] * f["c"]),
        "rain": -f["r"],
    }


def rhs(t, coef: Mapping, f: Mapping, cfg: Mapping):
    tt = terms(t, coef, f, cfg)
    return tt["solar"] + tt["advection"] + tt["longwave"] + tt["rain"]


def euler_step(t, coef: Mapping, f: Mapping, cfg: Mapping, dt: float = 1.0):
    return t + dt * rhs(t, coef, f, cfg)


def forcing_arrays(forcing) -> tuple[dict, int]:
    """Validate a forcing table: 1-D columns of equal length with no NaN."""
    missing = [k for k in FORCING if k not in forcing]
    if missing:
        raise ValueError(f"forcing is missing columns {missing}")
    f = {k: np.asarray(forcing[k], dtype=float) for k in FORCING}
    n = len(f["t_adv"])
    for k, default in OPTIONAL_FORCING.items():
        f[k] = np.asarray(forcing[k], dtype=float) if k in forcing else np.full(n, default)
    for k, v in f.items():
        if v.ndim != 1 or len(v) != n:
            raise ValueError(f"forcing column {k} must be 1-D of length {n}")
        if np.isnan(v).any():
            raise ValueError(f"forcing column {k} contains NaN")
    return f, n


def integrate(t0: float, coef: Mapping, forcing, cfg: Mapping) -> pd.DataFrame:
    """Deterministic mean path with RK45.

    forcing: mapping or DataFrame of 1-D columns of length n. Row i is the forcing valid at minute
    origin + i + 1. Returns n rows (index 1..n, minutes after origin) with column `t`, each ODE
    term and each forcing series.
    """
    f, n = forcing_arrays(forcing)
    grid = np.arange(1, n + 1, dtype=float)

    def at(s: float) -> dict:
        return {k: np.interp(s, grid, v) for k, v in f.items()}

    sol = solve_ivp(lambda s, y: rhs(y, coef, at(s), cfg), (0.0, float(n)), [float(t0)],
                    method="RK45", t_eval=grid, max_step=1.0, rtol=1e-8, atol=1e-8)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    t = sol.y[0]
    out = {"t": t, **terms(t, coef, f, cfg), **f}
    return pd.DataFrame(out, index=pd.RangeIndex(1, n + 1, name="minute"))
