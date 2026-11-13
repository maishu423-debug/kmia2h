"""One-minute process model for the ensemble filter (SPEC.md Phase 5).

State per member, in this order:
    T, Tdot, T_m1, T_m2, T_m3, C_eff, a, b, c, kappa, X_slow, X_fast, T_m4, dT_adv

The first twelve are the spec's. T is the smooth ODE temperature and the sensor sees
Y = T + X_slow + X_fast. T_m1..T_m4 are past values of Y. T_m4 and dT_adv are additions: see
DECISIONS.md.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from kmia2h.physics import ode

STATE = ("T", "Tdot", "T_m1", "T_m2", "T_m3", "C_eff", "a", "b", "c", "kappa", "X_slow", "X_fast", "T_m4", "dT_adv")
IDX = {name: i for i, name in enumerate(STATE)}
COEFS = ("a", "b", "c", "kappa")
REGIME_FIELDS = ("theta_f", "sigma_f", "theta_s", "sigma_s")
# Inflation skips the coefficients: unobserved ones (a at night) would otherwise grow without limit.
INFLATED = tuple(IDX[s] for s in STATE if s not in COEFS)


def ou_step(x, theta, sigma, z, dt: float = 1.0):
    """Exact one-step transition of dX = -theta X ds + sigma dW, driven by unit-variance z."""
    theta = np.asarray(theta, float)
    phi = np.exp(-theta * dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(theta > 0, -np.expm1(-2.0 * theta * dt) / (2.0 * theta), dt)
    return phi * x + np.asarray(sigma, float) * np.sqrt(var) * z


def sensor_temperature(x: np.ndarray) -> np.ndarray:
    """Y = T + X_slow + X_fast, for an array whose last axis is the state."""
    return x[..., IDX["T"]] + x[..., IDX["X_slow"]] + x[..., IDX["X_fast"]]


def coefficients(x: np.ndarray, cfg: Mapping) -> dict:
    out = {k: x[..., IDX[k]] for k in COEFS}
    out["lam"] = cfg["physics"]["prior"]["lam"]
    return out


def constrain(x: np.ndarray, cfg: Mapping) -> np.ndarray:
    """Clip coefficients to their physical bounds and C_eff to [0, 1], in place."""
    for k, (lo, hi) in cfg["physics"]["bounds"].items():
        x[..., IDX[k]] = np.clip(x[..., IDX[k]], lo, hi)
    x[..., IDX["C_eff"]] = np.clip(x[..., IDX["C_eff"]], 0.0, 1.0)
    return x


def init_ensemble(n: int, t0: float, c0: float, cfg: Mapping, rng: np.random.Generator) -> np.ndarray:
    """Ensemble around the coefficient prior, with lag states at the member's temperature."""
    sig, prior = cfg["filter"]["init_sigma"], cfg["physics"]["prior"]
    x = np.zeros((n, len(STATE)))
    x[:, IDX["T"]] = t0 + rng.normal(0.0, sig["T"], n)
    x[:, IDX["C_eff"]] = c0 + rng.normal(0.0, sig["C_eff"], n)
    x[:, IDX["X_slow"]] = rng.normal(0.0, sig["X_slow"], n)
    x[:, IDX["X_fast"]] = rng.normal(0.0, sig["X_fast"], n)
    for k in COEFS:
        x[:, IDX[k]] = prior[k] + rng.normal(0.0, sig[k], n)
    x[:, IDX["dT_adv"]] = rng.normal(0.0, cfg["filter"]["t_adv_sigma"], n)
    y = sensor_temperature(x)
    for k in ("T_m1", "T_m2", "T_m3", "T_m4"):
        x[:, IDX[k]] = y
    return constrain(x, cfg)


def step(x: np.ndarray, f: Mapping, rp: Mapping, cfg: Mapping, rng: np.random.Generator) -> np.ndarray:
    """Advance members one minute. f holds the forcing valid at the new minute (scalars).

    rp: theta_f, sigma_f, theta_s, sigma_s of the regime in force. Forcing c may be NaN, in which
    case C_eff persists.
    """
    fl = cfg["filter"]
    n, d = x.shape
    z = rng.standard_normal((n, 7))
    out = x.copy()
    y = sensor_temperature(x)
    out[:, IDX["T_m4"]] = x[:, IDX["T_m3"]]
    out[:, IDX["T_m3"]] = x[:, IDX["T_m2"]]
    out[:, IDX["T_m2"]] = x[:, IDX["T_m1"]]
    out[:, IDX["T_m1"]] = y

    c_eff = x[:, IDX["C_eff"]]
    forced = float(f["c"])
    relax = (forced - c_eff) / fl["c_relax_min"] if np.isfinite(forced) else 0.0
    out[:, IDX["C_eff"]] = c_eff + relax + fl["c_sigma"] * z[:, 0]

    theta = 1.0 / fl["t_adv_tau_min"]
    d_adv = ou_step(x[:, IDX["dT_adv"]], theta, fl["t_adv_sigma"] * np.sqrt(2.0 * theta), z[:, 1])
    out[:, IDX["dT_adv"]] = d_adv

    forcing = {**f, "c": c_eff, "t_adv": f["t_adv"] + x[:, IDX["dT_adv"]]}
    t = x[:, IDX["T"]]
    t_new = ode.euler_step(t, coefficients(x, cfg), forcing, cfg) + fl["t_sigma"] * z[:, 2]
    out[:, IDX["T"]] = t_new
    out[:, IDX["Tdot"]] = t_new - t

    out[:, IDX["X_fast"]] = ou_step(x[:, IDX["X_fast"]], rp["theta_f"], rp["sigma_f"], z[:, 3])
    out[:, IDX["X_slow"]] = ou_step(x[:, IDX["X_slow"]], rp["theta_s"], rp["sigma_s"], z[:, 4])

    rw = fl["coef_rw_sigma"]
    u = rng.standard_normal((n, len(COEFS)))
    for i, k in enumerate(COEFS):
        out[:, IDX[k]] = x[:, IDX[k]] + rw[k] * u[:, i]
    return constrain(out, cfg)


def make_step(forcing: Mapping, regime: Mapping, cfg: Mapping) -> Callable:
    """step_fn(x, k, rng) advancing from minute index k to k + 1.

    forcing: columns s_clr, c, t_adv, t_sky, r and optionally b_scale, kappa_scale; one value per
    minute index. regime: theta_f, sigma_f, theta_s, sigma_s, also one value per minute index.
    """
    columns = {k: np.asarray(v, float) for k, v in forcing.items()}
    rp = {k: np.asarray(regime[k], float) for k in REGIME_FIELDS}

    def step_fn(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        f = {name: v[k + 1] for name, v in columns.items()}
        return step(x, f, {name: v[k + 1] for name, v in rp.items()}, cfg, rng)

    return step_fn


def regime_label(is_day: bool, sector: str, sky_class: str) -> str:
    """Readable regime string for a minute, e.g. day-onshore-sct."""
    return f"{'day' if is_day else 'night'}-{sector}-{sky_class}"


def lookup_regime(table: Mapping, season: str, is_day: bool, sky_class: str, wind: str) -> Mapping:
    """Finest available regime, following the Phase 3 hierarchy and its key format."""
    day = str(bool(is_day))
    for key in (f"{season}|{day}|{sky_class}|{wind}", f"{season}|{day}|{sky_class}", f"{season}|{day}"):
        if key in table:
            return table[key]
    raise KeyError(f"no regime parameters for {season}|{day}")


def regime_series(table: Mapping, season, is_day, sky_class, wind) -> dict:
    """Per-minute residual parameters. Regimes switch here and never touch the ensemble."""
    rows = [lookup_regime(table, s, d, k, w) for s, d, k, w in zip(season, is_day, sky_class, wind)]
    return {f: np.array([r[f] for r in rows], float) for f in REGIME_FIELDS}
