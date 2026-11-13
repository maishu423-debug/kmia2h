"""Monte Carlo through the exact settlement operator (SPEC.md Phase 7).

Paths are drawn from the filter's analysis ensemble, carried through the ODE and the residual
process one minute at a time, rounded to whole degF like the sensor, and passed to `ops.settle`.
Bin probabilities are counts of the settled results.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kmia2h.filter.process_model import IDX, coefficients, sensor_temperature
from kmia2h.interfaces import Target
from kmia2h.physics import cloud_advect, ode, rain
from kmia2h.residual import residual_sim


@dataclass
class SimResult:
    """forecast follows SPEC.md's schema. The other fields are arrays for later phases."""

    forecast: dict
    high: np.ndarray
    low: np.ndarray
    max_unrounded: np.ndarray
    min_unrounded: np.ndarray
    paths: np.ndarray            # (n_paths, lead + window) whole degF
    checks: dict
    forcing_used: dict


def _as_time(value) -> str:
    return pd.Timestamp(value).isoformat()


def _resample(snapshot: Mapping, n_paths: int, cfg: Mapping, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Members drawn with replacement, with T jittered by a fraction of the analysis spread."""
    ens = np.asarray(snapshot["ensemble"], float)
    idx = rng.integers(0, len(ens), n_paths)
    x = ens[idx].copy()
    spread = ens[:, IDX["T"]].std(ddof=1) if len(ens) > 1 else 0.0
    x[:, IDX["T"]] += cfg["simulate"]["jitter_frac"] * spread * rng.standard_normal(n_paths)
    return x, idx


def cloud_paths(c: np.ndarray, n_paths: int, cfg: Mapping, rng: np.random.Generator,
                cloud: Mapping | None = None) -> np.ndarray:
    """(n_paths, n) cloud forcing: the mean series with motion and level perturbed per path.

    cloud: cf_disc, cf_field, u, v (and optionally tcdc), the inputs of cloud_advect.project. The
    difference between perturbed and unperturbed motion is added to c, so anything else already in
    c (convective growth) is kept.
    """
    s = cfg["simulate"]
    n = len(c)
    out = np.broadcast_to(c, (n_paths, n)).copy()
    if cloud is not None and np.isfinite([cloud["u"], cloud["v"]]).all():
        du = rng.normal(0.0, s["motion_sigma_ms"], n_paths)
        dv = rng.normal(0.0, s["motion_sigma_ms"], n_paths)
        args = (cloud["cf_disc"], cloud["cf_field"])
        tcdc = cloud.get("tcdc")
        base = cloud_advect.project(*args, cloud["u"], cloud["v"], n, cfg, tcdc)
        out += cloud_advect.project(*args, cloud["u"] + du, cloud["v"] + dv, n, cfg, tcdc) - base
    out += rng.normal(0.0, s["cloud_sigma"], n_paths)[:, None]
    return np.clip(out, 0.0, 1.0)


def _integrate(x: np.ndarray, forcing: Mapping, c_path: np.ndarray, r_path: np.ndarray, n: int, cfg: Mapping,
               rng: np.random.Generator) -> np.ndarray:
    """ODE paths (n_paths, n) with each member's own coefficients."""
    coef = coefficients(x, cfg)
    t = x[:, IDX["T"]].copy()
    c_eff = x[:, IDX["C_eff"]].copy()
    tau_adv = cfg["filter"]["t_adv_tau_min"]
    extra = cfg["simulate"]["t_adv_extra_sigma"] * rng.standard_normal(len(x))
    d_adv = x[:, IDX["dT_adv"]] + extra
    b_scale = np.broadcast_to(np.asarray(forcing.get("b_scale", 1.0), float), (n,))
    k_scale = np.broadcast_to(np.asarray(forcing.get("kappa_scale", 1.0), float), (n,))
    out = np.zeros((len(x), n))
    for i in range(n):
        f = {"s_clr": forcing["s_clr"][i], "c": c_eff, "t_adv": forcing["t_adv"][i] + d_adv * np.exp(-i / tau_adv),
             "t_sky": forcing["t_sky"][i], "r": r_path[:, i], "b_scale": b_scale[i], "kappa_scale": k_scale[i]}
        t = ode.euler_step(t, coef, f, cfg)
        c_eff = np.clip(c_eff + (c_path[:, i] - c_eff) / cfg["filter"]["c_relax_min"], 0.0, 1.0)
        out[:, i] = t
    return out


def _table(values: np.ndarray) -> dict:
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(k)): float(c) / len(values) for k, c in zip(keys, counts)}


def _joint(high: np.ndarray, low: np.ndarray) -> dict:
    pairs, counts = np.unique(np.column_stack([high, low]), axis=0, return_counts=True)
    return {f"{int(h)}|{int(l)}": float(c) / len(high) for (h, l), c in zip(pairs, counts)}


def _quantiles(values: np.ndarray, levels) -> dict:
    return {f"{q:02d}": float(np.quantile(values, q / 100.0)) for q in levels}


def run(snapshot: Mapping, forcing: Mapping, rp: Mapping, ops: Target, cfg: Mapping, origin, seed: int | None = None,
        n_paths: int | None = None, elapsed_min: int = 0, regime: str = "", latest_obs: float | None = None,
        floor_range: tuple[float, float] | None = None, cloud: Mapping | None = None,
        rain_cells: pd.DataFrame | None = None) -> SimResult:
    """One forecast.

    snapshot: {"ensemble": (N, state), "paths": (N, m) smoothed member temperatures ending at the
    origin}. paths is needed only for elapsed_min > 0.
    forcing: s_clr, c, t_adv, t_sky, r (and optionally b_scale, kappa_scale), 1-D of length n; row i
    is valid at minute origin + i + 1 and n sets the remaining window.
    rp: residual regime parameters, as in residual_sim.simulate.
    rain_cells: arrival_min, drop_f. With cells given, rain is re-drawn per path and forcing r is not used.
    elapsed_min: minutes of the market window already over at origin; they count toward settlement.
    """
    sim, target = cfg["sim"], cfg["target"]
    seed = sim["seed"] if seed is None else int(seed)
    n_paths = sim["paths_live"] if n_paths is None else int(n_paths)
    rng = np.random.default_rng(seed)
    rule, lead, window = target["rounding"], int(sim["lead_min"]), int(target["roll_window"])
    fc, n = ode.forcing_arrays(forcing)

    x, members = _resample(snapshot, n_paths, cfg, rng)
    c_path = cloud_paths(fc["c"], n_paths, cfg, rng, cloud)
    b_ref = cfg["physics"]["prior"]["b"]
    r_path = (rain.rate_paths(rain_cells, n, n_paths, b_ref, rng, cfg) if rain_cells is not None and len(rain_cells)
              else np.broadcast_to(fc["r"], (n_paths, n)).copy())

    t_paths = _integrate(x, fc, c_path, r_path, n, cfg, rng)
    noise = residual_sim.simulate(rp, n_paths, n, rng, init_slow=x[:, IDX["X_slow"]], init_fast=x[:, IDX["X_fast"]])
    smooth = t_paths + noise["x_slow"] + noise["x_fast"]
    future = residual_sim.quantise(smooth + noise["white"], ops, rule)

    y0 = sensor_temperature(x)
    if elapsed_min > 0:
        past = np.asarray(snapshot["paths"], float)
        need = elapsed_min + lead
        if past.shape[1] < need:
            raise ValueError(f"elapsed window needs {need} path minutes, snapshot has {past.shape[1]}")
        unrounded_past = past[members][:, -need:]
    else:
        unrounded_past = np.column_stack([x[:, IDX["T_m3"]], x[:, IDX["T_m2"]], x[:, IDX["T_m1"]], y0])
    past_whole = residual_sim.quantise(unrounded_past, ops, rule)
    paths = np.concatenate([past_whole, future], axis=1)

    high, low = ops.settle(paths, rule)
    roll = residual_sim.rolling_mean(paths, window)[:, lead:]
    max_u, min_u = roll.max(axis=1), roll.min(axis=1)

    mean_path = smooth.mean(axis=0)
    mean_series = np.concatenate([unrounded_past.mean(axis=0), mean_path])
    mean_roll = residual_sim.rolling_mean(mean_series, window)[lead:]

    levels = cfg["simulate"]["quantiles"]
    origin_ts = pd.Timestamp(origin)
    forecast = {
        "origin_utc": _as_time(origin_ts),
        "window_end_utc": _as_time(origin_ts + pd.Timedelta(minutes=n)),
        "n_paths": n_paths,
        "p_high": _table(high),
        "p_low": _table(low),
        "p_joint": _joint(high, low),
        "q_max": _quantiles(max_u, levels),
        "q_min": _quantiles(min_u, levels),
        "edge_high": float(ops.edge(mean_roll.max())),
        "edge_low": float(ops.edge(mean_roll.min())),
        "regime": regime,
        "mean_path": [float(v) for v in mean_path],
    }
    checks = sanity_checks(paths, smooth, mean_path, r_path, cfg, latest_obs, floor_range)
    return SimResult(forecast, high, low, max_u, min_u, paths, checks,
                     {"c": c_path.mean(axis=0), "r": r_path.mean(axis=0), **{k: fc[k] for k in ("s_clr", "t_adv", "t_sky")}})


def sanity_checks(paths: np.ndarray, smooth: np.ndarray, mean_path: np.ndarray, r_path: np.ndarray, cfg: Mapping,
                  latest_obs: float | None, floor_range: tuple[float, float] | None) -> dict:
    """The per-run checks of SPEC.md Phase 7 step 6. None means the check had no input."""
    s = cfg["simulate"]
    n = r_path.shape[1]
    steps = np.abs(np.diff(paths[:, -(n + 1):], axis=1))
    quiet = np.abs(r_path) <= 0.0
    excess = float(smooth.max(axis=1).std())
    return {
        "first_minute_near_obs": None if latest_obs is None else bool(abs(mean_path[0] - latest_obs) <= s["max_obs_gap_f"]),
        "steps_within_limit": bool((steps[quiet] <= s["max_step_f"]).all()),
        "spread_of_path_max": excess,
        "spread_in_floor_range": None if floor_range is None else bool(floor_range[0] <= excess <= floor_range[1]),
    }


def validate_forecast(fc: Mapping) -> list[str]:
    """Problems with a forecast dict: empty when it matches SPEC.md's schema."""
    problems = []
    required = ("origin_utc", "window_end_utc", "n_paths", "p_high", "p_low", "p_joint", "q_max", "q_min",
                "edge_high", "edge_low", "regime", "mean_path")
    for k in required:
        if k not in fc:
            problems.append(f"missing {k}")
    if problems:
        return problems
    for name in ("p_high", "p_low", "p_joint"):
        total = sum(fc[name].values())
        if abs(total - 1.0) > 1e-9:
            problems.append(f"{name} sums to {total}")
        if any(v < 0 for v in fc[name].values()):
            problems.append(f"{name} has a negative probability")
    for key in fc["p_joint"]:
        high, low = key.split("|")
        if int(high) < int(low):
            problems.append(f"p_joint has high below low at {key}")
    return problems
