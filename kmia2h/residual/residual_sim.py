"""Simulate the residual process and turn paths into sensor values (SPEC.md Phase 6).

Regime parameters (`rp`) are scalars or arrays with one value per step, so a regime can change
inside a path. Fields: theta_f, sigma_f, theta_s, sigma_s, white, lam_per_hour, p_drop, mean_drop,
mean_rise, jump_min, nu (None or NaN for Gaussian).
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from kmia2h.filter.process_model import ou_step
from kmia2h.interfaces import Target

MINUTES_PER_HOUR = 60.0


def _per_step(rp: Mapping, name: str, n_steps: int, default=0.0) -> np.ndarray:
    v = rp.get(name, default)
    return np.broadcast_to(np.asarray(np.nan if v is None else v, float), (n_steps,))


def innovations(rng: np.random.Generator, nu: float, n: int) -> np.ndarray:
    """Unit-variance Gaussian, or Student-t with nu > 2 scaled to unit variance."""
    if np.isfinite(nu):
        return rng.standard_t(nu, n) * np.sqrt((nu - 2.0) / nu)
    return rng.standard_normal(n)


def jump_sizes(rng: np.random.Generator, n: int, p_drop: float, mean_drop: float, mean_rise: float,
               jump_min: float) -> np.ndarray:
    """Signed sizes: threshold plus an exponential excess, negative with probability p_drop."""
    drop = rng.random(n) < p_drop
    excess = rng.standard_exponential(n) * np.where(drop, mean_drop, mean_rise)
    return np.where(drop, -1.0, 1.0) * (jump_min + excess)


def mean_jump(p_drop: float, mean_drop: float, mean_rise: float, jump_min: float) -> float:
    """E[J] of the distribution sampled by jump_sizes."""
    return (1.0 - p_drop) * (jump_min + mean_rise) - p_drop * (jump_min + mean_drop)


def simulate(rp: Mapping, n_paths: int, n_steps: int, rng: np.random.Generator,
             init_slow: np.ndarray | None = None, init_fast: np.ndarray | None = None) -> dict:
    """Arrays (n_paths, n_steps) x_slow, x_fast, white, and the jump count per path and step.

    Without init_* a path starts from the stationary distribution of the first step's regime.
    """
    p = {k: _per_step(rp, k, n_steps) for k in ("theta_f", "sigma_f", "theta_s", "sigma_s", "white",
                                                "lam_per_hour", "p_drop", "mean_drop", "mean_rise", "jump_min")}
    nu = _per_step(rp, "nu", n_steps, np.nan)

    def stationary(sigma, theta):
        return sigma[0] / np.sqrt(2.0 * theta[0]) * rng.standard_normal(n_paths) if theta[0] > 0 else np.zeros(n_paths)

    xs = stationary(p["sigma_s"], p["theta_s"]) if init_slow is None else np.array(init_slow, float)
    xf = stationary(p["sigma_f"], p["theta_f"]) if init_fast is None else np.array(init_fast, float)
    slow, fast, jumps = np.zeros((n_paths, n_steps)), np.zeros((n_paths, n_steps)), np.zeros((n_paths, n_steps), int)
    for i in range(n_steps):
        xf = ou_step(xf, p["theta_f"][i], p["sigma_f"][i], innovations(rng, nu[i], n_paths))
        xs = ou_step(xs, p["theta_s"][i], p["sigma_s"][i], rng.standard_normal(n_paths))
        rate = p["lam_per_hour"][i] / MINUTES_PER_HOUR
        if rate > 0:
            # X is a departure from a centred mean, so jumps must not move its average. The
            # compensator removes their expected drift each minute.
            xs = xs - rate * mean_jump(p["p_drop"][i], p["mean_drop"][i], p["mean_rise"][i], p["jump_min"][i])
            count = rng.poisson(rate, n_paths)
            total = int(count.sum())
            if total:
                sizes = jump_sizes(rng, total, p["p_drop"][i], p["mean_drop"][i], p["mean_rise"][i], p["jump_min"][i])
                np.add.at(xs, np.repeat(np.arange(n_paths), count), sizes)
                jumps[:, i] = count
        slow[:, i], fast[:, i] = xs, xf
    white = np.sqrt(p["white"])[None, :] * rng.standard_normal((n_paths, n_steps))
    return {"x_slow": slow, "x_fast": fast, "white": white, "jumps": jumps}


def quantise(t: np.ndarray, ops: Target, rule: str) -> np.ndarray:
    """Whole-degF one-minute values, exactly as the sensor stores them."""
    return np.asarray(ops.round_rule(t, rule), float)


def rolling_mean(t: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean along the last axis. The first window - 1 entries are NaN."""
    a = np.asarray(t, float)
    cum = np.cumsum(np.concatenate([np.zeros_like(a[..., :1]), a], axis=-1), axis=-1)
    out = np.full_like(a, np.nan)
    out[..., window - 1:] = (cum[..., window:] - cum[..., :-window]) / window
    return out


def floor_errors(mean_path: np.ndarray, paths: np.ndarray, lead: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """e_H and e_L: extreme of the rolling mean of paths minus that of the mean path.

    Arrays have `lead` extra minutes in front, which are excluded from the extremes.
    """
    r = rolling_mean(paths, window)[..., lead:]
    m = rolling_mean(mean_path, window)[..., lead:]
    return r.max(axis=-1) - m.max(axis=-1), r.min(axis=-1) - m.min(axis=-1)
