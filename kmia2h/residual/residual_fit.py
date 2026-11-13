"""Fit the residual process per regime (SPEC.md Phase 6).

X(s) = X_slow(s) + X_fast(s), with
    dX_fast = -theta_f X_fast ds + sigma_f dW_f
    dX_slow = -theta_s X_slow ds + sigma_s dW_s + J dN_s
in degF and minutes. N_s is active only by day under SCT/BKN cloud.

Input x is the 1-minute residual T - m (degF), NaN where missing. Fitted parameters are plain
dicts, one per regime, with the fields listed in INTERFACES.md.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import optimize, stats

HOURS_PER_MIN = 1.0 / 60.0
ABS_Q25_OF_UNIT_GAUSSIAN = 0.31864   # lower quartile of |Z|, Z standard normal


def autocovariance(x: np.ndarray, maxlag: int, segments: np.ndarray | None = None) -> np.ndarray:
    """c[0..maxlag]. NaNs are skipped. Pairs from different segments (days, gaps) are not used."""
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    c = np.zeros(maxlag + 1)
    for k in range(maxlag + 1):
        a, b = x[:len(x) - k], x[k:]
        p = a * b
        if segments is not None:
            p = np.where(segments[:len(x) - k] == segments[k:], p, np.nan)
        c[k] = np.nanmean(p)
    return c


def fit_two_exponentials(c: np.ndarray, cfg: Mapping) -> dict:
    """Fit C(k) = A_f exp(-theta_f k) + A_s exp(-theta_s k) on lags >= 1.

    Returns the Phase 3 schema: A_f, theta_f, A_s, theta_s, white, var0. The excess of c[0] over
    A_f + A_s is white noise.
    """
    r = cfg["residual"]
    k = np.arange(1, len(c), dtype=float)

    def f(k, af, tf, a_s, ts):
        return af * np.exp(-tf * k) + a_s * np.exp(-ts * k)

    p0 = np.array(r["acf_p0"], float) * np.array([c[1], 1.0, c[1], 1.0])
    (af, tf, a_s, ts), _ = optimize.curve_fit(f, k, c[1:], p0=p0, bounds=(0, np.inf), maxfev=r["acf_maxfev"])
    if tf < ts:
        af, tf, a_s, ts = a_s, ts, af, tf
    return {"A_f": float(af), "theta_f": float(tf), "A_s": float(a_s), "theta_s": float(ts),
            "white": float(max(c[0] - af - a_s, 0.0)), "var0": float(c[0])}


def ou_params(acov: Mapping, cfg: Mapping) -> dict:
    """SDE parameters from a Phase 3 autocovariance row, using var = sigma^2 / (2 theta).

    A component with weight below min_weight is dropped and its variance moves to the other one.
    `white` is the part still to be simulated: quantisation (1/12) comes from rounding instead.
    """
    r = cfg["residual"]
    total = acov["A_f"] + acov["A_s"]
    a_f, a_s = acov["A_f"], acov["A_s"]
    if total > 0 and a_s / total < r["min_weight"]:
        a_f, a_s = total, 0.0
    elif total > 0 and a_f / total < r["min_weight"]:
        a_f, a_s = 0.0, total
    return {
        "theta_f": float(acov["theta_f"]),
        "sigma_f": float(np.sqrt(2.0 * acov["theta_f"] * a_f)),
        "theta_s": float(acov["theta_s"]),
        "sigma_s": float(np.sqrt(2.0 * acov["theta_s"] * a_s)),
        "white": float(max(acov["white"] - r["quantisation_var"], 0.0)),
    }


def window_difference(x: np.ndarray, window: int) -> np.ndarray:
    """d[i] = mean(x[i+1 .. i+window]) - mean(x[i-window+1 .. i]). NaN wherever a window is incomplete.

    For a step of size J between minutes i and i+1 in otherwise flat data, d[i] is exactly J.
    """
    x = np.asarray(x, float)
    n = len(x)
    cum = np.r_[0.0, np.cumsum(np.nan_to_num(x))]
    bad = np.r_[0, np.cumsum(np.isnan(x))]
    d = np.full(n, np.nan)
    if n < 2 * window:
        return d
    i = np.arange(window - 1, n - window)
    after, before = cum[i + 1 + window] - cum[i + 1], cum[i + 1] - cum[i + 1 - window]
    ok = (bad[i + 1 + window] - bad[i + 1 - window]) == 0
    d[i[ok]] = (after[ok] - before[ok]) / window
    return d


def effective_threshold(x: np.ndarray, cfg: Mapping) -> float:
    """The spec's threshold, raised if the window difference is too noisy for it to mean a jump.

    The noise level comes from the lower quartile of |d|, which stays usable even when jumps
    disturb half of the minutes.
    """
    r = cfg["residual"]
    d = window_difference(x, r["jump_window_min"])
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float(r["jump_threshold_f"])
    noise = np.quantile(np.abs(d), 0.25) / ABS_Q25_OF_UNIT_GAUSSIAN
    return float(max(r["jump_threshold_f"], r["jump_noise_z"] * noise))


def detect_jumps(x: np.ndarray, cfg: Mapping, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(minute index, size) of each jump. A run of same-sign exceedances counts once, at its peak."""
    r = cfg["residual"]
    d = window_difference(x, r["jump_window_min"])
    thr = effective_threshold(x, cfg) if threshold is None else threshold
    idx = np.flatnonzero(np.abs(np.nan_to_num(d)) > thr)
    if idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    new_run = np.r_[True, (np.diff(idx) > 1) | (np.sign(d[idx[1:]]) != np.sign(d[idx[:-1]]))]
    peaks = [run[np.argmax(np.abs(d[run]))] for run in np.split(idx, np.flatnonzero(new_run)[1:])]
    peaks = np.array(peaks, dtype=int)
    return peaks, d[peaks]


def fit_jump_sizes(sizes: np.ndarray, thr: float) -> dict:
    """Two-sided shifted exponential: P(drop), and each side's mean size above the threshold."""
    sizes = np.asarray(sizes, float)
    drops, rises = -sizes[sizes < 0], sizes[sizes > 0]
    return {
        "p_drop": float(len(drops) / len(sizes)) if len(sizes) else 0.5,
        "mean_drop": float(drops.mean() - thr) if len(drops) else 0.0,
        "mean_rise": float(rises.mean() - thr) if len(rises) else 0.0,
        "jump_min": thr,
    }


def jump_active(is_day, sky_class, cfg: Mapping) -> np.ndarray:
    return np.asarray(is_day, bool) & np.isin(np.asarray(sky_class, dtype=object).astype(str), cfg["residual"]["jump_sky"])


def _dead_time_corrected(events: int, minutes: int, window: int) -> float:
    """True rate per hour from a count. Jumps closer together than the window merge into one."""
    if minutes == 0:
        return 0.0
    observed = events / minutes
    return observed / max(1.0 - observed * window, 1e-6) / HOURS_PER_MIN


def fit_jumps(x: np.ndarray, is_day, sky_class, hour, cfg: Mapping) -> dict:
    """Rate per hour overall and by sky class and local hour, and the size distribution.

    Only minutes where jumps are active count, both for events and for exposure.
    """
    window = int(cfg["residual"]["jump_window_min"])
    active = jump_active(is_day, sky_class, cfg)
    sky = np.asarray(sky_class, dtype=object).astype(str)
    hour = np.asarray(hour, int)
    thr = effective_threshold(x, cfg)
    where, size = detect_jumps(x, cfg, thr)
    keep = active[where] if where.size else np.array([], dtype=bool)
    where, size = where[keep], size[keep]
    usable = active & ~np.isnan(np.asarray(x, float))
    table: dict[str, dict[str, float]] = {}
    for s in cfg["residual"]["jump_sky"]:
        by_hour: dict[str, float] = {}
        for h in np.unique(hour[usable & (sky == s)]):
            mask = usable & (sky == s) & (hour == h)
            by_hour[str(int(h))] = float(_dead_time_corrected(int(np.isin(where, np.flatnonzero(mask)).sum()),
                                                              int(mask.sum()), window))
        table[s] = by_hour
    observed = np.asarray(x, float)
    return {"lam_per_hour": float(_dead_time_corrected(len(where), int(usable.sum()), window)), "lam_by_sky_hour": table,
            "active_fraction": float(usable.sum() / max(int((~np.isnan(observed)).sum()), 1)),
            **fit_jump_sizes(size, thr)}


def jump_variance(jumps: Mapping, theta_s: float) -> float:
    """Variance that the jumps add to X_slow: lam E[J^2] / (2 theta_s), lam per minute."""
    lam = jumps["lam_per_hour"] * HOURS_PER_MIN * jumps.get("active_fraction", 1.0)
    m = jumps["jump_min"]

    def second_moment(mu):
        return m ** 2 + 2.0 * m * mu + 2.0 * mu ** 2

    p = jumps["p_drop"]
    return lam * (p * second_moment(jumps["mean_drop"]) + (1.0 - p) * second_moment(jumps["mean_rise"])) / (2.0 * theta_s)


def fit_tails(x: np.ndarray, theta_f: float, cfg: Mapping, seed: int = 0) -> dict:
    """Excess kurtosis of X_fast innovations and, if above the limit, Student-t degrees of freedom.

    Minutes around detected jumps are left out: a jump is not a heavy-tailed innovation.
    """
    r = cfg["residual"]
    x = np.asarray(x, float)
    e = x[1:] - np.exp(-theta_f) * x[:-1]
    near_jump = np.zeros(len(x), bool)
    w = int(r["jump_window_min"])
    for i in detect_jumps(x, cfg)[0]:
        near_jump[max(i - w, 0):i + w + 1] = True
    e = e[np.isfinite(e) & ~near_jump[1:]]
    excess = float(stats.kurtosis(e, fisher=True))
    if excess <= r["kurtosis_limit"]:
        return {"excess_kurtosis": excess, "nu": None}
    rng = np.random.default_rng(seed)
    sample = rng.choice(e, size=min(r["tail_sample"], len(e)), replace=False)
    return {"excess_kurtosis": excess, "nu": float(stats.t.fit(sample, floc=0.0)[0])}


def fit_regime(x: np.ndarray, acov: Mapping, is_day, sky_class, hour, cfg: Mapping) -> dict:
    """Complete parameter set for one regime, starting from its Phase 3 autocovariance row.

    The autocovariance of X_slow includes the variance the jumps produce. That part is taken out
    before sigma_s is computed, or the simulator would count it twice.
    """
    jumps = fit_jumps(x, is_day, sky_class, hour, cfg)
    without = {**acov, "A_s": max(acov["A_s"] - jump_variance(jumps, acov["theta_s"]), 0.0)}
    p = ou_params(without, cfg)
    return {**p, **jumps, **fit_tails(x, p["theta_f"], cfg)}


def validate_against_floor(simulated: np.ndarray, empirical: np.ndarray, cfg: Mapping) -> dict:
    """Two-sample KS comparison of simulated and empirical e_H (or e_L)."""
    ks = float(stats.ks_2samp(np.asarray(simulated).ravel(), np.asarray(empirical).ravel()).statistic)
    return {"ks": ks, "passed": ks < cfg["residual"]["ks_limit"]}
