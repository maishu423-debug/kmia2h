"""Settlement target (SPEC.md Phase 1). Every later phase calls these functions.

reported(): rolling mean of the last 5 whole-degF one-minute values (NaN if < 4 valid), rounded.
settle():   max / min of reported() over the last w minutes of a (..., 4 + w) array.
"""
from __future__ import annotations

import numpy as np

from kmia2h.config import CFG

RULES = ("half_up", "half_even", "truncate", "via_c")


def round_rule(x, rule: str):
    """Round degF values under a named rule. NaN stays NaN. Works on arrays and pandas objects."""
    is_pd = hasattr(x, "to_numpy") and hasattr(x, "index")
    a = np.asarray(x, dtype=float)
    if rule == "half_up":
        r = np.floor(a + 0.5)
    elif rule == "half_even":
        r = np.round(a)
    elif rule == "truncate":
        r = np.trunc(a)
    elif rule == "via_c":
        c = np.floor((a - 32.0) * 5.0 / 9.0 * 10.0 + 0.5) / 10.0
        r = np.floor(c * 9.0 / 5.0 + 32.0 + 0.5 + 1e-9)
    else:
        raise ValueError(f"unknown rounding rule {rule!r}")
    if is_pd:
        return x.__class__(r, index=x.index) if r.ndim == 1 else r
    return r


def rolling_mean(t1: np.ndarray, window: int | None = None, min_valid: int | None = None) -> np.ndarray:
    """Trailing mean of the last `window` values along the last axis, NaN if < min_valid valid."""
    window = window or CFG.target.roll_window
    min_valid = min_valid or CFG.target.roll_min_valid
    a = np.asarray(t1, dtype=float)
    valid = ~np.isnan(a)
    pad = [(0, 0)] * (a.ndim - 1) + [(1, 0)]
    cs = np.cumsum(np.pad(np.where(valid, a, 0.0), pad), axis=-1)
    cv = np.cumsum(np.pad(valid.astype(np.int32), pad), axis=-1)
    idx = np.arange(1, a.shape[-1] + 1)
    lo = np.maximum(idx - window, 0)
    s, c = cs[..., idx] - cs[..., lo], cv[..., idx] - cv[..., lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = s / c
    return np.where(c >= min_valid, m, np.nan)


def reported(t1: np.ndarray, rule: str) -> np.ndarray:
    """t1: (..., n) one-minute degF, NaN = missing. Rolling mean of the last 5 values
    (NaN if < 4 valid), rounded by `rule`. Output aligned to the last minute."""
    return round_rule(rolling_mean(t1), rule)


def settle(t1: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """t1: (..., 4 + w) = minutes t-3 .. t+w for origin t (w = 120 normally).
    Returns (high, low) = max / min of reported() over minutes t+1 .. t+w."""
    lead = CFG.target.lead_min
    r = reported(t1, rule)[..., lead:]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(r, axis=-1), np.nanmin(r, axis=-1)


def edge(x):
    """|frac(x) - 0.5| of an unrounded extreme: 0 = on a rounding boundary, 0.5 = bin centre."""
    return np.abs(np.asarray(x, dtype=float) % 1 - 0.5)
