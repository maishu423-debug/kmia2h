"""Test support: the settlement rule copied from SPEC.md Phase 1, a config loader and stubs.

Passing this module as `ops=` satisfies kmia2h.interfaces.Target.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROLL_WINDOW = 5
ROLL_MIN_VALID = 4
LEAD_MIN = 4
FRAMEWORK_CONFIG = Path(__file__).resolve().parents[1] / "config.framework.yaml"

# The keys of the main repo's config.yaml that the framework reads.
MAIN_KEYS = {
    "tz_lst_offset_hours": -5,
    "neighbours": ["KOPF", "KFLL", "KHWO", "KTMB", "KHST", "KFXE", "KPMP"],
    "target": {"rounding": "half_up", "roll_window": ROLL_WINDOW, "roll_min_valid": ROLL_MIN_VALID},
    "sim": {"paths_backtest": 2000, "paths_live": 4000, "horizon_min": 120, "lead_min": LEAD_MIN, "seed": 20261109},
    "enkf": {"members": 300},
    "features": {"sector_onshore": [30, 150], "sector_offshore": [210, 330], "wet_months": [5, 6, 7, 8, 9, 10]},
}


def load_cfg() -> dict:
    with open(FRAMEWORK_CONFIG, encoding="utf8") as f:
        return {**MAIN_KEYS, **yaml.safe_load(f)}


def round_rule(x, rule: str):
    """half_up is floor(x + 0.5). NaN stays NaN."""
    if rule != "half_up":
        raise ValueError(f"only half_up is copied for tests, got {rule!r}")
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def rolling_mean(t1: np.ndarray) -> np.ndarray:
    a = np.asarray(t1, dtype=float)
    valid = ~np.isnan(a)
    pad = [(0, 0)] * (a.ndim - 1) + [(1, 0)]
    cs = np.cumsum(np.pad(np.where(valid, a, 0.0), pad), axis=-1)
    cv = np.cumsum(np.pad(valid.astype(np.int32), pad), axis=-1)
    idx = np.arange(1, a.shape[-1] + 1)
    lo = np.maximum(idx - ROLL_WINDOW, 0)
    s, c = cs[..., idx] - cs[..., lo], cv[..., idx] - cv[..., lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = s / c
    return np.where(c >= ROLL_MIN_VALID, m, np.nan)


def reported(t1: np.ndarray, rule: str) -> np.ndarray:
    return round_rule(rolling_mean(t1), rule)


def settle(t1: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray]:
    r = reported(t1, rule)[..., LEAD_MIN:]
    return np.nanmax(r, axis=-1), np.nanmin(r, axis=-1)


def edge(x):
    return np.abs(np.asarray(x, dtype=float) % 1 - 0.5)


class StubWindow:
    """A market window fixed in minutes around any origin."""

    def __init__(self, elapsed: int = 0, remaining: int = 120):
        self.elapsed, self.remaining = elapsed, remaining

    def elapsed_min(self, origin: pd.Timestamp) -> int:
        return self.elapsed

    def remaining_min(self, origin: pd.Timestamp) -> int:
        return self.remaining
