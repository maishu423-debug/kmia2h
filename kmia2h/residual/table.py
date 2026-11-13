"""Per-regime residual parameters shared by the filter and the simulator (REBUILD step 3: one set
of noise settings). params/residual.json if fitted, else derived from params/residual_acov.json."""
from __future__ import annotations

import json

import numpy as np

from kmia2h.config import path
from kmia2h.residual import residual_fit

DEFAULTS = {"lam_per_hour": 0.0, "p_drop": 0.5, "mean_drop": 0.0, "mean_rise": 0.0, "jump_min": 1.0, "nu": None}


def load(cfg) -> dict:
    """Regime table. residual.sigma_scale (tuned on 2024, params/filter.json) scales the OU noise
    of both components; the filter and the simulator read the same table."""
    p = path("params", "residual.json")
    if p.exists():
        t = json.loads(p.read_text())["regimes"]
    else:
        ac = json.loads(path("params", "residual_acov.json").read_text())
        t = {k: {**DEFAULTS, **residual_fit.ou_params(v, cfg)} for k, v in ac.items()}
    k = float(cfg["residual"].get("sigma_scale", 1.0))
    if k != 1.0:
        t = {g: {**v, "sigma_f": v["sigma_f"] * k, "sigma_s": v["sigma_s"] * k} for g, v in t.items()}
    return t


def lookup(table: dict, season: str, is_day, sky: str, wind: str) -> dict:
    day = str(bool(is_day) if not isinstance(is_day, str) else is_day == "True")
    for key in (f"{season}|{day}|{sky}|{wind}", f"{season}|{day}|{sky}", f"{season}|{day}"):
        if key in table:
            return table[key]
    raise KeyError(f"{season}|{day}")


def sim_params(table: dict, season, is_day, sky, wind, cfg) -> dict:
    """Simulator parameters at an origin. Jumps only by day under SCT/BKN."""
    r = dict(lookup(table, season, is_day, sky, wind))
    active = (str(is_day) == "True" or is_day is True) and sky in cfg["residual"]["jump_sky"]
    if not active:
        r["lam_per_hour"] = 0.0
    if r.get("nu") is not None and not np.isfinite(r["nu"]):
        r["nu"] = None
    return r
