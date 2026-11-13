"""Observation operators and the observation record (SPEC.md Phase 5).

Each operator maps the ensemble (N, state) at an observation's valid minute to predicted values
(N,). The rounding in the HFMETAR and METAR chains is carried as observation error, as the spec
directs, so the operators stay continuous. `hfmetar_chain` is the exact chain, for building
synthetic observations and for checking the size of that error.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kmia2h.filter.process_model import IDX, sensor_temperature
from kmia2h.interfaces import Target


@dataclass(frozen=True)
class Obs:
    """One observation. Minute indices count from the start of the filter run."""

    valid_k: int
    available_k: int
    kind: str
    value: float


def five_minute_mean(x: np.ndarray) -> np.ndarray:
    """Mean of the member's last five one-minute sensor values."""
    lags = [x[:, IDX[k]] for k in ("T_m1", "T_m2", "T_m3", "T_m4")]
    return (sensor_temperature(x) + sum(lags)) / 5.0


def smooth_temperature(x: np.ndarray) -> np.ndarray:
    return x[:, IDX["T"]]


def cloud_factor(x: np.ndarray) -> np.ndarray:
    return x[:, IDX["C_eff"]]


OPERATORS = {
    "hfmetar": five_minute_mean,   # whole degC every 5 min
    "metar": five_minute_mean,     # tenths of degC, hourly
    "goes": cloud_factor,          # cloud fraction over the station
    "hrrr": smooth_temperature,    # 2 m temperature at the matched valid time
}


def obs_sigma(kind: str, cfg: Mapping) -> float:
    return float(cfg["filter"]["obs_sigma"][kind])


def hfmetar_chain(mean_f, ops: Target, rule: str):
    """5-minute mean -> whole degF -> whole degC -> degF, as an ASOS HFMETAR reports it."""
    whole_f = ops.round_rule(mean_f, rule)
    whole_c = ops.round_rule((np.asarray(whole_f, float) - 32.0) * 5.0 / 9.0, rule)
    return np.asarray(whole_c, float) * 9.0 / 5.0 + 32.0


def obs_from_frame(d: pd.DataFrame, start: pd.Timestamp) -> list[Obs]:
    """Records from a frame with columns ts_utc, available_at, kind, value (NaN rows are skipped).

    start is the UTC time of minute index 0. Times round up to the next whole minute for
    available_at, so an observation is never used before it exists.
    """
    d = d[d["value"].notna()]
    valid = (d["ts_utc"] - start).dt.total_seconds().to_numpy() / 60.0
    avail = (d["available_at"] - start).dt.total_seconds().to_numpy() / 60.0
    return [Obs(int(round(v)), int(np.ceil(a)), str(k), float(y))
            for v, a, k, y in zip(valid, avail, d["kind"].to_numpy(), d["value"].to_numpy())]
