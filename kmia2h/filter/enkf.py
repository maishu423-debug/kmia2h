"""Ensemble Kalman filter with late-observation handling (SPEC.md Phase 5).

Perturbed-observation analysis, one scalar observation at a time. Members are kept for the last
lag_window_min minutes. An observation is applied at its valid minute and members are re-propagated
to now. Process noise depends only on (seed, minute), so re-propagation repeats the same noise
instead of adding more.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from kmia2h.filter.obs_operators import OPERATORS, Obs, obs_sigma
from kmia2h.filter.process_model import INFLATED, constrain, sensor_temperature


class EnKF:
    """state_fn advances members: step_fn(x, k, rng) -> x at minute k + 1."""

    def __init__(self, x0: np.ndarray, step_fn: Callable, cfg: Mapping, seed: int, k0: int = 0,
                 operators: Mapping | None = None, path_fn: Callable = sensor_temperature,
                 constrain_fn: Callable | None = None, inflated: Sequence[int] = INFLATED):
        self.cfg, self.seed, self.k = cfg, int(seed), int(k0)
        self.step_fn = step_fn
        self.operators = dict(OPERATORS if operators is None else operators)
        self.path_fn = path_fn
        self.constrain_fn = (lambda x: constrain(x, cfg)) if constrain_fn is None else constrain_fn
        self.inflated = list(inflated)
        self.inflation = float(cfg["filter"]["inflation"])
        self.lag_window = int(cfg["filter"]["lag_window_min"])
        self.path_window = int(cfg["filter"]["path_window_min"])
        self.members = {self.k: np.array(x0, dtype=float)}
        self.path = {self.k: self.path_fn(self.members[self.k]).copy()}
        self.events = 0
        self.dropped = 0
        self.innovation_log: list[dict] = []

    @property
    def state(self) -> np.ndarray:
        return self.members[self.k]

    def _rng(self, *key: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, *(int(v) for v in key)])

    def _propagate_from(self, k: int) -> np.ndarray:
        x = self.step_fn(self.members[k], k, self._rng(0, k))
        self.members[k + 1] = x
        self.path[k + 1] = self.path_fn(x).copy()
        return x

    def _trim(self) -> None:
        for k in [k for k in self.members if k < self.k - self.lag_window]:
            del self.members[k]
        for k in [k for k in self.path if k < self.k - self.path_window + 1]:
            del self.path[k]

    def advance(self) -> None:
        self._propagate_from(self.k)
        self.k += 1
        self._trim()

    def _analyse(self, x: np.ndarray, k: int, obs: Sequence[Obs]) -> np.ndarray:
        """Inflate once, then apply each observation at minute k in turn."""
        self.events += 1
        mean = x.mean(axis=0)
        x = x.copy()
        x[:, self.inflated] = mean[self.inflated] + self.inflation * (x[:, self.inflated] - mean[self.inflated])
        n = len(x)
        for j, o in enumerate(obs):
            hx = self.operators[o.kind](x)
            r = obs_sigma(o.kind, self.cfg)
            hm = hx.mean()
            ha = hx - hm
            s = ha @ ha / (n - 1) + r ** 2
            self.innovation_log.append({"valid_k": k, "kind": o.kind, "innovation": o.value - hm, "variance": s})
            eps = self._rng(1, k, self.events, j).normal(0.0, r, n)
            increment = o.value + (eps - eps.mean()) - hx
            gain = (x - x.mean(axis=0)).T @ ha / (n - 1) / s
            x = x + np.outer(increment, gain)
            self._smooth(k, ha, s, increment)
        return self.constrain_fn(x)

    def _smooth(self, k: int, ha: np.ndarray, s: float, increment: np.ndarray) -> None:
        """Apply this observation to the member paths at minutes up to and including k."""
        n = len(ha)
        for m, y in self.path.items():
            if m <= k:
                self.path[m] = y + increment * (((y - y.mean()) @ ha) / (n - 1) / s)

    def assimilate(self, obs: Sequence[Obs]) -> None:
        """Apply observations that have become available and bring members back to now."""
        by_valid: dict[int, list[Obs]] = defaultdict(list)
        for o in obs:
            if o.available_k > self.k:
                raise ValueError(f"look-ahead: observation available at {o.available_k}, filter is at {self.k}")
            if o.valid_k > self.k:
                raise ValueError(f"observation valid at {o.valid_k} is in the future of minute {self.k}")
            if o.valid_k in self.members:
                by_valid[o.valid_k].append(o)
            else:
                self.dropped += 1
        if not by_valid:
            return
        for k in range(min(by_valid), self.k + 1):
            if k in by_valid:
                self.members[k] = self._analyse(self.members[k], k, by_valid[k])
                self.path[k] = self.path_fn(self.members[k]).copy()
            if k < self.k:
                self._propagate_from(k)

    def snapshot(self) -> dict:
        """Analysis ensemble at the current minute and each member's smoothed recent path.

        paths has shape (N, m), oldest minute first, its last column being the current minute.
        """
        keys = sorted(self.path)
        return {"k": self.k, "ensemble": self.state.copy(), "paths": np.column_stack([self.path[k] for k in keys])}

    def innovations(self) -> pd.DataFrame:
        """Observed minus predicted for every observation, with its normalised value."""
        d = pd.DataFrame(self.innovation_log, columns=["valid_k", "kind", "innovation", "variance"])
        d["normalised"] = d["innovation"] / np.sqrt(d["variance"])
        return d


def spinup_start(first_origin_k: int, cfg: Mapping) -> int:
    return first_origin_k - int(cfg["filter"]["spinup_hours"] * 60)


def run(kf: EnKF, obs: Sequence[Obs], k_end: int, snapshot_at: Sequence[int] = ()) -> dict:
    """Step to k_end, assimilating each observation at the minute it becomes available."""
    by_available: dict[int, list[Obs]] = defaultdict(list)
    for o in obs:
        by_available[max(o.available_k, kf.k)].append(o)
    wanted, snaps = set(snapshot_at), {}
    if kf.k in by_available:
        kf.assimilate(by_available[kf.k])
    while kf.k < k_end:
        kf.advance()
        if kf.k in by_available:
            kf.assimilate(by_available[kf.k])
        if kf.k in wanted:
            snaps[kf.k] = kf.snapshot()
    return snaps
