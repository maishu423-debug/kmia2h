"""From predicted quantiles to bin probabilities (SPEC.md Phase 8, step 4)."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from kmia2h.mos.train import MosModel, anchor_values, matrix


def predict_quantiles(model: MosModel, df: pd.DataFrame) -> np.ndarray:
    """(rows, taus) predicted quantiles, sorted along each row so none cross."""
    x = matrix(df, model.features)
    anchor = anchor_values(df, model.anchor_column, model.anchor_fill)
    q = np.column_stack([anchor + model.offsets[t] + model.boosters[t].predict(x, raw_score=True) for t in model.taus])
    return np.sort(q, axis=1)


def _strictly_increasing(q: np.ndarray, cfg: Mapping) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(q, float)) + cfg["mos"]["quantile_eps"] * np.arange(len(q))


def cdf(x, taus, q, cfg: Mapping) -> np.ndarray:
    """Piecewise-linear CDF between the quantiles, with exponential tails beyond the outer ones.

    Each tail is the exponential that passes through the outer two quantiles of its side.
    """
    t, qs = np.asarray(taus, float), _strictly_increasing(q, cfg)
    floor = cfg["mos"]["min_tail_scale_f"]
    lower = max((qs[1] - qs[0]) / np.log(t[1] / t[0]), floor)
    upper = max((qs[-1] - qs[-2]) / np.log((1.0 - t[-2]) / (1.0 - t[-1])), floor)
    x = np.asarray(x, float)
    out = np.interp(x, qs, t)
    out = np.where(x < qs[0], t[0] * np.exp(np.minimum((x - qs[0]) / lower, 0.0)), out)
    return np.where(x > qs[-1], 1.0 - (1.0 - t[-1]) * np.exp(-np.maximum(x - qs[-1], 0.0) / upper), out)


def bin_table(taus, q, cfg: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """(bins k, P(k)) with P(k) = F(k + 0.5) - F(k - 0.5), the half_up settlement bins. Sums to 1."""
    offset = cfg["mos"]["bin_offset"]
    ks = np.arange(int(np.floor(q[0])) - offset, int(np.ceil(q[-1])) + offset + 1)
    p = cdf(ks + 0.5, taus, q, cfg) - cdf(ks - 0.5, taus, q, cfg)
    return ks, p / p.sum()


def copula_rho(max_u: np.ndarray, min_u: np.ndarray, rho_max: float) -> float:
    """Correlation of normal scores of the ensemble's (max, min) pairs. 0 if either has no spread."""
    def scores(v):
        ranks = stats.rankdata(v) / (len(v) + 1.0)
        return stats.norm.ppf(ranks)

    if np.ptp(max_u) == 0 or np.ptp(min_u) == 0:
        return 0.0
    return float(np.clip(np.corrcoef(scores(max_u), scores(min_u))[0, 1], -rho_max, rho_max))


def _z_edges(p: np.ndarray, z_max: float) -> np.ndarray:
    edges = stats.norm.ppf(np.clip(np.r_[0.0, np.cumsum(p)], 0.0, 1.0))
    return np.clip(np.nan_to_num(edges, posinf=z_max, neginf=-z_max), -z_max, z_max)


def joint_table(ks_high: np.ndarray, p_high: np.ndarray, ks_low: np.ndarray, p_low: np.ndarray, rho: float,
                cfg: Mapping) -> np.ndarray:
    """Gaussian-copula joint P(high = ks_high[i], low = ks_low[j]), by deterministic quadrature.

    Cells with high below low are impossible. They are set to zero and the table is refitted to
    the two marginals, so the joint table stays consistent with the reported p_high and p_low.
    """
    m = cfg["mos"]
    r = float(np.clip(rho, -m["copula_rho_max"], m["copula_rho_max"]))
    s = np.sqrt(1.0 - r ** 2)
    nodes, weights = np.polynomial.legendre.leggauss(m["copula_nodes"])
    z_h, z_l = _z_edges(p_high, m["copula_z_max"]), _z_edges(p_low, m["copula_z_max"])
    table = np.zeros((len(p_high), len(p_low)))
    for i in range(len(p_high)):
        half = (z_h[i + 1] - z_h[i]) / 2.0
        if half <= 0:
            continue
        z = z_h[i] + half * (nodes + 1.0)
        density = half * weights * stats.norm.pdf(z)
        upper = stats.norm.cdf((z_l[1:][None, :] - r * z[:, None]) / s)
        lower = stats.norm.cdf((z_l[:-1][None, :] - r * z[:, None]) / s)
        row = (density[:, None] * (upper - lower)).sum(axis=0)
        table[i] = row * (p_high[i] / row.sum()) if row.sum() > 0 else 0.0
    table[ks_high[:, None] < ks_low[None, :]] = 0.0
    return _fit_marginals(table, p_high, p_low, int(m["copula_ipf_iterations"]))


def _fit_marginals(table: np.ndarray, p_rows: np.ndarray, p_cols: np.ndarray, iterations: int) -> np.ndarray:
    """Iterative proportional fitting. Cells that are zero stay zero."""
    t = table / table.sum()
    for _ in range(iterations):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = t * np.nan_to_num(p_rows / t.sum(axis=1), nan=0.0, posinf=0.0)[:, None]
            t = t * np.nan_to_num(p_cols / t.sum(axis=0), nan=0.0, posinf=0.0)[None, :]
    return t / t.sum()


def to_dict(ks: np.ndarray, p: np.ndarray) -> dict:
    return {str(int(k)): float(v) for k, v in zip(ks, p) if v > 0}


def joint_to_dict(ks_high: np.ndarray, ks_low: np.ndarray, table: np.ndarray) -> dict:
    return {f"{int(ks_high[i])}|{int(ks_low[j])}": float(table[i, j]) for i, j in zip(*np.nonzero(table))}


def on_bins(table: Mapping, ks: np.ndarray) -> np.ndarray:
    """A probability dict ({"82": 0.7}) as an array over integer bins ks."""
    return np.array([float(table.get(str(int(k)), 0.0)) for k in ks])


def blended_marginal(physics: Mapping, ks: np.ndarray, p_mos: np.ndarray, weight: float) -> np.ndarray:
    """w MOS + (1 - w) physics on bins ks. Physics mass outside ks moves to the nearest end bin."""
    p = on_bins(physics, ks)
    keys = np.array([int(k) for k in physics], dtype=int)
    values = np.array([float(v) for v in physics.values()])
    p[0] += values[keys < ks[0]].sum()
    p[-1] += values[keys > ks[-1]].sum()
    return weight * p_mos + (1.0 - weight) * p
