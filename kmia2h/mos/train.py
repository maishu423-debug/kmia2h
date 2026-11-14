"""Quantile gradient boosting of the realised extremes (SPEC.md Phase 8, steps 1-3).

Training frame: one row per origin, with column `origin` (UTC), the target (`h_r` or `l_r`, the
unrounded realised H_r or L_r in degF) and the features listed in INTERFACES.md. Objects are
returned, never written: saving them is the caller's job.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

TARGETS = {"h_r": "q_max", "l_r": "q_min"}


@dataclass
class MosModel:
    """One booster per quantile level, all using the same ordered feature list.

    A prediction is anchor + offsets[tau] + booster output, where anchor is the physics median.
    """

    target: str
    taus: list[float]
    features: list[str]
    boosters: dict
    best_iterations: dict
    anchor_column: str
    anchor_fill: float
    offsets: dict


def physics_columns(target: str, cfg: Mapping) -> list[str]:
    prefix = TARGETS[target]
    return [f"{prefix}_{q:02d}" for q in cfg["mos"]["physics_levels"]]


def feature_list(df: pd.DataFrame, target: str, cfg: Mapping) -> list[str]:
    """Physics quantiles first, then the configured columns, then regime_* and avail_* columns."""
    missing = [c for c in physics_columns(target, cfg) if c not in df.columns]
    if missing:
        raise ValueError(f"missing physics quantile columns {missing}")
    extra = [c for c in sorted(df.columns) if any(c.startswith(p) for p in cfg["mos"]["group_prefixes"])]
    return physics_columns(target, cfg) + list(cfg["mos"]["feature_columns"]) + extra


def matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Feature matrix in a fixed order. A column that is absent is all NaN, which LightGBM handles."""
    return df.reindex(columns=features).to_numpy(float)


def monotone_constraints(features: list[str], target: str, cfg: Mapping) -> list[int]:
    physics = set(physics_columns(target, cfg))
    return [1 if f in physics else 0 for f in features]


def time_folds(times: pd.Series, k: int, embargo_min: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """k contiguous validation blocks. Training rows within the embargo of a block are dropped."""
    order = np.argsort(times.to_numpy())
    sorted_t = pd.DatetimeIndex(times.iloc[order])
    embargo = pd.Timedelta(minutes=embargo_min)
    folds = []
    for block in np.array_split(np.arange(len(order)), k):
        start, end = sorted_t[block[0]], sorted_t[block[-1]]
        outside = (sorted_t < start - embargo) | (sorted_t > end + embargo)
        folds.append((order[np.flatnonzero(outside)], order[block]))
    return folds


def _objective(tau: float):
    """Pinball gradient with a unit Hessian.

    LightGBM refuses monotone constraints on its built-in quantile objective, because that
    objective re-estimates leaf values after the constraints are applied. A custom objective keeps
    them, which is what the spec asks for.
    """
    def objective(preds, data):
        error = data.get_label() - preds
        return np.where(error > 0, -tau, 1.0 - tau), np.ones_like(preds)

    return objective


def _metric(tau: float):
    def metric(preds, data):
        return "pinball", pinball(data.get_label(), preds, tau), False

    return metric


def _params(tau: float, monotone: list[int], cfg: Mapping) -> dict:
    return {**cfg["mos"]["params"], "objective": _objective(tau), "metric": "None", "monotone_constraints": monotone}


def _dataset(x, y, init) -> lgb.Dataset:
    return lgb.Dataset(x, y, init_score=init, free_raw_data=False)


def _best_iteration(x, y, init, tau, monotone, folds, cfg: Mapping) -> int:
    m = cfg["mos"]
    rounds = []
    for tr, va in folds:
        booster = lgb.train(_params(tau, monotone, cfg), _dataset(x[tr], y[tr], init[tr]), num_boost_round=m["max_rounds"],
                            valid_sets=[_dataset(x[va], y[va], init[va])], feval=_metric(tau),
                            callbacks=[lgb.early_stopping(m["early_stopping_rounds"], verbose=False)])
        rounds.append(booster.best_iteration or m["max_rounds"])
    return max(int(np.median(rounds)), 1)


def _cv_loss(x, y, anchor, offsets_src, monotone, folds, params_cfg, taus) -> float:
    """Mean out-of-fold pinball over the selection quantiles, early-stopped inside each fold."""
    losses = []
    for tau in taus:
        for tr, va in folds:
            off = float(np.quantile(offsets_src[tr], tau))
            init = anchor + off
            booster = lgb.train(_params(tau, monotone, params_cfg), _dataset(x[tr], y[tr], init[tr]),
                                num_boost_round=params_cfg["mos"]["max_rounds"],
                                valid_sets=[_dataset(x[va], y[va], init[va])], feval=_metric(tau),
                                callbacks=[lgb.early_stopping(params_cfg["mos"]["early_stopping_rounds"], verbose=False)])
            q = init[va] + booster.predict(x[va], num_iteration=booster.best_iteration)
            losses.append(pinball(y[va], q, tau) * len(va))
    return float(np.sum(losses) / (len(taus) * len(y)))


def select_params(df: pd.DataFrame, target: str, cfg: Mapping) -> tuple[dict, list[dict]]:
    """Regularisation by time-blocked CV on the training rows only (REBUILD pre-Phase-9).

    Each candidate in cfg.mos.regularise.grid overrides cfg.mos.params; the one with the lowest
    out-of-fold pinball (mean over regularise.select_taus) wins. Folds are contiguous time blocks
    with an embargo of cfg.mos.embargo_min, so overlapping windows never straddle train and test.
    """
    import copy
    reg = cfg["mos"]["regularise"]
    d = df[df[target].notna()].reset_index(drop=True)
    features = feature_list(d, target, cfg)
    x, y = matrix(d, features), d[target].to_numpy(float)
    monotone = monotone_constraints(features, target, cfg)
    folds = time_folds(d["origin"], cfg["mos"]["cv_folds"], cfg["mos"]["embargo_min"])
    anchor_column = f"{TARGETS[target]}_50"
    anchor = anchor_values(d, anchor_column, float(d[anchor_column].mean()))
    table = []
    for cand in reg["grid"]:
        c = copy.deepcopy(dict(cfg))
        c["mos"] = {**cfg["mos"], "params": {**cfg["mos"]["params"], **cand}}
        loss = _cv_loss(x, y, anchor, y - anchor, monotone, folds, c, reg["select_taus"])
        table.append({**cand, "cv_pinball": loss})
    best = min(table, key=lambda r: r["cv_pinball"])
    return {k: v for k, v in best.items() if k != "cv_pinball"}, table


def anchor_values(df: pd.DataFrame, column: str, fill: float) -> np.ndarray:
    return df[column].fillna(fill).to_numpy(float)


def train(df: pd.DataFrame, target: str, cfg: Mapping) -> MosModel:
    """Time-blocked CV picks the number of rounds per quantile, then each quantile is refit on all rows."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {list(TARGETS)}")
    d = df[df[target].notna()].reset_index(drop=True)
    features = feature_list(d, target, cfg)
    x, y = matrix(d, features), d[target].to_numpy(float)
    monotone = monotone_constraints(features, target, cfg)
    m = cfg["mos"]
    selection = None
    fixed = (m.get("params_by_target") or {}).get(target)
    if fixed:  # frozen outcome of an earlier time-blocked selection
        m = {**m, "params": {**m["params"], **fixed}}
        cfg = {**dict(cfg), "mos": m}
    elif m.get("regularise", {}).get("enabled", False):
        chosen, selection = select_params(d, target, cfg)
        m = {**m, "params": {**m["params"], **chosen}}
        cfg = {**dict(cfg), "mos": m}
    folds = time_folds(d["origin"], m["cv_folds"], m["embargo_min"])
    anchor_column = f"{TARGETS[target]}_50"
    fill = float(d[anchor_column].mean())
    anchor = anchor_values(d, anchor_column, fill)
    boosters, best, offsets = {}, {}, {}
    for tau in m["quantiles"]:
        offsets[tau] = float(np.quantile(y - anchor, tau))
        init = anchor + offsets[tau]
        best[tau] = _best_iteration(x, y, init, tau, monotone, folds, cfg)
        boosters[tau] = lgb.train(_params(tau, monotone, cfg), _dataset(x, y, init), num_boost_round=best[tau])
    model = MosModel(target, list(m["quantiles"]), features, boosters, best, anchor_column, fill, offsets)
    model.params = dict(m["params"])
    model.selection = selection
    return model


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Mean pinball loss of quantile predictions q at level tau."""
    e = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))
