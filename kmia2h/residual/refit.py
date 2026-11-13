"""Residual refit on observations minus the filter's mean path (REBUILD step 3, SPEC Phase 6).

X = t1 - T_filter, with t1 the QC-cleaned KMIA 1-minute value and T_filter the filter's
ensemble-mean smooth temperature at that minute (filtered, live inputs only). Per regime at the
three Phase 3 levels: two-exponential autocovariance, jumps (day SCT/BKN), tails. The jump rate is
corrected for detector efficiency by simulation, and the step-check threshold becomes the 99.9th
percentile of the fitted jump sizes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kmia2h.timeutil import from_ns
from kmia2h.config import CFG, path, store
from kmia2h.data.qc import clean
from kmia2h.floor.noise_floor import LEVELS
from kmia2h.residual import residual_fit, residual_sim

QUANT = 1.0 / 12.0


def load_X(tag: str, first: str, last: str) -> pd.DataFrame:
    months = pd.period_range(first, last, freq="M")
    parts = []
    for p in months:
        f = store(f"backtest/{tag}/{p.year}{p.month:02d}.npz")
        if not f.exists():
            continue
        d = np.load(f)
        idx = from_ns(d["minutes"])
        parts.append(pd.DataFrame({"T": d["rec_T"], "Y": d["rec_Y"], "ok": d["forcing_ok"]}, index=idx))
    r = pd.concat(parts)
    k = pd.read_parquet(store("kmia_1min.parquet"), filters=[("ts_utc", ">=", r.index[0]), ("ts_utc", "<=", r.index[-1])])
    t1 = clean(k).reindex(r.index)
    f10 = pd.read_parquet(store("features_10m.parquet"), columns=["season", "is_day", "sky_class", "wind", "hour_lst"])
    reg = f10.reindex(r.index, method="ffill")
    X = pd.DataFrame({"x": (t1 - r["T"]).where(r.ok), "season": reg.season, "is_day": reg.is_day.astype(str),
                      "sky_class": reg.sky_class, "wind": reg.wind, "hour": reg.hour_lst}, index=r.index)
    return X


def segments(X: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    key = X[cols].astype(str).agg("|".join, axis=1).to_numpy()
    day = (X.index + pd.Timedelta(hours=CFG.tz_lst_offset_hours)).normalize().astype("int64").to_numpy()
    codes, uniq = pd.factorize(key)
    seg = pd.factorize(pd.Series(codes).astype(str) + "_" + pd.Series(day).astype(str))[0]
    return codes, uniq, seg


def acov_by_regime(x: np.ndarray, codes: np.ndarray, seg: np.ndarray, n_reg: int, maxlag: int) -> tuple:
    seg = np.where(np.isnan(x), -1, seg)
    mean = np.bincount(codes[seg >= 0], weights=x[seg >= 0], minlength=n_reg) / np.maximum(
        np.bincount(codes[seg >= 0], minlength=n_reg), 1)
    xd = np.where(seg >= 0, x - mean[codes], 0.0)
    c = np.zeros((n_reg, maxlag + 1))
    n0 = None
    for lag in range(maxlag + 1):
        a, b = xd[:len(xd) - lag], xd[lag:]
        ok = (seg[:len(seg) - lag] == seg[lag:]) & (seg[lag:] >= 0)
        s = np.bincount(codes[:len(codes) - lag][ok], weights=(a * b)[ok], minlength=n_reg)
        n = np.bincount(codes[:len(codes) - lag][ok], minlength=n_reg)
        with np.errstate(invalid="ignore", divide="ignore"):
            c[:, lag] = s / n
        if lag == 0:
            n0 = n
    return c, n0, mean


def separated(x: np.ndarray, seg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Values of one regime concatenated with a NaN gap between segments, so the jump detector
    never sees an artificial step at a boundary."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.array([])
    brk = np.r_[False, (np.diff(idx) > 1) | (np.diff(seg[idx]) != 0)]
    gap = int(CFG.residual.jump_window_min) + 1
    pos = np.arange(idx.size) + np.cumsum(brk) * gap
    out = np.full(pos[-1] + 1, np.nan)
    out[pos] = x[idx]
    return out, pos, idx


def efficiency(rp: dict, cfg, rng, minutes: int = 60_000) -> float:
    """Detected / true jump rate for a synthetic series with the regime's fitted parameters."""
    if rp["lam_per_hour"] <= 0:
        return 1.0
    n_paths, n_steps = 20, minutes // 20
    sim = residual_sim.simulate(rp, n_paths, n_steps, rng)
    x = sim["x_slow"] + sim["x_fast"] + sim["white"] + rng.uniform(-0.5, 0.5, (n_paths, n_steps))
    true = sim["jumps"].sum()
    det = 0
    thr = rp["jump_min"]
    for p in range(n_paths):
        det += len(residual_fit.detect_jumps(x[p], cfg, thr)[0])
    det_rate_corr = residual_fit._dead_time_corrected(det, n_paths * n_steps, int(cfg["residual"]["jump_window_min"]))
    true_rate = true / (n_paths * n_steps) * 60
    return float(np.clip(det_rate_corr / max(true_rate, 1e-9), 0.2, 5.0))


def fit(X: pd.DataFrame, cfg, min_minutes: int = 3000) -> dict:
    rng = np.random.default_rng(int(cfg["sim"]["seed"]))
    x = X.x.to_numpy(float)
    table, diag = {}, {}
    maxlag = int(cfg["residual"]["acf_maxlag"])
    for cols in LEVELS:
        codes, uniq, seg = segments(X, cols)
        c, n0, mean = acov_by_regime(x, codes, seg, len(uniq), maxlag)
        for i, g in enumerate(uniq):
            if g in table or n0[i] < min_minutes or "nan" in g or "unk" in g:
                continue
            try:
                ac = residual_fit.fit_two_exponentials(c[i], cfg)
            except Exception as e:
                diag[g] = {"error": repr(e)[:100]}
                continue
            m = (codes == i) & np.isfinite(x)
            xs, pos, idx = separated(x, seg, codes == i)
            is_day = np.zeros(len(xs), bool)
            sky = np.full(len(xs), "unk", dtype=object)
            hour = np.zeros(len(xs), int)
            is_day[pos] = X.is_day.to_numpy()[idx] == "True"
            sky[pos] = X.sky_class.to_numpy()[idx]
            hour[pos] = X.hour.to_numpy()[idx]
            p = residual_fit.fit_regime(xs, ac, is_day, sky, hour, cfg)
            if p.get("nu") is not None:
                # Student-t with nu <= 2 has no variance; the unit-variance scaling needs nu > 2.
                p["nu_fitted"] = p["nu"]
                p["nu"] = float(max(p["nu"], cfg["residual"].get("nu_min", 3.0)))
            raw_lam = p["lam_per_hour"]
            if raw_lam > 0:
                eff = efficiency({**p, "white": p["white"]}, cfg, rng)
                p["lam_per_hour"] = raw_lam / eff
                p["lam_by_sky_hour"] = {s: {h: v / eff for h, v in hh.items()} for s, hh in p["lam_by_sky_hour"].items()}
                p["jump_efficiency"] = eff
            p.update({"mean": float(mean[i]), "var0": float(c[i, 0]), "n_minutes": int(n0[i]),
                      "A_f": ac["A_f"], "A_s": ac["A_s"], "acov": [float(v) for v in c[i]]})
            table[g] = p
            diag[g] = {"n": int(n0[i]), "theta_f": p["theta_f"], "theta_s": p["theta_s"], "sigma_f": p["sigma_f"],
                       "sigma_s": p["sigma_s"], "white": p["white"], "lam": p["lam_per_hour"], "nu": p["nu"],
                       "mean": p["mean"]}
    return table, diag


def jump_q999(table: dict) -> float:
    """99.9th percentile of |J| under the fitted size distributions, pooled over active regimes."""
    rng = np.random.default_rng(0)
    sizes = []
    for p in table.values():
        if p.get("lam_per_hour", 0) > 0:
            sizes.append(np.abs(residual_sim.jump_sizes(rng, 20000, p["p_drop"], p["mean_drop"], p["mean_rise"],
                                                        p["jump_min"])))
    return float(np.quantile(np.concatenate(sizes), 0.999)) if sizes else float(CFG.simulate.max_step_f)


def main(tag: str, first: str, last: str, cfg) -> dict:
    X = load_X(tag, first, last)
    table, diag = fit(X, cfg)
    q = jump_q999(table)
    res = {"regimes": table, "fit_period": [first, last], "source": f"t1 - filter mean path ({tag})",
           "step_check_f": q}
    path("params", "residual.json").write_text(json.dumps(res, indent=1, default=float))
    return res, diag, X
