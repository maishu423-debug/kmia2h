"""Batch fit of the ODE coefficients (SPEC.md Phase 4, step 5).

From the smooth observed temperature at each hourly origin of the physics-fit period, Euler-
integrate 120 minutes with the forecast-time forcing and minimise the squared error against the
smooth 1-minute path, separately for day/night x sector. Writes params/ode_prior.json with the
pooled fit (the filter's prior means), the group fits and the filter bounds."""
from __future__ import annotations

import json

import numpy as np
from scipy import optimize

from kmia2h.config import CFG, path
from kmia2h.forcing import model_cfg

NAMES = ("a", "b", "c", "kappa", "lam")
LO = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
HI = np.array([1.0e-3, 0.2, 5.0e-3, 1.0, 1.0])
SCALE = np.array([5e-5, 1e-2, 3e-4, 0.5, 0.5])


def simulate(x: np.ndarray, F: dict, t0: np.ndarray, cfg: dict, n: int | None = None) -> np.ndarray:
    """Euler paths (windows, n) for coefficient vector x = (a, b, c, kappa, lam). Forcing ratios
    (s_clr already holds a's ratio) are NOT applied here: the fit estimates each group directly."""
    a, b, c, kappa, lam = x
    sb = cfg["physics"]["sigma_sb"]
    n = n or F["t_adv"].shape[1]
    t = t0.astype(float).copy()
    out = np.empty((len(t0), n))
    for i in range(n):
        s, cc, ta, tsk, r = F["s_clr"][:, i], F["c"][:, i], F["t_adv"][:, i], F["t_sky"][:, i], F["r"][:, i]
        ks = F["kappa_scale"][:, i]
        tk, sk = (t - 32) * 5 / 9 + 273.15, (tsk - 32) * 5 / 9 + 273.15
        k_eff = np.clip(kappa * ks, 0, 1)
        dt = a * s * (1 - k_eff * cc) - b * F["b_base"][:, i] * (t - ta) - c * sb * (tk ** 4 - sk ** 4) * (1 - lam * cc) - r
        t = t + dt
        out[:, i] = t
    return out


def fit_group(F: dict, t0: np.ndarray, target: np.ndarray, cfg: dict, x0: np.ndarray) -> dict:
    ok = np.isfinite(target)

    def resid(x):
        p = simulate(x, F, t0, cfg)
        return np.where(ok, p - target, 0.0).ravel()

    r = optimize.least_squares(resid, x0, bounds=(LO, HI), x_scale=SCALE, method="trf", max_nfev=200)
    rmse = float(np.sqrt(np.mean(r.fun[ok.ravel()] ** 2)))
    return {**{k: float(v) for k, v in zip(NAMES, r.x)}, "rmse": rmse, "n": int(len(t0)),
            "converged": bool(r.success), "status": int(r.status), "nfev": int(r.nfev)}


def prepare(w: dict, mask: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Raw cached forcing (no group ratios); b_base is the sea-breeze ratio."""
    F = {}
    for c in ("s_clr", "c", "t_adv", "t_sky", "r", "kappa_scale"):
        F[c] = w[f"f_{c}"][mask].astype(float)          # raw cache: kappa_scale = kappa_rel
    F["onset"] = np.abs(w["f_b_scale"][mask].astype(float) - 1.0) > 1e-9   # minutes after sea-breeze onset
    F["b_base"] = np.ones_like(F["t_adv"])               # group fits ignore the switch; b_onshore is fitted below
    m = w["y_m"][mask].astype(float)
    return F, m[:, 0], m[:, 1:]


def main(w: dict, out_name: str = "ode_prior.json") -> dict:
    cfg = model_cfg()
    o = w["origins"]
    hourly = (o.minute == 0)
    base = hourly & np.isfinite(w["y_m"]).all(1) & np.isfinite(w["f_t_adv"]).all(1) & np.isfinite(w["f_t_sky"]).all(1)
    x0 = np.array([CFG.physics.prior[k] for k in NAMES])
    F, t0, y = prepare(w, base)
    Fs = {k: v[:, 1:] for k, v in F.items()}
    pooled = fit_group(Fs, t0, y, cfg, x0)
    x_pooled = np.array([pooled[k] for k in NAMES])
    groups = {}
    for dn in ("True", "False"):
        for sect in ("onshore", "offshore", "other"):
            m = base & (w["g_is_day"] == dn) & (w["g_sector"] == sect)
            if m.sum() < 50:
                continue
            Fg, t0g, yg = prepare(w, m)
            g = fit_group({k: v[:, 1:] for k, v in Fg.items()}, t0g, yg, cfg, x_pooled)
            groups[f"{'day' if dn == 'True' else 'night'}|{sect}"] = g
    extra = {}
    # Thin-cloud kappa (GOES cloud class): grid the thin/thick ratio with the pooled coefficients.
    thin = F["kappa_scale"] < 0.999
    if thin.any():
        ratios = np.round(np.arange(0.1, 1.01, 0.1), 2)
        def sse(r):
            Fr = dict(Fs)
            Fr["kappa_scale"] = np.where(Fs["kappa_scale"] < 0.999, r, 1.0)
            p = simulate(x_pooled, Fr, t0, cfg)
            ok = np.isfinite(y)
            return float(np.mean((p - y)[ok] ** 2))
        errs = {float(r): sse(r) for r in ratios}
        r_best = min(errs, key=errs.get)
        extra["kappa_thin"] = float(r_best * pooled["kappa"])
        extra["kappa_thick"] = float(pooled["kappa"])
        extra["kappa_thin_ratio_mse"] = errs
    # Sea-breeze b multiplier after onset (b_base != 1 marks minutes after onset).
    onset = F["onset"]
    if onset.any():
        rows = onset.any(axis=1)
        Fo = {k: v[rows][:, 1:] for k, v in F.items()}
        on = Fo["onset"]
        g = groups.get("day|onshore", pooled)
        xg = np.array([g[k] for k in NAMES])
        mults = [1, 2, 3, 5, 8, 12, 20]
        def sse_b(mlt):
            Fr = dict(Fo)
            Fr["b_base"] = np.where(on, mlt, 1.0)
            p = simulate(xg, Fr, t0[rows], cfg)
            ok = np.isfinite(y[rows])
            return float(np.mean((p - y[rows])[ok] ** 2))
        eb = {m: sse_b(m) for m in mults}
        mb = min(eb, key=eb.get)
        extra["b_onshore"] = float(mb * max(g["b"], 1e-6))
        extra["b_onshore_mse"] = {str(k): v for k, v in eb.items()}
        extra["n_onset_windows"] = int(rows.sum())
    vals = {k: [g[k] for g in groups.values()] + [pooled[k]] for k in NAMES}
    bounds = {k: [0.0, float(min(HI[i], 3.0 * max(vals[k]) + 1e-12))] for i, k in enumerate(NAMES[:3])}
    bounds["kappa"] = [0.0, 1.0]
    res = {"pooled": pooled, "groups": groups, "bounds": bounds, "b_onshore": float(CFG.physics.prior.b_onshore),
           "fit_period": list(CFG.periods.physics_fit), "n_windows": int(base.sum()), **extra}
    if "b_onshore" in extra:
        res["bounds"]["b"][1] = float(max(res["bounds"]["b"][1], extra["b_onshore"] / max(pooled["b"], 1e-9) * pooled["b"]))
    path("params", out_name).write_text(json.dumps(res, indent=1))
    return res
