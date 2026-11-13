"""Phase 4: batch fit, deterministic mean path vs persistence+trend on 2024, and the ablation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kmia2h.backtest import windows
from kmia2h.config import CFG, out, path, store
from kmia2h.forcing import load_ode_prior, model_cfg
from kmia2h.physics import batch_fit
from kmia2h.target import rolling_mean

TERMS = {"none": (), "cloud": ("cloud",), "advection": ("advection",), "rain": ("rain",), "longwave": ("longwave",)}


def scaled(w: dict, mask: np.ndarray, prior: dict) -> dict:
    """Model forcing with day/night x sector ratios applied minute by minute."""
    F = {c: w[f"f_{c}"][mask].astype(float) for c in windows.FCOLS}
    # the group of the origin applies to the whole window, as in the batch fit
    is_day = np.repeat((w["g_is_day"][mask] == "True")[:, None], F["t_adv"].shape[1], 1)
    sect = w["g_sector"][mask]
    pooled = prior["pooled"]
    for name, col in (("a", "s_clr"), ("b", "b_scale"), ("kappa", "kappa_scale"), ("c", "c_scale")):
        ratio = np.ones_like(F["t_adv"])
        for key, g in prior["groups"].items():
            dn, s = key.split("|")
            m = (sect == s)[:, None] & (is_day == (dn == "day"))
            ratio = np.where(m, g[name] / pooled[name] if pooled[name] > 0 else 1.0, ratio)
        F[col] = F[col] * ratio
    return F


def paths(F: dict, t0: np.ndarray, cfg: dict, coef: dict, off: tuple = ()) -> np.ndarray:
    sb = cfg["physics"]["sigma_sb"]
    t = t0.astype(float).copy()
    n = F["t_adv"].shape[1]
    out_ = np.empty((len(t0), n))
    for i in range(n):
        c = 0.0 if "cloud" in off else F["c"][:, i]
        k = np.clip(coef["kappa"] * F["kappa_scale"][:, i], 0, 1)
        tk, sk = (t - 32) * 5 / 9 + 273.15, (F["t_sky"][:, i] - 32) * 5 / 9 + 273.15
        solar = coef["a"] * F["s_clr"][:, i] * (1 - k * c)
        adv = 0.0 if "advection" in off else -coef["b"] * F["b_scale"][:, i] * (t - F["t_adv"][:, i])
        lw = 0.0 if "longwave" in off else -coef["c"] * F["c_scale"][:, i] * sb * (tk ** 4 - sk ** 4) * (1 - coef["lam"] * c)
        rain = 0.0 if "rain" in off else -F["r"][:, i]
        t = t + solar + adv + lw + rain
        out_[:, i] = t
    return out_


def extremes(p: np.ndarray, t_lead: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lead = CFG.sim.lead_min
    x = np.concatenate([np.repeat(t_lead[:, None], lead, 1), p], 1)
    r = rolling_mean(x)[:, lead:]
    return r.max(1), r.min(1)


def main():
    w = windows.load("v1", "2022-01", "2024-12")
    o = w["origins"]
    fit_mask = o < pd.Timestamp("2024-01-01", tz="UTC")
    wf = {k: (v[fit_mask] if hasattr(v, "__len__") and len(v) == len(o) else v) for k, v in w.items()}
    res = batch_fit.main(wf)
    print(json.dumps({k: res[k] for k in ("pooled", "bounds")}, indent=1))
    for g, v in res["groups"].items():
        print(g, {k: (round(x, 7) if isinstance(x, float) else x) for k, x in v.items()})
    prior = load_ode_prior()
    cfg = model_cfg()
    coef = {k: prior["pooled"][k] for k in ("a", "b", "c", "kappa", "lam")}
    floor = json.loads(path("params", "floor.json").read_text())
    rows, abl = [], []
    for period, m in (("fit 2022-2023", fit_mask), ("validation 2024", ~fit_mask)):
        ok = m & w["y_good"] & np.isfinite(w["s_t_last"]) & np.isfinite(w["f_t_adv"]).all(1) & np.isfinite(w["f_t_sky"]).all(1)
        F = scaled(w, ok, prior)
        t0 = w["s_t_last"][ok]
        tr_m, h_r, l_r = w["y_m"][ok], w["y_h_r"][ok], w["y_l_r"][ok]
        t1 = w["y_t1"][ok][:, CFG.sim.lead_min:]
        season = w["g_season"][ok]
        day = w["g_is_day"][ok]
        grp = np.char.add(np.char.add(season.astype(str), "|"), day.astype(str))
        gap_h = np.array([floor.get(g, {}).get("eH", {}).get("median", 0.0) for g in grp])
        gap_l = np.array([floor.get(g, {}).get("eL", {}).get("median", 0.0) for g in grp])
        # baseline: persistence + 30-minute trend held for 60 minutes
        fe = pd.read_parquet(store("features_30m.parquet"), columns=["trend_30"]).reindex(w["origins"][ok])
        b = np.nan_to_num(fe.trend_30.to_numpy() / 30.0)
        s = np.arange(1, 121)
        base = t0[:, None] + b[:, None] * np.minimum(s, 60)[None, :]
        bh, bl = np.maximum(t0, t0 + 60 * b), np.minimum(t0, t0 + 60 * b)
        for name, off in TERMS.items():
            p = paths(F, t0, cfg, coef, off)
            mh, ml = extremes(p, t0)
            mh, ml = mh + gap_h, ml + gap_l
            if name == "none":
                bias_h, bias_l = np.nanmedian(h_r - mh), np.nanmedian(l_r - ml)
                rows.append({"period": period, "model": "ODE mean path", "n": int(ok.sum()),
                             "rmse_path_vs_m": float(np.sqrt(np.nanmean((p - tr_m) ** 2))),
                             "rmse_path_vs_t1": float(np.sqrt(np.nanmean((p - t1) ** 2))),
                             "rmse_H": float(np.sqrt(np.nanmean((mh - h_r) ** 2))),
                             "rmse_L": float(np.sqrt(np.nanmean((ml - l_r) ** 2))),
                             "bias_H": float(np.nanmean(mh - h_r)), "bias_L": float(np.nanmean(ml - l_r))})
            for g in np.unique(grp):
                gm = grp == g
                abl.append({"period": period, "removed": name, "group": g, "n": int(gm.sum()),
                            "rmse_H": float(np.sqrt(np.nanmean((mh[gm] - h_r[gm]) ** 2))),
                            "rmse_L": float(np.sqrt(np.nanmean((ml[gm] - l_r[gm]) ** 2)))})
        bgap_h = np.nanmedian(h_r - bh) if period.startswith("fit") else base_gap[0]
        bgap_l = np.nanmedian(l_r - bl) if period.startswith("fit") else base_gap[1]
        base_gap = (bgap_h, bgap_l)
        rows.append({"period": period, "model": "persistence + trend", "n": int(ok.sum()),
                     "rmse_path_vs_m": float(np.sqrt(np.nanmean((base - tr_m) ** 2))),
                     "rmse_path_vs_t1": float(np.sqrt(np.nanmean((base - t1) ** 2))),
                     "rmse_H": float(np.sqrt(np.nanmean((bh + bgap_h - h_r) ** 2))),
                     "rmse_L": float(np.sqrt(np.nanmean((bl + bgap_l - l_r) ** 2))),
                     "bias_H": float(np.nanmean(bh + bgap_h - h_r)), "bias_L": float(np.nanmean(bl + bgap_l - l_r))})
    r = pd.DataFrame(rows)
    a = pd.DataFrame(abl)
    r.to_csv(out(4, "mean_path_vs_baseline.csv"), index=False)
    a.to_csv(out(4, "ablation.csv"), index=False)
    print(r.to_string(index=False))
    piv = a[a.period.str.startswith("valid")].pivot_table(index="group", columns="removed", values=["rmse_H", "rmse_L"])
    print(piv.round(3).to_string())
    return r, a


if __name__ == "__main__":
    main()
