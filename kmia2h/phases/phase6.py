"""Phase 6 validation: simulate the fitted residual on the true smooth path and compare e_H, e_L
with the Phase 3 empirical distribution per regime (two-sample KS)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import kmia2h.target as T
from kmia2h.backtest import windows
from kmia2h.config import CFG, out, path, store
from kmia2h.forcing import model_cfg
from kmia2h.residual import residual_sim, table as rtable


def main(n_per=1500, seed=0):
    cfg = model_cfg()
    tab = rtable.load(cfg)
    w = windows.load("v1", "2022-01", "2023-12")
    ft = pd.read_parquet(store("floor_table_L21.parquet"), columns=["eH", "eL", "season", "is_day", "sky_class", "wind"])
    ft = ft[ft.index < pd.Timestamp("2024-01-01", tz="UTC")]
    rng = np.random.default_rng(seed)
    good = w["y_good"] & np.isfinite(w["y_m"]).all(1)
    rows = []
    key = np.char.add(np.char.add(w["g_season"].astype(str), "|"), w["g_is_day"].astype(str))
    for g in np.unique(key[good]):
        idx = np.flatnonzero(good & (key == g))
        idx = rng.choice(idx, min(n_per, len(idx)), replace=False)
        es_h, es_l = [], []
        for i in idx:
            rp = rtable.sim_params(tab, w["g_season"][i], w["g_is_day"][i], w["g_sky_class"][i], w["g_wind"][i], cfg)
            m = w["y_m"][i].astype(float)
            sim = residual_sim.simulate(rp, 4, len(m) + 4, rng)
            x = sim["x_slow"] + sim["x_fast"] + sim["white"]
            mm = np.r_[np.full(4, m[0]), m]
            paths = residual_sim.quantise(mm[None, :] + x, T, CFG.target.rounding)
            eh, el = residual_sim.floor_errors(mm[None, :].repeat(4, 0), paths, 4, 5)
            es_h += list(eh)
            es_l += list(el)
        s, dn = g.split("|")
        emp = ft[(ft.season == s) & (ft.is_day.astype(str) == dn)]
        kh = stats.ks_2samp(es_h, emp.eH).statistic
        kl = stats.ks_2samp(es_l, emp.eL).statistic
        rows.append({"regime": g, "n_sim": len(es_h), "ks_eH": kh, "ks_eL": kl, "sim_mean_eH": np.mean(es_h),
                     "emp_mean_eH": emp.eH.mean(), "sim_sd_eH": np.std(es_h), "emp_sd_eH": emp.eH.std(),
                     "sim_mean_eL": np.mean(es_l), "emp_mean_eL": emp.eL.mean(), "sim_sd_eL": np.std(es_l),
                     "emp_sd_eL": emp.eL.std(), "passed": bool(kh < cfg["residual"]["ks_limit"] and kl < cfg["residual"]["ks_limit"])})
    d = pd.DataFrame(rows)
    d.to_csv(out(6, "validation_vs_floor.csv"), index=False)
    return d


if __name__ == "__main__":
    print(main().round(3).to_string(index=False))
