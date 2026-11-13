"""Tune the GOES cloud-motion quality threshold on 2022-2023 (REBUILD step 5).

For each origin, advect the current cloud field for 30-120 min with the GOES motion vector when
its quality exceeds q, otherwise with 1.5 x the 10 m wind (HRRR at origin when present, else KMIA
METAR), and compare with the cloud fraction GOES later observes over the station. Writes
params/cloud.json {motion_q_min}."""
import json
import numpy as np
import pandas as pd
from kmia2h.config import CFG, path, store
from kmia2h.forcing import model_cfg, uv_from
from kmia2h.physics import cloud_advect

cfg = model_cfg()
g = pd.concat([pd.read_parquet(f) for f in sorted(store("goes_days").glob("202[23]*[0-9].parquet"))])
g = g[g.cf_disc.notna()].set_index("origin").sort_index()
f30 = pd.read_parquet(store("features_30m.parquet"), columns=["sknt", "drct"]).reindex(g.index)
u_w, v_w = uv_from(f30.sknt.fillna(0).to_numpy(), f30.drct.fillna(0).to_numpy())
hp = store("nwp_point.parquet")
if hp.exists():
    h = pd.read_parquet(hp)
    h = h[(h.valid_utc - h.cycle_utc) == pd.Timedelta(minutes=90)].set_index("valid_utc")
    hu = h.u10.reindex(g.index); hv = h.v10.reindex(g.index)
    u_w = np.where(hu.notna(), hu, u_w); v_w = np.where(hv.notna(), hv, v_w)
leads = [30, 60, 90, 120]
truth = {L: g.cf_disc.reindex(g.index + pd.Timedelta(minutes=L)).to_numpy() for L in leads}
fac = cfg["physics"]["cloud"]["wind_factor"]
pred_g, pred_w = [], []
for i, (o, r) in enumerate(g.iterrows()):
    field = cloud_advect.field_from_row(r, cfg)
    pw = cloud_advect.project(r.cf_disc, field, fac * u_w[i], fac * v_w[i], 120, cfg)[0]
    pg = (cloud_advect.project(r.cf_disc, field, r.motion_u, r.motion_v, 120, cfg)[0]
          if np.isfinite([r.motion_u, r.motion_v]).all() and np.hypot(r.motion_u, r.motion_v) <= 30 else pw)
    pred_g.append([pg[L - 1] for L in leads]); pred_w.append([pw[L - 1] for L in leads])
pred_g, pred_w = np.array(pred_g), np.array(pred_w)
T = np.column_stack([truth[L] for L in leads])
q = g.motion_q.fillna(0).to_numpy()
persist = np.repeat(g.cf_disc.to_numpy()[:, None], len(leads), 1)
rows = []
for thr in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.01]:
    use = (q > thr)[:, None]
    p = np.where(use, pred_g, pred_w)
    ok = np.isfinite(T) & np.isfinite(p)
    rows.append({"q_min": thr, "share_goes": float(use.mean()), "mae": float(np.abs(p - T)[ok].mean()),
                 **{f"mae_{L}": float(np.nanmean(np.abs(p[:, j] - T[:, j]))) for j, L in enumerate(leads)}})
r = pd.DataFrame(rows)
okp = np.isfinite(T)
print(r.round(4).to_string(index=False))
print("persistence mae", float(np.abs(persist - T)[okp].mean()))
best = float(r.loc[r.mae.idxmin(), "q_min"])
path("params", "cloud.json").write_text(json.dumps({"motion_q_min": best, "table": rows, "n_origins": int(len(g)),
                                                    "persistence_mae": float(np.abs(persist - T)[okp].mean())}, indent=1))
print("best", best)
