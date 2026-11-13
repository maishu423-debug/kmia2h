"""Rain drop-size regression on 2022-2023 (SPEC Phase 4 step 4.2).

For origins where rain.find_cells projects a cell over KMIA, the observed drop is the smooth
1-minute temperature at arrival minus its minimum over the following recovery time. Regress on
(max dBZ - 35, frac35); the residual std becomes drop_sigma_f. Writes params/rain.json."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kmia2h.backtest import windows
from kmia2h.config import CFG, path
from kmia2h.forcing import model_cfg
from kmia2h.physics import rain
from kmia2h.sources import _load


def main():
    cfg = model_cfg()
    w = windows.load("v1", "2022-01", "2023-12")
    o = w["origins"]
    m = _load("mrms_feat.parquet", o[0], o[-1])
    if m is None:
        return None
    m = m.set_index("origin")
    rows = []
    rc = cfg["physics"]["rain"]
    for i, t in enumerate(o):
        if t not in m.index or not w["y_good"][i]:
            continue
        r = m.loc[t]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[-1]
        r = r.to_dict()
        if not np.isfinite(np.hypot(r.get("motion_u", np.nan), r.get("motion_v", np.nan))) or np.hypot(r["motion_u"], r["motion_v"]) > 30:
            continue
        cells = rain.find_cells(r, 120, cfg)
        if len(cells) == 0:
            continue
        c = cells.sort_values("arrival_min").iloc[0]
        a = int(c.arrival_min)
        mm = w["y_m"][i].astype(float)
        if a >= 110:
            continue
        before = mm[max(a - 1, 0)]
        after = np.nanmin(mm[a:min(a + int(rc["ramp_min"] + rc["recovery_min"]), 120)])
        rows.append({"origin": t, "maxdbz": float(r[f"maxdbz_s{int(c.sector):02d}_r{int(c.ring)}"]),
                     "frac35": float(r.get(f"frac35_s{int(c.sector):02d}_r{int(c.ring)}", 0.0)),
                     "drop": float(max(before - after, 0.0))})
    d = pd.DataFrame(rows).dropna()
    X = np.column_stack([np.ones(len(d)), d.maxdbz - rc["dbz_min"], d.frac35])
    coef, *_ = np.linalg.lstsq(X, d.drop.to_numpy(), rcond=None)
    resid = d.drop.to_numpy() - X @ coef
    out = {"drop_coef": [float(v) for v in coef], "drop_sigma_f": float(resid.std()), "n": int(len(d)),
           "mean_drop": float(d.drop.mean())}
    path("params", "rain.json").write_text(json.dumps(out, indent=1))
    return out, d
