"""Step 6: re-tune the daytime residual noise scale (shared by filter and simulator) on 2024 with
all sources on. Criterion: daytime HFMETAR innovation variance nearest 1 with lag-1 < 0.2, then the
lowest daytime filter error. Usage: tune_day_noise.py SCALES SOURCES"""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from kmia2h.backtest import backtest
from kmia2h.config import store
from kmia2h.data.qc import clean
from kmia2h.physics import solar
from kmia2h.timeutil import from_ns

scales = [float(x) for x in sys.argv[1].split(",")]
use = tuple(sys.argv[2].split(","))
MONTHS = [(2024, 1), (2024, 4), (2024, 7), (2024, 10)]
jobs = [(f"tuneday_{i}", y, m, {"simulate": False, "overwrite": True, "sources": use,
                                "cfg_override": {"residual.sigma_scale_day": s}})
        for i, s in enumerate(scales) for y, m in MONTHS]
with ProcessPoolExecutor(8) as ex:
    list(ex.map(backtest.run_month, jobs))
k = pd.read_parquet(store("kmia_1min.parquet"), filters=[("ts_utc", ">=", pd.Timestamp("2024-01-01", tz="UTC"))])
t1 = clean(k)
f = pd.read_parquet(store("features_10m.parquet"), columns=["t_last"])
rows = []
for i, s in enumerate(scales):
    z, err, err_last = [], [], []
    for y, m in MONTHS:
        d = np.load(store(f"backtest/tuneday_{i}/{y}{m:02d}.npz"))
        idx = from_ns(d["minutes"])
        day = solar.compute(idx)["is_day"].to_numpy()
        sel = (d["inn_kind"] == "hfmetar") & day[np.clip(d["inn_k"] - (len(d["minutes"]) and 0) - 360, 0, len(idx) - 1)]
        zz = d["inn_val"] / np.sqrt(d["inn_var"])
        z.append(zz[sel])
        tt = t1.reindex(idx).to_numpy()
        e = (d["rec_Y"] - tt)[day][::10]
        el = (f.t_last.reindex(idx, method="ffill").to_numpy() - tt)[day][::10]
        err.append(e); err_last.append(el)
    z = np.concatenate(z); e = np.concatenate(err); el = np.concatenate(err_last)
    ok = np.isfinite(e) & np.isfinite(el)
    rows.append({"sigma_scale_day": s, "day_hf_var": z.var(), "day_hf_mean": z.mean(),
                 "day_hf_lag1": np.corrcoef(z[1:], z[:-1])[0, 1], "day_rmse_filter": np.sqrt(np.mean(e[ok] ** 2)),
                 "day_rmse_last_obs": np.sqrt(np.mean(el[ok] ** 2))})
print(pd.DataFrame(rows).round(3).to_string(index=False))
