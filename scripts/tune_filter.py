"""Grid over filter T noise and residual sigma scale on 2024 months (no simulation)."""
import itertools, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from kmia2h.backtest import backtest
from kmia2h.config import store
from kmia2h.data.qc import clean
from kmia2h.timeutil import from_ns

MONTHS = [(2024, 1), (2024, 4), (2024, 7), (2024, 10)]
grid = list(itertools.product([float(x) for x in sys.argv[1].split(",")], [float(x) for x in sys.argv[2].split(",")]))
jobs = []
for i, (ts, sc) in enumerate(grid):
    for y, m in MONTHS:
        jobs.append((f"tune_{i}", y, m, {"simulate": False, "overwrite": True,
                                          "cfg_override": {"filter.t_sigma": ts, "residual.sigma_scale": sc}}))
with ProcessPoolExecutor(8) as ex:
    list(ex.map(backtest.run_month, jobs))
k = pd.read_parquet(store("kmia_1min.parquet"), filters=[("ts_utc", ">=", pd.Timestamp("2024-01-01", tz="UTC"))])
t1 = clean(k)
f = pd.read_parquet(store("features_10m.parquet"), columns=["t_last"])
rows = []
for i, (ts, sc) in enumerate(grid):
    z, zk, err, err_last = [], [], [], []
    for y, m in MONTHS:
        d = np.load(store(f"backtest/tune_{i}/{y}{m:02d}.npz"))
        zz = d["inn_val"] / np.sqrt(d["inn_var"])
        z.append(zz[d["inn_kind"] == "hfmetar"])
        idx = from_ns(d["minutes"])
        tt = t1.reindex(idx).to_numpy()
        e = d["rec_Y"] - tt
        err.append(e[::10]); err_last.append((f.t_last.reindex(idx[::10], method="ffill").to_numpy() - tt[::10]))
    z = np.concatenate(z); e = np.concatenate(err); el = np.concatenate(err_last)
    ok = np.isfinite(e) & np.isfinite(el)
    rows.append({"t_sigma": ts, "sigma_scale": sc, "hf_var": z.var(), "hf_mean": z.mean(),
                 "hf_lag1": np.corrcoef(z[1:], z[:-1])[0, 1], "rmse_filter": np.sqrt(np.mean(e[ok] ** 2)),
                 "rmse_last_obs": np.sqrt(np.mean(el[ok] ** 2))})
print(pd.DataFrame(rows).round(3).to_string(index=False))
