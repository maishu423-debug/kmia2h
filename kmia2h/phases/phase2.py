"""Phase 2 checks: monthly coverage per source and station, synthesiser match, disk use."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from kmia2h.config import CFG, ROOT, out, store


def coverage() -> pd.DataFrame:
    rows = []
    k = pd.read_parquet(store("kmia_1min.parquet"), columns=["tmpf", "qc"])
    from kmia2h.data.qc import clean
    t = clean(k)
    m = t.groupby(t.index.tz_localize(None).to_period("M")).agg(lambda x: x.notna().mean())
    rows += [{"source": "kmia_1min", "station": CFG.station, "month": str(p), "coverage": v} for p, v in m.items()]
    obs = pd.read_parquet(store("obs.parquet"), columns=["station", "ts_utc", "report_type", "synthetic", "tmpf"])
    obs = obs[obs.tmpf.notna()]
    obs["month"] = obs.ts_utc.dt.tz_localize(None).dt.to_period("M")
    for (st, rt, syn), g in obs.groupby(["station", "report_type", "synthetic"]):
        c = g.groupby("month").ts_utc.apply(lambda x: x.dt.floor("5min" if rt == 1 else "h").nunique())
        exp = np.array([p.days_in_month for p in c.index]) * (288 if rt == 1 else 24)
        src = ("hfmetar" if rt == 1 else "metar") + ("_synthetic" if syn else "")
        rows += [{"source": src, "station": st, "month": str(p), "coverage": v / e} for (p, v), e in zip(c.items(), exp)]
    p = store("ndbc.parquet")
    if p.exists():
        n = pd.read_parquet(p)
        n = n[n.atmp_f.notna()]
        n["month"] = n.ts_utc.dt.tz_localize(None).dt.to_period("M")
        for st, g in n.groupby("station"):
            c = g.groupby("month").ts_utc.apply(lambda x: x.dt.floor("h").nunique())
            rows += [{"source": "ndbc_atmp", "station": st, "month": str(q), "coverage": v / (q.days_in_month * 24)}
                     for q, v in c.items()]
    d = pd.DataFrame(rows)
    d.to_csv(out(2, "coverage.csv"), index=False)
    return d


def disk_gb() -> float:
    return sum(f.stat().st_size for f in (ROOT / "data").rglob("*") if f.is_file()) / 1e9


if __name__ == "__main__":
    d = coverage()
    k = d[(d.source == "kmia_1min") & (d.month <= "2024-12")]
    print("kmia 1min months >=90%:", (k.coverage >= 0.9).mean().round(3), "of", len(k))
    piv = d[d.month <= "2024-12"].groupby(["source", "station"]).coverage.agg(["mean", lambda x: (x >= 0.7).mean()])
    print(piv)
    print("disk GB", round(disk_gb(), 2), "free", shutil.disk_usage(ROOT).free / 1e9)
