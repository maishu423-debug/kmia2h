"""Optional live sources (GOES, HRRR, MRMS) as callables for forcing.Inputs.extra.

Every accessor returns only rows with available_at <= the time asked for. A missing or stale
source returns None / NaN, and the forcing falls back as INTERFACES.md describes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kmia2h.config import CFG, store

MAX_GOES_AGE = pd.Timedelta(minutes=40)
MAX_MRMS_AGE = pd.Timedelta(minutes=30)
MAX_MOTION_MS = 30.0


def _load(name: str, lo, hi, tcol="available_at", tables: dict | None = None):
    key = name.replace(".parquet", "")
    if tables and key in tables:
        d = tables[key]
        if d is None:
            return None
        d = d.copy()
    else:
        p = store(name)
        if not p.exists():
            return None
        d = pd.read_parquet(p)
    for c in d.columns:
        if isinstance(d[c].dtype, pd.DatetimeTZDtype):
            d[c] = d[c].astype("datetime64[ns, UTC]")
    d = d[(d[tcol] >= lo - pd.Timedelta(hours=8)) & (d[tcol] <= hi)]
    return d.sort_values(tcol).reset_index(drop=True)


def _latest(d: pd.DataFrame, t: pd.Timestamp, max_age: pd.Timedelta):
    a = d["available_at"].dt.tz_convert(None).to_numpy()
    i = np.searchsorted(a, pd.Timestamp(t).tz_convert(None).as_unit("ns").to_datetime64(), side="right") - 1
    if i < 0:
        return None
    row = d.iloc[i]
    if t - row["ts_utc"] > max_age:
        return None
    return row


def factory(use: set[str], tables: dict | None = None, goes_max_age: pd.Timedelta | None = None):
    """use: subset of {'goes', 'hrrr', 'sea_breeze', 'mrms'}. Returns f(start, end) -> extra.
    tables: in-memory replacements for the stores (keys goes_feat, mrms_feat, nwp_point).
    goes_max_age: degraded mode (GOES outage): accept an older last scan at the origin."""
    g_age = MAX_GOES_AGE if goes_max_age is None else goes_max_age

    def make(start, end) -> dict:
        extra = {"sea_breeze": "sea_breeze" in use, "sea_breeze_hrrr": "sea_breeze" in use and "hrrr" in use}
        obs = []
        if "goes" in use:
            g = _load("goes_feat.parquet", start, end, tables=tables)
            if g is not None and len(g):
                g = g[g.cf_disc.notna()].reset_index(drop=True)
                gi = g.set_index("available_at")

                def goes_row(t, g=g):
                    r = _latest(g, t, g_age)
                    if r is None:
                        return None
                    r = r.to_dict()
                    sp = np.hypot(r.get("motion_u", np.nan), r.get("motion_v", np.nan))
                    if not np.isfinite(sp) or sp > MAX_MOTION_MS:
                        r["motion_q"] = np.nan
                    return r

                def goes_c(minutes, g=g):
                    t = pd.DataFrame({"t": pd.DatetimeIndex(minutes).astype("datetime64[ns, UTC]")})
                    m = pd.merge_asof(t, g[["available_at", "ts_utc", "cf_disc"]], left_on="t", right_on="available_at")
                    stale = (m.t - m.ts_utc) > MAX_GOES_AGE
                    return m.cf_disc.where(~stale).to_numpy(float)

                extra["goes_row"], extra["goes_c_series"] = goes_row, goes_c
                obs.append(pd.DataFrame({"ts_utc": g.ts_utc, "available_at": g.available_at, "kind": "goes",
                                         "value": g.cf_disc}))
        if "hrrr" in use:
            h = _load("nwp_point.parquet", start, end + pd.Timedelta(hours=6), "valid_utc", tables=tables)
            if h is not None and len(h):
                cycles = h.groupby("cycle_utc")

                def latest_cycle(t):
                    avail = h.cycle_utc + pd.Timedelta(minutes=CFG.latency_min.hrrr)
                    ok = h.cycle_utc[avail <= t]
                    if ok.empty:
                        return None
                    c = ok.max()
                    if t - c > pd.Timedelta(hours=6):
                        return None
                    return cycles.get_group(c)

                def interp(g, col, times):
                    x = g[g[col].notna()]
                    if len(x) < 2:
                        return np.full(len(times), np.nan)
                    xs = x.valid_utc.astype("int64").to_numpy() / 6e10
                    ts = pd.DatetimeIndex(times).as_unit("ns").asi8 / 6e10
                    return np.interp(ts, xs, x[col].to_numpy(float))

                def hrrr_window(origin, valid):
                    g = latest_cycle(origin)
                    if g is None:
                        return None
                    t2m = interp(g, "t2m", valid)
                    u, v = interp(g, "u10", valid), interp(g, "v10", valid)
                    u0, v0 = interp(g, "u10", [origin])[0], interp(g, "v10", [origin])[0]
                    tc = interp(g, "tcdc", valid)
                    tc0 = interp(g, "tcdc", [origin])[0]
                    direction = np.degrees(np.arctan2(-u, -v)) % 360.0
                    return {"t2m": t2m if np.isfinite(t2m).all() else None, "dir": direction if np.isfinite(direction).all() else None,
                            "u0": u0, "v0": v0, "tcdc": np.clip(tc, 0, 1) if np.isfinite(tc).all() else None,
                            "tcdc0": float(np.clip(tc0, 0, 1)) if np.isfinite(tc0) else None}

                def hrrr_wind(minutes):
                    lead = h.valid_utc - h.cycle_utc
                    x = h[(lead == pd.Timedelta(minutes=90)) & h.u10.notna()].sort_values("valid_utc")
                    t = pd.DataFrame({"t": pd.DatetimeIndex(minutes).astype("datetime64[ns, UTC]")})
                    m = pd.merge_asof(t, x[["valid_utc", "u10", "v10"]], left_on="t", right_on="valid_utc",
                                      tolerance=pd.Timedelta(minutes=45))
                    return m.u10.to_numpy(float), m.v10.to_numpy(float)

                extra["hrrr_window"], extra["hrrr_wind_series"] = hrrr_window, hrrr_wind
                lead = h.valid_utc - h.cycle_utc
                x = h[(lead == pd.Timedelta(minutes=90)) & h.t2m.notna()]
                obs.append(pd.DataFrame({"ts_utc": x.valid_utc, "available_at": x.cycle_utc + pd.Timedelta(minutes=CFG.latency_min.hrrr),
                                         "kind": "hrrr", "value": x.t2m}))
        if "mrms" in use:
            m = _load("mrms_feat.parquet", start, end, tables=tables)
            if m is not None and len(m):
                def mrms_row(t, m=m):
                    r = _latest(m, t, MAX_MRMS_AGE)
                    if r is None:
                        return None
                    r = r.to_dict()
                    sp = np.hypot(r.get("motion_u", np.nan), r.get("motion_v", np.nan))
                    if not np.isfinite(sp) or sp > MAX_MOTION_MS:
                        r["motion_u"] = r["motion_v"] = np.nan
                    return r
                extra["mrms_row"] = mrms_row
        if obs:
            extra["filter_obs"] = pd.concat(obs, ignore_index=True)
        return extra

    return make
