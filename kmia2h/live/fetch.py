"""Live source fetch for run.py --origin now (SPEC.md Phase 10, step 1).

Returns in-memory tables with the same columns as the stores (obs, ndbc, goes_feat, mrms_feat,
nwp_point). Each source is fetched in its own thread; a source that fails or times out is returned as
None and logged, and the run degrades as SPEC.md describes. KMIA 1-minute data is not used."""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import pandas as pd

from kmia2h.config import CFG

HISTORY = pd.Timedelta(hours=9)       # 6 h spin-up + trend / rain look-back
SOURCE_TIMEOUT_S = 60


def fetch_obs(origin: pd.Timestamp) -> pd.DataFrame:
    """KMIA and neighbour METAR / HFMETAR from IEM in one request."""
    from kmia2h.data import iem, parse_obs
    from kmia2h.data.fetch_iem_metar import iem_id
    stations = [CFG.station, *CFG.neighbours]
    lo = origin - HISTORY
    params = [*[("station", iem_id(s)) for s in stations], ("data", "all"),
              *[("report_type", r) for r in ("1", "3", "4")], ("tz", "UTC"), ("format", "onlycomma"),
              ("latlon", "no"), ("missing", "empty"), ("trace", "0.0001"),
              ("sts", f"{lo:%Y-%m-%dT%H:%MZ}"), ("ets", f"{origin + pd.Timedelta(minutes=1):%Y-%m-%dT%H:%MZ}")]
    text = iem._get(CFG.iem.base + "/cgi-bin/request/asos.py", params)
    raw = pd.read_csv(io.StringIO(text), low_memory=False, dtype=str)
    back = {iem_id(s): s for s in stations}
    parts = []
    for sid, g in raw.groupby("station"):
        buf = io.StringIO()
        g.to_csv(buf, index=False)
        buf.seek(0)
        parts.append(parse_obs.parse_file(buf, back.get(sid, sid)))
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates(["station", "ts_utc", "report_type"]).sort_values(["station", "ts_utc"])
    d = d[d.ts_utc <= origin].reset_index(drop=True)
    return d


AWC_URL = "https://aviationweather.gov/api/data/metar"


def fetch_awc(origin: pd.Timestamp) -> pd.DataFrame:
    """KMIA routine and special METARs from aviationweather.gov (receipt ~3 min after the observation,
    well ahead of IEM). Same station, same parser as the IEM CSV: not a substitute station."""
    import requests
    from kmia2h.data import parse_obs
    r = requests.get(AWC_URL, params={"ids": CFG.station, "format": "json", "hours": 9},
                     headers={"User-Agent": "kmia2h-forecast"}, timeout=30)
    r.raise_for_status()
    rows = []
    for m in r.json():
        cl = (m.get("clouds") or [])[:4]
        row = {"station": CFG.station[1:], "valid": pd.Timestamp(m["obsTime"], unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"),
               "metar": m.get("rawOb", ""), "tmpf": None if m.get("temp") is None else m["temp"] * 9 / 5 + 32,
               "dwpf": None if m.get("dewp") is None else m["dewp"] * 9 / 5 + 32,
               "sknt": m.get("wspd"), "drct": m.get("wdir") if isinstance(m.get("wdir"), (int, float)) else None,
               "wxcodes": m.get("wxString"), "p01i": m.get("precip")}
        for i in range(4):
            row[f"skyc{i + 1}"] = cl[i]["cover"] if i < len(cl) else None
            row[f"skyl{i + 1}"] = cl[i].get("base") if i < len(cl) else None
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    d = parse_obs.parse_file(buf, CFG.station)
    return d[d.ts_utc <= origin].reset_index(drop=True)


def fetch_ndbc(origin: pd.Timestamp) -> pd.DataFrame:
    from kmia2h.data import fetch_ndbc as nd
    parts = [r for st in CFG.buoys if (r := nd.fetch_realtime(st, f"{origin:%Y%m%d%H%M}")) is not None]
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.ts_utc >= origin - HISTORY) & (d.ts_utc <= origin)].copy()
    d["available_at"] = d.ts_utc + pd.Timedelta(minutes=CFG.latency_min.ndbc)
    return d.reset_index(drop=True)


def fetch_goes(origin: pd.Timestamp) -> pd.DataFrame:
    """The scans at the origin and 30 min earlier (the filter assimilates both)."""
    from kmia2h.data import fetch_goes as fg
    return fg.rows_for(pd.DatetimeIndex([origin - pd.Timedelta(minutes=30), origin]))


def fetch_mrms(origin: pd.Timestamp) -> pd.DataFrame:
    import s3fs
    from kmia2h.data import fetch_mrms as fm
    fs = s3fs.S3FileSystem(anon=True)
    days = sorted({(origin - pd.Timedelta(minutes=30)).normalize(), origin.normalize()})
    ft = fm.listing(days, fs)
    rows = [r for r in [fm.origin_row(origin, ft, fs)] if r is not None]
    d = pd.DataFrame(rows)
    if len(d):
        d["available_at"] = d.ts_utc + pd.Timedelta(minutes=CFG.latency_min.mrms)
    return d


def fetch_hrrr(origin: pd.Timestamp) -> pd.DataFrame:
    """The newest cycle that is available at the origin (cycle + latency) and the one before it,
    so a late cycle falls back to the previous one at a longer lead."""
    from kmia2h.data import fetch_nwp as fn
    newest = (origin - pd.Timedelta(minutes=CFG.latency_min.hrrr)).floor("h")
    rows = []
    for c in (newest - pd.Timedelta(hours=1), newest):
        try:
            rows += fn.cycle_job(c)
        except Exception as e:  # a cycle not yet on the bucket
            print("hrrr cycle", c, "failed", repr(e)[:80], flush=True)
    d = pd.DataFrame(rows)
    if len(d):
        d = d.groupby(["cycle_utc", "valid_utc"], as_index=False).first()
        d["available_at"] = d.cycle_utc + pd.Timedelta(minutes=CFG.latency_min.hrrr)
    return d


FETCHERS = {"obs": fetch_obs, "awc": fetch_awc, "ndbc": fetch_ndbc, "goes_feat": fetch_goes, "mrms_feat": fetch_mrms,
            "nwp_point": fetch_hrrr}


def fetch_all(origin: pd.Timestamp, timeout_s: float = SOURCE_TIMEOUT_S) -> tuple[dict, dict]:
    """All live sources in parallel. Returns (tables, log); a failed source is None in tables."""
    t0 = time.time()
    tables, log = {}, {}
    ex = ThreadPoolExecutor(len(FETCHERS))
    futs = {k: ex.submit(f, origin) for k, f in FETCHERS.items()}
    wait(futs.values(), timeout=timeout_s)
    for k, f in futs.items():
        if not f.done():
            tables[k], log[k] = None, f"timeout after {timeout_s:.0f}s"
            continue
        try:
            d = f.result()
            tables[k] = d if d is not None and len(d) else None
            log[k] = f"{0 if d is None else len(d)} rows"
        except Exception as e:
            tables[k], log[k] = None, f"failed: {repr(e)[:120]}"
    ex.shutdown(wait=False, cancel_futures=True)
    log["seconds"] = round(time.time() - t0, 1)
    awc = tables.pop("awc", None)
    parts = [x for x in (tables.get("obs"), awc) if x is not None and len(x)]
    if parts:   # IEM first: on the same valid minute and report type the IEM row is kept
        d = pd.concat(parts, ignore_index=True).drop_duplicates(["station", "ts_utc", "report_type"])
        for c in ("ts_utc", "available_at"):
            d[c] = d[c].astype("datetime64[ns, UTC]")
        tables["obs"] = d.sort_values(["station", "ts_utc"]).reset_index(drop=True)
    else:
        tables["obs"] = None
    for k in ("ndbc", "goes_feat", "mrms_feat", "nwp_point"):
        d = tables.get(k)
        if d is not None:
            for c in d.columns:
                if isinstance(d[c].dtype, pd.DatetimeTZDtype):
                    d[c] = d[c].astype("datetime64[ns, UTC]")
    return tables, log

