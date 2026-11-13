"""HRRR via our own .idx byte-range reader -> store/nwp_point.parquet (REBUILD step 5).

For each hourly cycle c: TMP 2 m and UGRD/VGRD 10 m from the subhourly files at leads 90-270 min in
30-minute steps, and TCDC (entire atmosphere) from the hourly wrfsfc files at leads 1-5 h. Each
GRIB message is fetched by HTTP range, decoded with eccodes, reduced to the KMIA grid point and the
3 x 3 mean, and discarded. available_at = cycle + 90 min. TCDC is stored as a 0-1 fraction."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import eccodes
import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from kmia2h.config import CFG, store
from kmia2h.data import stations

BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
LEADS_MIN = list(range(90, 271, 30))
TCDC_H = [1, 2, 3, 4, 5]
_IDX = {}
_S = None


def session():
    global _S
    if _S is None:
        _S = requests.Session()
    return _S


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
def get(url, rng=None):
    r = session().get(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else None, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content if rng else r.text


def index(url) -> list[tuple[int, str, str, str]]:
    t = get(url + ".idx")
    if t is None:
        return []
    out = []
    for line in t.splitlines():
        p = line.split(":")
        out.append((int(p[1]), p[3], p[4], p[5]))
    return out


def ranges(idx, var, level, fcst):
    for i, (off, v, lv, fc) in enumerate(idx):
        if v == var and lv == level and fc == fcst:
            end = idx[i + 1][0] - 1 if i + 1 < len(idx) else ""
            return off, end
    return None


def point(msg: bytes) -> tuple[float, float]:
    h = eccodes.codes_new_from_message(msg)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        if (ni, nj) not in _IDX:
            lat = eccodes.codes_get_array(h, "latitudes").reshape(nj, ni)
            lon = eccodes.codes_get_array(h, "longitudes").reshape(nj, ni)
            k = stations.load()[CFG.station]
            d = (lat - k["lat"]) ** 2 + ((lon - 360 if lon.max() > 180 else lon) - k["lon"]) ** 2
            _IDX[(ni, nj)] = np.unravel_index(np.argmin(d), d.shape)
        j, i = _IDX[(ni, nj)]
        v = eccodes.codes_get_values(h).reshape(nj, ni)
        return float(v[j, i]), float(v[j - 1:j + 2, i - 1:i + 2].mean())
    finally:
        eccodes.codes_release(h)


def cycle_job(cycle: pd.Timestamp) -> list[dict]:
    rows = []
    for lead in LEADS_MIN:
        fh = int(np.ceil(lead / 60))
        url = f"{BASE}/hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrfsubhf{fh:02d}.grib2"
        idx = index(url)
        if not idx:
            continue
        fc = f"{lead} min fcst" if lead % 60 else f"{lead // 60} hour fcst"
        row = {"cycle_utc": cycle, "valid_utc": cycle + pd.Timedelta(minutes=lead)}
        for var, lev, name in (("TMP", "2 m above ground", "t2m"), ("UGRD", "10 m above ground", "u10"),
                               ("VGRD", "10 m above ground", "v10")):
            r = ranges(idx, var, lev, fc)
            if r is None and lead % 60 == 0:
                r = ranges(idx, var, lev, f"{lead} min fcst")
            if r is None:
                continue
            msg = get(url, r)
            if msg:
                pt, m3 = point(msg)
                if name == "t2m":
                    pt, m3 = (pt - 273.15) * 9 / 5 + 32, (m3 - 273.15) * 9 / 5 + 32
                row[name], row[f"{name}_3x3"] = pt, m3
        rows.append(row)
    for fh in TCDC_H:
        url = f"{BASE}/hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrfsfcf{fh:02d}.grib2"
        idx = index(url)
        r = ranges(idx, "TCDC", "entire atmosphere", f"{fh} hour fcst") if idx else None
        if r is None:
            continue
        msg = get(url, r)
        if msg:
            pt, m3 = point(msg)
            rows.append({"cycle_utc": cycle, "valid_utc": cycle + pd.Timedelta(hours=fh), "tcdc": pt / 100.0,
                         "tcdc_3x3": m3 / 100.0})
    return rows


def day_job(day: pd.Timestamp) -> str:
    out = store("hrrr_days") / f"{day:%Y%m%d}.parquet"
    if out.exists():
        return "skip"
    t0 = time.time()
    rows = []
    for h in range(24):
        try:
            rows += cycle_job(day + pd.Timedelta(hours=h))
        except Exception as e:
            print("hrrr cycle failed", day + pd.Timedelta(hours=h), repr(e)[:100], flush=True)
    d = pd.DataFrame(rows)
    if d.empty:
        pd.DataFrame({"cycle_utc": []}).to_parquet(out.with_suffix(".empty.parquet"))
        return f"hrrr {day.date()} EMPTY"
    d = d.groupby(["cycle_utc", "valid_utc"], as_index=False).first()
    d["available_at"] = d.cycle_utc + pd.Timedelta(minutes=CFG.latency_min.hrrr)
    d.to_parquet(out)
    return f"hrrr {day.date()} {len(d)} {time.time() - t0:.0f}s"


def run(first, last, workers=6):
    store("hrrr_days").mkdir(exist_ok=True)
    days = pd.date_range(first, last, freq="D", tz="UTC")
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(day_job, days):
            print(r, flush=True)


def build() -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(store("hrrr_days").glob("*[0-9].parquet"))]
    d = pd.concat(parts, ignore_index=True).sort_values(["cycle_utc", "valid_utc"])
    d.to_parquet(store("nwp_point.parquet"), index=False)
    return d


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
