"""MRMS composite reflectivity -> store/mrms_feat.parquet (REBUILD step 5: same times as GOES).

For every 30-minute origin t: the newest 2-minute file whose valid time is <= t - latency, plus the
file 10 minutes before it (motion). Each file is decompressed in memory, decoded with eccodes,
reduced to sector/ring features around KMIA and discarded."""
from __future__ import annotations

import gzip
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import eccodes
import numpy as np
import pandas as pd
import pyproj
import s3fs

from kmia2h.config import CFG, store
from kmia2h.data import stations
from kmia2h.data.fetch_goes import RINGS, motion

PROD = "MergedReflectivityQCComposite_00.50"
BOX_KM = 100.0
_G = {}


def geometry(lat0, lon0, dlat, dlon, ni, nj):
    k = stations.load()[CFG.station]
    lon_k = k["lon"] % 360
    ci = int(round((lon_k - lon0) / dlon))
    ri = int(round((lat0 - k["lat"]) / dlat))
    hr = int(np.ceil(BOX_KM / 111.0 / dlat)) + 1
    hc = int(np.ceil(BOX_KM / (111.0 * np.cos(np.radians(k["lat"]))) / dlon)) + 1
    r0, r1, c0, c1 = ri - hr, ri + hr + 1, ci - hc, ci + hc + 1
    lats = lat0 - np.arange(r0, r1) * dlat
    lons = lon0 + np.arange(c0, c1) * dlon - 360
    LON, LAT = np.meshgrid(lons, lats)
    az, _, dist = pyproj.Geod(ellps="WGS84").inv(np.full(LON.shape, k["lon"]), np.full(LAT.shape, k["lat"]), LON, LAT)
    return {"r0": r0, "r1": r1, "c0": c0, "c1": c1, "dist": dist / 1000, "sector": (np.floor((az % 360) / 30).astype(int)) % 12,
            "ring": np.searchsorted(RINGS, dist / 1000, side="right") - 1, "dx_m": dlon * 111e3 * np.cos(np.radians(k["lat"]))}


def decode(raw: bytes) -> tuple[np.ndarray, dict]:
    h = eccodes.codes_new_from_message(gzip.decompress(raw))
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        key = (ni, nj)
        if key not in _G:
            _G[key] = geometry(eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
                               eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees"),
                               eccodes.codes_get(h, "jDirectionIncrementInDegrees"),
                               eccodes.codes_get(h, "iDirectionIncrementInDegrees"), ni, nj)
        g = _G[key]
        v = eccodes.codes_get_values(h).reshape(nj, ni)[g["r0"]:g["r1"], g["c0"]:g["c1"]].astype(float)
    finally:
        eccodes.codes_release(h)
    v[v < -900] = np.nan          # no coverage
    v[v < 0] = 0.0                # below detection -> no echo (dBZ floor)
    return v, g


def features(v: np.ndarray, g: dict) -> dict:
    out = {}
    for s in range(12):
        for r in range(4):
            m = (g["sector"] == s) & (g["ring"] == r)
            a = v[m]
            a = a[np.isfinite(a)]
            out[f"maxdbz_s{s:02d}_r{r}"] = float(a.max()) if a.size else np.nan
            out[f"meandbz_s{s:02d}_r{r}"] = float(a.mean()) if a.size else np.nan
            out[f"frac35_s{s:02d}_r{r}"] = float((a >= 35).mean()) if a.size else np.nan
    disc = v[g["dist"] <= 5.0]
    out["maxdbz_disc"] = float(np.nanmax(disc)) if np.isfinite(disc).any() else np.nan
    return out


def file_time(k: str) -> pd.Timestamp:
    return pd.to_datetime(k.split("_")[-1][:15], format="%Y%m%d-%H%M%S", utc=True)


def origin_row(o: pd.Timestamp, ft: pd.Series, fs) -> dict | None:
    """MRMS features for one origin from the file listing ft (time -> key): the newest file available
    at o, and motion from the file 9-16 min earlier."""
    lat = CFG.latency_min.mrms
    want = o - pd.Timedelta(minutes=max(lat, 10))
    c = ft[(ft.index <= want) & (ft.index > want - pd.Timedelta(minutes=10))]
    if c.empty:
        return None
    ts, k = c.index[-1], c.iloc[-1]
    p = ft[(ft.index <= ts - pd.Timedelta(minutes=9)) & (ft.index > ts - pd.Timedelta(minutes=16))]
    try:
        v, g = decode(fs.cat(k))
        row = {"origin": o, "ts_utc": ts, **features(v, g)}
        if not p.empty:
            v0, _ = decode(fs.cat(p.iloc[-1]))
            u, vv, q = motion(np.nan_to_num(v0) >= 20, np.nan_to_num(v) >= 20, g["dx_m"], (ts - p.index[-1]).total_seconds())
            row.update({"motion_u": u, "motion_v": vv, "motion_q": q})
        return row
    except Exception as e:
        return {"origin": o, "ts_utc": ts, "error": repr(e)[:100]}


def listing(days, fs) -> pd.Series:
    keys = []
    for d in days:
        try:
            keys += fs.ls(f"noaa-mrms-pds/CONUS/{PROD}/{d:%Y%m%d}/")
        except FileNotFoundError:
            pass
    return pd.Series(keys, index=pd.DatetimeIndex([file_time(k) for k in keys])).sort_index()


def day_job(day: pd.Timestamp) -> str:
    store("mrms_days").mkdir(parents=True, exist_ok=True)
    out = store("mrms_days") / f"{day:%Y%m%d}.parquet"
    if out.exists():
        return "skip"
    fs = s3fs.S3FileSystem(anon=True)
    t0 = time.time()
    keys = []
    for d in (day - pd.Timedelta(days=1), day):
        try:
            keys += fs.ls(f"noaa-mrms-pds/CONUS/{PROD}/{d:%Y%m%d}/")
        except FileNotFoundError:
            pass
    if not keys:
        pd.DataFrame({"origin": []}).to_parquet(out.with_suffix(".empty.parquet"))
        return f"mrms {day.date()} EMPTY"
    ft = pd.Series(keys, index=pd.DatetimeIndex([file_time(k) for k in keys])).sort_index()
    rows = [r for o in pd.date_range(day, day + pd.Timedelta(days=1), freq="30min", inclusive="left")
            if (r := origin_row(o, ft, fs)) is not None]
    d = pd.DataFrame(rows)
    d["available_at"] = d.ts_utc + pd.Timedelta(minutes=CFG.latency_min.mrms)
    d.to_parquet(out)
    return f"mrms {day.date()} {len(d)} {time.time() - t0:.0f}s"


def run(first, last, workers=4):
    store("mrms_days").mkdir(exist_ok=True)
    days = pd.date_range(first, last, freq="D", tz="UTC")
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(day_job, days):
            print(r, flush=True)


def build() -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(store("mrms_days").glob("*[0-9].parquet"))]
    d = pd.concat(parts, ignore_index=True).sort_values("ts_utc")
    d.to_parquet(store("mrms_feat.parquet"), index=False)
    return d


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
