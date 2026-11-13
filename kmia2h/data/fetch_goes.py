"""GOES-East ABI L2 CONUS -> store/goes_feat.parquet, lean plan (REBUILD step 5).

For every 30-minute origin t: the newest CONUS scan whose start is <= t - latency (10 min), plus
the scan 10 minutes before it (for motion). ACMC and ACHAC always, CODC only by day. Each file is
read by HDF5 range requests over a Miami subset box (~0.2-0.4 MB per file), reduced to feature
rows and never stored. Resumable through a per-day manifest of finished parquet parts."""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd
import pyproj
import s3fs

from kmia2h.config import CFG, store
from kmia2h.data import stations
from kmia2h.physics import solar

SWITCH = pd.Timestamp("2025-04-07 15:10", tz="UTC")
VAR = {"ACMC": "BCM", "ACHAC": "HT", "CODC": "COD"}
RINGS = [0, 10, 20, 40, 80]
BOX_KM = 100.0
_local = threading.local()


def fs():
    if not hasattr(_local, "fs"):
        _local.fs = s3fs.S3FileSystem(anon=True, default_block_size=64 * 1024)
    return _local.fs


def bucket(t: pd.Timestamp) -> str:
    return "noaa-goes16" if t < SWITCH else "noaa-goes19"


class Geometry:
    """Pixel indices of the station disc, 12 sectors x 4 rings, and the motion box, per grid."""

    def __init__(self, h: h5py.File):
        p = h["goes_imager_projection"].attrs
        hgt = float(p["perspective_point_height"][0])
        geos = pyproj.Proj(proj="geos", h=hgt, lon_0=float(p["longitude_of_projection_origin"][0]),
                           sweep=p["sweep_angle_axis"].decode() if isinstance(p["sweep_angle_axis"], bytes) else str(p["sweep_angle_axis"]),
                           a=float(p["semi_major_axis"][0]), b=float(p["semi_minor_axis"][0]))
        xs, ys = h["x"], h["y"]
        x = (xs[:].astype(float) * xs.attrs["scale_factor"][0] + xs.attrs["add_offset"][0]) * hgt
        y = (ys[:].astype(float) * ys.attrs["scale_factor"][0] + ys.attrs["add_offset"][0]) * hgt
        k = stations.load()[CFG.station]
        kx, ky = geos(k["lon"], k["lat"])
        ci, ri = int(np.argmin(np.abs(x - kx))), int(np.argmin(np.abs(y - ky)))
        dx = abs(x[1] - x[0])
        half = int(np.ceil(BOX_KM * 1000 / dx)) + 1
        self.r0, self.r1 = max(ri - half, 0), min(ri + half + 1, len(y))
        self.c0, self.c1 = max(ci - half, 0), min(ci + half + 1, len(x))
        X, Y = np.meshgrid(x[self.c0:self.c1], y[self.r0:self.r1])
        lon, lat = geos(X, Y, inverse=True)
        g = pyproj.Geod(ellps="WGS84")
        az, _, dist = g.inv(np.full(lon.shape, k["lon"]), np.full(lat.shape, k["lat"]), lon, lat)
        self.dist_km = dist / 1000.0
        self.sector = (np.floor((az % 360) / 30.0).astype(int)) % 12
        self.ring = np.searchsorted(RINGS, self.dist_km, side="right") - 1
        self.dx_m = dx


def regions(geo: Geometry, a: np.ndarray, prefix: str, how: str) -> dict:
    out = {}
    disc = geo.dist_km <= 5.0
    f = (lambda v: np.nanmean(v)) if how == "mean" else (lambda v: np.nanmedian(v))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[f"{prefix}_disc"] = float(f(a[disc])) if np.isfinite(a[disc]).any() else np.nan
            for s in range(12):
                for r in range(4):
                    m = (geo.sector == s) & (geo.ring == r)
                    v = a[m]
                    out[f"{prefix}_s{s:02d}_r{r}"] = float(f(v)) if np.isfinite(v).any() else np.nan
    return out


def read(key: str, var: str, geos_cache: dict) -> tuple[np.ndarray, Geometry]:
    with fs().open(key, "rb", block_size=64 * 1024, cache_type="blockcache") as f:
        h = h5py.File(f, "r")
        shape = h[var].shape
        if shape not in geos_cache:
            geos_cache[shape] = Geometry(h)
        geo = geos_cache[shape]
        v = h[var]
        raw = v[geo.r0:geo.r1, geo.c0:geo.c1].astype(float)
        fill = v.attrs.get("_FillValue")
        if fill is not None:
            raw[raw == fill[0]] = np.nan
        sf, ao = v.attrs.get("scale_factor"), v.attrs.get("add_offset")
        if sf is not None:
            raw = raw * sf[0] + (ao[0] if ao is not None else 0.0)
    return raw, geo


def motion(a: np.ndarray, b: np.ndarray, dx_m: float, dt_s: float) -> tuple[float, float, float]:
    """Phase cross-correlation of two cloud masks (earlier a, later b). Returns u, v (m/s, the
    direction the cloud moves) and the normalised correlation peak as quality."""
    a, b = np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0)
    a, b = a - a.mean(), b - b.mean()
    if a.std() == 0 or b.std() == 0:
        return np.nan, np.nan, 0.0
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A, B = np.fft.fft2(a * win), np.fft.fft2(b * win)
    R = A.conj() * B
    R /= np.maximum(np.abs(R), 1e-12)
    c = np.fft.ifft2(R).real
    i, j = np.unravel_index(np.argmax(c), c.shape)
    di = i if i <= c.shape[0] // 2 else i - c.shape[0]
    dj = j if j <= c.shape[1] // 2 else j - c.shape[1]
    q = float(c.max())
    # rows increase southward in the ABI fixed grid (y decreases with row)
    return dj * dx_m / dt_s, -di * dx_m / dt_s, q


def scan_keys(prod: str, t: pd.Timestamp) -> list[tuple[pd.Timestamp, str]]:
    b = bucket(t)
    d = f"{b}/ABI-L2-{prod}/{t:%Y}/{t.dayofyear:03d}/{t:%H}/"
    try:
        keys = fs().ls(d)
    except FileNotFoundError:
        return []
    out = []
    for k in keys:
        s = k.split("_s")[1][:13]
        ts = pd.Timestamp.strptime(s, "%Y%j%H%M%S").tz_localize("UTC")
        out.append((ts, k))
    return sorted(out)


def day_rows(day: pd.Timestamp, geos_cache: dict) -> pd.DataFrame:
    lat = CFG.latency_min.goes
    origins = pd.date_range(day, day + pd.Timedelta(days=1), freq="30min", inclusive="left")
    listing = {}
    rows = []
    sol = solar.compute(origins)
    for o, is_day in zip(origins, sol["is_day"].to_numpy()):
        want = o - pd.Timedelta(minutes=lat)
        cands = []
        for hh in (want.floor("h") - pd.Timedelta(hours=1), want.floor("h")):
            if hh not in listing:
                listing[hh] = {p: scan_keys(p, hh) for p in ("ACMC", "ACHAC", "CODC")}
            cands += listing[hh]["ACMC"]
        cands = [(ts, k) for ts, k in cands if ts <= want]
        if not cands:
            continue
        ts_now, k_now = cands[-1]
        prev = [(ts, k) for ts, k in cands if ts <= ts_now - pd.Timedelta(minutes=9)]
        row = {"ts_utc": ts_now, "origin": o}
        try:
            m_now, geo = read(k_now, "BCM", geos_cache)
            row.update(regions(geo, m_now, "cf", "mean"))
            if prev:
                ts_p, k_p = prev[-1]
                m_prev, _ = read(k_p, "BCM", geos_cache)
                u, v, q = motion(m_prev, m_now, geo.dx_m, (ts_now - ts_p).total_seconds())
                row.update({"motion_u": u, "motion_v": v, "motion_q": q, "motion_dt_s": (ts_now - ts_p).total_seconds()})
            for prod, var, pre, use in (("ACHAC", "HT", "cth", True), ("CODC", "COD", "cod", bool(is_day))):
                if not use:
                    continue
                ks = [(ts, k) for hh in (want.floor("h") - pd.Timedelta(hours=1), want.floor("h"))
                      for ts, k in listing[hh][prod] if abs((ts - ts_now).total_seconds()) < 60]
                if ks:
                    a, g2 = read(ks[0][1], var, geos_cache)
                    reg = regions(g2, a, pre, "median")
                    row.update({k: v for k, v in reg.items() if k.endswith("_disc") or pre == "cth"})
        except Exception as e:  # unreadable file: the row keeps what it has
            row["error"] = repr(e)[:100]
        rows.append(row)
    d = pd.DataFrame(rows)
    if len(d):
        d["available_at"] = d.ts_utc + pd.Timedelta(minutes=lat)
    return d


def run(first: str, last: str, workers: int = 16) -> None:
    out = store("goes_days")
    out.mkdir(exist_ok=True)
    days = pd.date_range(first, last, freq="D", tz="UTC")
    todo = [d for d in days if not (out / f"{d:%Y%m%d}.parquet").exists()]
    geos_cache: dict = {}
    lock = threading.Lock()

    def job(day):
        t = time.time()
        for attempt in range(3):
            try:
                d = day_rows(day, geos_cache)
                break
            except Exception as e:
                time.sleep(5 * (attempt + 1))
                d = None
        if d is None or d.empty:
            pd.DataFrame({"origin": [], "avail": []}).to_parquet(out / f"{day:%Y%m%d}.empty.parquet")
            print("goes", day.date(), "EMPTY", flush=True)
            return
        d.to_parquet(out / f"{day:%Y%m%d}.parquet")
        print("goes", day.date(), len(d), f"{time.time() - t:.0f}s", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, todo))


def build() -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(store("goes_days").glob("*[0-9].parquet"))]
    d = pd.concat(parts, ignore_index=True).sort_values("ts_utc")
    d.to_parquet(store("goes_feat.parquet"), index=False)
    return d


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16)
