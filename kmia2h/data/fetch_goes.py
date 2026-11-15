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
    # 5 km disc; on coarse grids (10 km cloud-top height) at least the nearest pixel
    disc = geo.dist_km <= max(5.0, 0.75 * geo.dx_m / 1000.0)
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

    def sub(cm, c0, cp):  # parabolic sub-pixel refinement of the peak
        den = cm - 2 * c0 + cp
        return 0.0 if den == 0 else float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))
    ni, nj = c.shape
    fi = sub(c[(i - 1) % ni, j], c[i, j], c[(i + 1) % ni, j])
    fj = sub(c[i, (j - 1) % nj], c[i, j], c[i, (j + 1) % nj])
    di = (i if i <= ni // 2 else i - ni) + fi
    dj = (j if j <= nj // 2 else j - nj) + fj
    q = float(c.max())
    # rows increase southward in the ABI fixed grid (y decreases with row)
    return dj * dx_m / dt_s, -di * dx_m / dt_s, q


def scan_keys(prod: str, t: pd.Timestamp) -> list[tuple[pd.Timestamp, str]]:
    b = bucket(t)
    d = f"{b}/ABI-L2-{prod}/{t:%Y}/{t.day_of_year:03d}/{t:%H}/"
    try:
        keys = fs().ls(d)
    except FileNotFoundError:
        return []
    out = []
    for k in keys:
        s = k.split("_s")[1][:13]
        ts = pd.to_datetime(s, format="%Y%j%H%M%S", utc=True)
        out.append((ts, k))
    return sorted(out)


def origin_row(o: pd.Timestamp, is_day: bool, listing: dict, geos_cache: dict) -> dict | None:
    """GOES features for one origin: the newest ACMC scan available at o (latency), cloud motion from
    the scan >= 9 min earlier, cloud-top height and (daytime) optical depth from the matching scans."""
    lat = CFG.latency_min.goes
    want = o - pd.Timedelta(minutes=lat)
    cands = []
    for hh in (want.floor("h") - pd.Timedelta(hours=1), want.floor("h")):
        if hh not in listing:
            listing[hh] = {p: scan_keys(p, hh) for p in ("ACMC", "ACHAC", "CODC")}
        cands += listing[hh]["ACMC"]
    cands = [(ts, k) for ts, k in cands if ts <= want]
    if not cands:
        return None
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
    return row


def rows_for(origins: pd.DatetimeIndex, geos_cache: dict | None = None) -> pd.DataFrame:
    geos_cache = {} if geos_cache is None else geos_cache
    listing: dict = {}
    sol = solar.compute(origins)
    rows = [r for o, is_day in zip(origins, sol["is_day"].to_numpy())
            if (r := origin_row(o, bool(is_day), listing, geos_cache)) is not None]
    d = pd.DataFrame(rows)
    if len(d):
        d["available_at"] = d.ts_utc + pd.Timedelta(minutes=CFG.latency_min.goes)
    return d


def day_rows(day: pd.Timestamp, geos_cache: dict) -> pd.DataFrame:
    return rows_for(pd.date_range(day, day + pd.Timedelta(days=1), freq="30min", inclusive="left"), geos_cache)


def job(day):
    """One day in its own process (h5py serialises threads on a global lock)."""
    out = store("goes_days")
    geos_cache: dict = {}
    if True:
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


def run(first: str, last: str, workers: int = 12) -> None:
    from concurrent.futures import ProcessPoolExecutor
    out = store("goes_days")
    out.mkdir(exist_ok=True)
    days = pd.date_range(first, last, freq="D", tz="UTC")
    todo = [d for d in days if not (out / f"{d:%Y%m%d}.parquet").exists()]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(job, todo, chunksize=1))


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
