"""Upstream advected temperature T_adv and the sea-breeze switch (SPEC.md Phase 4, step 3)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
SECONDS_PER_MIN = 60.0
M_PER_KM = 1000.0


def wind_speed_dir(u, v):
    """Speed (m/s) and meteorological direction (deg the wind blows FROM) from u, v."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    return np.hypot(u, v), np.degrees(np.arctan2(-u, -v)) % 360.0


def in_sector(direction, sector) -> np.ndarray:
    lo, hi = sector
    d = np.asarray(direction, float)
    return (d >= lo) & (d <= hi)


def wind_class(direction: float, cfg: Mapping) -> str:
    """onshore [30, 150], offshore [210, 330], northerly (330, 360) U [0, 30), else other."""
    f = cfg["features"]
    if in_sector(direction, f["sector_onshore"]):
        return "onshore"
    if in_sector(direction, f["sector_offshore"]):
        return "offshore"
    if direction > f["sector_offshore"][1] or direction < f["sector_onshore"][0]:
        return "northerly"
    return "other"


def upwind_set(direction: float, cfg: Mapping) -> dict:
    """{primary, fallback} station lists for a wind direction."""
    cls = wind_class(direction, cfg)
    if cls == "other":
        return {"primary": list(cfg["neighbours"]), "fallback": []}
    return dict(cfg["physics"]["upstream"]["sets"][cls])


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def lag_minutes(dist_km, speed_ms, cfg: Mapping):
    """Transit time from station to KMIA, capped. Calm air gives the cap."""
    cap = cfg["physics"]["upstream"]["max_lag_min"]
    d, s = np.asarray(dist_km, float), np.asarray(speed_ms, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lag = d * M_PER_KM / s / SECONDS_PER_MIN
    return np.where(s > 0, np.minimum(lag, cap), cap)


def lagged_series(obs: pd.DataFrame, origin: pd.Timestamp, station: str, lag: float, n: int,
                  cfg: Mapping) -> np.ndarray:
    """Station temperature at (s - lag) for minutes s = 1..n after origin.

    Only rows with available_at <= origin are used. If (s - lag) is later than the newest of those
    rows, the newest value is held. Values older than max_age_min are NaN.
    """
    d = obs[(obs["station"] == station) & (obs["available_at"] <= origin) & obs["tmpf"].notna()]
    if d.empty:
        return np.full(n, np.nan)
    d = d.sort_values("ts_utc")
    x = (d["ts_utc"] - origin).dt.total_seconds().to_numpy() / SECONDS_PER_MIN
    target = np.arange(1, n + 1, dtype=float) - lag
    out = np.interp(target, x, d["tmpf"].to_numpy(float))
    out[target < x[0]] = np.nan
    out[target - x[-1] > cfg["physics"]["upstream"]["max_age_min"]] = np.nan
    return out


def _idw(obs, origin, names: Sequence[str], n, kmia: Mapping, meta: Mapping, speed, cfg) -> np.ndarray:
    """Inverse-distance mean of lagged stations. Missing stations drop out minute by minute."""
    u = cfg["physics"]["upstream"]
    series, weights = [], []
    for s in names:
        if s not in meta:
            continue
        d = distance_km(kmia["lat"], kmia["lon"], meta[s]["lat"], meta[s]["lon"])
        series.append(lagged_series(obs, origin, s, float(lag_minutes(d, speed, cfg)), n, cfg))
        weights.append(1.0 / max(float(d), 1e-6) ** u["idw_power"])
    if not series:
        return np.full(n, np.nan)
    v, w = np.vstack(series), np.array(weights)[:, None]
    mask = ~np.isnan(v)
    total = (w * mask).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, (w * np.where(mask, v, 0.0)).sum(0) / total, np.nan)


def _resolve(obs, origin, stations_set: Mapping, n, kmia, meta, speed, cfg) -> np.ndarray:
    out = _idw(obs, origin, stations_set["primary"], n, kmia, meta, speed, cfg)
    if stations_set.get("fallback"):
        out = np.where(np.isnan(out), _idw(obs, origin, stations_set["fallback"], n, kmia, meta, speed, cfg), out)
    return out


def sea_breeze_onset(origin: pd.Timestamp, n: int, kmia_obs: pd.DataFrame, dir0: float, hrrr_dir: np.ndarray | None,
                     cfg: Mapping) -> int | None:
    """Index into minutes 1..n at which the sea breeze is on, or None.

    Observed onset (index 0): within the last 30 min KMIA wind turned into the onshore sector and
    the dewpoint rose by at least 2 degF. Forecast onset: HRRR wind turns onshore inside the window.
    """
    onshore = cfg["features"]["sector_onshore"]
    u = cfg["physics"]["upstream"]
    since = origin - pd.Timedelta(minutes=u["sea_breeze_window_min"])
    d = kmia_obs[(kmia_obs["available_at"] <= origin) & (kmia_obs["ts_utc"] >= since)].sort_values("ts_utc")
    d = d[d["drct"].notna()]
    if len(d) >= 2:
        on = in_sector(d["drct"].to_numpy(float), onshore)
        dew = d["dwpf"].to_numpy(float)
        if on[-1] and not on.all() and np.isfinite(dew).any() and dew[-1] - np.nanmin(dew) >= u["sea_breeze_dew_rise_f"]:
            return 0
    if hrrr_dir is not None and not in_sector(dir0, onshore):
        hits = np.flatnonzero(in_sector(hrrr_dir, onshore))
        if hits.size:
            return int(hits[0])
    return None


def build(origin: pd.Timestamp, n: int, obs: pd.DataFrame, kmia_obs: pd.DataFrame, meta: Mapping, kmia: Mapping,
          u10: float, v10: float, cfg: Mapping, t_last: float, hrrr_t2m: np.ndarray | None = None,
          hrrr_dir: np.ndarray | None = None) -> dict:
    """{t_adv, b_scale, onset, wind_class} for minutes 1..n after origin.

    obs: station, ts_utc, tmpf, available_at (buoy air temperature goes in tmpf).
    kmia_obs: ts_utc, drct, dwpf, available_at. meta: station -> lat, lon. kmia: lat, lon.
    Fallback order at each minute: upwind stations, HRRR 2 m T, last KMIA temperature.
    """
    speed, direction = wind_speed_dir(u10, v10)
    speed, direction = float(speed), float(direction)
    t_adv = _resolve(obs, origin, upwind_set(direction, cfg), n, kmia, meta, speed, cfg)
    marine = _resolve(obs, origin, cfg["physics"]["upstream"]["marine"], n, kmia, meta, speed, cfg)

    onset = sea_breeze_onset(origin, n, kmia_obs, direction, hrrr_dir, cfg)
    b_scale = np.ones(n)
    if onset is not None:
        after = np.arange(n) >= onset
        t_adv = np.where(after & ~np.isnan(marine), marine, t_adv)
        prior = cfg["physics"]["prior"]
        b_scale[after] = prior["b_onshore"] / prior["b"]

    if hrrr_t2m is not None:
        t_adv = np.where(np.isnan(t_adv), hrrr_t2m, t_adv)
    t_adv = np.where(np.isnan(t_adv), float(t_last), t_adv)
    return {"t_adv": t_adv, "b_scale": b_scale, "onset": onset, "wind_class": wind_class(direction, cfg)}
