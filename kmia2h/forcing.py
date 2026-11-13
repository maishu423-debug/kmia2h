"""Forcing for the ODE from live streams only (SPEC.md Phase 4 / INTERFACES.md).

`Inputs` holds everything a live run could see for one period. `window_forcing` builds the
120-minute forecast forcing for one origin with the framework modules; `filter_forcing` builds the
continuous one-minute forcing that drives the filter. Both read only rows with available_at <= the
minute they are valid for, and neither touches KMIA 1-minute data.

Until HRRR is downloaded (build step 4) the "HRRR 10 m wind" used for the upwind set and the lag
is KMIA's latest METAR wind (DECISIONS.md)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kmia2h.config import CFG, path, store
from kmia2h.data import stations
from kmia2h.features import SKY_RANK, SKY_NAME, season, sector, sky_class, wind_bucket
from kmia2h.physics import cloud_advect, ode, solar, upstream

KT_TO_MS = 0.514444
COVER = {"clear": "CLR", "sct": "SCT", "bkn": "BKN", "ovc": "OVC", "unk": ""}


def uv_from(sknt, drct):
    s = np.asarray(sknt, float) * KT_TO_MS
    d = np.radians(np.asarray(drct, float))
    return -s * np.sin(d), -s * np.cos(d)


def load_ode_prior() -> dict | None:
    p = path("params", "ode_prior.json")
    return json.loads(p.read_text()) if p.exists() else None


def model_cfg() -> dict:
    """CFG with the batch-fit prior and bounds (params/ode_prior.json) and params/filter.json
    overrides in place of the PLACEHOLDER values."""
    import copy
    cfg = copy.deepcopy(CFG)
    pr = load_ode_prior()
    if pr is not None:
        cfg["physics"]["prior"].update({k: pr["pooled"][k] for k in ("a", "b", "c", "kappa", "lam")})
        cfg["physics"]["prior"]["b_onshore"] = pr.get("b_onshore", cfg["physics"]["prior"]["b_onshore"])
        cfg["physics"]["bounds"].update({k: v for k, v in pr["bounds"].items()})
        if "kappa_thin" in pr:
            cfg["physics"]["cloud"]["kappa_thin"] = pr["kappa_thin"]
            cfg["physics"]["cloud"]["kappa_thick"] = pr["kappa_thick"]
    rp = path("params", "residual.json")
    if rp.exists():
        # REBUILD step 3: the step check uses the 99.9th percentile of fitted jumps (warning only)
        cfg["simulate"]["max_step_f"] = float(json.loads(rp.read_text()).get("step_check_f", cfg["simulate"]["max_step_f"]))
    fp = path("params", "filter.json")
    if fp.exists():
        f = json.loads(fp.read_text())
        for k, v in f.get("filter", {}).items():
            if isinstance(v, dict):
                cfg["filter"][k].update(v)
            else:
                cfg["filter"][k] = v
        for k, v in f.get("simulate", {}).items():
            cfg["simulate"][k] = v
    return cfg


def group_scales(prior: dict | None, is_day: np.ndarray, sect: str) -> dict:
    """Per-minute coefficient ratios (group fit / pooled fit) for day/night x sector."""
    n = len(is_day)
    out = {k: np.ones(n) for k in ("a_scale", "b_scale", "c_scale", "kappa_scale")}
    if prior is None:
        return out
    pooled = prior["pooled"]
    for dn in (True, False):
        g = prior["groups"].get(f"{'day' if dn else 'night'}|{sect}")
        if g is None:
            continue
        m = np.asarray(is_day) == dn
        for k in ("a", "b", "c", "kappa"):
            if pooled[k] > 0:
                out[f"{k}_scale"][m] = g[k] / pooled[k]
    return out


@dataclass
class Inputs:
    start: pd.Timestamp
    end: pd.Timestamp
    obs: pd.DataFrame = field(repr=False)
    up: pd.DataFrame = field(repr=False)          # neighbours + buoys: station, ts_utc, tmpf, available_at
    kmia: pd.DataFrame = field(repr=False)        # KMIA reports
    meta: dict = field(repr=False)
    sol: pd.DataFrame = field(repr=False)
    extra: dict = field(default_factory=dict, repr=False)   # goes / mrms / hrrr tables when present

    @classmethod
    def load(cls, start, end, obs=None, ndbc=None, extra=None):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        lo, hi = start - pd.Timedelta(hours=8), end + pd.Timedelta(hours=3)
        if obs is None:
            obs = pd.read_parquet(store("obs.parquet"), filters=[("ts_utc", ">=", lo), ("ts_utc", "<=", hi)])
        obs = obs[(obs.ts_utc >= lo) & (obs.ts_utc <= hi)]
        if ndbc is None and store("ndbc.parquet").exists():
            ndbc = pd.read_parquet(store("ndbc.parquet"), filters=[("ts_utc", ">=", lo), ("ts_utc", "<=", hi)])
        nb = obs[(obs.station != CFG.station) & obs.tmpf.notna()][["station", "ts_utc", "tmpf", "available_at"]]
        parts = [nb]
        if ndbc is not None and len(ndbc):
            b = ndbc[ndbc.atmp_f.notna()].rename(columns={"atmp_f": "tmpf"})[["station", "ts_utc", "tmpf", "available_at"]]
            parts.append(b)
        up = pd.concat(parts, ignore_index=True)
        for c in ("ts_utc", "available_at"):
            up[c] = up[c].astype("datetime64[ns, UTC]")
        k = obs[obs.station == CFG.station].copy()
        for c in ("ts_utc", "available_at"):
            k[c] = k[c].astype("datetime64[ns, UTC]")
        k = k.sort_values(["available_at", "ts_utc", "report_type"]).reset_index(drop=True)
        k["sky_class"] = sky_class(k[["skyc1", "skyc2", "skyc3", "skyc4"]])
        k.loc[k.skyc1.isna(), "sky_class"] = np.nan
        meta = stations.load()
        grid = pd.date_range(lo, hi, freq="min", tz="UTC")
        sol = solar.clear_sky(grid, meta[CFG.station])
        return cls(start, end, obs, up, k, meta, sol, extra or {})

    # ---- as-of helpers ------------------------------------------------------------------
    def asof(self, times: pd.DatetimeIndex, col: str, kinds: tuple | None = None) -> pd.DataFrame:
        """Latest KMIA report with `col` present, as of each time (available_at <= t)."""
        k = self.kmia[self.kmia[col].notna()]
        if kinds is not None:
            k = k[k.report_type.isin(kinds)]
        t = pd.DataFrame({"t": pd.DatetimeIndex(times).astype("datetime64[ns, UTC]")})
        r = pd.merge_asof(t, k[["available_at", "ts_utc", col]], left_on="t", right_on="available_at",
                          direction="backward")
        return r.set_index("t")

    def state_at(self, origin: pd.Timestamp) -> dict:
        """Live scalars at an origin: last T, dewpoint, wind, sky."""
        o = pd.DatetimeIndex([origin])
        t = self.asof(o, "tmpf").iloc[0]
        td = self.asof(o, "dwpf").iloc[0]
        w = self.asof(o, "sknt").iloc[0]
        dr = self.asof(o, "drct").iloc[0]
        sk = self.asof(o, "sky_class").iloc[0]
        u, v = uv_from(w.sknt if pd.notna(w.sknt) else 0.0, dr.drct if pd.notna(dr.drct) else 0.0)
        return {"t_last": float(t.tmpf), "t_ts": t.ts_utc, "dwpf": float(td.dwpf) if pd.notna(td.dwpf) else float(t.tmpf) - 5,
                "sknt": float(w.sknt) if pd.notna(w.sknt) else np.nan, "drct": float(dr.drct) if pd.notna(dr.drct) else np.nan,
                "u10": float(u), "v10": float(v), "sky_class": sk.sky_class if isinstance(sk.sky_class, str) else "unk"}


def regime_of(inp: Inputs, origin: pd.Timestamp, st: dict | None = None) -> dict:
    st = st or inp.state_at(origin)
    o = pd.DatetimeIndex([origin])
    is_day = bool(inp.sol["is_day"].reindex(o).iloc[0])
    return {"season": season(o)[0], "is_day": is_day, "sky_class": st["sky_class"],
            "wind": wind_bucket([st["sknt"]])[0], "sector": sector([st["drct"]], [st["sknt"]])[0]}


def apply_scales(f: dict, is_day: np.ndarray, sect: str, prior: dict | None = None) -> dict:
    """Raw forcing -> model forcing: multiply in the day/night x sector coefficient ratios."""
    prior = load_ode_prior() if prior is None else prior
    sc = group_scales(prior, is_day, sect)
    g = dict(f)
    g["s_clr"] = np.asarray(f["s_clr"], float) * sc["a_scale"]
    g["b_scale"] = np.asarray(f["b_scale"], float) * sc["b_scale"]
    g["kappa_scale"] = np.asarray(f["kappa_scale"], float) * sc["kappa_scale"]
    g["c_scale"] = np.asarray(f.get("c_scale", 1.0), float) * sc["c_scale"]
    return g


def window_forcing(inp: Inputs, origin: pd.Timestamp, cfg: dict, n: int | None = None, terms_off: tuple = (),
                   st: dict | None = None, scales: bool = True) -> dict:
    """Forcing for minutes origin+1 .. origin+n. terms_off drops forcing terms for the ablation:
    'cloud' (c = 0), 'advection' (b = 0), 'longwave' (c coefficient = 0), 'rain' (r = 0)."""
    n = n or CFG.sim.horizon_min
    st = st or inp.state_at(origin)
    valid = pd.date_range(origin + pd.Timedelta(minutes=1), periods=n, freq="min", tz="UTC")
    sol = inp.sol.reindex(valid)
    kmia_obs = inp.kmia[["ts_utc", "drct", "dwpf", "available_at"]]
    hr = inp.extra.get("hrrr_window")
    hrrr_t2m = hrrr_dir = None
    u10, v10 = st["u10"], st["v10"]
    if hr is not None:
        h = hr(origin, valid)
        if h is not None:
            hrrr_t2m, hrrr_dir = h.get("t2m"), h.get("dir")
            if np.isfinite(h.get("u0", np.nan)):
                u10, v10 = h["u0"], h["v0"]
    up = upstream.build(origin, n, inp.up, kmia_obs, inp.meta, inp.meta[CFG.station], u10, v10, cfg,
                        st["t_last"], hrrr_t2m, hrrr_dir if inp.extra.get("sea_breeze_hrrr") else None)
    if not inp.extra.get("sea_breeze", False):
        up["b_scale"] = np.ones(n)
    goes_row = inp.extra.get("goes_row")(origin) if inp.extra.get("goes_row") else None
    tcdc = tcdc0 = None
    if hr is not None and h is not None and h.get("tcdc") is not None:
        tcdc, tcdc0 = h["tcdc"], h.get("tcdc0")
    cl = cloud_advect.build(valid, goes_row, u10, v10, bool(sol["is_day"].iloc[0]), COVER.get(st["sky_class"], ""),
                            cfg, tcdc, tcdc0)
    c = np.nan_to_num(cl["c"], nan=0.0)
    is_day = sol["is_day"].to_numpy(bool)
    # kappa_rel: the cloud class's kappa relative to the thick-cloud kappa the batch fit estimates.
    kappa_rel = cl["kappa_scale"] * cfg["physics"]["prior"]["kappa"] / cfg["physics"]["cloud"]["kappa_thick"]
    f = {"s_clr": sol["ghi"].to_numpy(float), "c": c, "t_adv": up["t_adv"],
         "t_sky": np.full(n, float(ode.sky_temperature_f(st["t_last"], st["dwpf"], cfg))),
         "r": np.zeros(n), "b_scale": np.asarray(up["b_scale"], float), "kappa_scale": kappa_rel,
         "c_scale": np.ones(n)}
    if scales:
        # ratios follow the origin's group for the whole window, as in the batch fit
        f = apply_scales(f, np.full(n, bool(is_day[0])), regime_of(inp, origin, st)["sector"])
    rain_cells = None
    if inp.extra.get("mrms_row"):
        from kmia2h.physics import rain
        row = inp.extra["mrms_row"](origin)
        rain_cells = rain.find_cells(row, n, cfg) if row is not None else None
        if rain_cells is not None and len(rain_cells):
            f["r"] = rain.rate(rain_cells, n, cfg["physics"]["prior"]["b"], cfg)
    if "cloud" in terms_off:
        f["c"] = np.zeros(n)
    if "advection" in terms_off:
        f["b_scale"] = np.zeros(n)
    if "longwave" in terms_off:
        f["c_scale"] = np.zeros(n)
    if "rain" in terms_off:
        f["r"] = np.zeros(n)
    cloud = None
    if goes_row is not None and np.isfinite([cl["u"], cl["v"]]).all():
        cloud = {"cf_disc": float(goes_row.get("cf_disc", np.nan)), "cf_field": cloud_advect.field_from_row(goes_row, cfg),
                 "u": cl["u"], "v": cl["v"], "tcdc": tcdc}
    return {"forcing": f, "state": st, "valid": valid, "wind_class": up["wind_class"], "onset": up["onset"],
            "cloud_source": cl["source"], "cloud": cloud, "rain_cells": rain_cells, "hrrr_t2m": hrrr_t2m}


# ---- continuous forcing for the filter -----------------------------------------------------

def _lagged_station(up: pd.DataFrame, st: str, times: pd.DatetimeIndex, lag_min: np.ndarray, max_age: float):
    d = up[up.station == st].sort_values("ts_utc")
    if d.empty:
        return np.full(len(times), np.nan)
    tsec = d.ts_utc.astype("int64").to_numpy() / 6e10
    asec = d.available_at.astype("int64").to_numpy() / 6e10
    t = times.astype("int64").to_numpy() / 6e10
    # newest valid time available at each minute (latency is constant per station)
    order = np.argsort(asec)
    newest = np.maximum.accumulate(tsec[order])
    i = np.searchsorted(asec[order], t, side="right") - 1
    ts_max = np.where(i >= 0, newest[np.maximum(i, 0)], np.nan)
    target = np.minimum(t - lag_min, ts_max)
    v = np.interp(target, tsec, d.tmpf.to_numpy(float))
    v[(target < tsec[0]) | np.isnan(ts_max) | (t - ts_max > max_age)] = np.nan
    return v


def filter_forcing(inp: Inputs, minutes: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """One row per filter minute: s_clr, c, t_adv, t_sky, r, b_scale, kappa_scale, a_scale, c_scale,
    plus the regime columns season, is_day, sky_class, wind, sector and t_last."""
    n = len(minutes)
    t = inp.asof(minutes, "tmpf")
    td = inp.asof(minutes, "dwpf")
    w = inp.asof(minutes, "sknt")
    dr = inp.asof(minutes, "drct")
    sk = inp.asof(minutes, "sky_class")
    sknt, drct = w.sknt.to_numpy(float), dr.drct.to_numpy(float)
    u, v = uv_from(np.nan_to_num(sknt), np.nan_to_num(drct))
    if inp.extra.get("hrrr_wind_series") is not None:
        hu, hv = inp.extra["hrrr_wind_series"](minutes)
        u, v = np.where(np.isfinite(hu), hu, u), np.where(np.isfinite(hv), hv, v)
    speed, direction = upstream.wind_speed_dir(u, v)
    sol = inp.sol.reindex(minutes)
    kmia = inp.meta[CFG.station]
    # upwind class per minute
    classes = np.array([upstream.wind_class(d, cfg) for d in direction], dtype=object)
    t_adv = np.full(n, np.nan)
    u_cfg = cfg["physics"]["upstream"]
    for cls in ("onshore", "offshore", "northerly", "other"):
        m = classes == cls
        if not m.any():
            continue
        sets = ({"primary": list(cfg["neighbours"]), "fallback": []} if cls == "other" else dict(u_cfg["sets"][cls]))
        for which in ("primary", "fallback"):
            names = [s for s in sets.get(which, []) if s in inp.meta]
            if not names:
                continue
            num, den = np.zeros(n), np.zeros(n)
            for s in names:
                dist = float(upstream.distance_km(kmia["lat"], kmia["lon"], inp.meta[s]["lat"], inp.meta[s]["lon"]))
                lag = upstream.lag_minutes(dist, speed, cfg)
                val = _lagged_station(inp.up, s, minutes, lag, u_cfg["max_age_min"])
                wgt = 1.0 / max(dist, 1e-6) ** u_cfg["idw_power"]
                ok = np.isfinite(val)
                num += np.where(ok, wgt * np.nan_to_num(val), 0.0)
                den += np.where(ok, wgt, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                val = np.where(den > 0, num / den, np.nan)
            t_adv = np.where(m & np.isnan(t_adv), val, t_adv)
    t_last = t.tmpf.to_numpy(float)
    t_adv = np.where(np.isnan(t_adv), t_last, t_adv)
    sky = sk.sky_class.astype(object).where(sk.sky_class.notna(), "unk").to_numpy()
    cover = cfg["physics"]["cloud"]["metar_cover"]
    c = np.array([cover.get(COVER.get(s, ""), np.nan) for s in sky], float)
    goes_c = inp.extra.get("goes_c_series")
    if goes_c is not None:
        gc = goes_c(minutes)
        c = np.where(np.isfinite(gc), gc, c)
    is_day = sol["is_day"].to_numpy(bool)
    sect = sector(drct, sknt)
    prior = load_ode_prior()
    sc = {k: np.ones(n) for k in ("a_scale", "b_scale", "c_scale", "kappa_scale")}
    for s in ("onshore", "offshore", "other"):
        m = sect == s
        if m.any():
            g = group_scales(prior, is_day[m], s)
            for k in sc:
                sc[k][m] = g[k]
    tdv = td.dwpf.to_numpy(float)
    tdv = np.where(np.isnan(tdv), t_last - 5.0, tdv)
    f = pd.DataFrame({
        "s_clr": sol["ghi"].to_numpy(float) * sc["a_scale"], "c": c, "t_adv": t_adv,
        "t_sky": ode.sky_temperature_f(t_last, tdv, cfg), "r": np.zeros(n),
        "b_scale": sc["b_scale"], "kappa_scale": sc["kappa_scale"], "c_scale": sc["c_scale"],
        "season": season(minutes), "is_day": is_day, "sky_class": sky, "wind": wind_bucket(sknt), "sector": sect,
        "t_last": t_last}, index=minutes)
    f["t_sky"] = f["t_sky"].fillna(f["t_last"] - 20.0)
    return f
