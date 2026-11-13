"""Point-in-time features (SPEC.md Phase 2 step 9). Every value at origin t is an as-of join on
available_at <= t. KMIA 1-minute data is never used here."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kmia2h.config import CFG, store

SKY_RANK = {"CLR": 0, "SKC": 0, "NCD": 0, "FEW": 0, "SCT": 1, "BKN": 2, "OVC": 3, "VV": 3}
SKY_NAME = {0: "clear", 1: "sct", 2: "bkn", 3: "ovc"}
RAIN_CODES = ("RA", "TS", "SH")


def sky_class(skyc: pd.DataFrame) -> pd.Series:
    """Dominant cover of skyc1-4: CLR/FEW -> clear, SCT -> sct, BKN -> bkn, OVC/VV -> ovc."""
    rank = pd.DataFrame({c: skyc[c].astype("string").str.strip().str.upper().map(SKY_RANK) for c in skyc.columns})
    r = rank.max(axis=1)
    return r.map(SKY_NAME).astype(object).where(r.notna(), "unk")


def sector(drct, sknt=None) -> np.ndarray:
    """onshore 30-150, offshore 210-330, other (including calm / variable / missing)."""
    f = CFG.features
    d = np.asarray(drct, float)
    on = (d >= f.sector_onshore[0]) & (d <= f.sector_onshore[1])
    off = (d >= f.sector_offshore[0]) & (d <= f.sector_offshore[1])
    out = np.where(on, "onshore", np.where(off, "offshore", "other")).astype(object)
    if sknt is not None:
        out = np.where(np.asarray(sknt, float) == 0, "other", out)
    return out


def bucket(x, edges, labels) -> np.ndarray:
    x = np.asarray(x, float)
    i = np.searchsorted(np.asarray(edges, float), x, side="right")
    out = np.array(labels, dtype=object)[i]
    return np.where(np.isnan(x), "unk", out)


def wind_bucket(sknt) -> np.ndarray:
    e = CFG.features.wind_buckets_kt
    return bucket(sknt, e, [f"lt{e[0]}", f"{e[0]}to{e[1]}", f"{e[1]}to{e[2]}", f"gt{e[2]}"])


def dd_bucket(dd) -> np.ndarray:
    e = CFG.features.dd_buckets_f
    return bucket(dd, e, [f"lt{e[0]}", f"{e[0]}to{e[1]}", f"{e[1]}to{e[2]}", f"gt{e[2]}"])


def season(ts: pd.DatetimeIndex) -> np.ndarray:
    lst = ts + pd.Timedelta(hours=CFG.tz_lst_offset_hours)
    return np.where(np.isin(lst.month, CFG.features.wet_months), "wet", "dry").astype(object)


def _asof(origins: pd.DataFrame, d: pd.DataFrame, cols: list[str], prefix: str = "") -> pd.DataFrame:
    d = d.sort_values("available_at")[["available_at", *cols]].rename(columns={c: prefix + c for c in cols})
    return pd.merge_asof(origins, d, left_on="origin", right_on="available_at", direction="backward").drop(
        columns="available_at")


class Features:
    def __init__(self, obs: pd.DataFrame | None = None, ndbc: pd.DataFrame | None = None,
                 extra: dict[str, pd.DataFrame] | None = None):
        self.obs = pd.read_parquet(store("obs.parquet")) if obs is None else obs
        if ndbc is None:
            p = store("ndbc.parquet")
            ndbc = pd.read_parquet(p) if p.exists() else None
        self.ndbc = ndbc
        self.extra = extra or {}

    def features_at(self, origins) -> pd.DataFrame:
        origins = pd.DatetimeIndex(origins)
        o = pd.DataFrame({"origin": origins}).sort_values("origin").reset_index(drop=True)
        o["origin"] = o["origin"].astype("datetime64[ns, UTC]")
        k = self.obs[self.obs.station == CFG.station].copy()
        k["available_at"] = k["available_at"].astype("datetime64[ns, UTC]")
        k["ts_utc"] = k["ts_utc"].astype("datetime64[ns, UTC]")
        kt = k[k.tmpf.notna()].sort_values("available_at")
        # Prefer the higher-precision METAR when two reports share a valid minute.
        kt = kt.sort_values(["ts_utc", "report_type"]).drop_duplicates("ts_utc", keep="last")

        f = _asof(o, kt.assign(t_ts=kt.ts_utc), ["tmpf", "t_ts", "report_type", "synthetic"])
        f = f.rename(columns={"tmpf": "t_last", "report_type": "t_last_type", "synthetic": "t_last_synth"})
        f["t_age_min"] = (f.origin - f.t_ts).dt.total_seconds() / 60.0
        # Trends: value at (t_ts - 30 / 60 min), as of the same availability.
        series = kt.set_index("ts_utc")["tmpf"].sort_index()
        for m in (30, 60):
            back = pd.DataFrame({"ts": f.t_ts - pd.Timedelta(minutes=m)}).reset_index()
            ok = back.ts.notna()
            s = series.reset_index().rename(columns={"ts_utc": "ts", "tmpf": "v"})
            s["ts"] = s["ts"].astype("datetime64[ns, UTC]")
            got = pd.merge_asof(back[ok].sort_values("ts"), s, on="ts", direction="backward",
                                tolerance=pd.Timedelta(minutes=10)).set_index("index")["v"]
            f[f"trend_{m}"] = f.t_last - got.reindex(f.index)
        kd = k[k.dwpf.notna()]
        f = _asof(f, kd, ["dwpf"])
        f["dd"] = f.t_last - f.dwpf
        kw = k[k.sknt.notna()]
        f = _asof(f, kw, ["sknt", "drct"])
        f["sector"] = sector(f.drct, f.sknt)
        ks = k[k.skyc1.notna()].copy()
        ks["sky_class"] = sky_class(ks[["skyc1", "skyc2", "skyc3", "skyc4"]])
        f = _asof(f, ks, ["sky_class", "skyc1", "skyl1"])
        f["sky_class"] = f.sky_class.fillna("unk")
        # dominant raw cover code, used by the METAR cloud fallback
        rank = {v: k_ for k_, v in SKY_NAME.items()}
        f["sky_code"] = f.sky_class.map({"clear": "CLR", "sct": "SCT", "bkn": "BKN", "ovc": "OVC"}).fillna("")
        wx = k["wxcodes"].astype("string").fillna("")
        rain = wx.str.contains("|".join(RAIN_CODES), regex=True) | (k.p01i.fillna(0) > 0.0001)
        kr = k.assign(rain_ts=k.ts_utc.where(rain))
        kr = kr[kr.rain_ts.notna()]
        f = _asof(f, kr, ["rain_ts"])
        f["rain_recent"] = (f.origin - f.rain_ts) <= pd.Timedelta(minutes=CFG.features.rain_recent_min)
        f = f.drop(columns=["rain_ts"])
        f["avail_kmia_obs"] = f.t_age_min <= CFG.features.max_t_age_min

        # Neighbours and buoys.
        for st in CFG.neighbours:
            n = self.obs[(self.obs.station == st) & self.obs.tmpf.notna()]
            if n.empty:
                f[f"nb_{st}"] = np.nan
                continue
            f = _asof(f, n.assign(ts=n.ts_utc), ["tmpf", "ts"], prefix=f"nb_{st}_")
            age = (f.origin - f[f"nb_{st}_ts"]).dt.total_seconds() / 60
            f[f"nb_{st}"] = f[f"nb_{st}_tmpf"].where(age <= CFG.features.max_t_age_min)
            f = f.drop(columns=[f"nb_{st}_tmpf", f"nb_{st}_ts"])
        f["avail_neighbours"] = f[[f"nb_{s}" for s in CFG.neighbours]].notna().any(axis=1)
        if self.ndbc is not None and len(self.ndbc):
            for st in CFG.buoys:
                b = self.ndbc[(self.ndbc.station == st) & self.ndbc.atmp_f.notna()]
                if b.empty:
                    f[f"buoy_{st}"] = np.nan
                    continue
                f = _asof(f, b.assign(ts=b.ts_utc), ["atmp_f", "wtmp_f", "ts"], prefix=f"by_{st}_")
                age = (f.origin - f[f"by_{st}_ts"]).dt.total_seconds() / 60
                f[f"buoy_{st}"] = f[f"by_{st}_atmp_f"].where(age <= CFG.features.max_t_age_min)
                f[f"sst_{st}"] = f[f"by_{st}_wtmp_f"].where(age <= CFG.features.max_t_age_min)
                f = f.drop(columns=[f"by_{st}_atmp_f", f"by_{st}_wtmp_f", f"by_{st}_ts"])
            f["avail_ndbc"] = f[[c for c in f.columns if c.startswith("buoy_")]].notna().any(axis=1)
        else:
            f["avail_ndbc"] = False

        for name, d in self.extra.items():
            cols = [c for c in d.columns if c not in ("available_at", "ts_utc", "cycle_utc", "valid_utc")]
            f = _asof(f, d, cols, prefix=f"{name}_")
            f[f"avail_{name}"] = f[f"{name}_{cols[0]}"].notna()

        # Calendar and sun.
        from kmia2h.physics import solar
        idx = pd.DatetimeIndex(f.origin)
        s0 = solar.compute(idx)
        s1 = solar.compute(idx + pd.Timedelta(minutes=CFG.sim.horizon_min))
        f["sun_elev"] = s0["elevation"].to_numpy()
        f["sun_elev_end"] = s1["elevation"].to_numpy()
        f["is_day"] = s0["is_day"].to_numpy()
        f["season"] = season(idx)
        lst = idx + pd.Timedelta(hours=CFG.tz_lst_offset_hours)
        hr = lst.hour + lst.minute / 60.0
        f["hour_lst"] = lst.hour
        f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
        doy = lst.dayofyear
        f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        f["wind"] = wind_bucket(f.sknt)
        f["dd_bucket"] = dd_bucket(f.dd)
        return f.set_index("origin")


def features_at(origins) -> pd.DataFrame:
    return Features().features_at(origins)


def build_grids(start=None, end=None) -> None:
    start = start or CFG.periods.station_start
    end = end or CFG.periods.validation[1]
    fe = Features()
    for step, name in ((CFG.sim.origin_step_floor_min, "features_10m.parquet"),
                       (CFG.sim.origin_step_backtest_min, "features_30m.parquet")):
        idx = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1),
                            freq=f"{step}min", inclusive="left")
        fe.features_at(idx).to_parquet(store(name))
