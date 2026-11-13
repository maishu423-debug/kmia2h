"""Clear-sky solar geometry and irradiance from pvlib (Ineichen, monthly Linke turbidity)."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pvlib


def clear_sky(index: pd.DatetimeIndex, site: Mapping, linke_turbidity=None) -> pd.DataFrame:
    """GHI (W/m2), zenith, elevation (deg) and is_day (elevation > 0) at UTC times.

    site: lat, lon (deg), elevation_m. linke_turbidity overrides the pvlib monthly climatology.
    """
    loc = pvlib.location.Location(site["lat"], site["lon"], altitude=site["elevation_m"], tz="UTC")
    pos = loc.get_solarposition(index, method="nrel_numpy")
    linke = (pvlib.clearsky.lookup_linke_turbidity(index, loc.latitude, loc.longitude)
             if linke_turbidity is None else linke_turbidity)
    cs = loc.get_clearsky(index, model="ineichen", solar_position=pos, linke_turbidity=linke)
    return pd.DataFrame({
        "ghi": cs["ghi"].to_numpy(),
        "zenith": pos["apparent_zenith"].to_numpy(),
        "elevation": pos["apparent_elevation"].to_numpy(),
        "is_day": (pos["apparent_elevation"] > 0).to_numpy(),
    }, index=index)


# ---- Main-repo compatibility -----------------------------------------------------------------
# features.py and data/qc.py already call solar.compute(index) and solar.cached(start, end). These
# keep working when this file replaces the main repo's solar.py. The framework itself only calls
# clear_sky(), and the imports below only resolve inside the main repo.

def location() -> pvlib.location.Location:
    from kmia2h.config import CFG
    from kmia2h.data import stations
    s = stations.load()[CFG.station]
    return pvlib.location.Location(s["lat"], s["lon"], altitude=s["elevation_m"], tz="UTC")


def compute(index: pd.DatetimeIndex, site: Mapping | None = None) -> pd.DataFrame:
    if site is None:
        loc = location()
        site = {"lat": loc.latitude, "lon": loc.longitude, "elevation_m": loc.altitude}
    return clear_sky(index, site)


def cached(start, end) -> pd.DataFrame:
    """Solar table on the 1-minute grid, cached to store/solar_1min.parquet and extended as needed."""
    from kmia2h.config import store
    p = store("solar_1min.parquet")
    grid = pd.date_range(start, end, freq="min", tz="UTC", name="ts_utc")
    if p.exists():
        d = pd.read_parquet(p)
        if d.index.min() <= grid[0] and d.index.max() >= grid[-1]:
            return d.loc[grid[0]:grid[-1]]
    d = compute(grid)
    d.to_parquet(p)
    return d
