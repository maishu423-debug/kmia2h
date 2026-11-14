"""NDBC buoys (VAKF1, FWYF1) -> store/ndbc.parquet (station, ts_utc, atmp_f, wtmp_f, wspd, wdir,
available_at). FWYF1 stopped reporting in 2022; a missing year is skipped and logged."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from kmia2h.config import CFG, path, store

MISSING = {"ATMP": 999.0, "WTMP": 999.0, "WSPD": 99.0, "WDIR": 999.0}


def _get(url: str, f) -> bool:
    if f.exists():
        return True
    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(r.content)
    return True


def fetch_year(st: str, year: int) -> pd.DataFrame | None:
    f = path("raw", "ndbc", f"{st.lower()}h{year}.txt.gz")
    if not _get(f"https://www.ndbc.noaa.gov/data/historical/stdmet/{st.lower()}h{year}.txt.gz", f):
        return None
    return parse(f, st)


def fetch_month(st: str, year: int, month: int) -> pd.DataFrame | None:
    """Current-year months not yet in the yearly archive: data/stdmet/<Mon>/<st><m hex><yyyy>.txt.gz."""
    mon = pd.Timestamp(year=year, month=month, day=1).strftime("%b")
    name = f"{st.lower()}{month:x}{year}.txt.gz"
    f = path("raw", "ndbc", "monthly", name)
    return parse(f, st) if _get(f"https://www.ndbc.noaa.gov/data/stdmet/{mon}/{name}", f) else None


def fetch_realtime(st: str, tag: str) -> pd.DataFrame | None:
    """Last 45 days (realtime2), saved once per tag (the download date) so a rebuild is reproducible."""
    f = path("raw", "ndbc", "realtime", f"{st.upper()}_{tag}.txt")
    return parse(f, st) if _get(f"https://www.ndbc.noaa.gov/data/realtime2/{st.upper()}.txt", f) else None


def parse(f, st: str) -> pd.DataFrame:
    d = pd.read_csv(f, sep=r"\s+", skiprows=[1], na_values=["MM"],
                    compression="gzip" if str(f).endswith(".gz") else None)
    d = d.rename(columns={"#YY": "YY"})
    ts = pd.to_datetime(dict(year=d.YY, month=d.MM, day=d.DD, hour=d.hh, minute=d.mm), utc=True)
    o = pd.DataFrame({"station": st, "ts_utc": ts})
    for c, name in (("ATMP", "atmp_f"), ("WTMP", "wtmp_f")):
        v = d[c].astype(float).where(d[c].astype(float) < MISSING[c])
        o[name] = v * 9 / 5 + 32
    o["wspd"] = d["WSPD"].astype(float).where(d["WSPD"].astype(float) < MISSING["WSPD"])
    o["wdir"] = d["WDIR"].astype(float).where(d["WDIR"].astype(float) < MISSING["WDIR"])
    return o


def main(first=2016, last=None) -> tuple[pd.DataFrame, list]:
    """Yearly archives, then monthly files for the current year, then the realtime file. Earlier
    sources win on duplicates. Gaps (e.g. a month not yet archived) stay missing -> avail_ndbc False."""
    end = pd.Timestamp(CFG.periods.data_end)
    last = last or end.year
    parts, missing = [], []
    for st in CFG.buoys:
        for y in range(first, last + 1):
            o = fetch_year(st, y)
            if o is None:
                missing.append(f"{st} {y}")
                if y == end.year:
                    for m in range(1, end.month + 1):
                        om = fetch_month(st, y, m)
                        if om is None:
                            missing.append(f"{st} {y}-{m:02d}")
                        else:
                            parts.append(om)
                continue
            parts.append(o)
        if last == end.year:
            r = fetch_realtime(st, end.strftime("%Y%m%d"))
            if r is not None:
                parts.append(r)
    end_ts = pd.Timestamp(CFG.periods.data_end, tz="UTC") + pd.Timedelta(days=1)
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["station", "ts_utc"]).sort_values(["station", "ts_utc"])
    d = d[d.ts_utc < end_ts]
    d["available_at"] = d.ts_utc + pd.Timedelta(minutes=CFG.latency_min.ndbc)
    d.to_parquet(store("ndbc.parquet"), index=False)
    return d, missing


if __name__ == "__main__":
    d, m = main()
    print(d.groupby("station").agg(n=("ts_utc", "size"), first=("ts_utc", "min"), last=("ts_utc", "max"),
                                   atmp=("atmp_f", lambda x: x.notna().mean())))
    print("missing", m)
