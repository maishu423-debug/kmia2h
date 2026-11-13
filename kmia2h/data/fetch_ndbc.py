"""NDBC buoys (VAKF1, FWYF1) -> store/ndbc.parquet (station, ts_utc, atmp_f, wtmp_f, wspd, wdir,
available_at). FWYF1 stopped reporting in 2022; a missing year is skipped and logged."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from kmia2h.config import CFG, path, store

MISSING = {"ATMP": 999.0, "WTMP": 999.0, "WSPD": 99.0, "WDIR": 999.0}


def fetch_year(st: str, year: int) -> pd.DataFrame | None:
    f = path("raw", "ndbc", f"{st.lower()}h{year}.txt.gz")
    if not f.exists():
        r = requests.get(f"https://www.ndbc.noaa.gov/data/historical/stdmet/{st.lower()}h{year}.txt.gz", timeout=120)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(r.content)
    d = pd.read_csv(f, sep=r"\s+", skiprows=[1], compression="gzip")
    d = d.rename(columns={"#YY": "YY"})
    ts = pd.to_datetime(dict(year=d.YY, month=d.MM, day=d.DD, hour=d.hh, minute=d.mm), utc=True)
    o = pd.DataFrame({"station": st, "ts_utc": ts})
    for c, name in (("ATMP", "atmp_f"), ("WTMP", "wtmp_f")):
        v = d[c].astype(float).where(d[c].astype(float) < MISSING[c])
        o[name] = v * 9 / 5 + 32
    o["wspd"] = d["WSPD"].astype(float).where(d["WSPD"].astype(float) < MISSING["WSPD"])
    o["wdir"] = d["WDIR"].astype(float).where(d["WDIR"].astype(float) < MISSING["WDIR"])
    return o


def main(first=2016, last=2024) -> tuple[pd.DataFrame, list]:
    parts, missing = [], []
    for st in CFG.buoys:
        for y in range(first, last + 1):
            o = fetch_year(st, y)
            if o is None:
                missing.append(f"{st} {y}")
                continue
            parts.append(o)
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["station", "ts_utc"]).sort_values(["station", "ts_utc"])
    d["available_at"] = d.ts_utc + pd.Timedelta(minutes=CFG.latency_min.ndbc)
    d.to_parquet(store("ndbc.parquet"), index=False)
    return d, missing


if __name__ == "__main__":
    d, m = main()
    print(d.groupby("station").agg(n=("ts_utc", "size"), first=("ts_utc", "min"), last=("ts_utc", "max"),
                                   atmp=("atmp_f", lambda x: x.notna().mean())))
    print("missing", m)
