"""KMIA 1-minute (NCEI via IEM) -> raw monthly chunks -> store/kmia_1min.parquet."""
from __future__ import annotations

import gzip
import sys

import numpy as np
import pandas as pd

from kmia2h.config import CFG, path, store
from kmia2h.data.iem import fetch_chunk, month_chunks

VARS = ["tmpf", "dwpf", "sknt", "drct", "precip"]


def download(start: str, end: str) -> None:
    for a, b in month_chunks(start, end):
        params = [("station", CFG.iem_station), *[("vars", v) for v in VARS], ("sample", "1min"),
                  ("what", "download"), ("tz", "UTC")]
        fetch_chunk("1min", "/cgi-bin/request/asos1min.py", params, f"{CFG.iem_station}_{a:%Y%m}", a, b)
        print("1min", a.strftime("%Y-%m"), flush=True)


def build() -> pd.DataFrame:
    files = sorted(path("raw", "iem", "1min").glob("*.csv.gz"))
    parts = []
    for f in files:
        d = pd.read_csv(f, na_values=["M", ""], low_memory=False)
        if d.empty:
            continue
        d = d.rename(columns={"valid(UTC)": "ts_utc"})
        d["ts_utc"] = pd.to_datetime(d["ts_utc"], utc=True)
        parts.append(d[["ts_utc", *VARS]])
    d = pd.concat(parts).drop_duplicates("ts_utc").set_index("ts_utc").sort_index()
    for v in VARS:
        d[v] = pd.to_numeric(d[v], errors="coerce")
    grid = pd.date_range(d.index.min().floor("D"), d.index.max().ceil("D") - pd.Timedelta(minutes=1),
                         freq="min", tz="UTC", name="ts_utc")
    d = d.reindex(grid)
    from kmia2h.data.qc import flags
    d["qc"] = flags(d)
    d.to_parquet(store("kmia_1min.parquet"))
    return d


if __name__ == "__main__":
    if sys.argv[1] == "download":
        download(sys.argv[2], sys.argv[3])
    else:
        build()
