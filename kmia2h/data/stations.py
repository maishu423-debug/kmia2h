"""Station metadata -> params/stations.json. ASOS from the IEM FL_ASOS network table, buoys from the
NDBC station table. Nothing is hard-coded."""
from __future__ import annotations

import json
import re

import requests

from kmia2h.config import CFG, path


def fetch() -> dict:
    out = {}
    d = requests.get(f"{CFG.iem.base}/api/1/network/FL_ASOS.json", timeout=60).json()["data"]
    want = {s[1:]: s for s in [CFG.station, *CFG.neighbours]}
    for r in d:
        if r["id"] in want:
            out[want[r["id"]]] = {"lat": r["latitude"], "lon": r["longitude"], "elevation_m": r["elevation"],
                                  "name": r["name"], "kind": "asos"}
    t = requests.get("https://www.ndbc.noaa.gov/data/stations/station_table.txt", timeout=60).text
    for line in t.splitlines():
        parts = line.split("|")
        if parts and parts[0].upper() in CFG.buoys:
            m = re.match(r"\s*([\d.]+)\s*([NS])\s+([\d.]+)\s*([EW])", parts[6])
            lat = float(m.group(1)) * (1 if m.group(2) == "N" else -1)
            lon = float(m.group(3)) * (1 if m.group(4) == "E" else -1)
            out[parts[0].upper()] = {"lat": lat, "lon": lon, "elevation_m": 0.0, "name": parts[4], "kind": "buoy"}
    missing = {CFG.station, *CFG.neighbours, *CFG.buoys} - set(out)
    if missing:
        raise RuntimeError(f"no metadata for {missing}")
    path("params", "stations.json").write_text(json.dumps(out, indent=1))
    return out


def load() -> dict:
    p = path("params", "stations.json")
    if not p.exists():
        return fetch()
    return json.loads(p.read_text())


if __name__ == "__main__":
    print(json.dumps(fetch(), indent=1))
